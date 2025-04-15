from encoder import Encoder
from decoder import CustomDecoder
from datasets import load_dataset
from tqdm import tqdm

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("DESUCLUB/combined_emoji_data")
df = ds["train"].to_pandas()

"""Without GPT2"""

import sys

CustomDecoder = CustomDecoder

from transformers import BertTokenizer
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
vocab_size = tokenizer.vocab_size

# Import model definitions
from encoder import Encoder
from decoder import CustomDecoder

# Instantiate encoder and decoder (make sure layer config matches trained weights)
encoder = Encoder(voc_size=vocab_size, embed_size=128, device=device).to(device)
decoder = CustomDecoder(
    vocab_size=vocab_size,
    encoder_dim=2048,
    d_model=512,            # match your training setup
    num_decoder_layers=4    # adjust this if needed
).to(device)

# Load trained weights
encoder.load_state_dict(torch.load("trained_encoder_model.pt", map_location=device))
decoder.load_state_dict(torch.load("fixed_decoder.pt", map_location=device))

# Set models to evaluation mode
encoder.eval()
decoder.eval()

# ==== Inference Example ====
text = "The company is making profit!"
tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    encoder_output = encoder(tokens["input_ids"])      # (B, 2048)
    memory = decoder.adapter(encoder_output)           # (B, 512)
    if memory.dim() == 2:
        memory = memory.unsqueeze(1)                   # (B, 1, 512)

    generated_ids = decoder.generate(memory)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Generated:", generated_text)

def evaluate_full_pipeline(df, encoder, decoder, tokenizer, device):
    encoder.eval()
    decoder.eval()

    predictions = []
    references = []


    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row["text"]

        # Tokenize input text
        tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

        # ---- Step 1: Encode ----
        with torch.no_grad():
            encoder_output = encoder(tokenized["input_ids"])      # (B, 2048)
            memory = decoder.adapter(encoder_output)              # (B, 512)
            if memory.dim() == 2:
                memory = memory.unsqueeze(1)                      # (B, 1, 512)

        # ---- Step 2: Decode ----
        with torch.no_grad():
            generated_ids = decoder.generate(memory)              # (B, seq_len)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # print(f"\n📝 Input: {input_text}\n🔁 Output: {generated_text.strip()}")

        predictions.append(generated_text.strip())
        references.append(input_text.strip())

    # ---- Step 3: HF metrics ----
    import evaluate
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    exact = evaluate.load("exact_match")

    results = {
        "BLEU": bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"],
        "METEOR": meteor.compute(predictions=predictions, references=references)["meteor"],
        "ROUGE-L": rouge.compute(predictions=predictions, references=references)["rougeL"],
        "Exact Match": exact.compute(predictions=predictions, references=references)["exact_match"],
    }

    print("\n📊 Full Pipeline Evaluation:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    return results

# Filter out rows with missing or null "text" values
df_clean = df[df["text"].notnull()]

# Then sample
df_sample = df_clean.sample(n=10000, random_state=42)

# Run evaluation
evaluate_full_pipeline(
    df=df_sample,
    encoder=encoder,
    decoder=decoder,
    tokenizer=tokenizer,
    device=device
)

"""with GPT-2"""

""" from google.colab import files
files.upload()

from llm import GPT2Adapter

def evaluate_full_pipeline(df, encoder, decoder, tokenizer, device, gpt2_adapter=None):
    encoder.eval()
    decoder.eval()
    if gpt2_adapter is not None:
        gpt2_adapter.eval()
        # ✅ Add this projection to match decoder expectations
        projection_to_encoder_dim = nn.Linear(768, 2048).to(device)

    predictions = []
    references = []

    from tqdm import tqdm
    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row["text"]

        # Tokenize input text
        tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

        # ---- Step 1: Encode ----
        with torch.no_grad():
            encoder_output = encoder(tokenized["input_ids"])  # (B, 2048)

            if gpt2_adapter is not None:
                x = gpt2_adapter(encoder_output)               # (B, 768)
                x = projection_to_encoder_dim(x)               # ✅ (B, 2048)
                encoder_output = decoder.adapter(x)            # (B, 512)
            else:
                encoder_output = decoder.adapter(encoder_output)

            if encoder_output.dim() == 2:
                encoder_output = encoder_output.unsqueeze(1)   # (B, 1, 512)

        # ---- Step 2: Decode ----
        with torch.no_grad():
            generated_ids = decoder.generate(encoder_output)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # print(f"\n📝 Input: {input_text}\n🔁 Output: {generated_text.strip()}")
        predictions.append(generated_text.strip())
        references.append(input_text.strip())

    # ---- Step 3: HF metrics ----
    import evaluate
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    exact = evaluate.load("exact_match")

    results = {
        "BLEU": bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"],
        "METEOR": meteor.compute(predictions=predictions, references=references)["meteor"],
        "ROUGE-L": rouge.compute(predictions=predictions, references=references)["rougeL"],
        "Exact Match": exact.compute(predictions=predictions, references=references)["exact_match"],
    }

    print("\n📊 Full Pipeline Evaluation:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    return results

import torch.nn as nn

results_with_gpt = evaluate_full_pipeline(
    df=df.sample(n=10000),
    encoder=encoder,
    decoder=decoder,
    tokenizer=tokenizer,
    device=device,
    gpt2_adapter=gpt2_adapter
)

 """