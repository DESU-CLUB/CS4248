import torch
from datasets import load_dataset
import evaluate
from tqdm import tqdm

def evaluate_ae_pipeline(df, model, tokenizer, device):
    """
    Evaluate the Autoencoder model on BLEU score.
    
    Args:
        df (pd.DataFrame): DataFrame containing the input text.
        model (Autoencoder): The Autoencoder model.
        tokenizer: The tokenizer used for the model.
        device: The device (CPU/GPU) to run the evaluation on.
    
    Returns:
        dict: BLEU score results.
    """
    model.eval()

    predictions = []
    references = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row["text"]

        # Tokenize input text
        tokenized = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)

        # ---- Step 1: Generate Output ----
        with torch.no_grad():
            generated_text_list = model(input_text)  # Forward pass through the Autoencoder
            generated_text = generated_text_list[0]  # Extract the first generated text

        # Append predictions and references
        predictions.append(generated_text.strip())
        references.append(input_text.strip())

    # ---- Step 2: Compute BLEU Score ----
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=[[r] for r in references])

    print("\n📊 Autoencoder BLEU Evaluation:")
    print(f"BLEU: {results['bleu']:.4f}")

    return results

# ==== Load Dataset ====
ds = load_dataset("DESUCLUB/combined_emoji_data")
df = ds["train"].to_pandas()

# Filter out rows with missing or null "text" values
df_clean = df[df["text"].notnull()]

# Sample a subset of the data for evaluation
df_sample = df_clean.sample(n=1000, random_state=42)  # Adjust `n` as needed

# ==== Load Autoencoder Model ====
from AE import Autoencoder  # Import your Autoencoder class
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Instantiate the Autoencoder model
model = Autoencoder(device=device)

# ==== Run Evaluation ====
results = evaluate_ae_pipeline(
    df=df_sample,
    model=model,
    tokenizer=tokenizer,
    device=device
)