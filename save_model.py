from huggingface_hub import HfApi

# Initialize the Hugging Face API
api = HfApi()

# Upload the file using your specific repo name
api.upload_file(
    path_or_fileobj="trained_llama_decoder_model.pt",  # your local file path
    path_in_repo="trained_decoder_model_llama.pt",  # name of the file in the repo
    repo_id="DESUCLUB/emoji_decoder_elco",  # your specific repo
    repo_type="model",
)

print("Model successfully uploaded to HuggingFace Hub: DESUCLUB/emoji_decoder_elco")
