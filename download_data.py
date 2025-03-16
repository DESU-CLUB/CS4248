from datasets import load_dataset
from huggingface_hub import login

ds = load_dataset("KomeijiForce/Text2Emoji")
print(ds["train"][0]["topic"])

# Add more data to the dataset
# Clean data
