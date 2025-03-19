# Combine ElCO_adapted_edited and text2emoji datasets
import datasets
from datasets import Dataset

import pandas as pd

elco_adapted_edited = pd.read_csv("datasets/ELCo_adapted_edited.csv")
text2emoji = pd.read_csv("datasets/text2emoji.csv")

# Combine the datasets
combined_df = pd.concat([elco_adapted_edited, text2emoji], ignore_index=True)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(combined_df)

# Save as pyarrow format
dataset.save_to_disk("combined_emoji_data")

# Upload to Hugging Face Hub
dataset.push_to_hub("combined_emoji_data")