import pandas as pd

# Use this script to split a dataset into emotionally dense and non-emotionally dense datasets
# Reference: The Multidimensional Lexicon ofEmojis: A New Tool to Assess the Emotional Content of Emojis 
# doi: 10.3389/fpsyg.2022.921388

# Load your dataset
df = pd.read_csv("./datasets/ELCo_adapted_edited.csv")  # Edit this to split the combined dataset

# Define emotionally dense emojis
emotionally_dense_emojis = {"⚖️", "😡", "🤬", "🤮", "🤢", "👿", "✝️", "⚠️", "🎂", "🎈", "🎁", "🍀"}

# Function to check if any dense emoji is present
def contains_dense_emoji(emoji_str):
    return any(emoji in emoji_str for emoji in emotionally_dense_emojis)

# Apply the function to split the dataset
df_dense = df[df['emoji'].apply(contains_dense_emoji)]
df_not_dense = df[~df['emoji'].apply(contains_dense_emoji)]

# Optionally save or explore
df_dense.to_csv("./datasets/dense_emoji_subset.csv", index=False)
df_not_dense.to_csv("./datasets/non_dense_emoji_subset.csv", index=False)
