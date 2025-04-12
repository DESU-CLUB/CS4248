from datasets import load_dataset
import pandas as pd

ds = pd.read_csv("ELCo_adapted_edited.csv")
df = pd.read_csv("text2emoji.csv")

cm = pd.read_csv("common_emojis.csv")
rr = pd.read_csv("rare_emojis.csv")

common_emojis = set('😂❤️😍🤣😊🙏💕😭😘')

def classify_emoji(emoji_sentence):
    return any(char in common_emojis for char in emoji_sentence)

df['classification'] = df['emoji'].apply(lambda x: 'common' if classify_emoji(x) else 'rare')

common_df = df[df['classification'] == 'common']
rare_df = df[df['classification'] == 'rare']

#common_df.to_csv('common_emojis.csv', index=False)
#rare_df.to_csv('rare_emojis.csv', index=False)
