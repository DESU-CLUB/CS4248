import pandas as pd
import re

# Load your emoji dataset
df = pd.read_csv("text2emoji.csv")

# Skin-tone modifiers and labels
skin_tone_modifiers = {
    '🏻': 'light_skin_tone',
    '🏼': 'medium_light_skin_tone',
    '🏽': 'medium_skin_tone',
    '🏾': 'medium_dark_skin_tone',
    '🏿': 'dark_skin_tone'
}

# Compile regex pattern
skin_tone_pattern = re.compile("|".join(skin_tone_modifiers.keys()))

# General classification
df['skin_tone_general'] = df['emoji'].apply(
    lambda x: 'has_skin_tone' if skin_tone_pattern.search(x) else 'no_skin_tone'
)

# Specific skin-tone classification
for modifier, tone_label in skin_tone_modifiers.items():
    df[tone_label] = df['emoji'].apply(lambda x: modifier in x)

# Create separate DataFrames
has_skin_tone_df = df[df['skin_tone_general'] == 'has_skin_tone']
no_skin_tone_df = df[df['skin_tone_general'] == 'no_skin_tone']

light_skin_tone_df = df[df['light_skin_tone']]
medium_light_skin_tone_df = df[df['medium_light_skin_tone']]
medium_skin_tone_df = df[df['medium_skin_tone']]
medium_dark_skin_tone_df = df[df['medium_dark_skin_tone']]
dark_skin_tone_df = df[df['dark_skin_tone']]

"""
# Columns to save
cols_to_save = ['text', 'emoji']

# Save general classification datasets (only 'text' and 'emoji')
has_skin_tone_df[cols_to_save].to_csv('has_skin_tone_emojis.csv', index=False)
no_skin_tone_df[cols_to_save].to_csv('no_skin_tone_emojis.csv', index=False)

# Save specific skin tone datasets (only 'text' and 'emoji')
light_skin_tone_df[cols_to_save].to_csv('light_skin_tone_emojis.csv', index=False)
medium_light_skin_tone_df[cols_to_save].to_csv('medium_light_skin_tone_emojis.csv', index=False)
medium_skin_tone_df[cols_to_save].to_csv('medium_skin_tone_emojis.csv', index=False)
medium_dark_skin_tone_df[cols_to_save].to_csv('medium_dark_skin_tone_emojis.csv', index=False)
dark_skin_tone_df[cols_to_save].to_csv('dark_skin_tone_emojis.csv', index=False)
"""