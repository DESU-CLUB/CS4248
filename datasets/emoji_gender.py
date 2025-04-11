import pandas as pd

# Use this script to split a dataset into gendered and non-gendered datasets
# Reference: <emoji>-ELCo: Towards Gender Equality in Emoji Composition
# reference paper about ELCo from John dealing w gender emoji. their dataset is called ELCo-AN but cannot find the dataset

# Load your dataset
df = pd.read_csv("./datasets/ELCo_adapted_edited.csv")  # Edit this to split the combined dataset

gendered_emojis = {
    # Basic gender representations
    "👨", "👩", "🧑",
    "👧", "👦", "🧒",
    "👴", "👵", "🧓",

    # Professions (ZWJ sequences, gendered)
    "👨‍⚕️", "👩‍⚕️", "🧑‍⚕️",
    "👨‍🏫", "👩‍🏫", "🧑‍🏫",
    "👨‍🍳", "👩‍🍳", "🧑‍🍳",
    "👨‍🎓", "👩‍🎓", "🧑‍🎓",
    "👨‍🏭", "👩‍🏭", "🧑‍🏭",
    "👨‍💻", "👩‍💻", "🧑‍💻",
    "👨‍💼", "👩‍💼", "🧑‍💼",
    "👨‍🔧", "👩‍🔧", "🧑‍🔧",
    "👨‍🎤", "👩‍🎤", "🧑‍🎤",
    "👨‍🎨", "👩‍🎨", "🧑‍🎨",
    "👨‍🚀", "👩‍🚀", "🧑‍🚀",
    "👨‍⚖️", "👩‍⚖️", "🧑‍⚖️",
    "👨‍🚒", "👩‍🚒", "🧑‍🚒",
    "👨‍🌾", "👩‍🌾", "🧑‍🌾",
    "👨‍✈️", "👩‍✈️", "🧑‍✈️",

    # Miscellaneous gender-related
    "👱‍♀️", "👱‍♂️",
    "🙎‍♂️", "🙎‍♀️",
    "🙍‍♂️", "🙍‍♀️",
    "🙅‍♂️", "🙅‍♀️",
    "🙆‍♂️", "🙆‍♀️",
    "💁‍♂️", "💁‍♀️",
    "🙋‍♂️", "🙋‍♀️",
    "🙇‍♂️", "🙇‍♀️",
    "🤦‍♂️", "🤦‍♀️",
    "🤷‍♂️", "🤷‍♀️"
}


# Function to detect if a row contains any gendered emoji
def contains_gendered_emoji(emoji_sequence):
    return any(gendered in emoji_sequence for gendered in gendered_emojis)

# Filter
df_gendered = df[df['emoji'].apply(contains_gendered_emoji)]
df_non_gendered = df[~df['emoji'].apply(contains_gendered_emoji)]

# Save results
df_gendered.to_csv("./datasets/emoji_gendered.csv", index=False)
df_non_gendered.to_csv("./datasets/emoji_nongendered.csv", index=False)
