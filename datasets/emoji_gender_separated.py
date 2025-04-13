import pandas as pd

# Load your dataset
# df = pd.read_csv("text2emoji.csv")

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/DESUCLUB/combined_emoji_data/data/train-00000-of-00001.parquet")

# Complete sets of gendered emojis based on Unicode v15
male_emojis = {
    "👨", "👨‍⚕️", "👨‍🎓", "👨‍🏫", "👨‍⚖️", "👨‍🌾", "👨‍🍳", "👨‍🔧", "👨‍🏭", "👨‍💼", "👨‍🔬", "👨‍💻", "👨‍🎤",
    "👨‍🎨", "👨‍✈️", "👨‍🚀", "👨‍🚒", "👮‍♂️", "🕵️‍♂️", "💂‍♂️", "👷‍♂️", "🤴", "👳‍♂️", "👲", "🧔‍♂️", "👱‍♂️",
    "👨‍🦰", "👨‍🦱", "👨‍🦳", "👨‍🦲", "🙍‍♂️", "🙎‍♂️", "🙅‍♂️", "🙆‍♂️", "💁‍♂️", "🙋‍♂️", "🙇‍♂️", "🤦‍♂️",
    "🤷‍♂️", "💆‍♂️", "💇‍♂️", "🚶‍♂️", "🏃‍♂️", "🕺", "🧖‍♂️", "🧘‍♂️", "🛀", "🛌", "🤹‍♂️", "🤸‍♂️", "🚴‍♂️",
    "🚵‍♂️", "🏋️‍♂️", "🤼‍♂️", "🤽‍♂️", "🤾‍♂️", "🤺", "🏌️‍♂️", "🏄‍♂️", "🏊‍♂️", "🏇", "🚣‍♂️",
    "🧗‍♂️", "👬", "💏", "👨‍❤️‍👨", "👨‍❤️‍💋‍👨"
}

female_emojis = {
    "👩", "👩‍⚕️", "👩‍🎓", "👩‍🏫", "👩‍⚖️", "👩‍🌾", "👩‍🍳", "👩‍🔧", "👩‍🏭", "👩‍💼", "👩‍🔬", "👩‍💻", "👩‍🎤",
    "👩‍🎨", "👩‍✈️", "👩‍🚀", "👩‍🚒", "👮‍♀️", "🕵️‍♀️", "💂‍♀️", "👷‍♀️", "👸", "👳‍♀️", "🧕", "👱‍♀️", "👩‍🦰",
    "👩‍🦱", "👩‍🦳", "👩‍🦲", "🙍‍♀️", "🙎‍♀️", "🙅‍♀️", "🙆‍♀️", "💁‍♀️", "🙋‍♀️", "🙇‍♀️", "🤦‍♀️", "🤷‍♀️",
    "💆‍♀️", "💇‍♀️", "🚶‍♀️", "🏃‍♀️", "💃", "🧖‍♀️", "🧘‍♀️", "🛀", "🛌", "🤹‍♀️", "🤸‍♀️", "🚴‍♀️", "🚵‍♀️",
    "🏋️‍♀️", "🤼‍♀️", "🤽‍♀️", "🤾‍♀️", "🤺", "🏌️‍♀️", "🏄‍♀️", "🏊‍♀️", "🏇", "🚣‍♀️", "🧗‍♀️",
    "👭", "💏", "👩‍❤️‍👩", "👩‍❤️‍💋‍👩", "🤰", "🤱", "🤶", "🧙‍♀️", "🧚‍♀️", "🧛‍♀️", "🧜‍♀️", "🧝‍♀️", "🧞‍♀️", "🧟‍♀️"
}

# Classification function explicitly distinguishing male or female
def classify_gender(emoji_sequence):
    for emoji in male_emojis:
        if emoji in emoji_sequence:
            return 'male'
    for emoji in female_emojis:
        if emoji in emoji_sequence:
            return 'female'
    return 'other'  # Exclude neutral explicitly

# Apply gender classification
df['gender'] = df['emoji'].apply(classify_gender)

# Filter clearly only male and female datasets
df_male = df[df['gender'] == 'male']
df_female = df[df['gender'] == 'female']

"""

# Save datasets explicitly with only "text" and "emoji" columns
df_male[['text', 'emoji']].to_csv("./datasets/emoji_male.csv", index=False)
df_female[['text', 'emoji']].to_csv("./datasets/emoji_female.csv", index=False)

"""

df_male[['text', 'emoji']].to_csv("./datasets/emoji_male.csv", index=False)
df_female[['text', 'emoji']].to_csv("./datasets/emoji_female.csv", index=False)
