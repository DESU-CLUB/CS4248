import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

## Generate short phrase/sentence using ChatGPT given the emoji from ELCo and the corresponding label ##

# Load the ELCo dataset
elco = pd.read_csv("elco/ELCo.csv")
print(elco.head())
# env path
load_dotenv()

client = OpenAI()


# Function to generate a short phrase or sentence from emoji and label
def generate_phrase_from_emoji_and_label(emoji, label):
    # Use the OpenAI API to generate a sentence
    prompt = f"Generate a short phrase or sentence using the emoji: {emoji} and the label concept: {label}. The sentence should be relevant to the emoji and label."

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4" if you prefer
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,  # Limit the length of the generated text
        n=1,  # Generate one sentence
        stop=None,
        temperature=0.7,  # Adjust for more creativity in sentence generation
    )

    # Extract and return the generated sentence
    sentence = response.choices[0].message.content
    print(sentence)
    sentence = sentence.strip()
    return sentence


def hallucinate(dataset):
    # Create a new column for the generated sentences
    dataset = dataset.assign(
        generated_sentence=dataset.apply(
            lambda row: generate_phrase_from_emoji_and_label(row["EM"], row["EN"]),
            axis=1,
        )
    )

    # Save the updated DataFrame to a new CSV file
    dataset.to_csv("ELCo_with_sentences.csv", index=False)

    # Print the first few rows of the updated DataFrame
    dataset.head()


def test_hallucinate():
    elco_test = elco.head()
    hallucinate(elco_test)


if __name__ == "__main__":
    test_hallucinate()
