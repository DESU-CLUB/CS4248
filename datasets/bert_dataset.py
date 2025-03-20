import pandas as pd

## Replace emoji column with text while keeping the label for Bert to tokenise ##

# Create the CSV file with only emoji and text
def format_csv():
    df = pd.read_csv("elco_adapted/ELCo_with_sentences.csv")

    # Select only the "generated_sentence" and "EM" columns
    filtered_df = df[["generated_sentence", "Description"]].rename(columns={"generated_sentence": "text"}).rename(columns={"Description": "emoji"})

    # Save the new CSV file
    filtered_df.to_csv("ELCo_bert_adapted.csv", index=False)

    print(filtered_df.head())  # Display the first few rows


if __name__ == "__main__":
    format_csv()    