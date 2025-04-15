from datasets import load_dataset
from openai import AsyncOpenAI
import json
from tqdm.asyncio import tqdm_asyncio
import re
from typing import List, Dict, Tuple, Optional
import asyncio
from pydantic import BaseModel, Field

# Try to import instructor, with a fallback if it's not installed
try:
    import instructor
    has_instructor = True
except ImportError:
    has_instructor = False
    print("Instructor not installed. Installing it...")
    import subprocess
    subprocess.check_call(["pip", "install", "instructor"])
    import instructor
    has_instructor = True

# Define Pydantic models for structured output
class EmojiEmotion(BaseModel):
    """Model representing an emotion associated with an emoji"""
    emotion: str = Field(..., description="The emotion name")
    intensity: int = Field(..., ge=1, le=10, description="Intensity of the emotion on a scale of 1-10")
    explanation: str = Field(..., description="Brief explanation of why this emoji conveys this emotion")

class EmojiAnalysis(BaseModel):
    """Model representing the analysis of an emoji"""
    emoji: str = Field(..., description="The emoji character")
    visual_description: str = Field(..., description="Short visual description of what the emoji depicts")
    top_emotions: List[EmojiEmotion] = Field(..., min_items=3, max_items=3, description="Top 3 emotions associated with this emoji")

# Initialize AsyncOpenAI client
client = AsyncOpenAI()

# Apply instructor patch if available
if has_instructor:
    instructor_client = instructor.patch(client)

# Configure batch size for concurrent API calls
BATCH_SIZE = 10

def load_emoji_dataset():
    """Load the Text2Emoji dataset"""
    return load_dataset("KomeijiForce/Text2Emoji")

def extract_unique_emojis(dataset) -> List[str]:
    """Extract all unique emojis from the dataset"""
    # Debug and print dataset structure first
    print("Dataset keys:", dataset.keys())
    print("Train keys:", dataset["train"].column_names)
    
    # Look for emoji columns first by prioritizing the most likely names
    emoji_column_candidates = ["emoji", "emojis", "label", "output", "target"]
    text_column_candidates = ["text", "input", "prompt", "source"]
    
    emoji_column = None
    # First try to find a dedicated emoji column
    for col in emoji_column_candidates:
        if col in dataset["train"].column_names:
            emoji_column = col
            print(f"Found emoji column: {emoji_column}")
            break
    
    # If we found an emoji column, use it
    if emoji_column:
        all_texts = dataset["train"][emoji_column]
    # If not, look for a text column
    else:
        text_column = None
        for col in text_column_candidates:
            if col in dataset["train"].column_names:
                text_column = col
                print(f"Found text column: {text_column}")
                break
        
        if text_column:
            all_texts = dataset["train"][text_column]
        else:
            # Print available columns and take first one as a fallback
            print("Available columns:", dataset["train"].column_names)
            # Show samples from all columns to help identify the right one
            for col in dataset["train"].column_names:
                sample = dataset["train"][col][0]
                print(f"Sample from {col}:", sample, type(sample))
            
            # Use the first column as fallback
            fallback_col = dataset["train"].column_names[0]
            print(f"Using fallback column: {fallback_col}")
            all_texts = dataset["train"][fallback_col]
    
    # Extract all emojis from all texts
    all_emojis = set()
    
    # Use a regex pattern to find emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+"
    )
    
    count = 0
    for item in all_texts:
        # Make sure we're working with a string
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict) and "text" in item:
            text = item["text"]
        elif isinstance(item, dict) and "emoji" in item:
            text = item["emoji"]
        else:
            # Skip non-string items
            print(f"Skipping non-string item: {type(item)}, value: {item}")
            continue
        
        if not isinstance(text, str):
            print(f"Skipping non-string text: {type(text)}, value: {text}")
            continue
            
        # Find emojis in the text
        emojis = emoji_pattern.findall(text)
        for emoji in emojis:
            # Add each individual emoji character
            for char in emoji:
                all_emojis.add(char)
                count += 1
    
    print(f"Extracted {count} emoji instances, {len(all_emojis)} unique emojis")
    return list(all_emojis)

async def analyze_emoji(emoji: str) -> Dict[str, str]:
    """Get emotion and description for an emoji using GPT asynchronously"""
    try:
        if has_instructor:
            # Use instructor for structured output
            response = await instructor_client.chat.completions.create(
                model="gpt-4o",
                response_model=EmojiAnalysis,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes emojis in detail."},
                    {"role": "user", "content": f"For the emoji {emoji}, provide:\n1. A short visual description of what the emoji depicts\n2. The top 3 emotions it conveys, with intensity ratings and brief explanations"}
                ]
            )
            return {
                "emoji": emoji,
                "visual_description": response.visual_description,
                "top_emotions": [
                    {"emotion": e.emotion, "intensity": e.intensity, "explanation": e.explanation} 
                    for e in response.top_emotions
                ]
            }
        else:
            # Fallback to regular completion with JSON
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes emojis in detail."},
                    {"role": "user", "content": f"""For the emoji {emoji}, provide:
1. A short visual description of what the emoji depicts
2. The top 3 emotions it conveys, with intensity ratings (1-10) and brief explanations

Respond in JSON format with this structure:
{{
  "visual_description": "...",
  "top_emotions": [
    {{
      "emotion": "...",
      "intensity": X,
      "explanation": "..."
    }},
    ...
  ]
}}"""}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            return {
                "emoji": emoji,
                "visual_description": result.get("visual_description", ""),
                "top_emotions": result.get("top_emotions", [])
            }
    except Exception as e:
        print(f"Error analyzing emoji {emoji}: {e}")
        return {
            "emoji": emoji,
            "visual_description": f"Error: {str(e)}",
            "top_emotions": []
        }

async def process_batch(emojis: List[str]) -> List[Dict[str, str]]:
    """Process a batch of emojis concurrently"""
    tasks = [analyze_emoji(emoji) for emoji in emojis]
    return await asyncio.gather(*tasks)

async def process_emojis_async(unique_emojis: List[str]) -> List[Dict[str, str]]:
    """Process all emojis in batches asynchronously"""
    results = []
    
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(unique_emojis), BATCH_SIZE):
        batch = unique_emojis[i:i+BATCH_SIZE]
        batch_results = await process_batch(batch)
        results.extend(batch_results)
        
        # Print progress
        print(f"Processed {min(i+BATCH_SIZE, len(unique_emojis))}/{len(unique_emojis)} emojis")
    
    return results

async def main_async():
    print("Loading dataset...")
    dataset = load_emoji_dataset()
    
    # Debug: Examine dataset structure
    print("\nDataset structure:")
    for split in dataset.keys():
        print(f"Split: {split}")
        print(f"  Columns: {dataset[split].column_names}")
        print(f"  Number of examples: {len(dataset[split])}")
        # Show first 3 examples
        for i in range(min(3, len(dataset[split]))):
            print(f"\n  Example {i+1}:")
            for col in dataset[split].column_names:
                print(f"    {col}: {dataset[split][col][i]}")
    
    print("\nExtracting unique emojis...")
    unique_emojis = extract_unique_emojis(dataset)
    print(f"Found {len(unique_emojis)} unique emojis")
    
    # Debug: Display the found emojis
    if len(unique_emojis) > 0:
        print("First 20 unique emojis found:", "".join(unique_emojis[:20]))
    else:
        print("WARNING: No emojis were found. Check your dataset columns.")
    
    # Only proceed with analysis if we found emojis
    if len(unique_emojis) > 0:
        print("\nAnalyzing emojis with GPT...")
        results = await process_emojis_async(unique_emojis)
        
        # Save results to a JSON file
        with open("emoji_analysis.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Analysis complete. Results saved to emoji_analysis.json")
        
        # Display a few examples
        print("\nSample results:")
        for result in results[:min(5, len(results))]:
            print(f"\n{result['emoji']} - {result['visual_description']}")
            print("Top emotions:")
            for emotion in result.get('top_emotions', [])[:3]:
                print(f"  {emotion.get('emotion', 'Unknown')}: {emotion.get('intensity', 0)}/10 - {emotion.get('explanation', 'No explanation')}")
    else:
        print("Skipping analysis since no emojis were found.")

def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
