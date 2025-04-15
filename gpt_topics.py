from openai import AsyncOpenAI
import instructor
from datasets import load_dataset, concatenate_datasets
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple
from collections import defaultdict
import asyncio
from tqdm.asyncio import tqdm_asyncio
import time

ds = load_dataset("KomeijiForce/Text2Emoji")  # ds1
ds2 = load_dataset("DESUCLUB/combined_emoji_data")

# Initialize the async OpenAI clients
base_client = AsyncOpenAI()
# Create an instructor-enabled async client
client = instructor.patch(AsyncOpenAI())

# Batch size for concurrent API calls
BATCH_SIZE = 10
# Minimum number of examples required for a topic to be valid
MIN_TOPIC_SIZE = 100
# Maximum number of topics for visualization
MAX_TOPICS = 20

class TopicResponse(BaseModel):
    """Response model for topic assignments"""
    topic: str = Field(..., description="The best matching topic for the text")

class Topics(BaseModel):
    values: List[str] = Field(default_factory=list)
    topic_counts: Dict[str, int] = Field(default_factory=lambda: defaultdict(int))
    
    def update_values(self, new_value: str):
        if new_value not in self.values:
            self.values.append(new_value)
            self.topic_counts[new_value] = 1
        else:
            self.topic_counts[new_value] += 1
    
    def cleanup_small_topics(self):
        """Remove topics with too few examples and return them for redistribution"""
        small_topics = [topic for topic, count in self.topic_counts.items() 
                       if count < MIN_TOPIC_SIZE]
        
        # Remove small topics from values
        for topic in small_topics:
            self.values.remove(topic)
            del self.topic_counts[topic]
            
        return small_topics

# Initialize Topics with existing dataset values
def initialize_topics():
    topics_obj = Topics()
    
    # Get initial topics from ds1
    if "topic" in ds["train"].column_names:
        initial_topics = ds["train"].unique("topic")
        
        # Count frequency of each topic
        topic_counts = ds["train"].to_pandas()["topic"].value_counts().to_dict()
        
        for topic in initial_topics:
            topics_obj.values.append(topic)
            topics_obj.topic_counts[topic] = topic_counts.get(topic, 0)
    
    return topics_obj

# Global topics object
TOPICS = initialize_topics()

# Create a lookup from text to topic using ds1
def create_text_to_topic_lookup(dataset):
    """Create a dictionary mapping texts to their topics from ds1 using fast dataset operations"""
    if "topic" in dataset["train"].column_names:
        # Convert to pandas DataFrame and create a text->topic mapping
        df = dataset["train"].to_pandas()[["text", "topic"]]
        # Set the text column as the index and convert to dictionary
        return df.set_index("text")["topic"].to_dict()
    return {}

# Global lookup dictionary
TEXT_TO_TOPIC = create_text_to_topic_lookup(ds)

async def get_topics_new(text: str) -> str:
    """
    Generate a topic for the given text using GPT.
    The similarity check will be handled separately by embeddings.
    """
    global TOPICS, TEXT_TO_TOPIC
    
    # First check if this text already has a topic in ds1
    if text in TEXT_TO_TOPIC:
        topic = TEXT_TO_TOPIC[text]
        # Update topic counts
        TOPICS.update_values(topic)
        return topic
    
    # If no existing topic, generate one using GPT
    response = await base_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Create a single word or short phrase topic category for this text: '{text}'"}
        ]
    )
    new_topic = response.choices[0].message.content.strip()
    
    # Update topic counts
    TOPICS.update_values(new_topic)
    
    return new_topic

async def process_batch(texts, process_func):
    """Process a batch of texts concurrently"""
    tasks = [process_func(text) for text in texts]
    return await asyncio.gather(*tasks)

async def reassign_topic_async(example):
    """Async function to reassign a topic for a given example"""
    prompt = f"""
    I have a text: "{example['text']}"
    
    And these existing topic categories: {', '.join(TOPICS.values)}
    
    Which single category best fits this text? Choose only from the categories listed above.
    """
    
    # Use instructor to directly get a TopicResponse object
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_model=TopicResponse
    )
    
    # Extract the topic directly from the Pydantic model
    best_topic = response.topic
    
    # Validate that the returned topic exists in our list
    if best_topic not in TOPICS.values:
        # If GPT returns an invalid topic, fall back to the largest topic
        best_topic = max(TOPICS.topic_counts.items(), key=lambda x: x[1])[0]
    
    # Use the Topics update_values method to properly update the topic
    TOPICS.update_values(best_topic)
    
    return {"topic": best_topic}

async def redistribute_small_topics_async(dataset, small_topics):
    """Asynchronously reassign examples from small topics to larger topics"""
    # Get examples with small topics
    small_topic_examples = dataset.filter(lambda x: x["topic"] in small_topics)
    print(f"Redistributing {len(small_topic_examples)} examples from {len(small_topics)} small topics")
    
    # Convert to list for easier async processing
    examples_list = list(small_topic_examples)
    results = []
    
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(examples_list), BATCH_SIZE):
        batch = examples_list[i:i+BATCH_SIZE]
        batch_results = await tqdm_asyncio.gather(
            *[reassign_topic_async(example) for example in batch],
            desc=f"Redistributing batch {i//BATCH_SIZE+1}/{(len(examples_list)-1)//BATCH_SIZE+1}"
        )
        results.extend(batch_results)
    
    # Create a new dataset from the results
    from datasets import Dataset
    return Dataset.from_dict({
        "text": [example["text"] for example in examples_list],
        "topic": [result["topic"] for result in results]
    })

async def process_dataset_async(dataset, subset_size=None):
    """Asynchronously process a dataset to assign topics"""
    # Take a subset if requested
    if subset_size and subset_size < len(dataset):
        import random
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = dataset.select(indices)
    
    print(f"Processing {len(dataset)} examples")
    
    # Convert to list for easier async processing
    examples_list = list(dataset)
    results = []
    
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(examples_list), BATCH_SIZE):
        batch = examples_list[i:i+BATCH_SIZE]
        batch_results = await tqdm_asyncio.gather(
            *[get_topics_new(example["text"]) for example in batch],
            desc=f"Processing batch {i//BATCH_SIZE+1}/{(len(examples_list)-1)//BATCH_SIZE+1}"
        )
        results.extend(batch_results)
    
    # Create a dataset with the results
    from datasets import Dataset
    processed_ds = Dataset.from_dict({
        "text": [example["text"] for example in examples_list],
        "topic": results
    })
    
    return processed_ds

async def process_and_assign_topics_async(dataset, subset_size=None):
    """Asynchronously process a dataset to assign topics and handle small topics"""
    start_time = time.time()
    
    # First pass: assign initial topics
    processed_ds = await process_dataset_async(dataset["train"], subset_size)
    
    # Clean up small topics
    small_topics = TOPICS.cleanup_small_topics()
    
    # Redistribute examples from small topics
    if small_topics:
        redistributed = await redistribute_small_topics_async(processed_ds, small_topics)
        # Merge redistributed examples back into the dataset
        processed_ds = processed_ds.filter(lambda x: x["topic"] not in small_topics)
        # Use concatenate_datasets instead of the concatenate method
        processed_ds = concatenate_datasets([processed_ds, redistributed])
    
    duration = time.time() - start_time
    print(f"Processing completed in {duration:.2f} seconds")
    
    return processed_ds

async def merge_similar_topics(dataset, max_topics=MAX_TOPICS):
    """
    Merge similar topics if we have more than max_topics
    Returns a dataset with merged topics
    """
    # Get topic distribution
    topic_counts = dataset.to_pandas()["topic"].value_counts()
    topics = list(topic_counts.index)
    
    # If we have fewer topics than the maximum, no need to merge
    if len(topics) <= max_topics:
        print(f"Number of topics ({len(topics)}) is within the maximum ({max_topics}). No merging needed.")
        return dataset
    
    print(f"Too many topics ({len(topics)}). Merging to {max_topics} topics...")
    
    # Find topic pairs to merge based on semantic similarity
    topics_to_merge = topics[max_topics:]  # Smallest topics to merge
    
    # Create prompt to get merging suggestions
    topic_list_str = ", ".join(topics[:max_topics])  # Topics to keep
    topics_to_merge_str = ", ".join(topics_to_merge)  # Topics to merge
    
    prompt = f"""
    I need to merge these smaller topics into the larger topics for visualization purposes.
    
    Larger topics to keep ({len(topics[:max_topics])}): {topic_list_str}
    
    Smaller topics to merge ({len(topics_to_merge)}): {topics_to_merge_str}
    
    For each smaller topic, tell me which larger topic it should merge with based on semantic similarity.
    Format your response as a JSON where the keys are the smaller topics and the values are the larger topics they should merge with.
    """
    
    # Get merging suggestions
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_model=dict
        )
        
        # The response will be a dictionary mapping smaller topics to larger topics
        topic_mapping = response
        
        print("Topic merging map:")
        for small_topic, large_topic in topic_mapping.items():
            print(f"  {small_topic} → {large_topic}")
        
        # Apply the mapping to the dataset
        def merge_topic(example):
            if example["topic"] in topic_mapping:
                return {"topic": topic_mapping[example["topic"]]}
            return {"topic": example["topic"]}
        
        merged_dataset = dataset.map(merge_topic)
        
        # Get the new topic distribution
        new_topic_counts = merged_dataset.to_pandas()["topic"].value_counts()
        print(f"\nAfter merging, we have {len(new_topic_counts)} topics.")
        
        return merged_dataset
        
    except Exception as e:
        print(f"Error merging topics: {e}")
        print("Continuing with original topics")
        return dataset

async def add_topics_to_ds2_async(subset_size=None, max_topics=MAX_TOPICS):
    """Add topics to ds2, reusing topics from ds1 where possible"""
    start_time = time.time()
    
    print("Adding topics to ds2...")
    
    # If we want to process a subset
    ds2_train = ds2["train"]
    if subset_size and subset_size < len(ds2_train):
        import random
        indices = random.sample(range(len(ds2_train)), subset_size)
        ds2_train = ds2_train.select(indices)
    
    # Faster implementation using map directly
    def assign_topic(example):
        # Check if text already exists in ds1
        if example["text"] in TEXT_TO_TOPIC:
            return {"topic": TEXT_TO_TOPIC[example["text"]]}
        return {"topic": "__needs_processing__"}  # Use a string marker instead of None
    
    # First pass: assign existing topics from ds1
    print("Adding existing topics from ds1...")
    ds2_with_topics = ds2_train.map(assign_topic)
    
    # Count how many examples we need to process with GPT
    need_processing = [ex for ex in ds2_with_topics if ex["topic"] == "__needs_processing__"]
    already_assigned = len(ds2_with_topics) - len(need_processing)
    
    print(f"Found existing topics for {already_assigned} examples")
    print(f"Need to generate topics for {len(need_processing)} examples")
    
    # If we have examples that need processing
    if need_processing:
        # Extract examples that need processing
        need_processing_ds = ds2_with_topics.filter(lambda ex: ex["topic"] == "__needs_processing__")
        
        # Process in batches
        examples_list = list(need_processing_ds)
        results = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(examples_list), BATCH_SIZE):
            batch = examples_list[i:i+BATCH_SIZE]
            batch_results = await tqdm_asyncio.gather(
                *[get_topics_new(example["text"]) for example in batch],
                desc=f"Processing batch {i//BATCH_SIZE+1}/{(len(examples_list)-1)//BATCH_SIZE+1}"
            )
            results.extend(batch_results)
        
        # Update the topics
        for i, example in enumerate(examples_list):
            example["topic"] = results[i]
            TOPICS.update_values(results[i])
        
        # Create updated dataset from examples with topics
        from datasets import Dataset
        processed_new = Dataset.from_dict({
            "text": [ex["text"] for ex in examples_list],
            "topic": [ex["topic"] for ex in examples_list]
        })
        
        # Create dataset with existing topics
        existing_ds = ds2_with_topics.filter(lambda ex: ex["topic"] != "__needs_processing__")
        
        # Combine both datasets using concatenate_datasets
        final_ds = concatenate_datasets([existing_ds, processed_new])
    else:
        final_ds = ds2_with_topics
    
    # Clean up small topics
    small_topics = TOPICS.cleanup_small_topics()
    print(f"Small topics identified for redistribution: {small_topics}")

    # Redistribute examples from small topics
    if small_topics:
        print(f"Starting redistribution for {len(small_topics)} small topics")
        redistributed = await redistribute_small_topics_async(final_ds, small_topics)
        print(f"Redistribution complete, processed {len(redistributed)} examples")
        
        # Merge redistributed examples back into the dataset
        filtered_ds = final_ds.filter(lambda x: x["topic"] not in small_topics)
        print(f"Filtered out examples with small topics: {len(final_ds) - len(filtered_ds)} examples removed")
        
        # Use concatenate_datasets instead of the concatenate method
        final_ds = concatenate_datasets([filtered_ds, redistributed])
        print(f"Final dataset after redistribution: {len(final_ds)} examples")
    else:
        print("No small topics found, skipping redistribution")
    
    # If we have more than max_topics, merge similar topics
    final_topics = final_ds.unique("topic")
    if len(final_topics) > max_topics:
        final_ds = await merge_similar_topics(final_ds, max_topics)
    
    duration = time.time() - start_time
    print(f"Processing completed in {duration:.2f} seconds")
    
    return final_ds

async def main_async(subset_size=None, max_topics=MAX_TOPICS):
    """Async main function"""
    # Add topics to ds2, reusing topics from ds1 where possible
    processed_ds2 = await add_topics_to_ds2_async(subset_size, max_topics)
    
    # Get final list of topics
    topics = processed_ds2.unique("topic")
    print(f"Final topics ({len(topics)}):")
    for topic in sorted(topics):
        print(f"- {topic}")
    
    # Print topic distribution
    topic_counts = processed_ds2.to_pandas()["topic"].value_counts()
    print("\nTopic distribution:")
    for topic, count in topic_counts.items():
        print(f"{topic}: {count} examples")
    
    # Save the enhanced dataset
    processed_ds2.save_to_disk("combined_emoji_data_with_topics")
    print("Enhanced dataset saved to: combined_emoji_data_with_topics")
    
    return topics

def main(subset_size=None, max_topics=MAX_TOPICS):
    """Entry point that runs the async main function"""
    return asyncio.run(main_async(subset_size, max_topics))

if __name__ == "__main__":
    # Optional: specify a subset size for faster testing
    # Change to None to process the entire dataset
    SUBSET_SIZE = None
    # Maximum number of topics to visualize
    MAX_TOPIC_COUNT = 20  # Increased to accommodate original 17 topics
    main(SUBSET_SIZE, MAX_TOPIC_COUNT)
