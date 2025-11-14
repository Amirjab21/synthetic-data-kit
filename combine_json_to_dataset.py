import json

# Define the file paths
file1_path = '/Users/amirjabarivasal/Documents/synthetic-data-kit/data/generated/docx_qa_pairs_chapter_1_and_2.json'
file2_path = '/Users/amirjabarivasal/Documents/synthetic-data-kit/data/generated/docx_qa_pairs_chapter3-8.json'

# Function to load and parse JSON content from a file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Load data from both files
data1 = load_json(file1_path)
data2 = load_json(file2_path)

# Combine both datasets
data_combined = data1 + data2

# Extracting 'prompt' and 'qa_pairs' into a list of dictionaries
extracted_data = []
for entry in data_combined:
    prompt = entry.get('prompt')
    qa_pairs = entry.get('qa_pairs', {}).get('qa_pairs', [])
    for qa in qa_pairs:
        question = qa.get('question')
        answer = qa.get('answer')
        extracted_data.append({
            'instruction': prompt,
            'input': question,
            'output': answer
        })

print(len(extracted_data))

# Print or handle extracted_data as needed
# print(json.dumps(extracted_data, indent=2))

# Save the combined data to a new JSON file
output_file = 'data/generated/qa_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=2)

print(f"Combined data saved to: {output_file}")




from datasets import Dataset, Audio
from pathlib import Path
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Login to Hugging Face (you'll need to set your token as an environment variable)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set the HF_TOKEN environment variable")

login(HF_TOKEN)


# Prepare the dataset


# Create the dataset
dataset = Dataset.from_list(extracted_data)

# Push to the Hub
dataset.push_to_hub(
    "Amirjab21/LLM_training",  # Replace with your username and desired repo name
    private=True,  # Set to False if you want the dataset to be public
    split="train",
)
