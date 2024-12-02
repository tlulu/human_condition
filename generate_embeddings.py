import openai
import tiktoken
import re
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Preprocessing - Extract and Chunk Text from PDF
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def create_chapter_section_map(full_text):
    # Define patterns
    chapter_pattern = r"^CHAPTER\s+\d+\.\s+.+$"  # Matches lines like 'CHAPTER 1. title'
    section_pattern = r"^Section\s+\d+\.\s+.+$"  # Matches lines like 'Section 1. title'

    # Split into lines
    lines = full_text.splitlines()

    # Initialize variables
    current_chapter = None
    chapter_map = {}

    # Iterate through lines to populate the map
    for line in lines:
        line = line.strip()

        # Match chapter lines
        if re.match(chapter_pattern, line):
            current_chapter = line
            if current_chapter not in chapter_map:
                chapter_map[current_chapter] = []

        # Match section lines (only if within a chapter)
        elif re.match(section_pattern, line) and current_chapter:
            chapter_map[current_chapter].append(line)

    return chapter_map


def generate_chunks(raw_text, chapter_map):
    chapter_chunks = {}
    lines = raw_text.splitlines()

    # Current state tracking
    current_chapter = None
    current_section = None
    section_text = []

    for line in lines:
        line = line.strip()

        # Check for chapter header
        if line in chapter_map:
            print(f"Detected chapter: {line}") 
            # Save the previous chapter and section
            if current_chapter:
                if current_section:
                    chapter_chunks[current_chapter][current_section] = "\n".join(section_text)
                    current_section = None
                section_text = []
            current_chapter = line
            chapter_chunks[current_chapter] = {}

        # Check for section header (only if inside a valid chapter)
        elif current_chapter and line in chapter_map[current_chapter]:
            # Save the previous section
            if current_section:
                chapter_chunks[current_chapter][current_section] = "\n".join(section_text)
            current_section = line
            section_text = []

        # Add text to the current section
        elif current_chapter and current_section:
            section_text.append(line)

    # Save the last section of the last chapter
    if current_chapter and current_section:
        chapter_chunks[current_chapter][current_section] = "\n".join(section_text)

    return chapter_chunks

def replace_curly_apostrophe(text):
    """
    Replace all instances of curly apostrophe (’) with single apostrophe (').
    """
    return text.replace("’", "'")

# Step 2: Generate Embeddings

def generate_embeddings(chunks, openai_api_key):
    """
    Generates embeddings for each text chunk using OpenAI's Embedding API.
    """
    openai.api_key = openai_api_key
    embeddings = []

    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    for chapter, sections in chunks.items():
        for section, text in sections.items():
            enriched_text = f"{chapter}\n{section}\n{text}"
            tokens = encoding.encode(enriched_text)

            if len(tokens) > 8192:
                print(f"{chapter} - {section} EXCEEDS token limit {len(tokens)}. Skipping..")
                continue

            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=enriched_text
            )
            embeddings.append({
                "id": f"{chapter}-{section}",
                "text": enriched_text,
                "embedding": response["data"][0]["embedding"],
                "metadata": {
                    "chapter": chapter,
                    "section": section
                }
            })
            print(f"Processed {chapter} - {section}")

    return embeddings


def save_chunks_to_json(chunks, output_file):
    """
    Save chapter-section chunks to a JSON file.

    Args:
        chapter_chunks (dict): A dictionary containing chapter-section data.
        output_file (str): The name of the JSON file to save the chunks.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=4, ensure_ascii=False)
        print(f"Chunks successfully written to {output_file}")
    except Exception as e:
        print(f"An error occurred while saving chunks: {e}")


# Main Workflow

def main():
    # File paths and API key
    pdf_file = "the_human_condition.txt"  # Replace with your PDF file
    output_file = "embeddings.json"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Step 1: Preprocess the PDF
    print("Reading from file")
    raw_text = read_txt_file(pdf_file)
    print("Text length:", len(raw_text))

    formatted_text = replace_curly_apostrophe(raw_text)

    # Separate text into sections
    print("Extracting sections...")
    chapter_section_map = create_chapter_section_map(formatted_text)
    print("\nChapter-Section Map:")
    for chapter, sections in chapter_section_map.items():
        print(f"{chapter}:")
        for section in sections:
            print(f"  - {section}")
 

    # Generate a list of chunks 
    print("Chunking text into manageable pieces...")
    chunks = generate_chunks(formatted_text, chapter_section_map)
    # print(chunks["CHAPTER 0. Introduction"])
    # print(chunks["CHAPTER 1. The Human Condition"]['Section 1. Vita Activa and the HUMAN CONDITION'])
    # print(chunks["CHAPTER 1. The Human Condition"]['Section 2. The term Vita Activa'])
    # print(chunks['CHAPTER 1. The Human Condition'].keys())
    # print(chunks['CHAPTER 2. The Public and the Private Realm'].keys())

    for key,value in chunks.items():
        print(key, len(value))
            

    save_chunks_to_json(chunks, "local_db.json")

    # Step 2: Generate embeddings for the chunks
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, openai_api_key)

    # Save embeddings to a JSON file
    print(f"Saving embeddings to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(embeddings, f)

    print("Preprocessing and embedding generation complete!")

if __name__ == "__main__":
    main()
