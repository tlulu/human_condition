from pinecone import Pinecone
import json
import os
from dotenv import load_dotenv

PINECONE_INDEX_NAME = "humancondition"

load_dotenv()

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def upload_embeddings(embeddings):
    pinecone_apy_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_apy_key)
    index = pc.Index(PINECONE_INDEX_NAME)
    for embedding in embeddings:
        pinecone_data = [
            {
                "id": embedding["id"],  # Unique ID for this record
                "values": embedding["embedding"],  # The embedding vector
                "metadata": embedding["metadata"],  # Metadata
            }
        ]
        index.upsert(pinecone_data)
        print(f"Uploaded embedding: {embedding['id']}")
    

def main():
    embedding_file = read_file("embeddings.json")
    upload_embeddings(embedding_file)

    print("Finished uploading embeddings!")

if __name__ == "__main__":
    main()