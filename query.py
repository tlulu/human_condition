from openai import OpenAI
from pinecone import Pinecone
import json
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX_NAME = "humancondition"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"
TOP_K = 1  # Number of chunks to retrieve

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def retrieve_chunks_from_local(local_db_chunks, matches):
    retrieved_chunks = []
    for match in matches:
        chapter = match["metadata"]["chapter"]
        section = match["metadata"]["section"]

        try:
            text = local_db_chunks[chapter][section]
            retrieved_chunks.append(f"Chapter: {chapter}, Section: {section}\n{text}")
        except KeyError:
            print(f"Warning: Could not find text for Chapter: {chapter}, Section: {section}")

    return retrieved_chunks

def query_index(openai_client, index, query_text):
    try:
        embedding_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query_text
        )
        query_embedding = embedding_response.data[0].embedding

        # Perform the search
        result = index.query(
            vector=query_embedding,
            top_k=TOP_K,
            include_metadata=True
        )

        # Retrieve and display chunks
        print(f"\nPinecone Results for '{query_text}':")
        for match in result["matches"]:
            print(f"ID: {match['id']}")
            print(f"Score: {match['score']}")
            print(f"Metadata: {match['metadata']}\n---\n")

        # Ask GPT-4
        local_db_chunks = read_json("local_db.json")
        retrieved_chunks = retrieve_chunks_from_local(local_db_chunks, result["matches"])

        context = "\n---\n".join(retrieved_chunks)
        prompt = f"You are an expert on the book 'The Human Condition' by Hannah Arendt.\n\nContext:\n{context}\n\nQuestion: {query_text}"
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        print(messages)
        gpt_response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        print(f"\nGPT Answer:\n{gpt_response.choices[0].message.content}")

    except Exception as e:
        print(f"An error occurred: {e}")



def initialize_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name) 

def main():
    openai_client = OpenAI()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    index = initialize_pinecone(pinecone_api_key, PINECONE_INDEX_NAME)

    print("I am the Human Condition expert. Type 'exit' to quit.")
    while True:
        query_text = input("\nEnter your query: ").strip()
        if query_text.lower() == "exit":
            print("Exiting...")
            break

        query_index(openai_client, index, query_text)

if __name__ == "__main__":
    main()