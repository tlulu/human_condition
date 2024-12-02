## Running

Virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Create a .env file with your API keys
```
OPENAI_API_KEY=<key>
PINECONE_API_KEY=<key>
```

Run
```
python3 generate_embeddings.py
python3 upload_embeddings.py
python3 query.py
```
