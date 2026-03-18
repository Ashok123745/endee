from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

# 1. Load the CLEANED dataset
if os.path.exists("data/news_cleaned.csv"):
    df = pd.read_csv("data/news_cleaned.csv")
else:
    # Fallback if cleaning hasn't run
    df = pd.read_csv("data/news.csv")

# 2. Take a sample for RAG (Retrieval)
# 1500 rows is enough for local testing without lagging
df = df.sample(min(1500, len(df)), random_state=42).reset_index(drop=True)

texts = df['text'].astype(str).tolist()

# 3. Load embedding model
# This model converts text into math vectors (embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Creating embeddings for RAG...")
embeddings = model.encode(texts, show_progress_bar=True)
print("Embeddings created ✅")

# 4. Store vectors in memory
vector_store = list(zip(embeddings, texts))

# 5. THE MISSING FUNCTION: search_similar
def search_similar(query, top_k=3):
    """
    Finds the most similar news articles in the database 
    based on the user's input query.
    """
    # Convert user input into a vector
    query_vec = model.encode([query])[0]

    scores = []
    for emb, text in vector_store:
        # Cosine similarity using dot product
        score = np.dot(query_vec, emb)
        scores.append((score, text))

    # Sort results by highest score first
    scores.sort(key=lambda x: x[0], reverse=True)

    # Return only the top K results
    return [text for _, text in scores[:top_k]]