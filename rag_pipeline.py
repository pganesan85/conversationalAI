from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(embedder)

# Load and embed chunks
def embed_chunks(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve top-N chunks
def retrieve_chunks(query, chunks, index, embeddings, top_n=5):
    query_vec = embedder.encode([query])
    _, indices = index.search(np.array(query_vec), top_n)
    # Return only the answer part (second element of tuple)
    return [chunks[i][1] for i in indices[0]]

