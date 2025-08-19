import streamlit as st
from rag_pipeline import embed_chunks, retrieve_chunks
#from fine_tuned_model import generate_answer_ft

# Load data
chunks = [("Hello", "world"), ("How", "are you?")]  # Load your financial text chunks
index, embeddings = embed_chunks(chunks)

st.title("📊 Financial Q&A Assistant")

method = st.radio("Choose Method", ["RAG", "Fine-Tuned"])

query = st.text_input("Enter your financial question:")

if query:
    if method == "RAG":
        retrieved = retrieve_chunks(query, chunks, index, embeddings)
        context = "\n".join(retrieved)
        st.write("Retrieved Context:", context)
        # Generate answer using a generative model (e.g., GPT-2)
        # You can plug in your response generation logic here
    else:
      #  answer = generate_answer_ft(query)
         st.write("Answer:", answer)
