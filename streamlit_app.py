import streamlit as st
from rag_pipeline import embed_chunks, retrieve_chunks
from fine_tuned_model import generate_answer_ft

# Inject custom CSS to style the text input
st.markdown("""
    <style>
    div[data-testid="stTextInput"] > div > input {
        border: 2px solid #FF5733;  /* Change color here */
        border-radius: 6px;
        padding: 6px;
        outline: none;
    }
    div[data-testid="stTextInput"] > div > input:focus {
        border-color: #33C3F0;  /* Highlight color on focus */
        box-shadow: 0 0 5px rgba(51, 195, 240, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)


# Load data
chunks = [("Hello", "world"), ("How", "are you?")]  # Load your financial text chunks
index, embeddings = embed_chunks(chunks)

st.title("📊 Comparitive Financial Q&A System")

method = st.radio("Choose Method", ["RAG", "Fine-Tuned"])

query = st.text_input("Enter your financial question:")

if query:
    if method == "RAG":
        retrieved = retrieve_chunks(query, chunks, index, embeddings)
        context ="\n".join([str(item) for item in retrieved])
        st.write("Retrieved Context:", context)
        # Generate answer using a generative model (e.g., GPT-2)
        # You can plug in your response generation logic here
    else:
         answer = generate_answer_ft(query)
         st.write("Answer:", answer)
