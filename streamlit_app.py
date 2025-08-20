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
        border-color: #33C3F0;
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

# Add "Both" option
method = st.radio("Choose Method", ["RAG", "Fine-Tuned", "Both"])

query = st.text_input("Enter your financial question:")

if query:
    if method == "RAG":
        retrieved = retrieve_chunks(query, chunks, index, embeddings)
        rag_context = "\n".join([str(item) for item in retrieved])
        st.subheader("RAG Output")
        st.write(rag_context)

    elif method == "Fine-Tuned":
        ft_answer = generate_answer_ft(query)
        st.subheader("Fine-Tuned Output")
        st.write(ft_answer)

    elif method == "Both":
        # Get both outputs
        retrieved = retrieve_chunks(query, chunks, index, embeddings)
        rag_context = "\n".join([str(item) for item in retrieved])
        ft_answer = generate_answer_ft(query)

        # Display in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("RAG Output")
            st.write(rag_context)

        with col2:
            st.subheader("Fine-Tuned Output")
            st.write(ft_answer)
