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
chunks =  [
    ("What was the company’s total revenue in 2023?", "The company’s total revenue in 2023 was $4.13 billion."),
    ("How did revenue change compared to 2022?", "Revenue grew by 10% compared to 2022."),
    ("What was the primary driver of revenue growth?", "The primary driver was strong demand in the North American market."),
    ("Which segment contributed the most to revenue?", "The consumer electronics segment contributed the most, accounting for 45% of total revenue."),
    ("What percentage of revenue came from international markets?", "International markets contributed 35% of total revenue."),
    
    ("What were the company’s operating expenses in 2023?", "Operating expenses were $1.2 billion in 2023."),
    ("How did cost of goods sold (COGS) change?", "COGS increased by 8% due to higher raw material costs."),
    ("What was the operating income for 2023?", "Operating income was $900 million in 2023."),
    ("What was the net profit margin in 2023?", "The net profit margin was 18% in 2023."),
    ("What contributed to the increase in profitability?", "Improved supply chain efficiency and reduced logistics costs contributed to higher profitability."),
    
    ("What were the company’s total assets in 2023?", "Total assets were $15.6 billion in 2023."),
    ("What were the company’s total liabilities?", "Total liabilities were $6.8 billion."),
    ("What was the company’s shareholder equity?", "Shareholder equity stood at $8.8 billion."),
    ("What was the current ratio in 2023?", "The current ratio was 2.1, indicating strong liquidity."),
    ("How much cash and cash equivalents did the company hold?", "The company held $2.3 billion in cash and cash equivalents."),
    
    ("What was the operating cash flow in 2023?", "Operating cash flow was $1.5 billion."),
    ("How much was spent on capital expenditures (CapEx)?", "The company spent $600 million on capital expenditures."),
    ("What was the free cash flow in 2023?", "Free cash flow was $900 million."),
    ("Did the company generate positive investing cash flow?", "No, investing cash flow was negative due to acquisitions."),
    ("What was the financing cash flow in 2023?", "Financing cash flow was -$400 million, primarily due to debt repayments."),
    
    ("What was the company’s total debt in 2023?", "Total debt was $4.2 billion."),
    ("What was the debt-to-equity ratio?", "The debt-to-equity ratio was 0.48."),
    ("Did the company reduce its debt?", "Yes, the company reduced debt by $300 million in 2023."),
    ("How much interest expense did the company incur?", "Interest expense was $120 million."),
    ("What was the interest coverage ratio?", "The interest coverage ratio was 7.5, indicating strong ability to cover interest payments."),
    
    ("Did the company pay dividends in 2023?", "Yes, the company paid $1.20 per share in dividends."),
    ("What was the dividend payout ratio?", "The dividend payout ratio was 35%."),
    ("How many shares are outstanding?", "The company has 200 million shares outstanding."),
    ("Did the company announce a share buyback program?", "Yes, a $500 million share repurchase program was announced."),
    ("How much was returned to shareholders in 2023?", "A total of $740 million was returned through dividends and buybacks."),
    
    ("What was the company’s market capitalization at year-end?", "Market capitalization was $45 billion."),
    ("What was the earnings per share (EPS) in 2023?", "EPS was $3.75 in 2023."),
    ("How did EPS change compared to 2022?", "EPS grew by 12% compared to the prior year."),
    ("What is the company’s P/E ratio?", "The P/E ratio was 15."),
    ("Which geographic region showed the fastest growth?", "The Asia-Pacific region showed the fastest growth at 20%."),
    
    ("What risks were highlighted in 2023?", "Key risks included currency fluctuations and rising interest rates."),
    ("Did supply chain issues affect operations?", "Yes, supply chain issues caused higher inventory costs."),
    ("What is the company’s growth outlook for 2024?", "The company expects revenue growth of 8-10% in 2024."),
    ("Is the company planning new product launches?", "Yes, several new products are planned for the consumer electronics line."),
    ("How is the company addressing sustainability goals?", "The company is investing $200 million in renewable energy initiatives."),
    
    ("What was the return on equity (ROE) in 2023?", "ROE was 16% in 2023."),
    ("What was the return on assets (ROA)?", "ROA was 9%."),
    ("What was the gross profit margin?", "Gross profit margin was 55%."),
    ("How did operating margin change?", "Operating margin improved from 20% to 22%."),
    ("What was the asset turnover ratio?", "The asset turnover ratio was 0.27."),
    
    ("Did the company make any acquisitions in 2023?", "Yes, the company acquired a software firm for $250 million."),
    ("What percentage of revenue is from recurring sources?", "30% of revenue comes from recurring subscriptions."),
    ("How much was spent on research and development (R&D)?", "R&D spending was $400 million."),
    ("Did the workforce grow in 2023?", "Yes, the workforce grew by 8% to 25,000 employees."),
    ("What was the company’s ESG rating?", "The company received an ESG rating of A-."),
] # Load your financial text chunks

index, embeddings = embed_chunks(chunks)

st.title("📊 Financial Q&A System")

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
