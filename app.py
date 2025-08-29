import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env
load_dotenv()

# Use OPENAI_API_KEY and OPENAI_BASE_URL for OpenRouter compatibility
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    st.error("ERROR: OPENAI_API_KEY environment variable is NOT set.")
    st.stop()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")

OPENROUTER_API_URL = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"

st.title("Contract Clause Legalese Checker and Modifier")
st.write("Upload your contract document (.txt) to get a review highlighting unfair or illegal clauses with fair alternatives.")

uploaded_file = st.file_uploader("Upload your contract (.txt format)", type=["txt"])

if uploaded_file is not None:
    contract_text = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded Contract", contract_text, height=300)

    # Split text into chunks for retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_text(contract_text)

    # Initialize HuggingFace embeddings and build vector store
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings_model)

    # Retrieve relevant chunks about illegal/unfair clauses
    query = "Identify illegal or unfair clauses, flag them, and suggest fair replacements."
    retrieved_docs = vector_store.similarity_search(query, k=3)

    # Compose prompt for DeepSeek R1 0528 free model
    prompt = (
        "You are a contract law expert. Analyze the following contract clauses:\n\n"
        "1. Identify illegal or unfair clauses and explain why they are problematic.\n"
        "2. Provide clear, fair alternative clauses to replace them.\n"
        "3. Summarize overall contract fairness and potential risks.\n\n"
        "Clauses for review:\n"
    )
    for i, doc in enumerate(retrieved_docs):
        prompt += f"Clause chunk {i+1}:\n{doc.page_content}\n\n"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        review_text = result["choices"][0]["message"]["content"]
        st.subheader("Contract Analysis and Suggested Revisions")
        st.markdown(review_text)
    except requests.exceptions.Timeout:
        st.error("OpenRouter API request timed out. Please try again later.")
    except requests.exceptions.RequestException as e:
        st.error(f"OpenRouter API request error: {e}")
else:
    st.info("Please upload a contract text file to get started.")
