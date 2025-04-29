import streamlit as st
import requests

st.title("📄 Ask Your PDF (FastAPI + Claude RAG)")

# FastAPI server URL
API_URL = "http://127.0.0.1:8000"    # Localhost (adjust if deployed)

# Upload PDF
st.header("Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Upload"):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        response = requests.post(f"{API_URL}/upload/", files=files)
        if response.status_code == 200:
            st.success("Uploaded and indexed successfully!")
        else:
            st.error(f"Upload failed: {response.text}")

# Ask a Question
st.header("Ask a question")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question:
        params = {"question": question}
        response = requests.get(f"{API_URL}/query/", params=params)
        if response.status_code == 200:
            data = response.json()
            st.success("Answer:")
            st.write(data["answer"])
        else:
            st.error(f"Query failed: {response.text}")
    else:
        st.warning("Please enter a question first.")
