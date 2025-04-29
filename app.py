# app.py
from fastapi import FastAPI, UploadFile, File
import shutil, os, fitz, numpy as np
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import llm_answer
      # <-- export this in your shell

app = FastAPI()
vectordb = None            # simple global demo store; later persist to disk

# ----------------------------- upload ---------------------------------
@app.post("/upload/")
async def upload_and_index(file: UploadFile = File(...)):
    os.makedirs("store", exist_ok=True)
    path = f"store/{file.filename}"
    with open(path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # 1️⃣  ---- build one Document per PDF page ----
    pdf   = fitz.open(path)
    pages = []
    for i in range(pdf.page_count):
        txt = pdf.load_page(i).get_text("text")
        pages.append(
            Document(
                page_content=txt,
                metadata={"file_name": file.filename, "page_no": i + 1}
            )
        )

    # 2️⃣  ---- split each page into semantic windows ----
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40)
    chunks   = splitter.split_documents(pages)     # metadata carried forward

    # 3️⃣  ---- embed & index ----
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    global vectordb
    vectordb = FAISS.from_documents(chunks, embedder)

    return {"message": f"Indexed {len(chunks)} chunks from {file.filename}"}

# ----------------------------- query ----------------------------------
@app.get("/query/")
async def query(question: str):
    if vectordb is None:
        return {"error": "Upload a PDF first"}

    retrieved = vectordb.similarity_search(question, k=8)

    # ---- label each chunk with file + page before sending to GPT ----
    def label(doc: Document) -> str:
        fn   = doc.metadata.get("file_name", "doc")
        page = doc.metadata.get("page_no", "?")
        return f"[{fn} - page {page}] {doc.page_content}"

    context = "\n\n".join(label(d) for d in retrieved)

    # ---- call the LLM ----
    answer = llm_answer(context, question)

    return {
        "answer": answer
    }
