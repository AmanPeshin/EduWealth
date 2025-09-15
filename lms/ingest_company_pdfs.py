import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from .config import COMPANY_PDF_DIR, VECTOR_DIR, OPENAI_API_KEY, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def infer_subtopic(title: str, page_content: str) -> str:
    for key in ["NPV", "IRR", "WACC", "CAPM", "DCF", "Capital Structure", "Working Capital", "Derivatives"]:
        if key.lower() in (title + " " + page_content[:300]).lower():
            return key
    return "General"

def build_company_vectorstore() -> Chroma:
    os.makedirs(VECTOR_DIR, exist_ok=True)
    loader = PyPDFDirectoryLoader(COMPANY_PDF_DIR)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    enriched = []
    for d in chunks:
        topic = "Corporate Finance"
        subtopic = infer_subtopic(d.metadata.get("title",""), d.page_content)
        d.metadata.update({"topic": topic, "subtopic": subtopic, "source": "company_pdf"})
        enriched.append(d)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(enriched, embedding=embeddings, persist_directory=VECTOR_DIR)
    vectordb.persist()
    return vectordb
