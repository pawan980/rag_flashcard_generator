# vector_db.py

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# -----------------------------------------------------
# Load documents (PDF or TXT)
# -----------------------------------------------------
def load_docs(path: str):
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path)
    return loader.load()


# -----------------------------------------------------
# Chunking
# -----------------------------------------------------
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


# -----------------------------------------------------
# Build vector DB using Gemini embeddings
# -----------------------------------------------------
def build_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore


# -----------------------------------------------------
# Retriever
# -----------------------------------------------------
def build_retriever(vectorstore, k=5):
    return vectorstore.as_retriever(search_kwargs={"k": k})


# -----------------------------------------------------
# One-call function for entire pipeline
# -----------------------------------------------------
def create_vector_db(path: str):
    print("Loading documents...")
    docs = load_docs(path)

    print("Chunking...")
    chunks = chunk_docs(docs)

    print("Building vector DB (Chroma + ollama embeddings)...")
    vectorstore = build_vectorstore(chunks)

    return build_retriever(vectorstore)
