# loaders.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(path: str):
    """Load PDF and return documents."""
    loader = PyPDFLoader(path)
    return loader.load()

def split_documents(documents, chunk_size: int = 3000, chunk_overlap: int = 1000):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
