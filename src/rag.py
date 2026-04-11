# RAG (Retrieval Augmented Generation) pipeline.
#
# Author
# ------
# Kishanthan Kingston
#
# Copyright
# ---------
# © 2026 Kishanthan Kingston
#
# License
# -------
# MIT
#
# Description
# -----------
# Loads medical documents, splits them into chunks, stores them
# in a ChromaDB vector database, and retrieves relevant context
# for a given question.

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Path to the converted medical text files
DOCS_PATH = "data/medical_docs"

# Path where the ChromaDB vector database will be stored on disk
DB_PATH = "data/chroma_db"

# Embedding model used to convert text chunks into vectors (runs locally, free)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """
    Initialize and return the HuggingFace embedding model.

    Returns
    -------
    HuggingFaceEmbeddings
        An embedding model instance used to convert text into vectors.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore():
    """
    Load medical documents, split them into chunks, and store them
    in a ChromaDB vector database on disk.

    This function should only be run once, or when the documents are updated.

    Returns
    -------
    Chroma
        A ChromaDB vector store instance.
    """
    # Load all .txt files from the medical docs folder
    loader = DirectoryLoader(DOCS_PATH, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # Split documents into small overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,  # Max characters per chunk
        chunk_overlap=100,  # Overlap to preserve context between chunks
        separators=["\n---\n", "\n\n", "\n"],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Convert chunks to vectors and store them in ChromaDB
    vectorstore = Chroma.from_documents(
        chunks, get_embeddings(), persist_directory=DB_PATH
    )
    print(f"Vector store saved to {DB_PATH}.")

    return vectorstore


def load_vectorstore():
    """
    Load an existing ChromaDB vector store from disk.

    Returns
    -------
    Chroma
        A ChromaDB vector store instance.
    """
    return Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings())


def get_context(vectorstore, question, k=3):
    """
    Retrieve the most relevant document chunks for a given question.

    Parameters
    ----------
    vectorstore : Chroma
        The vector store to search in.
    question : str
        The user's question.
    k : int
        Number of chunks to retrieve (default: 3).

    Returns
    -------
    str
        The retrieved chunks joined as a single string.
    """
    results = vectorstore.similarity_search(question, k=k)

    # Join the retrieved chunks into a single context string
    return "\n\n".join([doc.page_content for doc in results])


if __name__ == "__main__":
    print("Building vector store...")
    vs = build_vectorstore()
    print("Done.\n")

    question = "What are the symptoms of diabetes?"
    context = get_context(vs, question)
    print(f"Question: {question}\n")
    print(f"Context retrieved:\n{context}")
