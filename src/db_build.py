import yaml
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import SingleStoreDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever


load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SINGLESTOREDB_URL = os.getenv("SINGLESTOREDB_URL")
BASE_URL = os.getenv("BASE_URL")


with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)
RETURN_SOURCE_DOCUMENTS = config.get("RETURN_SOURCE_DOCUMENTS")
VECTOR_COUNT = config.get("VECTOR_COUNT")
DATA_PATH = config.get("DATA_PATH")
CHUNK_SIZE = config.get("CHUNK_SIZE")
CHUNK_OVERLAP = config.get("CHUNK_OVERLAP")


# Build vector database
def run_db_build():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    os.environ["SINGLESTOREDB_URL"] = SINGLESTOREDB_URL
    vectorstore = SingleStoreDB.from_documents(
        texts, embeddings, distance_strategy="DOT_PRODUCT", table_name="demo0"
    )
    retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 2})

    keyword_retriever = BM25Retriever.from_documents(texts)
    keyword_retriever.k = 2

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vectordb, keyword_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


if __name__ == "__main__":
    run_db_build()
