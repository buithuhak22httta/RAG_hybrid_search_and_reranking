import yaml
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.singlestoredb import SingleStoreDB
from src.prompts import qa_template
from src.db_build import run_db_build
from src.api_llm import CustomAPILanguageModel
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import dill
from langchain.retrievers import BM25Retriever, EnsembleRetriever


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=qa_template, input_variables=["context", "question"]
    )
    return prompt


load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SINGLESTOREDB_URL = os.getenv("SINGLESTOREDB_URL")
BASE_URL = os.getenv("BASE_URL")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)
RETURN_SOURCE_DOCUMENTS = config.get("RETURN_SOURCE_DOCUMENTS")
VECTOR_COUNT = config.get("VECTOR_COUNT")
DATA_PATH = config.get("DATA_PATH")
CHUNK_SIZE = config.get("CHUNK_SIZE")
CHUNK_OVERLAP = config.get("CHUNK_OVERLAP")
HF_TOKEN = config.get("HF_TOKEN")
MODEL_NAME = config.get("MODEL_NAME")
MODEL_TYPE = config.get("MODEL_TYPE")
RERANKING = config.get("RERANKING")


def build_retrieval_qa(llm, prompt, retriever):
    dbqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        # vectordb.as_retriever(search_kwargs={"k": VECTOR_COUNT}),
        return_source_documents=RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={"prompt": prompt},
    )
    return dbqa


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    # with open("keyword_retriever_table.pkl", "rb") as f:
    with open("keyword_retriever.pkl", "rb") as f:
        keyword_retriever = dill.load(f)
    vectorstore = SingleStoreDB(
        embeddings,
        distance_strategy="DOT_PRODUCT",
        table_name="demo0",
        # table_name="demo0_table",
    )
    retriever_vectordb = vectorstore.as_retriever(search_kwargs={"k": 2})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vectordb, keyword_retriever], weights=[0.5, 0.5]
    )

    if MODEL_TYPE == "no_OpenAI":
        llm = CustomAPILanguageModel(
            base_url=BASE_URL, api_key=OPENAI_API_KEY, model=MODEL_NAME
        )
    else:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, base_url=BASE_URL
        )

    # llm = HuggingFaceHub(
    #     repo_id="mistralai/Codestral-22B-v0.1",
    #     model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
    #     huggingfacehub_api_token="",
    # )
    # ensemble_retriever = run_db_build()
    compressor = CohereRerank(cohere_api_key=COHERE_API_KEY)
    if RERANKING == "yes":
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )
        qa_prompt = set_qa_prompt()

        dbqa = build_retrieval_qa(llm, qa_prompt, compression_retriever)
    else:
        qa_prompt = set_qa_prompt()

        dbqa = build_retrieval_qa(llm, qa_prompt, ensemble_retriever)

    return dbqa
