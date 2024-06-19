import yaml
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import SingleStoreDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
import dill
from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import uuid
import logging
from api_llm import CustomAPILanguageModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
MODEL_NAME = config.get("MODEL_NAME")
MODEL_TYPE = config.get("MODEL_TYPE")


def custom_generate(llm, prompt):
    try:
        return llm._call(prompt)
    except Exception as e:
        raise ValueError(f"Error in custom_generate: {e}")


def run_db_build():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # pdf_files = [
        #     os.path.join(DATA_PATH, f)
        #     for f in os.listdir(DATA_PATH)
        #     if f.endswith(".pdf")
        # ]
        pdf_files = [
            os.path.join(DATA_PATH, f)
            for f in os.listdir(DATA_PATH)
            if f.endswith(".pdf") and f.startswith("CTKM")
        ]

        logger.info(f"Found {len(pdf_files)} PDF files in {DATA_PATH}")

        all_documents = []

        for filename in pdf_files:
            logger.info(f"Processing file: {filename}")
            raw_pdf_elements = partition_pdf(
                filename=filename,
                extract_images_in_pdf=False,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=4000,
                new_after_n_chars=3800,
                combine_text_under_n_chars=2000,
            )

            class Element:
                def __init__(self, type, text):
                    self.type = type
                    self.text = text

            categorized_elements = []
            for element in raw_pdf_elements:
                if "unstructured.documents.elements.Table" in str(type(element)):
                    categorized_elements.append(
                        Element(type="table", text=str(element))
                    )
                elif "unstructured.documents.elements.CompositeElement" in str(
                    type(element)
                ):
                    categorized_elements.append(Element(type="text", text=str(element)))

            table_elements = [e for e in categorized_elements if e.type == "table"]
            text_elements = [e for e in categorized_elements if e.type == "text"]

            logger.info(
                f"Found {len(table_elements)} tables and {len(text_elements)} text elements"
            )

            prompt_text = """You are the assistant assigned to write a paragraph to record all the information in the data table in Vietnamese. \
                            Please rewrite all table information in Vietnamese. Table chunk: {element} """
            prompt = ChatPromptTemplate.from_template(prompt_text)

            if MODEL_TYPE == "no_OpenAI":
                llm = CustomAPILanguageModel(
                    base_url=BASE_URL, api_key=OPENAI_API_KEY, model=MODEL_NAME
                )
            else:
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, base_url=BASE_URL
                )

            table_summaries = []
            for table in table_elements:
                try:
                    print(table.text)
                    print("------------------")
                    prompt_value = prompt.format(element=table.text)
                    logger.info(
                        f"Summarizing table with content: {table.text[:100]}..."
                    )
                    summary = custom_generate(llm, prompt_value)
                    table_summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error summarizing table: {e}")

            summary_tables = [
                Document(page_content=s, metadata={"doc_id": str(uuid.uuid4())})
                for s in table_summaries
            ]

            original_texts = [
                Document(page_content=i.text, metadata={"doc_id": str(uuid.uuid4())})
                for i in text_elements
            ]

            all_documents.extend(summary_tables)
            all_documents.extend(original_texts)

        os.environ["SINGLESTOREDB_URL"] = SINGLESTOREDB_URL
        vectorstore = SingleStoreDB.from_documents(
            all_documents,
            embeddings,
            distance_strategy="DOT_PRODUCT",
            table_name="demo0_table",
        )

        keyword_retriever = BM25Retriever.from_documents(all_documents)
        keyword_retriever.k = 2

        with open("keyword_retriever_table.pkl", "wb") as f:
            dill.dump(keyword_retriever, f)

        logger.info("Database build complete.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    run_db_build()
