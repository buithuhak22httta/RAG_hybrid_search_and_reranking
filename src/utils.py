from src.retrieval import setup_dbqa
from urllib import response
import timeit
import argparse


def query_dbqa(user_input):
    # Setup DBQA
    start = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({"query": user_input})
    end = timeit.default_timer()

    # Process source documents
    source_docs = response["source_documents"]
    source_texts = []
    for i, doc in enumerate(source_docs):
        source_text = f"\nSource Document {i+1}\n"
        source_text += f"Source Text: {doc.page_content}\n"
        source_text += f'Document Name: {doc.metadata["source"]}\n'
        source_text += f'Page Number: {doc.metadata["page"]}\n'
        source_text += "=" * 60
        source_texts.append(source_text)

    answer = response["result"]
    retrieval_time = f"Time to retrieve response: {end - start:.2f} seconds"

    return answer, "\n\n".join(source_texts), retrieval_time
