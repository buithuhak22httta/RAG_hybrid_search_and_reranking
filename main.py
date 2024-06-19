from urllib import response
import timeit
import argparse
from src.retrieval import setup_dbqa
from src.db_build import run_db_build


if __name__ == "__main__":
    # run_db_build()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        default="UNIQLO Vincom Metropolis có địa chỉ ở đâu?",
        help="Enter the query to pass into the LLM",
    )
    args = parser.parse_args()

    # Setup DBQA
    start = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({"query": args.input})
    end = timeit.default_timer()

    # Process source documents
    source_docs = response["source_documents"]
    for i, doc in enumerate(source_docs):
        print(f"\nSource Document {i+1}\n")
        print(f"Source Text: {doc.page_content}")
        print(f'Document Name: {doc.metadata["source"]}')
        print(f'Page Number: {doc.metadata["page"]}\n')
        print("=" * 60)

    print(f'\nAnswer: {response["result"]}')
    print("=" * 50)

    print(f"Time to retrieve response: {end - start}")
