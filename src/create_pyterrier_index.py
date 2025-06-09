import json
import pandas as pd
import pyterrier as pt
import argparse
import os

def create_index(json_file_path: str, index_path: str, overwrite: bool = False):
    """
    Creates a PyTerrier index from documents in a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing the documents.
                              The JSON should be a dictionary of doc_id: doc_text.
        index_path (str): Path where the created index will be saved.
        overwrite (bool): Whether to overwrite an existing index at index_path.
    """
    if not pt.started():
        pt.init()

    print(f"Loading documents from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    if not isinstance(corpus, dict):
        raise ValueError("JSON file should contain a dictionary of document_id: document_text.")

    print(f"Loaded {len(corpus)} documents.")

    # Convert to DataFrame format expected by PyTerrier
    # PyTerrier expects 'docno' and 'text' columns
    df_docs = pd.DataFrame([
        {'docno': str(doc_id), 'text': text}
        for doc_id, text in corpus.items()
    ])
    
    
    
    print("Creating index...")
    # Ensure the parent directory for the index exists
    index_parent_dir = os.path.dirname(index_path)
    if index_parent_dir and not os.path.exists(index_parent_dir):
        os.makedirs(index_parent_dir)
        print(f"Created directory: {index_parent_dir}")

    # Create an iterable indexer
    # The index_path here is a directory where PyTerrier will store index files.
    # For example, if index_path is "./my_index", files like "./my_index/data.properties" will be created.
    indexer = pt.IterDictIndexer(index_path, overwrite=overwrite, meta={'docno': 20, 'text': 4096})
    
    # Index the documents
    # The index() method expects an iterator of dictionaries
    index_ref = indexer.index(df_docs.to_dict(orient='records'))
    print(f"Index created successfully at: {index_ref.toString()}")
    print(f"Index files are located in the directory: {index_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a PyTerrier index from a JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the input JSON file (doc_id: text).")
    parser.add_argument("index_output_path", type=str, help="Path to save the generated index (directory).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index if it exists.")

    args = parser.parse_args()

    try:
        create_index(args.json_file, args.index_output_path, args.overwrite)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Example usage:
    # python create_index.py path/to/your/corpus.json ./my_index_location --overwrite
