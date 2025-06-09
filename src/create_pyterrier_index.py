#!/usr/bin/env python3
import json
import os
import argparse
import pandas as pd
import pyterrier as pt

def create_index(json_file_path: str, index_dir: str, overwrite: bool = False):
    """
    Creates (or reloads) a PyTerrier index from a JSON file.

    Args:
      json_file_path: path to a JSON file of {doc_id: doc_text}
      index_dir:      directory under which the Terrier index will be stored
      overwrite:      if True, delete any existing index at index_dir first
    """
    # 1) start Terrier
    if not pt.started():
        pt.init()

    # 2) load and validate JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    if not isinstance(corpus, dict):
        raise ValueError("JSON must be a dict of {doc_id: text}")

    print(f"Loaded {len(corpus)} documents from {json_file_path!r}")

    # 3) build a DataFrame with 'docno' and 'text'
    df = pd.DataFrame([
        {"docno": str(doc_id), "text": text}
        for doc_id, text in corpus.items()
    ])

    # 4) ensure output directory exists
    if overwrite and os.path.isdir(index_dir):
        print(f"Overwriting existing index at {index_dir!r}")
    os.makedirs(index_dir, exist_ok=True)

    # 5) index!
    indexer = pt.IterDictIndexer(index_dir, overwrite=overwrite)
    index_ref = indexer.index(df.to_dict(orient="records"))
    print(f"Index built at: {index_ref.toString()}")

    return index_ref

def main():
    p = argparse.ArgumentParser(
        description="Build/load a PyTerrier index from JSON (doc_id→text)."
    )
    p.add_argument("json_file",
                   help="path to JSON file containing {'id': 'text', …}")
    p.add_argument("index_dir",
                   help="directory under which to build the Terrier index")
    p.add_argument("--overwrite", "-O", action="store_true",
                   help="if set, delete any existing index in index_dir first")
    args = p.parse_args()

    try:
        create_index(args.json_file, args.index_dir, overwrite=args.overwrite)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

if __name__ == "__main__":
    main()
