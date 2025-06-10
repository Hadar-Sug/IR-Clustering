import gzip
import pytest
import sys
import types

# Provide dummy ir_datasets module required by loader import
sys.modules.setdefault("ir_datasets", types.ModuleType("ir_datasets"))

from src.data_loader.loader import DataLoader


def test_dataloader_missing_docs(tmp_path):
    qfile = tmp_path / "queries.tsv"
    qfile.write_text("q1\ttext\n")

    topfile = tmp_path / "top100.gz"
    with gzip.open(topfile, "wt", encoding="utf8") as f:
        f.write("q1 Q0 D1 0 1.0\n")

    qrelsfile = tmp_path / "qrels.txt"
    qrelsfile.write_text("q1 0 D1 1\n")

    missing_json = tmp_path / "docs.json"
    missing_trec = tmp_path / "docs.trec.gz"

    with pytest.raises(FileNotFoundError):
        DataLoader(
            str(qfile),
            str(topfile),
            str(qrelsfile),
            str(missing_trec),
            str(missing_json)
        )

