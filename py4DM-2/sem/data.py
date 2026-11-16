"""Loading the course dataset.

The dataset has two splits:
  index  1,184 images that go into Qdrant
  query     50 images that are used for queries

Columns:
  image_id, filename, image (PIL, 512x512), category, split,
  pixel_size_nm, pixel_size_source, mag, wd_mm, eht_kv, detector, acquired_at,
  embedding, source_image_id, source_filename, related_doi, md5
mag, wd_mm, eht_kv, detector and acquired_at are None when the instrument did not record them.
"""
from pathlib import Path

from datasets import Dataset, load_dataset

from .config import DATASET


def load_split(split: str, dataset: str = DATASET) -> Dataset:
    """Return one split ("index" or "query") as a `datasets.Dataset`.

    `dataset` is a Hugging Face dataset id, or a local folder with
    `<split>.parquet` files (useful before the dataset is published).
    """
    local = Path(dataset)
    if local.is_dir():
        files = {s: str(local / f"{s}.parquet") for s in ("index", "query") if (local / f"{s}.parquet").exists()}
        return load_dataset("parquet", data_files=files, split=split)
    return load_dataset(dataset, split=split)
