"""Exercise 1.1 - Explore the dataset.

Run:  python s1/explore.py

Expected output (values may differ slightly):
    index: 1184 rows   query: 50 rows
    columns: ['image_id', 'filename', 'image', 'category', 'split', 'pixel_size_nm', ...]
    sem_00001 biological 37.037 nm   embedding length 4096
    Counter({'biological': 120, 'films_coated_surface': 120, ..., 'fibres': 113})
    sem_q_01185 cosine(now, dataset) = 1.0000
"""
from collections import Counter

from sem.data import load_split
from sem.embeddings import cosine, embed_image


def main() -> None:
    # 1. Load the two splits
    index = load_split("index")
    query = load_split("query")

    # 2. Rows per split and column names
    # TODO (1.1.2): print the number of rows of each split and index.column_names
    raise NotImplementedError("1.1.2 rows and columns: not implemented yet")

    # 3. One row: image_id, category, pixel_size_nm, length of the embedding
    # TODO (1.1.3): row = index[0]; print the four values
    raise NotImplementedError("1.1.3 is not implemented yet")

    # 4. Rows per category (hint: Counter(index["category"]))
    # TODO (1.1.4)
    raise NotImplementedError("1.1.4 is not implemented yet")

    # 5. Reproducibility: ask ORFEO for the embedding of one query image and
    #    compare it with the embedding stored in the dataset for the same row.
    # TODO (1.1.5): q = query[0]; now = embed_image(q["image"]); print cosine(now, q["embedding"])
    raise NotImplementedError("1.1.5 is not implemented yet")


if __name__ == "__main__":
    main()
