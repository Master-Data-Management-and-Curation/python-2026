"""Exercise 1.1 - Explore the dataset (solution)."""
from collections import Counter

from sem.data import load_split
from sem.embeddings import cosine, embed_image


def main() -> None:
    index = load_split("index")
    query = load_split("query")

    print(f"index: {len(index)} rows   query: {len(query)} rows")
    print("columns:", index.column_names)

    row = index[0]
    print(row["image_id"], row["category"], f'{row["pixel_size_nm"]} nm',
          "  embedding length", len(row["embedding"]))

    print(Counter(index["category"]))

    q = query[0]
    now = embed_image(q["image"])                 # one request to ORFEO
    print(f'{q["image_id"]} cosine(now, dataset) = {cosine(now, q["embedding"]):.4f}')


if __name__ == "__main__":
    main()
