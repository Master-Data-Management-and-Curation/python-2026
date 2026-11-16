"""Exercises 1.3 and 1.4 - Search, with and without metadata filters.

Run:  python s1/search.py

Expected output (one block per query):
    --- image query sem_q_01205 (nanowires)
    0.785  sem_00618  particles                      36.70 nm
    0.766  sem_00487  nanowires                      36.70 nm
    ...
    --- text: SEM image of elongated thin wires
    0.623  sem_00148  fibres                        500.00 nm
    ...
    --- filter: nanowires
    ...
    fibres with pixel size in [5, 20] nm: 3
"""
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from sem.config import COLLECTION
from sem.data import load_split
from sem.embeddings import embed_image, embed_text
from sem.qdrant import get_client

client = get_client()                        # QdrantClient(url="http://localhost:8080")


def search(vector: list[float], limit: int = 5, query_filter: Filter | None = None):
    """Top-`limit` points by cosine similarity to `vector`, payload included.
    Returns a list of ScoredPoint (fields: id, score, payload)."""
    # TODO (1.3.2): return client.query_points(COLLECTION, query=vector, limit=limit,
    #                                          with_payload=True, query_filter=query_filter).points
    raise NotImplementedError("1.3.2 search is not implemented yet")


def show(hits) -> None:
    """Print one line per hit: score, image_id, category, pixel_size_nm."""
    # TODO (1.3.3): for h in hits: use h.score and h.payload["image_id"], ...
    raise NotImplementedError("1.3.3 show is not implemented yet")


def main() -> None:
    # The first nanowire image of the query split (never indexed)
    query = load_split("query")
    first = list(query["category"]).index("nanowires")
    q = query[first]
    vec = embed_image(q["image"])

    # ---- 1.3 search -------------------------------------------------------
    print(f"--- image query {q['image_id']} ({q['category']})")
    show(search(vec))

    print("--- text: SEM image of elongated thin wires")
    # TODO (1.3.4): embed the sentence with embed_text, then search and show
    raise NotImplementedError("1.3.4 text query is not implemented yet")

    print("--- text: a cat on a sofa")
    # TODO (1.3.5): same with the nonsense sentence. Look at the scores.
    raise NotImplementedError("1.3.5 nonsense query is not implemented yet")

    # ---- 1.4 filters ------------------------------------------------------
    # TODO (1.4.1): is_nanowire = FieldCondition(key="category", match=MatchValue(value="nanowires"))
    # TODO (1.4.2): is_fine = FieldCondition(key="pixel_size_nm", range=Range(lt=10.0))
    # TODO (1.4.3): run the image query (vec) three times with query_filter=Filter(must=[...]):
    #               nanowires only, pixel size < 10 nm only, both
    raise NotImplementedError("1.4.1-1.4.3 filters are not implemented yet")

    # TODO (1.4.5): count fibres with pixel_size_nm in [5, 20] without any vector:
    #               client.count(COLLECTION, count_filter=Filter(must=[...]), exact=True).count
    raise NotImplementedError("1.4.5 count is not implemented yet")


if __name__ == "__main__":
    main()
