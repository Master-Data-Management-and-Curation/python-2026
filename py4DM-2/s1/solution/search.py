"""Exercises 1.3 and 1.4 - Search, with and without metadata filters (solution)."""
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from sem.config import COLLECTION
from sem.data import load_split
from sem.embeddings import embed_image, embed_text
from sem.qdrant import get_client

client = get_client()


def search(vector: list[float], limit: int = 5, query_filter: Filter | None = None):
    res = client.query_points(COLLECTION, query=vector, limit=limit,
                              with_payload=True, query_filter=query_filter)
    return res.points


def show(hits) -> None:
    for h in hits:
        p = h.payload
        print(f"{h.score:.3f}  {p['image_id']}  {p['category']:<28} {p['pixel_size_nm']:>7.2f} nm")


def main() -> None:
    query = load_split("query")
    first = list(query["category"]).index("nanowires")
    q = query[first]
    vec = embed_image(q["image"])

    # ---- 1.3 search -------------------------------------------------------
    print(f"--- image query {q['image_id']} ({q['category']})")
    show(search(vec))

    print("--- text: SEM image of elongated thin wires")
    show(search(embed_text("SEM image of elongated thin wires")))

    print("--- text: a cat on a sofa")
    show(search(embed_text("a cat on a sofa")))

    # ---- 1.4 filters ------------------------------------------------------
    is_nanowire = FieldCondition(key="category", match=MatchValue(value="nanowires"))
    is_fine = FieldCondition(key="pixel_size_nm", range=Range(lt=10.0))

    print("--- filter: nanowires")
    show(search(vec, query_filter=Filter(must=[is_nanowire])))
    print("--- filter: pixel size < 10 nm")
    show(search(vec, query_filter=Filter(must=[is_fine])))
    print("--- filter: both")
    show(search(vec, query_filter=Filter(must=[is_nanowire, is_fine])))

    is_fibre = FieldCondition(key="category", match=MatchValue(value="fibres"))
    is_mid = FieldCondition(key="pixel_size_nm", range=Range(gte=5.0, lte=20.0))
    n = client.count(COLLECTION, count_filter=Filter(must=[is_fibre, is_mid]), exact=True).count
    print("fibres with pixel size in [5, 20] nm:", n)


if __name__ == "__main__":
    main()
