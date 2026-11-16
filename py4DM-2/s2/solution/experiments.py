"""Exercise 2.5 - Experiments: measure retrieval, then look at the answers (solution)."""
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sem.config import CATEGORIES, COLLECTION
from sem.data import load_split
from sem.embeddings import embed_text

from rag import answer, client          # s2/solution/rag.py, next to this file


def precision_at_k(hits, expected: str) -> float:
    return sum(h.payload["category"] == expected for h in hits) / len(hits)


def image_precision(query, k: int) -> float:
    scores = []
    for q in query:
        hits = client.query_points(COLLECTION, query=q["embedding"], limit=k, with_payload=True).points
        scores.append(precision_at_k(hits, q["category"]))
    return sum(scores) / len(scores)


def text_precision(template: str, k: int) -> dict[str, float]:
    result = {}
    for cat in CATEGORIES:
        hits = client.query_points(COLLECTION, query=embed_text(template.format(cat.replace("_", " "))),
                                   limit=k, with_payload=True).points
        result[cat] = precision_at_k(hits, cat)
    return result


def main() -> None:
    query = load_split("query").select_columns(["image_id", "category", "embedding"])

    # 2.5.2 image queries: the 50 query images, their stored vectors
    for k in (1, 5, 10):
        print(f"image queries   precision@{k:<2} = {image_precision(query, k):.2f}")

    # 2.5.3 the same question with k = 1, 5, 20; a filter; an out-of-domain question
    question = "Which instrument settings (EHT, detector) were used for nanowire images?"
    for k in (1, 5, 20):
        print(f"\nk={k}:", answer(question, k=k))
    only_nanowires = Filter(must=[FieldCondition(key="category", match=MatchValue(value="nanowires"))])
    print("\nk=5, filter nanowires:", answer(question, query_filter=only_nanowires))
    print("\nout of domain:", answer("What is the boiling point of water at the top of Mount Everest?"))

    # 2.5.4 text queries: one sentence per category; change the wording and compare
    for template in ("{}", "SEM image of {}"):
        p = text_precision(template, k=5)
        print(f"text '{template}'".ljust(28), f"mean precision@5 = {sum(p.values()) / len(p):.2f}  ",
              " ".join(f"{c[:8]}={v:.1f}" for c, v in p.items()))


if __name__ == "__main__":
    main()
