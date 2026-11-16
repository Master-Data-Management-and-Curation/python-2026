"""Exercise 2.5 - Experiments: measure retrieval, then look at the answers.

Run:  python s2/experiments.py          (after s2/rag.py works)

Do 2.5.1, then run (2.5.2 and 2.5.3 are given). Then 2.5.4.

Expected output (the answers are generated: your wording will differ):
    image queries   precision@1  = 0.70
    image queries   precision@5  = 0.60
    image queries   precision@10 = 0.57

    k=1: ... sem_00354 ... mems_devices_and_electrodes, not nanowires ... cannot be answered ...
    k=5: The retrieved images do not include any images categorized as nanowires ...
    k=20: ... the only nanowire image is sem_00587 ... 10.0 kV ... InLens.
    k=5, filter nanowires: ... 10 kV with InLens and 5 kV with SE2 ... (n/a) ...
    out of domain: The retrieved images do not contain any information about ...
    ...
    NotImplementedError: 2.5.4 text_precision is not implemented yet

After 2.5.4, two more lines at the end:
    text '{}'                    mean precision@5 = 0.38   biologic=0.0 fibres=0.6 ...
    text 'SEM image of {}'       mean precision@5 = 0.40   biologic=0.4 fibres=0.8 ...
"""
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sem.config import CATEGORIES, COLLECTION
from sem.data import load_split
from sem.embeddings import embed_text

from rag import answer, client          # your s2/rag.py, next to this file


def precision_at_k(hits, expected: str) -> float:
    """Fraction of the hits whose category is the expected one."""
    # TODO (2.5.1): return sum(h.payload["category"] == expected for h in hits) / len(hits)
    raise NotImplementedError("2.5.1 precision_at_k is not implemented yet")


def image_precision(query, k: int) -> float:
    """Mean precision@k over the query images. Their vectors are in the dataset: no call to ORFEO."""
    scores = []
    for q in query:
        hits = client.query_points(COLLECTION, query=q["embedding"], limit=k, with_payload=True).points
        scores.append(precision_at_k(hits, q["category"]))
    return sum(scores) / len(scores)


def text_precision(template: str, k: int) -> dict[str, float]:
    """precision@k of one sentence per category, e.g. template "SEM image of {}"."""
    # TODO (2.5.4): for each cat in CATEGORIES: embed template.format(cat.replace("_", " ")),
    #               query_points with limit=k, store precision_at_k(hits, cat) in a dict
    raise NotImplementedError("2.5.4 text_precision is not implemented yet")


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
