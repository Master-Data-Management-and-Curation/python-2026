"""Exercises 2.1-2.4 - RAG from scratch (solution)."""
import json
import time
from pathlib import Path

from openai import OpenAI
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sem.config import COLLECTION, LLM_MODEL, ORFEO_API_KEY, ORFEO_BASE_URL, ROOT
from sem.data import load_split
from sem.embeddings import embed_image, embed_text
from sem.qdrant import get_client

client = get_client()
llm = OpenAI(base_url=ORFEO_BASE_URL, api_key=ORFEO_API_KEY or "missing-token", timeout=120)

LOG = ROOT / "output" / "provenance.jsonl"
MAX_TOKENS = 1000           # the model reasons first: those tokens count too
SYSTEM_PROMPT = (
    "You answer questions about a collection of SEM images. "
    "Use only the retrieved images listed in the context, and cite their image_id. "
    "If the context does not answer the question, say so."
)


def fmt(value, unit: str = "") -> str:
    """A payload value as text; 'n/a' when the instrument did not record it."""
    return "n/a" if value is None else f"{value}{unit}"


def retrieve(question: str, k: int = 5, query_filter: Filter | None = None, image=None):
    vector = embed_image(image) if image is not None else embed_text(question)
    res = client.query_points(COLLECTION, query=vector, limit=k,
                              with_payload=True, query_filter=query_filter)
    return res.points


def build_context(hits) -> str:
    lines = [f"Retrieved images (top-{len(hits)}):"]
    for i, h in enumerate(hits, start=1):
        p = h.payload
        lines.append(f"{i}. {p['image_id']} category={p['category']} "
                     f"pixel_size={p['pixel_size_nm']:.2f} nm mag={fmt(p['mag'])} "
                     f"wd={fmt(p['wd_mm'], ' mm')} eht={fmt(p['eht_kv'], ' kV')} "
                     f"detector={fmt(p['detector'])} score={h.score:.3f}")
    return "\n".join(lines)


def call_llm(question: str, context: str) -> str:
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def answer(question: str, k: int = 5, query_filter: Filter | None = None,
           image=None, image_id: str | None = None) -> str:
    hits = retrieve(question, k, query_filter, image)
    text = call_llm(question, build_context(hits))
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "query_image": image_id,
        "filter": query_filter.model_dump(exclude_none=True) if query_filter else None,
        "top_k": k,
        "hits": [h.payload["image_id"] for h in hits],
        "scores": [round(h.score, 3) for h in hits],
        "model": LLM_MODEL,
        "answer": text,
    }
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    return text


def main() -> None:
    question = "Which categories and pixel sizes appear among images of elongated thin wires?"
    hits = retrieve(question)                               # 2.1
    context = build_context(hits)                           # 2.2
    print(context)
    print("\nQ:", question)
    print("A:", call_llm(question, context))                # 2.3

    # 2.4: the same three steps in one call, with a provenance record

    query = load_split("query")
    q = query[list(query["category"]).index("nanowires")]
    is_nanowire = Filter(must=[FieldCondition(key="category", match=MatchValue(value="nanowires"))])
    question = "What do the images similar to this one have in common?"
    print(f"\nQ: {question}  (image {q['image_id']}, filter: nanowires)")
    print("A:", answer(question, query_filter=is_nanowire, image=q["image"], image_id=q["image_id"]))
    print(f"\nprovenance log: {Path(LOG).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
