"""Exercises 2.1-2.4 - RAG from scratch: retrieve, build the context, call the LLM, log.

Run:  python s2/rag.py

Expected output (the answers are generated: your wording will differ):
    Retrieved images (top-5):                                                     <- 2.1 and 2.2
    1. sem_00230 category=fibres pixel_size=101.01 nm mag=n/a wd=n/a eht=n/a detector=n/a score=0.438
    2. sem_00203 category=fibres pixel_size=392.16 nm mag=n/a wd=n/a eht=n/a detector=n/a score=0.432
    ...
    Q: Which categories and pixel sizes appear among images of elongated thin wires?
    A: ... fibres ... sem_00230 ... sem_00203 ... mems_devices_and_electrodes ... sem_00402 ...   <- 2.3

    Q: What do the images similar to this one have in common?  (image sem_q_01205, filter: nanowires)
    A: ... nanowires ... sem_00487 ... sem_00580 ...                                         <- 2.4

    provenance log: output/provenance.jsonl                     (one new line per answer() call)
"""
import json
import time
from pathlib import Path

from openai import OpenAI
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sem.config import COLLECTION, LLM_MODEL, ORFEO_API_KEY, ORFEO_BASE_URL, ROOT
from sem.data import load_split
from sem.embeddings import embed_image, embed_text
from sem.qdrant import get_client

client = get_client()                        # Qdrant, as in Session 1
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
    """Top-k points for the question, or for `image` when one is given."""
    # TODO (2.1.1): vector = embed_image(image) if image is not None else embed_text(question)
    # TODO (2.1.2): return client.query_points(COLLECTION, query=vector, limit=k,
    #                                          with_payload=True, query_filter=query_filter).points
    raise NotImplementedError("2.1 retrieve is not implemented yet")


def build_context(hits) -> str:
    """The hits as text for the LLM: a header line, then one numbered line per hit."""
    # TODO (2.2.1): lines = [f"Retrieved images (top-{len(hits)}):"]
    # TODO (2.2.2): for i, h in enumerate(hits, start=1): append one line with
    #               image_id, category, pixel_size (.2f nm), mag, wd (mm), eht (kV), detector, score (.3f).
    #               Use fmt(...) for mag, wd_mm, eht_kv, detector: they can be None.
    # TODO (2.2.3): return "\n".join(lines)
    raise NotImplementedError("2.2 build_context is not implemented yet")


def call_llm(question: str, context: str) -> str:
    """One chat-completions request; returns the text of the answer."""
    # TODO (2.3.1): messages = a "system" message with SYSTEM_PROMPT and a "user" message
    #               with the context, a blank line, then "Question: ..."
    # TODO (2.3.2): response = llm.chat.completions.create(model=LLM_MODEL, messages=messages,
    #                                                      temperature=0, max_tokens=MAX_TOKENS)
    # TODO (2.3.3): return response.choices[0].message.content
    raise NotImplementedError("2.3 call_llm is not implemented yet")


def answer(question: str, k: int = 5, query_filter: Filter | None = None,
           image=None, image_id: str | None = None) -> str:
    """retrieve -> build_context -> call_llm, and one line of provenance in LOG."""
    # TODO (2.4.1): hits = retrieve(...); text = call_llm(question, build_context(hits))
    # TODO (2.4.2): record = {"time", "question", "query_image": image_id,
    #               "filter": query_filter.model_dump(exclude_none=True) if query_filter else None,
    #               "top_k", "hits": [image_ids], "scores": [rounded to 3], "model": LLM_MODEL, "answer"}
    # TODO (2.4.3): LOG.parent.mkdir(exist_ok=True); append json.dumps(record) + "\n" to LOG
    #               (open(LOG, "a")), then return text
    raise NotImplementedError("2.4 answer is not implemented yet")


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
