"""Check that the environment is ready for Session 1.

    python verify.py

Checks: Qdrant reachable, ORFEO endpoint reachable with API token, embedding and chat
models available, dataset loads, embedding dimension consistent for images
and text, a fresh embedding reproduces the one stored in the dataset.
Prints "Verified: ..." on success, a clear message and exit code 1 otherwise.
"""
import sys
import time

from sem.config import COLLECTION, DATASET, EMBED_MODEL, LLM_MODEL, ORFEO_API_KEY, ORFEO_BASE_URL, QDRANT_URL
from sem.data import load_split
from sem.embeddings import client as orfeo
from sem.embeddings import cosine, embed_image, embed_text
from sem.qdrant import get_client

EXPECTED = {"index": 1184, "query": 50}


def fail(msg: str) -> None:
    print(f"\nNOT verified: {msg}")
    sys.exit(1)


def main() -> None:
    print(f"dataset  {DATASET}\nqdrant   {QDRANT_URL}\norfeo    {ORFEO_BASE_URL}\n")

    # 1. Qdrant 
    client = get_client()
    for attempt in range(15):
        try:
            names = [c.name for c in client.get_collections().collections]
            break
        except Exception as e:  
            error = e
            time.sleep(2)
    else:
        fail(f"Qdrant not reachable at {QDRANT_URL}: {error}\n  -> docker compose up -d   (docker not found? see s1/README.md, section 2)")
    print(f"[ok] qdrant: {len(names)} collection(s) {names}" + (f"  ({COLLECTION} present)" if COLLECTION in names else ""))

    # 2. ORFEO endpoint and token
    if not ORFEO_API_KEY:
        fail("ORFEO_API_KEY is empty\n  -> cp .env.example .env, then paste your token in .env")
    try:
        models = [m.id for m in orfeo.models.list().data]
    except Exception as e:  
        fail(f"ORFEO endpoint not usable at {ORFEO_BASE_URL}: {e}\n  -> check the token in .env and the network")
    if EMBED_MODEL not in models:
        fail(f"model {EMBED_MODEL} is not served by the endpoint. Available: {models}")
    if LLM_MODEL not in models:
        fail(f"chat model LLM_MODEL={LLM_MODEL!r} is not served by the endpoint (Session 2). Available: {models}")
    print(f"[ok] orfeo: token accepted, models available: {models}")

    # 3. Dataset
    try:
        index = load_split("index")
        query = load_split("query")
    except Exception as e:  
        fail(f"dataset {DATASET} cannot be loaded: {e}\n  -> check DATASET in .env and the network")
    sizes = {"index": len(index), "query": len(query)}
    flag = "ok" if sizes == EXPECTED else "warn"
    print(f"[{flag}] dataset: index={sizes['index']} query={sizes['query']}" + ("" if flag == "ok" else f"  (expected {EXPECTED}; fine for a fake/local dataset)"))

    # 4. Embedding calls: text and image, same dimension as the stored vectors
    q = query[0]
    dim = len(q["embedding"])
    try:
        text_vector = embed_text("a test sentence")
        image_vector = embed_image(q["image"])
    except Exception as e:  
        fail(f"embedding request failed: {e}")
    if len(text_vector) != dim or len(image_vector) != dim:
        fail(f"embedding dimension mismatch: dataset {dim}, text {len(text_vector)}, image {len(image_vector)} (different model?)")
    print(f"[ok] embedding dimension: {dim} (images and text)")

    # 5. Reproducibility: the same image gives the stored vector
    c = cosine(image_vector, q["embedding"])
    if c < 0.99:
        fail(f"the embedding of {q['image_id']} computed now differs from the stored one (cosine={c:.4f}): different model or request")
    print(f"[ok] reproducibility: cosine(now, dataset) = {c:.4f} on {q['image_id']}")

    print(f"\nVerified: qdrant, orfeo, dataset ({sizes['index']} index + {sizes['query']} query), embedding model {EMBED_MODEL}, chat model {LLM_MODEL}")


if __name__ == "__main__":
    main()
