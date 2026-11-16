"""Qdrant client.

Two fallbacks for machines without Docker:
    QDRANT_URL=path:data/qdrant_local   local on-disk mode (one process at a time)
    QDRANT_URL=:memory:                 in-memory, lost when the process ends
"""
from qdrant_client import QdrantClient

from .config import QDRANT_URL


def get_client(url: str = QDRANT_URL) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(":memory:")
    if url.startswith("path:"):
        return QdrantClient(path=url[len("path:"):])
    return QdrantClient(url=url)
