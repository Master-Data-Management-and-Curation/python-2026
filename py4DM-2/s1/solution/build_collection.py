"""Exercise 1.2 - Create the collection and insert the points (solution)."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sem.config import COLLECTION, PAYLOAD_FIELDS
from sem.data import load_split
from sem.qdrant import get_client

BATCH = 256


def create_collection(client: QdrantClient, name: str, size: int) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))


def make_points(batch: dict, start: int) -> list[PointStruct]:
    n = len(batch["image_id"])
    return [PointStruct(id=start + i,
                        vector=batch["embedding"][i],
                        payload={field: batch[field][i] for field in PAYLOAD_FIELDS})
            for i in range(n)]


def main(client: QdrantClient | None = None) -> int:
    index = load_split("index")
    rows = index.select_columns(PAYLOAD_FIELDS + ["embedding"])
    client = client or get_client()

    size = len(rows[0]["embedding"])
    print("vector size:", size)
    create_collection(client, COLLECTION, size)

    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        client.upsert(COLLECTION, points=make_points(batch, start))
        print(f"upserted {min(start + BATCH, len(rows))} / {len(rows)}")

    client.create_payload_index(COLLECTION, "category", "keyword")
    client.create_payload_index(COLLECTION, "pixel_size_nm", "float")

    n = client.count(COLLECTION).count
    print(f"points in {COLLECTION}:", n)
    return n


if __name__ == "__main__":
    main()
