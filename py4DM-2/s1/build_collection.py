"""Exercise 1.2 - Create the collection and insert the points.

Run:  python s1/build_collection.py

Expected output:
    vector size: 4096
    upserted 256 / 1184
    ...
    upserted 1184 / 1184
    points in sem_images: 1184

Check in the dashboard: http://localhost:8080/dashboard (browser on your laptop, ssh to the VM open)
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sem.config import COLLECTION, PAYLOAD_FIELDS
from sem.data import load_split
from sem.qdrant import get_client

BATCH = 256


def create_collection(client: QdrantClient, name: str, size: int) -> None:
    """Create `name` with cosine distance and vectors of length `size`.
    If the collection already exists, delete it first (we rebuild from scratch)."""
    # TODO (1.2.2): client.collection_exists(name) / client.delete_collection(name)
    #               client.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
    raise NotImplementedError("1.2.2 create_collection is not implemented yet")


def make_points(batch: dict, start: int) -> list[PointStruct]:
    """Turn a batch of dataset rows (a dict of columns) into PointStruct objects.

    id      = start + position in the batch (an integer)
    vector  = batch["embedding"][i]
    payload = {field: batch[field][i] for field in PAYLOAD_FIELDS}
    """
    # TODO (1.2.3)
    raise NotImplementedError("1.2.3 make_points is not implemented yet")


def main() -> None:
    index = load_split("index")
    # Keep only what goes to Qdrant: the payload fields and the vector (not the images)
    rows = index.select_columns(PAYLOAD_FIELDS + ["embedding"])
    client = get_client()                    # QdrantClient(url="http://localhost:8080")

    size = len(rows[0]["embedding"])
    print("vector size:", size)
    create_collection(client, COLLECTION, size)

    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]    # a dict: column name -> list of values
        # TODO (1.2.4): upsert the points of this batch
        #               client.upsert(COLLECTION, points=make_points(batch, start))
        raise NotImplementedError("1.2.4 upsert is not implemented yet")
        print(f"upserted {min(start + BATCH, len(rows))} / {len(rows)}")

    # TODO (1.2.5): payload indexes for the fields we will filter on
    #               client.create_payload_index(COLLECTION, "category", "keyword")
    #               client.create_payload_index(COLLECTION, "pixel_size_nm", "float")
    raise NotImplementedError("1.2.5 payload indexes are not implemented yet")

    print(f"points in {COLLECTION}:", client.count(COLLECTION).count)


if __name__ == "__main__":
    main()
