"""Session 3 - classify the dataset images through the BentoML API and write the label
into the Qdrant payload.

Run:  1) bentoml serve              (in s3/, leave it running)
      2) python classify.py         (Qdrant up, sem_images built)

The model lives in the service (s3/service.py): this script sends each image to its
/classify endpoint and collects the answers. Qdrant is used at the end, only to write
    predicted_category    the category returned by the service
    predicted_confidence  its softmax probability (0-1)
    classifier            the model id returned by the service
into the payload of each image, matched by image_id. The ground truth `category`
is left untouched.
"""
from collections import defaultdict

import bentoml
import numpy as np
from qdrant_client.models import (FieldCondition, Filter, MatchValue, PayloadSchemaType,
                                  SetPayload, SetPayloadOperation)

from sem.config import COLLECTION
from sem.data import load_split
from sem.qdrant import get_client

API_URL = "http://localhost:8081"
TRAIN_SPLIT = "index"       # the split loaded into Qdrant


def norm(label: str) -> str:
    """'Films and Coated Surface' / 'films_and_coated_surface' -> same string."""
    return label.lower().replace("-", " ").replace("_", " ").strip().replace(" ", "_")


def predict_split(client, ds) -> tuple[list[tuple[str, float]], str]:
    """One request per image to the service. Returns ([(label, confidence)], model id)."""
    out, model = [], None
    for i in range(len(ds)):
        r = client.classify(image=ds[i]["image"])
        out.append((r["category"], r["confidence"]))
        model = r["model"]
    return out, model


def write_labels(image_ids, predictions, classifier: str, chunk: int = 100) -> None:
    """One set_payload per image, on the point whose payload image_id matches."""
    client = get_client()
    # a filter on image_id scans every point unless the field is indexed (safe to repeat)
    client.create_payload_index(COLLECTION, field_name="image_id",
                                field_schema=PayloadSchemaType.KEYWORD)
    ops = [SetPayloadOperation(set_payload=SetPayload(
        payload={"predicted_category": label, "predicted_confidence": round(conf, 3),
                 "classifier": classifier},
        filter=Filter(must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))])))
        for image_id, (label, conf) in zip(image_ids, predictions)]
    for i in range(0, len(ops), chunk):         # small requests: the client times out after a few seconds
        client.batch_update_points(COLLECTION, ops[i:i + chunk])


def main() -> None:
    index = load_split(TRAIN_SPLIT).select_columns(["image_id", "category", "image"])
    query = load_split("query").select_columns(["image_id", "category", "image"])

    with bentoml.SyncHTTPClient(API_URL) as client:
        for name, ds in (("query", query), ("index", index)):
            pred, model = predict_split(client, ds)
            ok = [norm(p[0]) == norm(c) for p, c in zip(pred, ds["category"])]
            print(f"{name:<6} accuracy = {np.mean(ok):.2f}  (n={len(ds)})")
            if name == "index":
                index_pred, index_model, per_cat = pred, model, defaultdict(list)
                for c, o in zip(ds["category"], ok):
                    per_cat[c].append(o)

    predicted = {norm(p[0]) for p in index_pred}
    data_labels = {norm(c) for c in index["category"]}
    if predicted - data_labels:
        print("WARNING: predicted labels not in the dataset, accuracy is not reliable:",
              sorted(predicted - data_labels))
    print(f"\n  {'category':<28} {'accuracy':>8} {'n':>5}")
    print("  " + "-" * 43)
    for cat, v in per_cat.items():
        print(f"  {cat[:24]:<24} {np.mean(v):.2f}  (n={len(v)})")

    write_labels(index["image_id"], index_pred, index_model)
    print(f"written predicted_category / predicted_confidence for {len(index_pred)} points in '{COLLECTION}'")


if __name__ == "__main__":
    main()
