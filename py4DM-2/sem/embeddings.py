"""Embeddings of images and text, computed by a model served on ORFEO.

One request = one embedding. The request body is a short conversation:
    system     the instruction given to the embedding model
    user       what we want to embed: a text, or an image ( encoded in base64 as a data URL)
    assistant  empty: the embedding is read where the answer would begin
"""
import base64
import io
from typing import Sequence

import numpy as np
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from PIL import Image

from .config import EMBED_MODEL, ORFEO_API_KEY, ORFEO_BASE_URL

INSTRUCTION = "Represent the user's input."

client = OpenAI(base_url=ORFEO_BASE_URL, api_key=ORFEO_API_KEY or "missing-token", timeout=120)


def _embed(content: list[dict], instruction: str = INSTRUCTION) -> list[float]:
    """Send one chat-embedding request and return the vector."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]
    response = client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body={"model": EMBED_MODEL, "messages": messages, "encoding_format": "float",
              "continue_final_message": True, "add_special_tokens": True},
    )
    return response.data[0].embedding


def image_to_data_url(image: Image.Image) -> str:
    """PIL image -> 'data:image/png;base64,...' (how an image travels inside JSON)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def embed_image(image: Image.Image | str) -> list[float]:
    """Embed a PIL image (or an image path). Returns a list of floats."""
    if isinstance(image, str):
        image = Image.open(image)
    return _embed([{"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                   {"type": "text", "text": ""}])


def embed_text(text: str, instruction: str = INSTRUCTION) -> list[float]:
    """Embed a sentence in the same space as the images."""
    return _embed([{"type": "text", "text": text}], instruction)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
