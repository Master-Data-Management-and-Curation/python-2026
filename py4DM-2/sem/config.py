"""Settings for the course, read from the environment (and from a .env file if present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Hugging Face dataset id, or a local folder containing index.parquet and query.parquet
DATASET = os.getenv("DATASET", "RitAreaSciencePark/sem-mdmc-2026")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:8080")
COLLECTION = os.getenv("COLLECTION", "sem_images")

# Model services on ORFEO (OpenAI-compatible API served by vLLM)
ORFEO_BASE_URL = os.getenv("ORFEO_BASE_URL", "https://orfeo-llm.areasciencepark.it/vllm/v1")
ORFEO_API_KEY = os.getenv("ORFEO_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash-0731")  

# Payload fields copied from the dataset into Qdrant
PAYLOAD_FIELDS = ["image_id", "filename", "category", "pixel_size_nm", "pixel_size_source",
                  "mag", "wd_mm", "eht_kv", "detector"]

CATEGORIES = ["biological", "fibres", "films_coated_surface", "mems_devices_and_electrodes",
              "nanowires", "particles", "patterned_surface", "porous_sponge", "powder", "tips"]
