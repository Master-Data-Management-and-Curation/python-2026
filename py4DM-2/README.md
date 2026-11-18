# Python for Data Management II — MDMC 2026–27

Python workflows to manage scientific data through vector search, RAG and model serving.


| Session | Folder | Exercise |
|---|---|---|
| 1 · Vector databases and semantic retrieval | `s1/` | a Qdrant collection of SEM images |
| 2 · RAG from scratch | `s2/` | retrieval + LLM, no framework |
| 3 · Model serving  | `s3/` |  a classifier behind `POST /predict` |

Each practice session folder has a `README.md` with the exercises, starter files with `# TODO` stubs, and a `solution/`.

## Setup

See [`s1/README.md`](s1/README.md), section 2. In short: `pip install -e .`,  ORFEO API token in `.env`,
`docker compose up -d` (Qdrant on port 8080 of the VM, dashboard at <http://localhost:8080/dashboard> ), then `python verify.py`.

## Model services

LLM models are served on ORFEO behind an OpenAI-compatible API (vLLM):
`https://orfeo-llm.areasciencepark.it/vllm/v1`, available using the `ORFEO_API_KEY` token.

## Dataset

SEM images from the CNR-IOM corpus from which the NFFA-EUROPE SEM dataset was published (Aversa et al., *Scientific Data* 5, 180172 (2018), CC-BY 4.0,
DOI 10.23728/b2share.19cc2afd23e34b92b36a1dfd0113a89f). 

Course subset: 1234 images in 10 categories, distributed in up to 125 images per category. Images are 512 × 512 crops of the frame at full resolution.

Published on Hugging Face: <https://huggingface.co/datasets/RitAreaSciencePark/sem-mdmc-2026>
