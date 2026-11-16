# Session 1 — Vector databases and semantic retrieval

## 1. Exercise Outcome

A Qdrant collection, `sem_images`, with 1,184 SEM images represented by their
embedding vectors and two layers of metadata: instrument and
curated.

The embedding model is `Qwen/Qwen3-VL-Embedding-8B`, served on ORFEO.
You call it with your token and the `openai` Python package. The embeddings of the 1,184 index images are pre-computed and published in the dataset: only queries need to be embedded in class.

## 2. Setup

**Installation**

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER       # docker without sudo, from your next login
```

Now **exit and ssh again**: the group change takes effect only in a new login. Then check:

```bash
docker --version                    # Docker version 29.1.3, ...
docker compose version              # Docker Compose version 2.40.3...
docker run --rm hello-world         # Hello from Docker!   (without sudo)
```

If the last one says `permission denied ... docker.sock`, you did not log in again.

Then clone the course code. It is the folder `py4DM-2` of the master's `python-2026` repository:

```bash
git clone https://github.com/Master-Data-Management-and-Curation/python-2026.git
cd python-2026/py4DM-2
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
cp .env.example .env                # then paste your ORFEO token after ORFEO_API_KEY=
```

If `python-2026` is already on your VM (from Py4DM I), run `git pull` inside
it, then go on from `cd python-2026/py4DM-2`.

**Start**

```bash
cd python-2026/py4DM-2 && source .venv/bin/activate
docker compose up -d                # Qdrant on port 8080 of the VM
python verify.py
```

✅ `verify.py` ends with `Verified: qdrant, orfeo, dataset (1184 index + 50 query), ...`.
If it ends with `NOT verified`, the message says what is missing and what to do.

The first `verify.py` downloads the dataset; it is cached afterwards.
Qdrant dashboard: <http://localhost:8080/dashboard>, in the browser **on your laptop** while ssh
is open: your ssh config forwards port 8080 of the VM to your laptop. Your code on the VM uses the same
address (`QDRANT_URL` in `.env`).

## 3. Project structure

```
.
├── .env                      # your token and settings (copied from .env.example)
├── compose.yaml              # Qdrant container
├── verify.py                 # checks the environment
├── sem/                      # shared helpers
│   ├── config.py             # settings
│   ├── data.py               # load dataset
│   ├── embeddings.py         # embeddings, cosine
│   └── qdrant.py             # qdrant client
└── s1/
    ├── explore.py            # TODO: Ex. 1.1
    ├── build_collection.py   # TODO: Ex. 1.2
    ├── search.py             # TODO: Ex. 1.3-4
    └── solution/             # Solutions
```

Every exercise to complete is marked `# TODO (x.y.z)` and raises `NotImplementedError` until you replace it.
Run the scripts from the project root: `python s1/explore.py`.

## 4. The dataset

| Column | Layer | Example |
|---|---|---|
| `image_id`, `filename` | identity | `sem_00421`, `sem_00421.png` |
| `image` | raw data | PIL image, 512 × 512 grayscale: a crop of the frame at full resolution, without the instrument bar |
| `category` | curated | `nanowires` |
| `pixel_size_nm`, `mag`, `wd_mm`, `eht_kv`, `detector` | instrument | `25.09`, `14630`, `3.198`, `2.0`, `InLens` |
| `pixel_size_source` | provenance (of `pixel_size_nm`) | `instrument` (tag 34118, 705 images) or `computed` (529 images: the instrument did not write it) |
| `acquired_at` | instrument | `2015-08-21T10:19:47` |
| `embedding` | derived | 4096 floats |
| `split` | — | `index` (1,184 rows) or `query` (50 rows, never indexed) |
| `source_image_id`, `source_filename`, `related_doi`, `md5` | provenance | `59348`, `A66_8DIV_CNT_45.tif` |

> Not every image has every instrument field because the instrument did not record them: real metadata is incomplete.


## 5. Exercise 1.1 — Explore the dataset

**File:** `s1/explore.py`

| Step | Do | Hint |
|---|---|---|
| 1.1.2 | Print rows per split and `index.column_names` | `len(index)` |
| 1.1.3 | Print `image_id`, `category`, `pixel_size_nm`, `len(embedding)` of `index[0]` | |
| 1.1.4 | Count rows per category | `Counter(index["category"])` |
| 1.1.5 | Embed `query[0]["image"]` again, now; cosine with the stored `query[0]["embedding"]` | `embed_image`, `cosine` |


## 6. Exercise 1.2 — Create the collection, insert the points

**File:** `s1/build_collection.py`

| Step | Do | Hint |
|---|---|---|
| 1.2.2 | `create_collection`: delete if it exists, create with cosine distance | `client.collection_exists`, `VectorParams(size=..., distance=Distance.COSINE)` |
| 1.2.3 | `make_points`: one `PointStruct` per row of the batch | id = `start + i`; payload = `{f: batch[f][i] for f in PAYLOAD_FIELDS}` |
| 1.2.4 | Upsert each batch | `client.upsert(COLLECTION, points=points)` |
| 1.2.5 | Payload indexes on `category` (keyword) and `pixel_size_nm` (float) | `client.create_payload_index(COLLECTION, "category", "keyword")` |

`main()` first keeps only the columns that go to Qdrant (`index.select_columns(...)`). A batch (`rows[start:start+256]`) is a **dict of columns**, not a list of rows:
`batch["image_id"]` is a list of 256 ids.

> The image bytes are **not** in the payload. Qdrant holds the coordinate and the metadata;
> the raw data stays in the dataset, linked through `image_id`.

## 7. Exercise 1.3 — Embed a query, search

**File:** `s1/search.py`

| Step | Do | Hint |
|---|---|---|
| 1.3.2 | `search`: top-`limit` points by similarity | `client.query_points(COLLECTION, query=vector, limit=limit, with_payload=True, query_filter=query_filter).points` |
| 1.3.3 | `show`: one line per hit | `h.score`, `h.payload["image_id"]`, ... |
| 1.3.4 | Text query *"SEM image of elongated thin wires"* | `embed_text` |
| 1.3.5 | Nonsense query *"a cat on a sofa"* | look at the scores |

The script uses the first *nanowires* image of the `query` split as the image query.

## 8. Exercise 1.4 — Search with metadata filters

**File:** `s1/search.py` (continue)

| Step | Do | Hint |
|---|---|---|
| 1.4.1 | Condition `category == "nanowires"` | `FieldCondition(key="category", match=MatchValue(value="nanowires"))` |
| 1.4.2 | Condition `pixel_size_nm < 10` | `FieldCondition(key="pixel_size_nm", range=Range(lt=10.0))` |
| 1.4.3 | Same image query with each filter, then both | `Filter(must=[...])` → `query_filter=` |
| 1.4.4 | Compare with the unfiltered list | which hits disappear? how do scores change? |
| 1.4.5 | Count `fibres` with pixel size in [5, 20] nm, no vector | `client.count(COLLECTION, count_filter=Filter(must=[...]), exact=True).count` |

> `must` = AND, `should` = OR, `must_not` = NOT. The same `Filter` object works for
> `query_points`, `count` and `scroll`.

