# Session 2 — RAG from scratch

## 1. What you build today

A question-answering pipeline over the `sem_images` collection of Session 1, in plain Python,
without any RAG framework using four functions:

- **retrieve**: embed the question (or an image) and search the collection, as in exercise 1.3.
- **build_context**: turn the payloads of the hits into text the LLM can read.
- **call_llm**: one chat-completions request to the LLM served on ORFEO.
- **answer**: the three together, plus one line of provenance per answer.

Then you measure retrieval (precision@k) and see how k, filters and the wording change the answers.

The LLM is `deepseek-ai/DeepSeek-V4-Flash-0731` (`LLM_MODEL` in `.env`), on the same ORFEO endpoint
and with the same token as the embedding model.

## 2. Start

```bash
ssh <your-vm>
cd python-2026/py4DM-2 && source .venv/bin/activate
docker compose up -d                # Qdrant, with sem_images from this morning
python verify.py                    # lists sem_images among the collections
```

If `sem_images` is missing, rebuild it: `python s1/solution/build_collection.py` (5 seconds).

## 3. Files

```
s2/
├── rag.py              # TODO: Ex. 2.1–4
├── experiments.py      # TODO: Ex. 2.5
└── solution/           # full solutions
output/provenance.jsonl # answer() log
```

Run from the project root: `python s2/rag.py`.

## 4. Calling an LLM: one request

A chat request uses the same
`openai` client, but a different method:

```python
from openai import OpenAI
llm = OpenAI(base_url=ORFEO_BASE_URL, api_key=ORFEO_API_KEY)

response = llm.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    messages=[{"role": "system", "content": "You answer questions about SEM images. ..."},
              {"role": "user", "content": "<context>\n\nQuestion: ..."}],
    temperature=0,          # least random: the same request gives (almost) the same answer
    max_tokens=1000,        # upper bound on the generated tokens
)
response.choices[0].message.content       # the answer, a string
response.choices[0].finish_reason         # "stop": finished; "length": cut by max_tokens
response.usage                            # prompt_tokens, completion_tokens, ...
```

- **system**: general instructions on how to generate answers. **user**: context + question.
- The model has **no memory**: everything it should know goes into `messages`, every time.
- This model **reasons before answering**. The reasoning is returned separately
  (`message.reasoning`) and its tokens count in `max_tokens`. With a small `max_tokens`
  (e.g. 40) the reasoning uses them all: `content` is `None` and `finish_reason` is `"length"`.

## 5. Exercise 2.1 — Retrieve

**File:** `s2/rag.py`

| Step | Do | Hint |
|---|---|---|
| 2.1.1 | Vector of the question, or of the image if one is given | `embed_text(question)`, `embed_image(image)` |
| 2.1.2 | Top-k points, payload included, optional filter | `client.query_points(...)` exactly as in 1.3.2 |

## 6. Exercise 2.2 — Build the context 
| Step | Do | Hint |
|---|---|---|
| 2.2.1 | A header line | `f"Retrieved images (top-{len(hits)}):"` |
| 2.2.2 | One numbered line per hit: id, category, pixel size, mag, WD, EHT, detector, score | `fmt(p["mag"])`, `fmt(p["wd_mm"], " mm")` |
| 2.2.3 | Join the lines | `"\n".join(lines)` |

`main()` prints the context before the answer: 

```
Retrieved images (top-5):
1. sem_00230 category=fibres pixel_size=101.01 nm mag=n/a wd=n/a eht=n/a detector=n/a score=0.438
2. sem_00203 category=fibres pixel_size=392.16 nm mag=n/a wd=n/a eht=n/a detector=n/a score=0.432
3. sem_00402 category=mems_devices_and_electrodes pixel_size=609.76 nm mag=604.0 wd=n/a eht=n/a detector=n/a score=0.430
...
```

> `mag`, `wd_mm`, `eht_kv`, `detector` are `None` for many images. Writing `n/a` tells the LLM *unknown*; leaving the field out, or writing `None` can be misleading.

## 7. Exercise 2.3 — Call the LLM

| Step | Do | Hint |
|---|---|---|
| 2.3.1 | Two messages: system and user | `{"role": "system", "content": SYSTEM_PROMPT}`, user = `f"{context}\n\nQuestion: {question}"` |
| 2.3.2 | The request | `llm.chat.completions.create(model=LLM_MODEL, messages=..., temperature=0, max_tokens=MAX_TOKENS)` |
| 2.3.3 | The text of the answer | `response.choices[0].message.content` |

## 8. Exercise 2.4 — Answer, with provenance 

| Step | Do | Hint |
|---|---|---|
| 2.4.1 | retrieve → build_context → call_llm | pass `k`, `query_filter`, `image` through |
| 2.4.2 | A provenance record | keys: `time`, `question`, `query_image`, `filter`, `top_k`, `hits`, `scores`, `model`, `answer` |
| 2.4.3 | Append it to `LOG` as one JSON line, return the text | `open(LOG, "a")`, `json.dumps(record) + "\n"` |

Check `output/provenance.jsonl` for new lines like the following:

```json
{"time": "...", "question": "What do the images similar to this one have in common?", "query_image": "sem_q_01205",
 "filter": {"must": [{"key": "category", "match": {"value": "nanowires"}}]}, "top_k": 5,
 "hits": ["sem_00487", "sem_00580", "sem_00553", "sem_00579", "sem_00572"],
 "scores": [0.766, 0.749, 0.747, 0.743, 0.739], "model": "deepseek-ai/DeepSeek-V4-Flash-0731", "answer": "..."}
```
## 9. Exercise 2.5 — Experiments

**File:** `s2/experiments.py`

| Step | Do | Hint |
|---|---|---|
| 2.5.1 | `precision_at_k(hits, expected)` | fraction of hits with `category == expected` |
| 2.5.2 | Run: image queries, precision@1, @5, @10  | the 50 query images; their vectors are in the dataset |
| 2.5.3 | Run: the same question with k = 1, 5, 20, with a filter, and an out-of-domain question | read the answers next to the hits in `output/provenance.jsonl` |
| 2.5.4 | `text_precision(template, k)`: one sentence per category | `template.format(cat.replace("_", " "))`, then `embed_text`, `query_points` |

Then change one thing at a time and run again:

- another wording in 2.5.4 (`"a scanning electron micrograph of {}"`, `"{} at high magnification"`);
- `k` in 2.5.3, or a filter on `pixel_size_nm`;
- `SYSTEM_PROMPT` in `rag.py`: remove *"If the context does not answer the question, say so"* and ask
  the out-of-domain question again.
