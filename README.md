# LLM Tourism Assistant

> End-to-end AI pipeline: Reddit scraping → RAG → dataset generation → fine-tuning → deployment

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Model](https://img.shields.io/badge/Model-Qwen2--1.5B--Instruct-purple?style=flat-square)
![Vector DB](https://img.shields.io/badge/VectorDB-Qdrant-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)

---

## Overview

A tourism question-answering assistant built from scratch as a personal learning project. The goal was not to ship a perfect product — it was to understand and implement every layer of a modern LLM system hands-on: data collection, RAG pipeline, dataset engineering, supervised fine-tuning, and deployment.

The system combines a fine-tuned small language model with a retrieval-augmented generation pipeline backed by a Qdrant vector store. At inference time, user queries are embedded, matched against indexed travel knowledge, and the retrieved context is injected into the model prompt before generation.

> ⚠️ This is a learning project. It is not production-ready and not affiliated with any company or platform.

---

## Architecture

```
User Query
    │
    ▼
Gradio UI (HF Spaces)
    │
    ▼
Python ChatBot
    │
    ├──────────────────────────┐
    ▼                          ▼
Embedding Model          Qdrant Cloud
(all-MiniLM-L6-v2)      (Vector Search — top-5)
    │                          │
    └──────────┬───────────────┘
               ▼
       Context Assembly
    (chunks + prompt template)
               │
               ▼
  HF Inference Endpoint
  (Qwen2-1.5B fine-tuned)
               │
               ▼
          Response → User
```

---

## Pipeline Stages

| Stage | Description |
|---|---|
| 1. Cleaning | Length filter, dedup, HTML strip, encoding fix |
| 2. Embedding | Generates 384-dim vectors with all-MiniLM-L6-v2 |
| 3. Indexing | Uploads vectors + payload to Qdrant Cloud |
| 4. Dataset gen | Ollama + Qwen2.5 generates grounded Q&A pairs |
| 5. Fine-tuning | FP16 LoRA on Qwen2-1.5B-Instruct, Kaggle T4 |


---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Data cleaning | Node.js  | Posts + comments from r/travel, r/Morocco, r/solotravel |
| Embedding model | all-MiniLM-L6-v2 | 384-dim vectors, CPU inference |
| Vector store | Qdrant Cloud | ~1,200 chunks, cosine similarity |
| Dataset gen LLM | Qwen2.5-7B via Ollama | Local inference, grounded generation |
| Fine-tune base | Qwen2-1.5B-Instruct | FP16 LoRA, r=16, alpha=32 |
| Training stack | transformers, peft, trl, accelerate | Kaggle T4 GPU (free tier) |
| Backend | Python | Retrieval + prompt assembly middleware |
| Frontend | Gradio (HF Spaces) | Conversational chat UI |
| Model hosting | Hugging Face Inference Endpoints | Merged LoRA + base weights |

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) (for dataset generation only)
- Qdrant Cloud account (free tier)
- Hugging Face account

### 1. Clone and install

```bash
git clone https://github.com/yowww1094/llm-tourism-assistant
cd llm-tourism-assistant
cd chatbot && pip install -r requirements.txt
cd dataset_generator && npm install
```

### 2. Configure environment

```bash
cd dataset_generator && cp .env.example .env
```

Fill in your `.env`:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION_NAME=your_collection_name
```

### 3. Run the data pipeline

Start first with naming your data file with: reddit.csv

```bash
cd dataset_generator

# Clean and structure
node scripts/prepare-rag-docs.js

# Embed and index
node scripts/chunk-rag-docs.js
node scripts/embed-and-store.js

# Test searching the embedded data
node scripts/search-rag.js
```

### 4. Generate training dataset

```bash
# Requires Ollama running locally with Qwen2.5 pulled
ollama pull qwen2.5:7b
node scripts/generate-dataset.js
```

### 5. Fine-tune

See the [Finetuning Folder](#finetuning-folder) section below for a full explanation of every file and how to get each one before running this step.

```bash
# Option A — run locally (requires a GPU with 15+ GB VRAM)
python finetuning/train.py --config finetuning/config.yaml

# Option B — run on Kaggle (recommended, free T4 GPU)
# Upload finetuning/notebooks/training_kaggle.ipynb to Kaggle,
# enable the T4 x2 accelerator, then run all cells.
```

### 6. Run the app

```bash
# Start chatbot
cd chatbot && python app.py
```

---

## Training Configuration

```yaml
model: Qwen/Qwen2-1.5B-Instruct
precision: fp16
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, v_proj]
epochs: 3
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.03
max_seq_length: 512
```

---

## Results

| Metric | Value |
|---|---|
| Training loss (epoch 3) | ~0.85 |
| Validation loss (epoch 3) | ~1.29 |
| Validation perplexity | ~2.3 |
| Training examples | ~150–200 |
| Indexed chunks | ~1,200 |


---

## Key Design Decision: RAG + Fine-Tuning

These two mechanisms solve different problems and are both necessary:

- **Fine-tuning** shapes *how* the model responds — tone, structure, domain vocabulary.
- **RAG** controls *what* the model knows — factual context retrieved at inference time.

Fine-tuning on 150–200 examples cannot inject meaningful new factual knowledge into a 1.5B model. Without RAG, the model would hallucinate or give generic answers for any query not well-represented in training. RAG provides the factual grounding; fine-tuning provides the behavioral style.

---

## Limitations

**The chatbot is not fully accurate.** The most significant reason is the training dataset: with only ~150–200 generated Q&A pairs sourced from a small, unverified Reddit corpus, the model has very limited factual coverage and generalises poorly to queries outside that narrow distribution. A larger, higher-quality dataset with verified facts would be the single most impactful improvement to the system.

Beyond the dataset, other known limitations include:

- **Retrieval gaps** — the embedding model (all-MiniLM-L6-v2) is not domain-adapted for tourism, so vocabulary mismatches cause retrieval misses on specialist queries
- **Hallucination** — the model generates plausible-sounding but incorrect facts (prices, dates, distances) when retrieved context does not provide explicit grounding
- **No answer abstention** — the model does not reliably refuse to answer when it doesn't know; it tends to produce a confident-sounding response regardless
- **Incomplete data validation** — only ~40 of the ~200 training examples were manually reviewed; the rest have unknown quality
- **High latency** — end-to-end response time averages 8–12 seconds due to sequential embedding, retrieval, and inference steps
- **English only** — no multilingual support; a real Morocco-focused assistant would need French, Arabic, and Darija

---

## Challenges

A few things that didn't go as planned:

- **QLoRA failed** — bitsandbytes had a CUDA driver incompatibility on the Kaggle T4 environment; switched to FP16 LoRA
- **Encoding bugs** — non-UTF-8 bytes from the Reddit API caused silent tokeniser failures, reducing the effective training set size
- **Gradio streaming artefacts** — partial tokens rendered as separate chat bubbles; streaming mode was disabled
- **Qdrant API version drift** — several community examples used deprecated patterns that returned empty results silently

---

## Future Improvements

- [ ] Expand dataset to 2,000–5,000 verified Q&A pairs from authoritative travel sources
- [ ] Replace all-MiniLM with a domain-adapted embedding model (E5-large, BGE-large)
- [ ] Add cross-encoder re-ranking after initial vector retrieval
- [ ] Add hybrid BM25 + dense retrieval for exact-match queries
- [ ] Fine-tune a 7B model with a stable QLoRA setup
- [ ] Add multilingual support (French, Arabic, Darija)
- [ ] Display source citations in the UI alongside answers
- [ ] Add a user feedback mechanism for answer quality

---

## Project Links

| Resource | Link |
|---|---|
| Hugging Face Model | [Model](https://huggingface.co/yowww1094/tourism-llm-fine-tuned-qwen2-1.5b-lora-merged) |
| Hugging Face Dataset | [Dataset](https://huggingface.co/datasets/yowww1094/tourism-llm-fine-tuning-dataset) |
| Live Demo | [Demo](https://huggingface.co/spaces/yowww1094/AI-tourism-chatbot) |
| Technical Report | [docs/technical_report.pdf](docs/technical_report.pdf) |

---

## License

MIT — see [LICENSE](LICENSE)
