# HairWise AI — Retrieval‑Augmented Hair Care Assistant

**Domain-specific trichology assistant — retrieval over clinical literature, not the internet.**
---

A practical, no‑hype project that answers real hair care questions using retrieval‑augmented generation (RAG). The system retrieves relevant snippets from a curated corpus and composes clear answers with an instruction‑tuned model. The goal was simple: reduce guesswork and generic “AI” responses, and return grounded, useful guidance.

**Try it →** [huggingface.co/spaces/akansha32/hairwise-ai](https://huggingface.co/spaces/akansha32/hairwise-ai)

---

## How it works

A user types something like *"my hair is frizzy and falling out lately."* The system expands that into clinical vocabulary, retrieves the three most relevant passages from 863 curated clinical passages, and synthesises a grounded answer using Llama-3.1-8B. When retrieval confidence is too low, it abstains rather than guessing.

```
Query
  → vocabulary expansion (layman → clinical terms)
  → FAISS retrieval        top 20 candidates
  → cross-encoder rerank   top 3 passages
  → confidence check       abstain if below 0.42
  → Groq Llama-3.1-8B      plain English synthesis
```

---

## Benchmark results

Evaluated on a 40-question layman benchmark across 9 clinical categories.

| Metric | Score |
|---|---|
| Domain accuracy | 100% |
| Semantic similarity | 76.3% |
| Query coverage | 97.5% (39/40) |
| Abstentions | 1 — correct |

The single abstention was *"how often should I actually wash my hair?"* — a question with no single clinical answer. The system correctly refused rather than generated something that sounded confident.

Per-category semantic similarity:

| Category | Score |
|---|---|
| Dandruff | 0.804 |
| Nutrition | 0.788 |
| Scalp Conditions | 0.787 |
| Products | 0.773 |
| Alopecia | 0.777 |
| Lifestyle | 0.749 |
| Hair Fall | 0.727 |
| When to See a Doctor | 0.721 |

---

## Corpus

| Source | Passages |
|---|---|
| IADVL Textbook of Trichology | 418 |
| V1 Primary Textbook | 136 |
| The Truth About Hair Loss | 49 |
| Research Papers (10) | 95 |
| Web — DermNet, PubMed, Wikipedia | 148 |
| Dermoscopy of Hair and Scalp | 17 |
| **Total** | **863** |

Papers span JAAD, International Journal of Trichology, and PMC — covering AGA treatment, seborrheic dermatitis, scalp psoriasis, trichoteiromania, alopecia areata, tinea capitis, and more.

---

## Architecture

**Retrieval**
- Embedder: `all-MiniLM-L6-v2` (384-dim, FAISS IndexFlatIP)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Two-stage: FAISS for recall (top 20), cross-encoder for precision (top 3)

**Vocabulary bridge**
- 35-term `LAYMAN_MAP` — maps everyday language to clinical retrieval terms before embedding
- `SCALP_MAP` — adds condition-relevant signals based on user's scalp type
- Without this, *"frizzy hair"* finds nothing. With it, it finds passages on hair shaft weathering, porosity, and moisture loss.

**Synthesis**
- Groq Llama-3.1-8B-Instant via API
- Constrained to retrieved passages only — no model memory
- Max 220 tokens, temperature 0.3

**Confidence thresholding**
- FAISS score below 0.42 → abstain, recommend trichologist
- Designed to fail safely rather than hallucinate confidently

---

## What was tried and what failed

**V1 fine-tuning — rouge2 = 0.000**

Every training example used one of three hardcoded generic questions. Same input, 200 different outputs. Zero learnable signal. The model had nothing to extract. Root cause: training data design, not model capacity.

**V2 fine-tuning — rouge2 = 0.0277, inference failed**

Hybrid generation (entity-aware templates + two-pass) fixed the signal problem. Rouge2 improved. Inference hallucinated about eyes and lungs when asked about hair loss.

Two stacked root causes: the training context was a generic placeholder rather than an actual retrieved passage, so the model learned context is irrelevant. At inference it ignored clinical text and generated from weights. And 250M parameters is not enough for clinical synthesis.

The fine-tuned model lives at `akansha32/hairwise-flan-t5-v2` as a documented experiment. It does not run in production.

**Production generator path**

Tried HuggingFace Inference API with Mistral-7B. Timed out silently on every call — the free tier is unreliable for 7B models. Fallback returned raw passage text. Switched to Groq. Initial implementation used `llama3-8b-8192` — decommissioned by Groq, returned 400 silently, six API calls confirmed as failures before the issue was found. One string change to `llama-3.1-8b-instant` fixed it.

---

## Q&A dataset

359 domain-specific trichology Q&A pairs generated via:
- Entity-aware templates across 22 medical conditions (327 pairs)
- Two-pass chunk-specific generation for undetected conditions (32 pairs)
- Filtered by J3 judge (Q-A semantic similarity), 3-gram repetition detector, citation filter, hallucination filter

Available at: [huggingface.co/datasets/akansha32/trichology-qa-v2](https://huggingface.co/datasets/akansha32/trichology-qa-v2)

Fields: `question`, `answer`, `source`, `book`, `method`, `judge_score`

---

## Notebooks

| Notebook | What it does |
|---|---|
| `hairwise_v2_nb1` | Corpus acquisition — PDFs, web scraping, chunking |
| `hairwise_v2_nb2` | Q&A generation, judge comparison, filtering |
| `hairwise_v2_nb3` | Fine-tuning experiment (documented failure) |
| `hairwise_v2_nb4` | 40-question benchmark evaluation |
| `hairwise_v2_nb5` | RAGAS evaluation |

All notebooks designed for Kaggle with T4 GPU.

---

## Stack

```
sentence-transformers    embedder + reranker
faiss-cpu                vector index
groq                     Llama-3.1 inference API
gradio                   deployed interface
pdfplumber               PDF extraction
beautifulsoup4           web scraping
transformers             flan-t5 for Q&A generation
ragas                    RAG evaluation framework
```

---

## Repository structure

```
hairwise-ai/
├── app.py                        deployed Gradio app
├── requirements.txt
├── corpus.jsonl                  863 clinical passages
├── notebooks/
│   ├── hairwise_v2_nb1.ipynb     corpus acquisition
│   ├── hairwise_v2_nb2_final.ipynb  Q&A generation + filtering
│   ├── hairwise_v2_nb3.ipynb     fine-tuning
│   ├── hairwise_v2_nb4.ipynb     benchmark evaluation
│   └── hairwise_v2_nb5_ragas.ipynb  RAGAS evaluation
└── results/
    ├── hv2_nb4_benchmark_results.json
    └── hv2_nb5_ragas_results.json
```

---

## Limitations

- Reference answers in the 40-question benchmark were LLM-generated, creating potential circularity in semantic similarity measurement. Coverage and domain accuracy are not affected.
- Synthesis quality is bounded by Groq free tier (Llama-3.1-8B). A production version would use a stronger generator.
- Corpus covers clinical trichology literature well. Product comparisons, clinical procedures, and environmental factors are underrepresented.
- Confidence threshold was not calibrated against human-labelled data.

---

## Built by

Akansha Sharma — M.Tech Computer Science, SVNIT Surat (2023–2025)  
Specialisation: NLP, RAG systems, LLM evaluation

[LinkedIn](https://www.linkedin.com/in/akansha-sharma2190/) · [GitHub](https://github.com/Akanshah520)
