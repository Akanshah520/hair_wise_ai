HairWise AI — Retrieval‑Augmented Hair Care Assistant

A practical, no‑hype project that answers real hair care questions using retrieval‑augmented generation (RAG). The system retrieves relevant snippets from a curated corpus and composes clear answers with an instruction‑tuned model. The goal was simple: reduce guesswork and generic “AI” responses, and return grounded, useful guidance.

## Why this exists
Most hair care answers online are either promotional or generic. This project uses a small, efficient ML stack to surface relevant, trustworthy information and summarize it clearly. It started as a personal need to stop wasting time and money on vague advice and evolved into a focused RAG pipeline that prioritizes retrieval quality and precise prompting.

## What it does
- Finds relevant context with semantic search (FAISS + Sentence-Transformers).
- Generates answers with an instruction‑tuned model (Flan‑T5‑Base).
- Uses conservative decoding and strict prompts to avoid rambling and template‑echoing.
- Includes an evaluation script for retrieval and generation quality (Top‑k, MRR, ROUGE‑L, BERTScore).

## Tech stack
- Embeddings: Sentence‑Transformers (all‑MiniLM‑L6‑v2)
- Vector search: FAISS
- Generation: Flan‑T5‑Base (inference‑only in the final version)
- Data: Requests + BeautifulSoup for scraping, Pandas/NumPy for processing
- Evaluation: scikit‑learn (nDCG), rouge‑score, BERTScore

## Project structure
- data/: raw content, processed chunks, FAISS index
- src/: retrieval, generation, cleaning, evaluation scripts
- notebooks/: exploration, preprocessing, evaluation
- models/: saved tokenizer/model (optional local cache)

## Getting started
1) Setup
- Python 3.9+
- pip install -r requirements.txt

2) Build or load the index
- Place processed chunked text at data/processed/cleaned_qa_pairs.csv
- Ensure FAISS index at data/processed/faiss_index/haircare.index

3) Ask a question
- from src.inference import generate_answer
- print(generate_answer("Best products for dry hair?"))

## How it’s evaluated
- Retrieval: Top‑3 Accuracy, MRR, nDCG — shows whether the system finds the right evidence.
- Generation: ROUGE‑L and BERTScore — measures content overlap and semantic similarity.
- Practical checks: latency and consistency under beam search.

## What worked
- Swapping a custom fine‑tuned T5‑small for Flan‑T5‑Base (no fine‑tuning) removed instruction‑following and context‑use issues.
- Cleaning and multi‑sentence chunk filtering improved retrieval quality significantly.
- Deterministic decoding (num_beams=8, no sampling) stabilized outputs and reduced repetition.

## What didn’t
- Over‑engineered prompts caused template echoing (“Product name…”) — simplified prompts performed better.
- Over‑aggressive cleaning led to “Insufficient context” — solved by increasing k and adding a raw‑context fallback.

## Results (representative)
- Top‑3 retrieval accuracy: ~85–90% on a held‑out question set
- BERTScore F1: ~0.75–0.80 on curated references
- Average response time: sub‑2 seconds on GPU for typical queries

## Roadmap
- Expand domain coverage and sources
- Lightweight re‑ranking on top of FAISS
- Optional guardrails and citations per sentence
- Personalization by hair type and routine

## Notes for reviewers
- Final system uses Flan‑T5‑Base with RAG; the earlier T5‑small fine‑tune was evaluated but not used in production due to instruction‑following gaps in multi‑context settings.
- The focus is reliability and retrieval faithfulness over flashy generative behavior.

Questions or suggestions? Open an issue — thoughtful feedback is welcome.
