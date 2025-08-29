# inLaw
Analyse contracts using AI and RAG input from text files.

# 🧠 AI Contract Analyzer

An open-source tool that flags problematic clauses in legal contracts using NLP and Retrieval-Augmented Generation (RAG). Built for legal tech enthusiasts, researchers, and ethical AI developers, this analyzer blends semantic search, clause heuristics, and generative feedback to help you spot risks and ambiguities in contracts.

---

## ✨ Features

- **Clause-Level Risk Detection**  
  Identifies vague, risky, or contradictory clauses using semantic similarity and rule-based heuristics.

- **Problem Tagging**  
  Flags issues like missing indemnity, unclear termination terms, or jurisdictional conflicts.

- **Embedding-Based Matching**  
  Uses Hugging Face sentence embeddings to compare clauses against a curated database of problematic patterns.

- **RAG for Text Files**  
  Supports Retrieval-Augmented Generation for `.txt` contracts, enabling context-aware clause analysis and generative feedback via OpenRouter.

- **Lightweight & Modular**  
  Designed to run locally with minimal setup. Easily extendable for new clause types or jurisdictions.

---

## 🛠️ How It Works

1. Preprocessing  
   Splits contract into clauses using regex and NLP chunking.

2. Embedding Generation  
   Uses free Hugging Face models like `sentence-transformers/all-MiniLM-L6-v2`.

3. Similarity Matching  
   Compares clause embeddings against a curated set of problematic examples.

4. RAG Pipeline  
   For uploaded `.txt` files, relevant clauses are retrieved and passed to a generative model via OpenRouter API for deeper analysis.

5. *Annotation  
   Highlights clauses with potential issues and provides brief explanations or suggestions.

---

## 📦 Installation

```bash
git clone https://github.com/ajmal2005/inLaw.git
cd inLaw
pip install -r requirements.txt
