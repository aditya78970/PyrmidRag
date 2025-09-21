# PyrmidRag
Pyramid Ingest is a lightweight document processing and retrieval pipeline that builds a multi-level "Knowledge Pyramid" from raw text. It extracts summaries, categories, and keywords per chunk, embeds each level, and enables fast semantic search with FAISS and fuzzy fallback. Ideal for intelligent search systems.


# Pyramid Ingest

**Pyramid Ingest** is a lightweight and modular document ingestion and semantic retrieval system that builds a multi-layered **Knowledge Pyramid** from raw text.

It converts raw documents into overlapping content windows, generates summaries, categories, and distilled keyword representations for each, embeds all layers using sentence-transformers, and enables **fast, flexible search** using FAISS and fuzzy matching.

---

## Description (short)

**Pyramid Ingest** is a lightweight pipeline that builds a multi-level "Knowledge Pyramid" from text chunks and enables fast, layered semantic retrieval with FAISS and fuzzy matching.

---

## Features

- 📄 **Supports multiple file types:** `.txt`, `.md`, `.pdf`, `.html`
- 🔁 **Sliding window chunking:** Character-based windows with overlap
- 🧱 **Multi-level Knowledge Pyramid:**
  - `raw` text windows
  - `summary` (placeholder summarizer)
  - `category` (rule-based keyword matcher)
  - `distilled` keywords (via TF-IDF)
- 🧠 **Semantic embeddings** using `sentence-transformers`
- ⚡ **Fast vector search** with FAISS
- 🔍 **Retrieval API** with fallback fuzzy matching using RapidFuzz
- 💾 Saves all nodes to disk in structured JSON format

---

## 🛠 Installation

### 1. Clone the repository

git clone https://github.com/yourusername/pyramid-ingest.git
cd pyramid-ingest

2. Install dependencies
pip install -r requirements.txt


Or manually:

pip install sentence-transformers faiss-cpu tqdm numpy pandas scikit-learn rapidfuzz PyPDF2 beautifulsoup4
Project Structure
.
├── pyramid_ingest.py         # Main script
├── gemma3/                   # Folder with input documents (default)
└── pyramid_index/            # Output index and node data

⚙Configuration

All config values are defined in the Config class in pyramid_ingest.py.

@dataclass
class Config:
    doc_dir: str = "./gemma3"               # Input documents folder
    page_chars: int = 2000                  # Chars per page (window)
    window_pages: int = 2                   # Sliding window width
    slide_stride_pages: int = 1             # Window stride
    top_k: int = 5                          # Top K for retrieval
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_topk: int = 6                     # TF-IDF top keywords
    index_dir: str = "./pyramid_index"      # Output folder

Knowledge Pyramid Levels
Level	Description
raw	Original overlapping text windows
summary	First 1–2 sentences (placeholder)
category	Keyword-matched label (e.g., finance, math)
distilled	Top-k TF-IDF keywords

Each level is independently embedded and indexed for retrieval.

Retrieval Logic

The PyramidRetriever enables semantic search across all pyramid levels:

Uses embedding similarity (FAISS inner product)

Falls back to lower levels if no good match

Optional fuzzy fallback using RapidFuzz if no vector hit

Levels are prioritized (default): ["summary", "distilled", "raw", "category"]

Usage
Run the full pipeline:
python pyramid_ingest.py


This will:

Ingest documents from gemma3/

Build knowledge nodes at all levels

Generate embeddings and build FAISS indices

Save everything in pyramid_index/

Start an interactive query loop

Example interactive session
Query (empty to exit)> budget allocation 2024

Top hits:
[summary] score=0.813 text_snip=The 2024 budget outlines a significant increase in R&D investment. Marketing sees a slight decrease.

[distilled] score=0.734 text_snip=budget, allocation, investment, marketing, 2024, increase
...

Output Files

After running, the following will be generated:

pyramid_index/
└── nodes.json         # All processed pyramid nodes
