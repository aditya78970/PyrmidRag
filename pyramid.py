"""
pyramid.py
A small, self-contained impl of:
- 2-page sliding window (char-based; can be swapped to token-based)
- Knowledge Pyramid: Raw Text -> Chunk Summary -> Category -> Distilled Keywords + embeddings
- FAISS indices per pyramid level, retrieval API
"""

import os
import re
import json
import glob
import uuid
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# embeddings + faiss
from sentence_transformers import SentenceTransformer
import faiss

# basic HTML/PDF reading
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# optional fuzzy fallback
from rapidfuzz import fuzz

# ---------- Config ----------
@dataclass
class Config:
    doc_dir: str = "./gemma3"
    page_chars: int = 2000        # chars per "page" (changeable)
    window_pages: int = 2         # sliding window width (2 pages)
    slide_stride_pages: int = 1   # slide by 1 page (overlap)
    top_k: int = 5
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_topk: int = 6
    index_dir: str = "./pyramid_index"

cfg = Config()

# ---------- Utilities ----------
def read_text_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md", ".rst"]:
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext in [".html", ".htm"]:
        raw = open(path, "r", encoding="utf-8", errors="ignore").read()
        soup = BeautifulSoup(raw, "html.parser")
        for s in soup(["script","style"]):
            s.decompose()
        return soup.get_text(separator=" ")
    if ext == ".pdf":
        try:
            reader = PdfReader(path)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)
        except Exception:
            return ""
    # fallback
    try:
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    except:
        return ""

def sliding_windows_by_chars(text: str, page_chars: int, window_pages: int, stride_pages: int):
    # split text into pages by characters
    pages = [text[i:i+page_chars] for i in range(0, len(text), page_chars)]
    windows = []
    for start in range(0, max(1, len(pages)), stride_pages):
        window_pages_slice = pages[start:start+window_pages]
        if not window_pages_slice:
            continue
        windows.append("\n".join(window_pages_slice))
        if start+window_pages >= len(pages):
            break
    return windows

# lightweight summarizer (placeholder)
def placeholder_summary(text: str, max_sentences=2) -> str:
    # naive sentence splitting
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:max_sentences]) if sents else (text[:200] + "...")

# simple rule-based category/theme
CATEGORY_KEYWORDS = {
    "finance": ["budget", "cost", "price", "profit", "revenue", "salary"],
    "music": ["music", "song", "volume", "play", "pause"],
    "home": ["light", "shutter", "camera", "thermostat", "door"],
    "legal": ["contract", "agreement", "liability", "law"],
    "math": ["sum", "multiply", "equation", "calculate", "rate"]
}
def rule_category_label(text: str):
    t = text.lower()
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for k in kws if k in t)
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "general"

# distilled knowledge via TF-IDF keywords
def top_tfidf_keywords(corpus: List[str], doc_text: str, topk=6):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())
    # get tfidf for this doc (find doc index)
    idx = corpus.index(doc_text)
    row = X[idx].toarray().reshape(-1)
    top_idx = np.argsort(row)[::-1][:topk]
    return feature_names[top_idx].tolist()

# ---------- Pyramid Data structures ----------
@dataclass
class PyramidNode:
    id: str
    doc_id: str
    level: str           # raw, summary, category, distilled
    text: str
    meta: dict

# ---------- Ingest + Build pyramid ----------
class PyramidBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embed_model = SentenceTransformer(cfg.embed_model)
        self.nodes: List[PyramidNode] = []

    def build_from_folder(self, folder: str):
        paths = []
        for ext in ("*.txt","*.md","*.pdf","*.html","*.htm"):
            paths += glob.glob(os.path.join(folder, ext))
        for path in tqdm(paths, desc="Documents"):
            text = read_text_file(path)
            if not text or len(text.strip()) < 20:
                continue
            doc_id = "doc-" + uuid.uuid4().hex[:8]
            windows = sliding_windows_by_chars(
                text, self.cfg.page_chars, self.cfg.window_pages, self.cfg.slide_stride_pages
            )
            # create raw nodes
            for widx, wtext in enumerate(windows):
                nid = f"{doc_id}-raw-{widx}"
                node = PyramidNode(id=nid, doc_id=doc_id, level="raw", text=wtext, meta={"path": path, "window": widx})
                self.nodes.append(node)
            # create summaries + categories + distillations per window
            corpus = windows[:]  # for TFIDF
            for widx, wtext in enumerate(windows):
                # chunk summary (placeholder)
                summ = placeholder_summary(wtext, max_sentences=2)
                s_id = f"{doc_id}-sum-{widx}"
                self.nodes.append(PyramidNode(id=s_id, doc_id=doc_id, level="summary", text=summ, meta={"window": widx}))
                # category
                cat = rule_category_label(wtext)
                c_id = f"{doc_id}-cat-{widx}"
                self.nodes.append(PyramidNode(id=c_id, doc_id=doc_id, level="category", text=cat, meta={"window": widx}))
            # distilled: compute TF-IDF top keywords per window
            if windows:
                for widx, wtext in enumerate(windows):
                    keywords = top_tfidf_keywords(corpus, wtext, topk=self.cfg.tfidf_topk)
                    dtext = ", ".join(keywords)
                    d_id = f"{doc_id}-dist-{widx}"
                    self.nodes.append(PyramidNode(id=d_id, doc_id=doc_id, level="distilled", text=dtext, meta={"window": widx}))
        print(f"Built {len(self.nodes)} nodes")
        return self.nodes

    def build_faiss_indices(self):
        # per-level build
        self.level_index = {}
        for level in ["raw","summary","category","distilled"]:
            texts = [n.text for n in self.nodes if n.level==level]
            if not texts:
                self.level_index[level] = None
                continue
            emb = self.embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)   # inner product on normalized vectors = cosine
            index.add(emb.astype("float32"))
            ids = [n.id for n in self.nodes if n.level==level]
            self.level_index[level] = {"index": index, "ids": ids, "emb": emb, "texts": texts}
        print("FAISS indices built per level.")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "nodes.json"), "w", encoding="utf8") as f:
            json.dump([asdict(n) for n in self.nodes], f, indent=2)

# ---------- Retrieval ----------
class PyramidRetriever:
    def __init__(self, builder: PyramidBuilder):
        self.builder = builder
        self.embed_model = builder.embed_model

    def retrieve(self, query: str, level_priority: List[str]=None, top_k=5):
        if level_priority is None:
            level_priority = ["summary","distilled","raw","category"]
        qv = self.embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        results = []
        for lvl in level_priority:
            info = self.builder.level_index.get(lvl)
            if info is None:
                continue
            D, I = info["index"].search(qv, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                nid = info["ids"][idx]
                results.append({"level": lvl, "id": nid, "score": float(score), "text": info["texts"][idx]})
            # early return if high-confidence hit
            if results and results[0]["score"] > 0.7:
                break
        # fallback fuzzy match on texts if no hits
        if not results or results[0]["score"] < 0.2:
            # naive fuzzy fallback across summary + distilled
            candidates = []
            for lvl in ["summary","distilled","raw"]:
                info = self.builder.level_index.get(lvl)
                if info is None: continue
                for tid, txt in zip(info["ids"], info["texts"]):
                    score = fuzz.partial_ratio(query.lower(), txt.lower()) / 100.0
                    candidates.append({"level": lvl, "id": tid, "score": score, "text": txt})
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
            return candidates
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# ---------- Example run ----------
if __name__ == "__main__":
    builder = PyramidBuilder(cfg)
    nodes = builder.build_from_folder(cfg.doc_dir)
    builder.build_faiss_indices()
    builder.save(cfg.index_dir)

    retriever = PyramidRetriever(builder)
    while True:
        q = input("\nQuery (empty to exit)> ").strip()
        if not q:
            break
        hits = retriever.retrieve(q, top_k=5)
        print("Top hits:")
        for h in hits[:5]:
            clean = h["text"][:180].replace('\n', ' ')
            print(f"[{h['level']}] score={h['score']:.3f} text_snip={clean}")

