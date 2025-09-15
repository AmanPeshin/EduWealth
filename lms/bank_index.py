import os
import numpy as np
import faiss
from typing import List, Tuple
from sqlalchemy.orm import Session
from .db import SessionLocal, QuestionItem
from .config import FAISS_DIR

class BankANN:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        self.ids: List[str] = []
        self.meta: List[Tuple[str, str, str]] = []

    def add(self, ids: List[str], vecs: List[List[float]], metas: List[Tuple[str,str,str]]):
        X = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(X)
        self.index.add(X)
        self.ids.extend(ids)
        self.meta.extend(metas)

    def search_filtered(self, q_vec: List[float], topic: str, subtopic: str, difficulty: str, topk: int = 50):
        X = np.array([q_vec], dtype=np.float32)
        faiss.normalize_L2(X)
        D, I = self.index.search(X, topk)
        hits = []
        for idx in I:
            if idx == -1:
                continue
            t, st, diff = self.meta[idx]
            if (t == topic) and (st == subtopic) and (diff == difficulty):
                hits.append(self.ids[idx])
        return hits

def build_bank_ann(dim: int = 1536) -> BankANN:
    os.makedirs(FAISS_DIR, exist_ok=True)
    ann = BankANN(dim)
    db: Session = SessionLocal()
    rows = db.query(QuestionItem).all()
    if not rows:
        return ann
    ids, vecs, metas = [], [], []
    for r in rows:
        if r.embedding:
            ids.append(r.item_id)
            vecs.append(r.embedding)
            metas.append((r.topic, r.subtopic, r.difficulty))
    if vecs:
        ann.add(ids, vecs, metas)
    return ann
