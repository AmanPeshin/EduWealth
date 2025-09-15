from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings
from .config import OPENAI_API_KEY, EMBED_MODEL

def embed_texts(texts: List[str]) -> List[List[float]]:
    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    return emb.embed_documents(texts)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def max_cosine(new_vec: List[float], prior_vecs: List[List[float]]) -> float:
    if not prior_vecs:
        return 0.0
    a = np.array(new_vec, dtype=np.float32)
    prior = np.array(prior_vecs, dtype=np.float32)
    sims = prior @ a / (np.linalg.norm(prior, axis=1) * np.linalg.norm(a) + 1e-12)
    return float(np.max(sims))

