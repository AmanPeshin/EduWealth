import math, hashlib, json
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from .db import QuestionItem
from .embeddings import embed_texts, max_cosine
from .config import CHAT_MODEL, OPENAI_API_KEY, COSINE_THRESHOLD_HARD

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def prob_correct_2pl(theta: float, a: float, b: float) -> float:
    return sigmoid(a * (theta - b))

def fisher_info_2pl(theta: float, a: float, b: float) -> float:
    p = prob_correct_2pl(theta, a, b)
    return (a * a) * p * (1.0 - p)

def update_theta_step(theta: float, a: float, b: float, y: int, lr: float) -> float:
    p = prob_correct_2pl(theta, a, b)
    grad = a * (y - p)
    return float(theta + lr * grad)

def stable_item_id(stem: str) -> str:
    return hashlib.sha256(stem.strip().lower().encode("utf-8")).hexdigest()[:24]

def llm_generate_mcqs(topic: str, subtopic: str, difficulty: str, k: int) -> List[Dict]:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
    sys = "You are a finance instructor. Create precise MCQs with 4 choices and one correct answer."
    usr = f"""
Generate {k*2} MCQs for topic '{topic}', subtopic '{subtopic}', difficulty '{difficulty}'.
Return JSON list: question, choices[28], answer_index (0-3), explanation.
Ensure questions are semantically distinct; vary stems, numbers, and rationale.
"""
    resp = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}])
    return json.loads(resp.content)

def pick_next_item_adaptive(
    db: Session,
    topic: str,
    subtopic: str,
    difficulty: str,
    theta: float,
    attempt_seen_items: List[Dict],
) -> Optional[Dict]:
    rows = db.query(QuestionItem).filter_by(topic=topic, subtopic=subtopic, difficulty=difficulty).limit(200).all()
    if not rows:
        return None
    seen_ids = {it["item_id"] for it in attempt_seen_items}
    seen_vecs = [it["embedding"] for it in attempt_seen_items if it.get("embedding")]

    best, best_info = None, -1.0
    for r in rows:
        if r.item_id in seen_ids:
            continue
        if r.embedding is not None:
            sim = max_cosine(r.embedding, seen_vecs)
            if sim >= COSINE_THRESHOLD_HARD:
                continue
        a = r.a if r.a is not None else 1.0
        b = r.b if r.b is not None else 0.0
        info = fisher_info_2pl(theta, a, b)
        if info > best_info:
            best_info, best = info, r
    if not best:
        return None
    return {
        "item_id": best.item_id,
        "question": best.payload["question"],
        "choices": best.payload["choices"],
        "correct_index": best.payload["answer_index"],
        "embedding": best.embedding,
        "a": best.a if best.a is not None else 1.0,
        "b": best.b if best.b is not None else 0.0,
    }
