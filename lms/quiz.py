import hashlib, json
from typing import List, Dict
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from .embeddings import embed_texts, max_cosine
from .config import CHAT_MODEL, OPENAI_API_KEY, COSINE_THRESHOLD_HARD
from .db import QuestionItem
from .bank_index import BankANN

def stable_item_id(stem: str) -> str:
    return hashlib.sha256(stem.strip().lower().encode("utf-8")).hex()[:24]

def llm_generate_mcqs(topic: str, subtopic: str, difficulty: str, k: int) -> List[Dict]:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
    sys = "You are a finance instructor. Create precise MCQs with 4 choices and one correct answer."
    usr = f"""
Generate {k*2} MCQs for topic '{topic}', subtopic '{subtopic}', difficulty '{difficulty}'.
Return JSON list: question, choices[10], answer_index (0-3), explanation.
Ensure questions are semantically distinct; vary stems, numbers, and rationale.
"""
    resp = llm.invoke([{"role":"system","content":sys},{"role":"user","content":usr}])
    return json.loads(resp.content)

def select_unique_items_for_attempt(
    db: Session,
    topic: str,
    subtopic: str,
    difficulty: str,
    needed: int,
    attempt_seen_items: List[Dict],
    bank_ann: BankANN | None = None,
) -> List[Dict]:
    collected: List[Dict] = []
    seen_vecs = [it["embedding"] for it in attempt_seen_items if it.get("embedding")]

    # Prefer bank by metadata
    bank_rows = db.query(QuestionItem).filter_by(topic=topic, subtopic=subtopic, difficulty=difficulty).limit(needed*3).all()
    for r in bank_rows:
        if len(collected) >= needed:
            break
        vec = r.embedding
        if vec:
            sim = max_cosine(vec, seen_vecs)
            if sim >= COSINE_THRESHOLD_HARD:
                continue
            collected.append({
                "item_id": r.item_id,
                "question": r.payload["question"],
                "choices": r.payload["choices"],
                "correct_index": r.payload["answer_index"],
                "embedding": vec
            })
            seen_vecs.append(vec)

    if len(collected) < needed:
        gen_k = max(needed - len(collected), 3)
        raw = llm_generate_mcqs(topic, subtopic, difficulty, gen_k)
        stems = [it["question"] for it in raw]
        gen_vecs = embed_texts(stems)
        for it, v in zip(raw, gen_vecs):
            sim = max_cosine(v, seen_vecs)
            if sim >= COSINE_THRESHOLD_HARD:
                continue
            it["item_id"] = stable_item_id(it["question"])
            it["embedding"] = v
            collected.append({
                "item_id": it["item_id"],
                "question": it["question"],
                "choices": it["choices"],
                "correct_index": it["answer_index"],
                "embedding": v
            })
            seen_vecs.append(v)
        # Upsert generated into bank
        for it in collected:
            if not db.query(QuestionItem).filter_by(item_id=it["item_id"]).first():
                db.add(QuestionItem(
                    item_id=it["item_id"],
                    source="generated",
                    topic=topic,
                    subtopic=subtopic,
                    difficulty=difficulty,
                    payload={"question": it["question"], "choices": it["choices"], "answer_index": it["correct_index"], "explanation": ""},
                    embedding=it["embedding"]
                ))
        db.commit()

    return collected[:needed]
