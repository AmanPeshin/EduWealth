from typing import TypedDict, List, Dict, Annotated
import operator
from datetime import datetime
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from .db import SessionLocal, QuizAttempt, AttemptResponse
from .quiz_adaptive import pick_next_item_adaptive, update_theta_step
from .config import QUIZ_LENGTH, CHECKPOINTER_BACKEND, REDIS_URL, SQLITE_CP_PATH, INIT_THETA, THETA_LR
from .progress import is_unlocked, record_attempt

def make_checkpointer():
    if CHECKPOINTER_BACKEND == "redis":
        from langgraph.checkpoint.redis import RedisSaver
        return RedisSaver.from_conn_string(REDIS_URL)
    if CHECKPOINTER_BACKEND == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver.from_conn_string(f"sqlite:///{SQLITE_CP_PATH}")
    from langgraph.checkpoint.memory import InMemorySaver
    return InMemorySaver()

class AttemptState(TypedDict):
    user_id: int
    topic: str
    subtopic: str
    difficulty: str
    needed: int
    served: Annotated[List[Dict], operator.add]  # items with a,b
    current_index: int
    current_answer: int | None
    correct_count: int
    theta: float
    complete: bool

def _db() -> Session:
    return SessionLocal()

def node_gate_unlock(s: AttemptState):
    db = SessionLocal()
    ok, unmet = is_unlocked(db, s["user_id"], s["topic"], s["subtopic"])
    if ok:
        return {}
    _ = interrupt({
        "type": "locked",
        "message": "This subtopic is locked. Complete prerequisites first.",
        "unmet": unmet
    })
    return {}

def node_init(s: AttemptState):
    return {
        "needed": QUIZ_LENGTH if not s.get("needed") else s["needed"],
        "served": s.get("served", []),
        "current_index": s.get("current_index", 0),
        "current_answer": None,
        "correct_count": s.get("correct_count", 0),
        "theta": s.get("theta", INIT_THETA),
        "complete": False
    }

def node_select_next(s: AttemptState):
    if len(s["served"]) > s["current_index"]:
        return {}
    item = pick_next_item_adaptive(
        db=_db(),
        topic=s["topic"],
        subtopic=s["subtopic"],
        difficulty=s["difficulty"],
        theta=s["theta"],
        attempt_seen_items=s["served"]
    )
    if not item:
        _ = interrupt({
            "type": "no_item_available",
            "message": "No suitable next item found for this adaptive step."
        })
        return {}
    return {"served": [item]}

def node_emit_and_wait(s: AttemptState):
    idx = s["current_index"]
    if idx >= s["needed"]:
        return {}
    q = s["served"][idx]
    _ = interrupt({
        "type": "await_answer",
        "index": idx,
        "question": q["question"],
        "choices": q["choices"],
    })
    return {}

def node_validate_update_and_advance(s: AttemptState):
    idx = s["current_index"]
    if idx >= len(s["served"]):
        return {}
    q = s["served"][idx]
    ans = s.get("current_answer", None)
    correct_count = s["correct_count"]
    theta = s["theta"]
    if ans is not None:
        y = 1 if ans == q["correct_index"] else 0
        if y == 1:
            correct_count += 1
        a = q.get("a", 1.0)
        b = q.get("b", 0.0)
        theta = update_theta_step(theta, a, b, y, lr=THETA_LR)
    next_idx = idx + 1
    done = next_idx >= s["needed"]
    return {"correct_count": correct_count, "current_index": next_idx, "current_answer": None, "theta": theta, "complete": done}

def build_quiz_graph_streaming_adaptive():
    g = StateGraph(AttemptState)
    g.add_node("gate", node_gate_unlock)
    g.add_node("init", node_init)
    g.add_node("select_next", node_select_next)
    g.add_node("emit_and_wait", node_emit_and_wait)
    g.add_node("validate_update_and_advance", node_validate_update_and_advance)
    g.add_edge(START, "gate")
    g.add_edge("gate", "init")
    g.add_edge("init", "select_next")
    g.add_edge("select_next", "emit_and_wait")
    g.add_edge("emit_and_wait", "validate_update_and_advance")
    g.add_conditional_edges(
        "validate_update_and_advance",
        lambda s: "end" if s["complete"] else "loop",
        {"end": END, "loop": "select_next"}
    )
    checkpointer = make_checkpointer()
    return g.compile(checkpointer=checkpointer)

def persist_attempt(user_id: int, topic: str, subtopic: str, served: List[Dict], answers: List[int], pass_mark: float = 0.6):
    db = _db()
    att = QuizAttempt(user_id=user_id, topic=topic, subtopic=subtopic, created_at=datetime.utcnow())
    db.add(att); db.flush()
    correct = 0
    for i, it in enumerate(served):
        ui = answers[i] if i < len(answers) else None
        is_correct = (ui == it["correct_index"]) if ui is not None else None
        if is_correct:
            correct += 1
        db.add(AttemptResponse(
            attempt_id=att.id,
            item_id=it["item_id"], question=it["question"], choices=it["choices"],
            correct_index=it["correct_index"], user_index=ui, correct=is_correct
        ))
    score = correct / max(1, len(served))
    att.score = score * 100.0
    att.passed = bool(score >= pass_mark)
    att.finished_at = datetime.utcnow()
    db.commit()
    record_attempt(db, user_id, topic, subtopic, att.score, pass_mark)
    return att.id
