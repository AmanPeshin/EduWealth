from sqlalchemy.orm import Session
from .db import UserProgress, Prerequisite, Subtopic

def is_unlocked(db: Session, user_id: int, topic: str, subtopic: str) -> tuple[bool, list[dict]]:
    reqs = db.query(Prerequisite).filter_by(target_topic=topic).filter(
        (Prerequisite.target_subtopic==subtopic) | (Prerequisite.target_subtopic=="ANY")
    ).all()
    unmet = []
    for r in reqs:
        if r.prereq_subtopic == "ANY":
            subs = db.query(Subtopic).join(Subtopic.topic).filter(Subtopic.topic.has(name=r.prereq_topic)).all()
            for s in subs:
                row = db.query(UserProgress).filter_by(user_id=user_id, topic=r.prereq_topic, subtopic=s.name).first()
                if not row or not row.completed:
                    unmet.append({"topic": r.prereq_topic, "subtopic": s.name})
        else:
            row = db.query(UserProgress).filter_by(user_id=user_id, topic=r.prereq_topic, subtopic=r.prereq_subtopic).first()
            if not row or not row.completed:
                unmet.append({"topic": r.prereq_topic, "subtopic": r.prereq_subtopic})
    return (len(unmet)==0, unmet)

def record_attempt(db: Session, user_id: int, topic: str, subtopic: str, score: float, pass_mark: float = 0.6):
    row = db.query(UserProgress).filter_by(user_id=user_id, topic=topic, subtopic=subtopic).first()
    if not row:
        row = UserProgress(user_id=user_id, topic=topic, subtopic=subtopic, attempts=0)
        db.add(row); db.flush()
    row.attempts += 1
    row.last_score = score
    if score >= pass_mark*100.0:
        row.completed = True
    db.commit()
