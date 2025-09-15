from .db import SessionLocal, Topic, Subtopic, Prerequisite

def seed_curriculum():
    db = SessionLocal()
    topics = {
        "Corporate Finance": ["Time Value of Money", "NPV", "IRR", "WACC", "Capital Structure", "CAPM"]
    }
    order = 0
    for t, subs in topics.items():
        topic = db.query(Topic).filter_by(name=t).first()
        if not topic:
            topic = Topic(name=t, order_index=order); db.add(topic); db.flush()
        for i, s in enumerate(subs):
            if not db.query(Subtopic).filter_by(topic_id=topic.id, name=s).first():
                db.add(Subtopic(topic_id=topic.id, name=s, order_index=i))
        order += 1
    db.commit()

    prereqs = [
        ("Corporate Finance","Time Value of Money","Corporate Finance","NPV"),
        ("Corporate Finance","NPV","Corporate Finance","IRR"),
        ("Corporate Finance","IRR","Corporate Finance","WACC"),
        ("Corporate Finance","Capital Structure","Corporate Finance","WACC"),
        ("Corporate Finance","WACC","Corporate Finance","CAPM"),
    ]
    for pt, ps, tt, ts in prereqs:
        if not db.query(Prerequisite).filter_by(prereq_topic=pt, prereq_subtopic=ps, target_topic=tt, target_subtopic=ts).first():
            db.add(Prerequisite(prereq_topic=pt, prereq_subtopic=ps, target_topic=tt, target_subtopic=ts))
    db.commit()
