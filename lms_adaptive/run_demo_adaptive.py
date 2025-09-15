from app.db import init_db, SessionLocal, User, UserProgress
from app.seed_curriculum import seed_curriculum
from app.ingest_company_pdfs import build_company_vectorstore
from app.graph_streaming_adaptive import build_quiz_graph_streaming_adaptive, persist_attempt
from langgraph.types import Command
from app.config import QUIZ_LENGTH

def ensure_user():
    db = SessionLocal()
    #Ensure user email id exists if not create user from frontend
    u = db.query(User).filter_by(email="student@example.com").first()
    if not u:
        u = User(email="student@example.com", name="Student One")
        db.add(u); db.commit(); db.refresh(u)
    db.close()
    return u.id

def get_next_progress(user_id):
    db = SessionLocal()
    # Get the first incomplete progress entry for the user, ordered by updated_at
    progress = (
        db.query(UserProgress)
        .filter_by(user_id=user_id, completed=False)
        .order_by(UserProgress.updated_at.asc())
        .first()
    )
    db.close()
    if progress:
        # You can set difficulty logic here, e.g., based on attempts or last_score
        u = db.query(User).filter_by(user_id=user_id).first()
        difficulty = u.user_level if u and u.user_level in ["beginner", "intermediate", "advanced"] else "intermediate"
        return progress.topic, progress.subtopic, difficulty
    else:
        # Fallback if no progress found
        return "Corporate Finance", "IRR", "intermediate"
    
def main():
    init_db()
    seed_curriculum()
    build_company_vectorstore()

    user_id = ensure_user()
    topic, subtopic, difficulty = get_next_progress(user_id)
    
    thread_id = f"adaptive-{user_id}-{topic}-{subtopic}"

    graph = build_quiz_graph_streaming_adaptive()
    state = {
        "user_id": user_id,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty": difficulty,
        "needed": QUIZ_LENGTH,
        "served": [],
        "current_index": 0,
        "current_answer": None,
        "correct_count": 0,
        "theta": 0.0,
        "complete": False
    }
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(state, config=config)
    answers = []
    while "__interrupt__" in result:
        intr = result["__interrupt__"].value
        if intr.get("type") == "locked":
            print("Locked:", intr["message"], intr["unmet"])
            return
        if intr.get("type") == "no_item_available":
            print("No next item available; ending early.")
            break
        i = intr["index"]
        print(f"\nQ{i+1}. {intr['question']}")
        for j, c in enumerate(intr["choices"]):
            print(f"  {j}) {c}")
        # Simulate an answer (choose 0) or you can import answer from request
        choice = 0
        answers.append(choice)
        result = graph.invoke(Command(resume={"current_answer": choice}), config=config)

    final = graph.get_state(config).values
    attempt_id = persist_attempt(user_id, topic, subtopic, final["served"], answers)
    print(f"Adaptive attempt saved: {attempt_id}, theta_end={final['theta']:.2f}")

if __name__ == "__main__":
    main()
