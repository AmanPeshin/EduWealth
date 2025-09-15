import json
from app.db import init_db, SessionLocal, User
from app.graph_streaming import build_quiz_graph_streaming, persist_attempt
from app.agents import build_assistant_graph
from app.seed_curriculum import seed_curriculum
from app.ingest_company_pdfs import build_company_vectorstore
from app.config import QUIZ_LENGTH

def ensure_user():
    db = SessionLocal()
    u = db.query(User).filter_by(email="student@example.com").first()
    if not u:
        u = User(email="student@example.com", name="Student One")
        db.add(u); db.commit(); db.refresh(u)
    db.close()
    return u.id

def main():
    init_db()
    seed_curriculum()
    build_company_vectorstore()

    user_id = ensure_user()
    topic, subtopic, difficulty = "Corporate Finance", "NPV", "intermediate"
    attempt_thread = f"attempt-{user_id}-{topic}-{subtopic}"
    assistant_thread = f"assist-{user_id}"

    quiz_graph = build_quiz_graph_streaming()
    assistant = build_assistant_graph()

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
        "complete": False
    }
    config = {"configurable": {"thread_id": attempt_thread}}

    result = quiz_graph.invoke(state, config=config)
    answers = []
    while "__interrupt__" in result:
        intr = result["__interrupt__"].value
        if intr.get("type") == "locked":
            print("Locked:", intr["message"])
            print("Unmet prerequisites:", intr["unmet"])
            return

        i = intr["index"]
        print(f"\nQ{i+1}. {intr['question']}")
        for j, c in enumerate(intr["choices"]):
            print(f"  {j}) {c}")

        # Assistant help
        msg = {"role":"user","content": f"Explain concept for: {intr['question']}"}
        aout = assistant({"messages":[msg]}, config={"configurable":{"thread_id": assistant_thread}})
        print("\nAssistant:", aout["messages"][-1]["content"])

        user_choice = 0
        answers.append(user_choice)
        result = quiz_graph.invoke(Command(resume={"current_answer": user_choice}), config=config)

    final_state = quiz_graph.get_state(config).values
    served = final_state["served"]
    attempt_id = persist_attempt(user_id, topic, subtopic, served, answers)
    print(f"\nSaved attempt_id={attempt_id}")

if __name__ == "__main__":
    main()
