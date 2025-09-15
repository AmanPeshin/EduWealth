from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey,
    JSON, Float, Boolean, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from .config import DB_URL

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    attempts = relationship("QuizAttempt", back_populates="user")

class Lesson(Base):
    __tablename__ = "lessons"
    id = Column(Integer, primary_key=True)
    title = Column(String, index=True)
    topic = Column(String, index=True)
    description = Column(Text)

class QuestionItem(Base):
    __tablename__ = "question_items"
    id = Column(Integer, primary_key=True)
    item_id = Column(String, unique=True, index=True)
    source = Column(String)  # "generated" | "curated"
    topic = Column(String, index=True)
    subtopic = Column(String, index=True)
    difficulty = Column(String, index=True)
    payload = Column(JSON)   # {question, choices, answer_index, explanation}
    embedding = Column(JSON) # vector as list[float]
    created_at = Column(DateTime, default=datetime.utcnow)

class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    topic = Column(String, index=True)
    subtopic = Column(String, index=True)
    passed = Column(Boolean, default=False)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    user = relationship("User", back_populates="attempts")
    responses = relationship("AttemptResponse", back_populates="attempt", cascade="all, delete-orphan")

class AttemptResponse(Base):
    __tablename__ = "attempt_responses"
    id = Column(Integer, primary_key=True)
    attempt_id = Column(Integer, ForeignKey("quiz_attempts.id"))
    item_id = Column(String, index=True)
    question = Column(Text)
    choices = Column(JSON)
    correct_index = Column(Integer)
    user_index = Column(Integer, nullable=True)
    correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    attempt = relationship("QuizAttempt", back_populates="responses")
    __table_args__ = (UniqueConstraint("attempt_id", "item_id", name="uniq_attempt_item"),)

# Progression
class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    order_index = Column(Integer, default=0)
    subtopics = relationship("Subtopic", back_populates="topic")

class Subtopic(Base):
    __tablename__ = "subtopics"
    id = Column(Integer, primary_key=True)
    topic_id = Column(Integer, ForeignKey("topics.id"))
    name = Column(String, index=True)
    order_index = Column(Integer, default=0)
    topic = relationship("Topic", back_populates="subtopics")
    __table_args__ = (UniqueConstraint("topic_id","name", name="uniq_topic_subtopic"),)

class Prerequisite(Base):
    __tablename__ = "prerequisites"
    id = Column(Integer, primary_key=True)
    prereq_topic = Column(String, index=True)
    prereq_subtopic = Column(String, index=True)  # "ANY" allowed
    target_topic = Column(String, index=True)
    target_subtopic = Column(String, index=True)  # "ANY" allowed

class UserProgress(Base):
    __tablename__ = "user_progress"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    topic = Column(String, index=True)
    subtopic = Column(String, index=True)
    attempts = Column(Integer, default=0)
    completed = Column(Boolean, default=False)
    last_score = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("user_id","topic","subtopic", name="uniq_user_topic_subtopic"),)

def init_db():
    Base.metadata.create_all(bind=engine)
