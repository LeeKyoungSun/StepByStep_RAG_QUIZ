# app/models/quiz.py
"""
퀴즈 DB 모델
RDS 구조에 맞춘 SQLAlchemy 모델
※ 주의: 이 스키마의 소유자는 Spring Boot + Flyway(JPA) 이고,
   이 파일은 '이미 만들어진 테이블'을 읽기 전용으로 매핑만 한다.
"""
from datetime import datetime

from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Text,
    Boolean,
    Integer,
    DateTime,
    Enum,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from app.db import Base


class QuizScenario(Base):
    """
    퀴즈 시나리오 (주제/키워드별 묶음)
    """
    __tablename__ = "quiz_scenario"
    __table_args__ = {"extend_existing": True}  # JPA 스키마 기준, AI는 따라가기만 함

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)

    # Relationships
    questions = relationship(
        "QuizQuestion",
        back_populates="scenario",
        cascade="all, delete-orphan",
    )
    attempts = relationship(
        "QuizAttempt",
        back_populates="scenario",
        cascade="all, delete-orphan",
    )


class QuizQuestion(Base):
    """
    퀴즈 질문
    """
    __tablename__ = "quiz_question"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    stem = Column(Text, nullable=False)           # JPA: @Column(name="stem", TEXT)
    correct_text = Column(Text, nullable=True)    # JPA: @Column(name="correct_text", TEXT)
    scenario_id = Column(
        BigInteger,
        ForeignKey("quiz_scenario.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Relationships
    scenario = relationship("QuizScenario", back_populates="questions")
    options = relationship(
        "QuizOption",
        back_populates="question",
        cascade="all, delete-orphan",
    )
    responses = relationship(
        "QuizResponse",
        back_populates="question",
        cascade="all, delete-orphan",
    )


class QuizOption(Base):
    """
    퀴즈 선택지
    """
    __tablename__ = "quiz_option"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)      # 선택지 내용
    label = Column(String(5), nullable=False)  # A, B, C, D
    is_correct = Column(Boolean, nullable=False, default=False)
    question_id = Column(
        BigInteger,
        ForeignKey("quiz_question.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Relationships
    question = relationship("QuizQuestion", back_populates="options")
    responses = relationship("QuizResponse", back_populates="option")


class QuizAttempt(Base):
    """
    사용자의 퀴즈 시도
    """
    __tablename__ = "quiz_attempt"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)  # FK to users table
    scenario_id = Column(
        BigInteger,
        ForeignKey("quiz_scenario.id", ondelete="CASCADE"),
        nullable=False,
    )

    started_at = Column(DateTime(6), nullable=False, default=datetime.utcnow)
    submitted_at = Column(DateTime(6), nullable=True)

    status = Column(
        Enum("IN_PROGRESS", "SUBMITTED", "CANCELLED", name="quiz_attempt_status"),
        nullable=False,
        default="IN_PROGRESS",
    )

    score_total = Column(Integer, nullable=True)  # 맞힌 개수
    score_max = Column(Integer, nullable=True)    # 전체 개수

    # Relationships
    scenario = relationship("QuizScenario", back_populates="attempts")
    responses = relationship(
        "QuizResponse",
        back_populates="attempt",
        cascade="all, delete-orphan",
    )


class QuizResponse(Base):
    """
    사용자의 개별 문제 응답
    """
    __tablename__ = "quiz_response"
    __table_args__ = {"extend_existing": True}

    attempt_id = Column(
        BigInteger,
        ForeignKey("quiz_attempt.id", ondelete="CASCADE"),
        primary_key=True,
    )
    question_id = Column(
        BigInteger,
        ForeignKey("quiz_question.id", ondelete="CASCADE"),
        primary_key=True,
    )

    option_id = Column(
        BigInteger,
        ForeignKey("quiz_option.id", ondelete="SET NULL"),
        nullable=True,
    )
    text_answer = Column(Text, nullable=True)  # 주관식 (현재 미사용)

    is_correct = Column(Boolean, nullable=True)
    score = Column(Integer, nullable=True)
    created_at = Column(DateTime(6), nullable=False, default=datetime.utcnow)

    # Relationships
    attempt = relationship("QuizAttempt", back_populates="responses")
    question = relationship("QuizQuestion", back_populates="responses")
    option = relationship("QuizOption", back_populates="responses")