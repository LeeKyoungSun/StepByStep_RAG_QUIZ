# app/db.py
import os
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Base 클래스 추가
Base = declarative_base()

# RDS 연결

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_readonly_session():
    """
    MySQL 세션 레벨에서 READ ONLY 트랜잭션으로 열어주는 helper.
    (DB 사용자 권한도 SELECT-only로 맞춰주면 더 안전)
    """
    db = SessionLocal()
    try:
        # MySQL 8 이상에서 지원
        db.execute(text("SET SESSION TRANSACTION READ ONLY"))
        yield db
    finally:
        db.close()