"""
Database connection setup (MySQL via SQLAlchemy).

Set the DERMASENSE_DATABASE_URL environment variable, e.g.:
    mysql+pymysql://dermasense_user:password@localhost:3306/dermasense
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DERMASENSE_DATABASE_URL",
    "mysql+pymysql://dermasense_user:password@localhost:3306/dermasense",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=280)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
