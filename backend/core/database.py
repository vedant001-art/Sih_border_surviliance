from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

import os

if os.getenv("VERCEL"):
    try:
        os.makedirs("/tmp", exist_ok=True)
    except Exception:
        pass
    db_url = "sqlite:////tmp/border_surveillance.db"
else:
    db_url = settings.DATABASE_URL

engine_args = {}
if db_url.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False, "timeout": 15}

engine = create_engine(db_url, **engine_args)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if db_url.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        if os.getenv("VERCEL"):
            cursor.execute("PRAGMA journal_mode=MEMORY")
        else:
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

