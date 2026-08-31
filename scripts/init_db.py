import sys
import os

# Add the project root to the python path so we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.database import engine, Base
from backend.models import schema
from loguru import logger

def init_db():
    logger.info("Creating database tables...")
    try:
        # Create all tables in the database
        Base.metadata.create_all(bind=engine)
        logger.info("Successfully created database tables!")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        logger.error("Please ensure PostgreSQL is running and the credentials in .env are correct.")

if __name__ == "__main__":
    init_db()
