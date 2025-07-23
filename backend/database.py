from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./blog.db"  # or use PostgreSQL: "postgresql://user:pass@host/db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})  # for SQLite
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
