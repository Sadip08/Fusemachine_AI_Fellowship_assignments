import os
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the database URL using environment variables
SQLALCHEMY_DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5433/{os.getenv('POSTGRES_DB')}"

#Create the SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

#Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind = engine)

#Create a base class for our models
Base = declarative_base()

#Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
