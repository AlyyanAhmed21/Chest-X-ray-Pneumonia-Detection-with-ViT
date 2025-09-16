# app/database.py

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import datetime
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# --- Database Connection ---
# Get the connection string from the environment variables
MONGODB_URL = os.getenv("MONGODB_CONNECTION_STRING")

if not MONGODB_URL:
    raise ValueError("MONGODB_CONNECTION_STRING not found in environment variables. Please check your .env file.")

# Create a client instance
client = AsyncIOMotorClient(MONGODB_URL)

# Get a handle to the database (it will be created if it doesn't exist)
# The database name 'pneumonia_db' should match the one in your connection string
database = client.pneumonia_db

# Get a handle to the collection (like a table in SQL)
patient_collection = database.get_collection("patient_records")


# --- Database Operations (now async) ---

async def add_patient_record(name: str, age: int, result: str, confidence: float) -> Dict:
    """
    Inserts a new patient record into the MongoDB collection.
    
    Returns the inserted document.
    """
    record_document = {
        "name": name,
        "age": age,
        "prediction_result": result,
        "confidence_score": confidence,
        "timestamp": datetime.datetime.utcnow()
    }
    
    # .insert_one is an async operation, so we must 'await' it
    result = await patient_collection.insert_one(record_document)
    
    # Find the newly created document to return it
    new_record = await patient_collection.find_one({"_id": result.inserted_id})
    return new_record


async def get_all_records() -> List[Dict]:
    """
    Retrieves all patient records, sorted by the most recent timestamp.
    """
    records = []
    # .find() returns a cursor, which we iterate over asynchronously
    cursor = patient_collection.find({}).sort("timestamp", -1) # -1 for descending order
    async for document in cursor:
        records.append(document)
    return records