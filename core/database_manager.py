import os
from pymongo import MongoClient, UpdateOne, errors
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables (e.g., for MONGO_URI)
load_dotenv()

# --- Configuration ---
MONGO_URI = os.environ.get("MONGO_ATLAS_URI") # Your MongoDB Atlas connection string
DB_NAME = "job_matching_app" # Or your preferred database name
PERSONALIZED_SEARCH_COLLECTION = "personalized_searches"
RECOMMENDED_JOBS_COLLECTION = "recommended_jobs_cache"

# Global client and db variables to reuse connection
client = None
db = None

def connect_db():
    """
    Establishes a connection to MongoDB Atlas.
    Returns the database object.
    """
    global client, db
    if db is not None:
        # Test connection with a simple command if already initialized
        try:
            client.admin.command('ping')
            # print("DEBUG: Reusing existing MongoDB connection.")
            return db
        except (errors.ConnectionFailure, errors.ServerSelectionTimeoutError) as e:
            print(f"DEBUG: Existing MongoDB connection lost or timed out: {e}. Reconnecting...")
            client = None # Force re-initialization
            db = None

    if not MONGO_URI:
        print("ERROR: MONGO_ATLAS_URI environment variable not set.")
        raise ValueError("MONGO_ATLAS_URI not set")

    try:
        print("Attempting to connect to MongoDB Atlas...")
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas!")
        db = client[DB_NAME]
        return db
    except errors.ConfigurationError as e_conf:
        print(f"MongoDB Configuration Error: {e_conf}")
        raise
    except errors.ConnectionFailure as e_conn:
        print(f"MongoDB Connection Failure: Could not connect to server: {e_conn}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        raise

# --- Personalized Search Results (Session-Specific) ---

def save_personalized_search_session(session_id: str, resume_score: float, extracted_skills: list,
                                   personalized_job_results: list, raw_resume_text: str = None):
    """
    Saves or updates a document containing resume data and personalized job results
    for a specific session. This is intended to be one document per session/resume upload.
    """
    if not session_id:
        print("ERROR: session_id is required to save personalized search.")
        return None

    try:
        database = connect_db()
        collection = database[PERSONALIZED_SEARCH_COLLECTION]

        search_document = {
            "session_id": session_id,
            "resume_data": {
                "score": resume_score,
                "extracted_skills": extracted_skills,
                "raw_text": raw_resume_text
            },
            "personalized_jobs": personalized_job_results, # List of job dicts
            "created_at": datetime.now(timezone.utc) # For TTL or manual cleanup
        }

        # Using update_one with upsert=True will create if not exists, or replace if exists.
        # This effectively means one active record per session_id.
        result = collection.update_one(
            {"session_id": session_id},
            {"$set": search_document},
            upsert=True
        )
        print(f"Personalized search session for '{session_id}' saved. Matched: {result.matched_count}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")
        return result.upserted_id if result.upserted_id else session_id # Return id or session_id
    except Exception as e:
        print(f"Error saving personalized search session for '{session_id}': {e}")
        return None

def get_personalized_search_session(session_id: str):
    """
    Retrieves the personalized search session data for a given session_id.
    """
    if not session_id:
        print("ERROR: session_id is required to retrieve personalized search.")
        return None
    try:
        database = connect_db()
        collection = database[PERSONALIZED_SEARCH_COLLECTION]
        document = collection.find_one({"session_id": session_id})
        if document:
            print(f"Personalized search session found for '{session_id}'.")
        else:
            print(f"No personalized search session found for '{session_id}'.")
        return document
    except Exception as e:
        print(f"Error retrieving personalized search session for '{session_id}': {e}")
        return None

def delete_personalized_search_session(session_id: str):
    """
    Deletes the personalized search session data for a given session_id.
    This would be called by your Flask app when a session ends or needs cleanup.
    """
    if not session_id:
        print("ERROR: session_id is required to delete personalized search.")
        return False
    try:
        database = connect_db()
        collection = database[PERSONALIZED_SEARCH_COLLECTION]
        result = collection.delete_one({"session_id": session_id})
        if result.deleted_count > 0:
            print(f"Personalized search session for '{session_id}' deleted successfully.")
            return True
        else:
            print(f"No personalized search session found for '{session_id}' to delete.")
            return False
    except Exception as e:
        print(f"Error deleting personalized search session for '{session_id}': {e}")
        return False

# --- Recommended Job Results (More Persistent Cache) ---

def save_recommended_job(job_data: dict, source_keywords: list):
    """
    Saves a job listing obtained from predefined/recommended keywords to a separate cache.
    Uses job URL as a unique identifier to avoid duplicates (upsert).
    These are NOT deleted with the personalized session data.
    """
    if not job_data or not job_data.get('url'):
        print("ERROR: Job data with a URL is required to save a recommended job.")
        return None

    try:
        database = connect_db()
        collection = database[RECOMMENDED_JOBS_COLLECTION]

        # Prepare document structure
        job_document = {
            "job_details": job_data,
            "source_keywords": source_keywords, # The predefined keywords that found this job
            "first_seen_at": datetime.now(timezone.utc), # Set only on insert
            "last_updated_at": datetime.now(timezone.utc) # Updated every time
        }

        # Upsert based on the job URL to avoid duplicates
        # $setOnInsert ensures first_seen_at is only set when a new document is created
        result = collection.update_one(
            {"job_details.url": job_data['url']},
            {
                "$set": {
                    "job_details": job_data,
                    "source_keywords": source_keywords,
                    "last_updated_at": datetime.now(timezone.utc)
                },
                "$setOnInsert": {"first_seen_at": datetime.now(timezone.utc)}
            },
            upsert=True
        )
        if result.upserted_id:
            print(f"New recommended job added to cache: {job_data['url']}")
        else:
            print(f"Recommended job updated in cache: {job_data['url']}")
        return result.upserted_id if result.upserted_id else job_data['url']
    except Exception as e:
        print(f"Error saving recommended job '{job_data.get('url')}': {e}")
        return None

def get_recommended_jobs_by_keywords(keywords: list, limit: int = 20):
    """
    Retrieves recommended jobs from the cache that were found using any of the given predefined keywords.
    """
    if not keywords:
        print("INFO: No keywords provided for fetching recommended jobs.")
        return []
    try:
        database = connect_db()
        collection = database[RECOMMENDED_JOBS_COLLECTION]
        # Find jobs where the 'source_keywords' array contains any of the provided keywords
        query = {"source_keywords": {"$in": keywords}}
        jobs = list(collection.find(query).sort("last_updated_at", -1).limit(limit))
        print(f"Found {len(jobs)} recommended jobs for keywords: {keywords}")
        return jobs
    except Exception as e:
        print(f"Error retrieving recommended jobs by keywords: {e}")
        return []


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("Database Manager - Direct Test Mode")

    # Ensure MONGO_ATLAS_URI is set in your .env file for this test to work
    if not MONGO_URI:
        print("Please set MONGO_ATLAS_URI in your .env file to run tests.")
    else:
        # Test Personalized Search
        test_session_id = "test_session_12345"
        sample_resume_score = 0.85
        sample_skills = ["Python", "MongoDB", "Flask"]
        sample_jobs = [
            {"title": "Python Developer", "company": "TestCorp", "url": "http://example.com/job1"},
            {"title": "Flask Engineer", "company": "WebInc", "url": "http://example.com/job2"}
        ]
        sample_raw_text = "This is the extracted resume text..."

        print(f"\n--- Testing Personalized Search for session: {test_session_id} ---")
        save_personalized_search_session(test_session_id, sample_resume_score, sample_skills, sample_jobs, sample_raw_text)
        retrieved_session = get_personalized_search_session(test_session_id)
        if retrieved_session:
            print(f"Retrieved {len(retrieved_session.get('personalized_jobs', []))} personalized jobs.")
            # print(retrieved_session) # Uncomment to see full data

        # Test Recommended Jobs
        print("\n--- Testing Recommended Jobs ---")
        rec_job_1 = {"title": "General Software Engineer", "company": "AnyCompany", "url": "http://example.com/recjob1", "description_text": "A general role."}
        rec_job_2 = {"title": "IT Support Specialist", "company": "HelpDesk Ltd", "url": "http://example.com/recjob2", "description_text": "Support role."}
        rec_job_1_again = {"title": "General Software Engineer (Updated)", "company": "AnyCompany", "url": "http://example.com/recjob1", "description_text": "Updated general role."}


        save_recommended_job(rec_job_1, ["software", "developer"])
        save_recommended_job(rec_job_2, ["IT", "support"])
        save_recommended_job(rec_job_1_again, ["software", "developer", "general"]) # This should update rec_job_1

        retrieved_recommended = get_recommended_jobs_by_keywords(["software", "IT"], limit=5)
        for job_doc in retrieved_recommended:
            print(f"  - Recommended Job: {job_doc['job_details']['title']} from {job_doc['job_details']['company']}")

        # Test Deletion (optional - uncomment to test)
        # print(f"\n--- Testing Deletion for session: {test_session_id} ---")
        # if delete_personalized_search_session(test_session_id):
        #     retrieved_after_delete = get_personalized_search_session(test_session_id)
        #     if not retrieved_after_delete:
        #         print("Session data successfully deleted and verified.")
        #     else:
        #         print("ERROR: Session data still exists after deletion attempt.")

        print("\n--- Test complete ---")