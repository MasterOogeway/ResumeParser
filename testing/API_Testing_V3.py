import json
from dotenv import load_dotenv

# --- IMPORTANT ---
# Make sure 'job_scraper_api_v1.py' (or whatever you named your main scraper script)
# is in the same directory as this test script, or accessible in your PYTHONPATH.
# We are importing functions from it.
try:
    from job_scraper_api_v1 import (
        # fetch_remotive_jobs, # Removed as Remotive block is removed
        fetch_arbeitnow_jobs,
        fetch_usajobs,
        fetch_github_jobs_mirror, # This one uses Arbeitnow as a proxy in your current scraper
        # fetch_jsearch_jobs, # Removed as JSearch block is removed
        # fetch_adzuna_jobs, # Removed as Adzuna block is removed
        USAJOBS_API_KEY, USAJOBS_USER_AGENT, # To check if keys are loaded
        # RAPIDAPI_JSEARCH_KEY, # Removed as JSearch block is removed
        # ADZUNA_APP_ID, ADZUNA_APP_KEY # Removed as Adzuna block is removed
    )
    print("Successfully imported functions from job_scraper_api_v1.py")
except ImportError as e:
    print(f"ERROR: Could not import from job_scraper_api_v1.py: {e}")
    print("Please ensure 'job_scraper_api_v1.py' is in the same directory or your PYTHONPATH.")
    print("Make sure all necessary functions and key variables are defined and imported from it.")
    exit()

# Load environment variables from .env file
# This should already be done in job_scraper_api_v1.py, but good to ensure here too for standalone test
load_dotenv()

def print_job_summary(api_name: str, jobs: list[dict]):
    """Prints a summary of jobs fetched from an API."""
    print(f"\n--- Results from {api_name} ---")
    if jobs:
        print(f"Found {len(jobs)} job(s).")
        print("First job details:")
        first_job = jobs[0]
        print(f"  Title: {first_job.get('title', 'N/A')}")
        print(f"  Company: {first_job.get('company', 'N/A')}")
        print(f"  Location: {first_job.get('location', 'N/A')}")
        print(f"  URL: {first_job.get('url', 'N/A')}")
        print(f"  Source: {first_job.get('source_site', 'N/A')}")
        print(f"  Extracted Skills: {', '.join(first_job.get('extracted_skills', []))[:70]}...")
    else:
        print(f"No jobs found or an error occurred for {api_name}.")
    print("-" * 30)

if __name__ == "__main__":
    print("--- API Connection Test Script ---")
    print("This script will attempt to fetch a few job listings from each configured API.")
    print("Make sure your .env file is set up with the necessary API keys.\n")

    # Define test parameters (as per the user-provided file)
    it_keywords = ["Software Developer", "IT Support", "Data Analyst", "Fresher", "Developer", "Python", "Java", "Software", "IT"]
    test_location = "India" # General location for testing
    test_limit = 2 # Fetch a small number of jobs for testing

    # Test Arbeitnow API (Original Section #2)
    # The user mentioned Arbeitnow_Jobs_API="https://arbeitnow.com/api/job-board-api"
    # This is the endpoint, not a key. Arbeitnow API does not require a key.
    print("\nTesting Arbeitnow API (no key required)...")
    arbeitnow_jobs = fetch_arbeitnow_jobs(keywords=it_keywords, limit=test_limit, location_query=test_location)
    print_job_summary("Arbeitnow API", arbeitnow_jobs)

    # Test USAJOBS API (Original Section #3)
    print("\nTesting USAJOBS API...")
    # Values for USAJOBS_API_KEY and USAJOBS_USER_AGENT will be loaded from .env
    if USAJOBS_API_KEY and USAJOBS_USER_AGENT:
        print(f"USAJOBS API Key: {'Found'}")
        print(f"USAJOBS User Agent: {'Found'}")
        # Note: USAJOBS is US-specific. "India" as location won't work well.
        # Using a generic keyword search with a US location for a more relevant test.
        usajobs_jobs = fetch_usajobs(keywords=["IT Specialist", "Computer Scientist"], limit=test_limit)
        print_job_summary("USAJOBS API", usajobs_jobs)
    else:
        print("Skipping USAJOBS API test: USAJOBS_API_KEY or USAJOBS_USER_AGENT not found in environment.")
        print_job_summary("USAJOBS API", [])

    # Test GitHub Jobs Mirror (uses Arbeitnow as proxy in your current scraper) (Original Section #6)
    # This test remains as is, as it's a conceptual test via Arbeitnow.
    print("\nTesting GitHub Jobs Mirror (uses Arbeitnow proxy)...")
    github_jobs = fetch_github_jobs_mirror(keywords=it_keywords + ["github"], limit=test_limit, location_query=test_location)
    print_job_summary("GitHub Jobs Mirror (via Arbeitnow)", github_jobs)


    print("\n--- API Connection Test Finished ---")
    print("Review the output above to check if APIs are responding and returning data.")