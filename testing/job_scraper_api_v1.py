import os
import requests
import json
import re
from urllib.parse import quote_plus # For URL encoding search terms
from dotenv import load_dotenv # To load .env file for local development

load_dotenv() # This loads all variables from .env into environment variables

# --- Configuration & Constants ---
# Load environment variables from .env file for local development
# Make sure to create a .env file with your API keys or set them in your environment
load_dotenv()

# API Key Placeholders - these should be set as environment variables
# For USAJOBS:
USAJOBS_API_KEY = os.environ.get('USAJOBS_API_KEY')
USAJOBS_USER_AGENT = os.environ.get('USAJOBS_USER_AGENT') # USAJOBS requires a user agent (your email)

# For JSearch (RapidAPI):
RAPIDAPI_JSEARCH_KEY = os.environ.get('RAPIDAPI_JSEARCH_KEY')
RAPIDAPI_HOST = "jsearch.p.rapidapi.com" # This can be a constant in your script

# For Adzuna:
ADZUNA_APP_ID = os.environ.get('ADZUNA_APP_ID')
ADZUNA_APP_KEY = os.environ.get('ADZUNA_APP_KEY')


# Default User-Agent for other requests
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 ResumeJobMatcher/1.0'

# Predefined skills for extraction - you should expand this list significantly,
# possibly by importing and processing your skills_db from resume_parser.py
PREDEFINED_SKILLS_KEYWORDS = [
    'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js', 'sql', 'nosql', 'c++', 'c#',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'machine learning', 'data analysis', 'data science',
    'project management', 'agile', 'scrum', 'communication', 'problem solving', 'teamwork',
    'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'html', 'css', 'django', 'flask',
    'spring boot', 'microservices', 'rest api', 'git', 'linux', 'cybersecurity','Software Development','Design','Product', 'fresher', 'entry level',
    'devops', 'ui/ux', 'mobile development', 'ios', 'android', 'swift', 'kotlin','Data Analysis','DevOps','Sysadmin'
]
PREDEFINED_SKILLS_LOWER = [skill.lower() for skill in PREDEFINED_SKILLS_KEYWORDS]

# --- Helper Functions ---

def make_request(url: str, headers: dict = None, params: dict = None, timeout: int = 15) -> dict | None:
    """
    Makes an HTTP GET request and returns the JSON response.
    """
    if headers is None:
        headers = {'User-Agent': DEFAULT_USER_AGENT}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"Timeout error fetching URL: {url}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - URL: {url}")
        if response is not None:
            print(f"Response content: {response.text[:500]}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
    except json.JSONDecodeError as e_json:
        print(f"Error decoding JSON from {url}: {e_json}")
        if response is not None:
            print(f"Response content that failed to parse: {response.text[:500]}")
    return None

def extract_skills_from_text(text: str, skill_list: list) -> list[str]:
    """
    Extracts predefined skills from a block of text using regex for whole word matching.
    """
    if not text:
        return []
    
    found_skills = set()
    text_lower = text.lower()
    
    sorted_skill_list = sorted(skill_list, key=len, reverse=True)
    
    for skill_keyword_lower in sorted_skill_list:
        try:
            pattern = r'\b' + re.escape(skill_keyword_lower) + r'\b'
            if re.search(pattern, text_lower):
                try:
                    original_casing_skill = PREDEFINED_SKILLS_KEYWORDS[PREDEFINED_SKILLS_LOWER.index(skill_keyword_lower)]
                    found_skills.add(original_casing_skill)
                except ValueError:
                    found_skills.add(skill_keyword_lower) 
        except re.error as e_re:
            print(f"Regex error for skill '{skill_keyword_lower}': {e_re}")
            continue

    return sorted(list(found_skills))

# --- API Specific Fetch Functions ---

def fetch_remotive_jobs(keywords: list[str], limit: int = 5) -> list[dict]:
    """
    Fetches remote jobs from Remotive API.
    API Docs: https://remotive.com/api/readme
    No API key required.
    """
    print(f"\nFetching jobs from Remotive for keywords: {keywords}...")
    base_url = "https://remotive.com/api/remote-jobs"
    search_query = " ".join(keywords) 
    params = {
        'search': search_query,
        'limit': limit
    }
    data = make_request(base_url, params=params)
    
    jobs = []
    if data and 'jobs' in data:
        for job_entry in data['jobs']:
            description = job_entry.get('description', '')
            cleaned_description = re.sub(r'<[^>]+>', ' ', description)
            cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()

            jobs.append({
                'title': job_entry.get('title'),
                'company': job_entry.get('company_name'),
                'location': job_entry.get('candidate_required_location', 'Remote'),
                'description_text': cleaned_description,
                'extracted_skills': extract_skills_from_text(cleaned_description, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('url'),
                'publication_date': job_entry.get('publication_date'),
                'source_site': 'Remotive API'
            })
        print(f"Found {len(jobs)} jobs from Remotive.")
    else:
        print("No jobs found or error fetching from Remotive.")
    return jobs

def fetch_arbeitnow_jobs(keywords: list[str], limit: int = 5, location_query: str = "India") -> list[dict]:
    """
    Fetches jobs from Arbeitnow API.
    API Docs: https://documenter.getpostman.com/view/18545278/UVJbJdKh
    No API key required.
    """
    print(f"\nFetching jobs from Arbeitnow for keywords: {keywords}, location: {location_query}...")
    base_url = "https://arbeitnow.com/api/job-board-api"
    search_query = " ".join(keywords)
    if location_query and location_query.lower() != "any":
        search_query += f" in {location_query}"

    params = {'q': search_query, 'page': 1}
    
    data = make_request(base_url, params=params)
    
    jobs = []
    if data and 'data' in data:
        fetched_count = 0
        for job_entry in data['data']:
            if fetched_count >= limit:
                break
            description = job_entry.get('description', '')
            jobs.append({
                'title': job_entry.get('title'),
                'company': job_entry.get('company_name'),
                'location': job_entry.get('location', location_query if location_query else "Not specified"),
                'description_text': description,
                'extracted_skills': extract_skills_from_text(description, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('url'),
                'publication_date': job_entry.get('created_at'),
                'source_site': 'Arbeitnow API'
            })
            fetched_count += 1
        print(f"Found {len(jobs)} jobs from Arbeitnow (up to limit {limit}).")
    else:
        print("No jobs found or error fetching from Arbeitnow.")
    return jobs

def fetch_usajobs(keywords: list[str], limit: int = 5, location_name: str = None) -> list[dict]:
    """
    Fetches jobs from USAJOBS API.
    Requires API key and User-Agent (email).
    """
    print(f"\nFetching jobs from USAJOBS for keywords: {keywords}...")
    if not USAJOBS_API_KEY or not USAJOBS_USER_AGENT:
        print("USAJOBS_API_KEY or USAJOBS_USER_AGENT not set in environment. Skipping USAJOBS.")
        return []

    base_url = "https://data.usajobs.gov/api/search"
    headers = {
        'Authorization-Key': USAJOBS_API_KEY,
        'User-Agent': USAJOBS_USER_AGENT,
       # 'Host': 'data.usajobs.gov'
    }
    search_query = " ".join(keywords)
    params = {
        'Keyword': search_query,
        'ResultsPerPage': limit
    }
    if location_name:
        params['LocationName'] = location_name
        print(f"Note: USAJOBS searches for US locations. '{location_name}' might not yield relevant results if it's not a US location.")

    data = make_request(base_url, headers=headers, params=params)
    
    jobs = []
    if data and 'SearchResult' in data and 'SearchResultItems' in data['SearchResult']:
        for job_entry_outer in data['SearchResult']['SearchResultItems']:
            job_entry = job_entry_outer.get('MatchedObjectDescriptor', {})
            description = job_entry.get('UserArea', {}).get('Details', {}).get('JobSummary', '')
            if not description:
                 duties = job_entry.get('UserArea', {}).get('Details', {}).get('MajorDuties', [])
                 if isinstance(duties, list): description = " ".join(duties)

            jobs.append({
                'title': job_entry.get('PositionTitle'),
                'company': job_entry.get('OrganizationName'),
                'location': job_entry.get('PositionLocationDisplay'),
                'description_text': description,
                'extracted_skills': extract_skills_from_text(description, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('PositionURI'),
                'publication_date': job_entry.get('PublicationStartDate'),
                'source_site': 'USAJOBS API'
            })
        print(f"Found {len(jobs)} jobs from USAJOBS.")
    else:
        print("No jobs found or error fetching from USAJOBS.")
    return jobs

def fetch_adzuna_jobs(keywords: list[str], limit: int = 5, location_query: str = "India", country_code: str = "in") -> list[dict]:
    """
    Fetches jobs from Adzuna API.
    API Docs: https://developer.adzuna.com/docs/search
    Requires App ID and App Key.
    """
    print(f"\nFetching jobs from Adzuna for keywords: {keywords}, location: {location_query}, country: {country_code}...")
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        print("ADZUNA_APP_ID or ADZUNA_APP_KEY not set in environment. Skipping Adzuna.")
        return []

    base_url = f"http://api.adzuna.com/v1/api/jobs/{country_code.lower()}/search/1" # Page 1
    # Adzuna 'what' parameter is a single string
    search_query = " ".join(keywords)
    
    params = {
        'app_id': ADZUNA_APP_ID,
        'app_key': ADZUNA_APP_KEY,
        'results_per_page': limit,
        'what': search_query,
        'content-type': 'application/json' # To request JSON response
    }
    # Adzuna uses 'where' for location
    if location_query and location_query.lower() != "any":
        params['where'] = location_query

    data = make_request(base_url, params=params) # Adzuna doesn't typically need special headers beyond User-Agent
    
    jobs = []
    if data and 'results' in data:
        for job_entry in data['results']:
            description = job_entry.get('description', '')
            # Adzuna description is usually plain text.
            
            jobs.append({
                'title': job_entry.get('title'),
                'company': job_entry.get('company', {}).get('display_name'),
                'location': job_entry.get('location', {}).get('display_name'),
                'description_text': description,
                'extracted_skills': extract_skills_from_text(description, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('redirect_url'), # Adzuna provides a redirect_url
                'publication_date': job_entry.get('created'), # 'created' field for publication date
                'source_site': 'Adzuna API'
            })
        print(f"Found {len(jobs)} jobs from Adzuna.")
    else:
        print("No jobs found or error fetching from Adzuna.")
        if data:
             print(f"Adzuna response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    return jobs


def fetch_github_jobs_mirror(keywords: list[str], limit: int = 5, location_query: str = "India") -> list[dict]:
    """
    Conceptual fetch for GitHub Jobs (uses Arbeitnow as a proxy if "github" in keywords).
    """
    print(f"\nFetching jobs from GitHub Jobs Mirror for keywords: {keywords}...")
    if "github" in [kw.lower() for kw in keywords]:
        print("Attempting to find GitHub-like jobs via Arbeitnow as a proxy...")
        return fetch_arbeitnow_jobs(keywords + ["developer", "engineer"], limit, location_query)

    print("Direct GitHub Jobs API is deprecated. Reliable public mirrors are scarce.")
    print("Consider searching developer-specific job boards or using broader APIs with dev keywords.")
    return []


def fetch_jsearch_jobs(keywords: list[str], limit: int = 5, location_query: str = "India") -> list[dict]:
    """
    Fetches jobs from JSearch API via RapidAPI.
    Requires RapidAPI key.
    """
    print(f"\nFetching jobs from JSearch (RapidAPI) for keywords: {keywords}...")
    if not RAPIDAPI_JSEARCH_KEY:
        print("RAPIDAPI_JSEARCH_KEY not set in environment. Skipping JSearch.")
        return []

    base_url = f"https://{RAPIDAPI_HOST}/search"
    search_query = " ".join(keywords)
    if location_query and location_query.lower() != "any":
        search_query += f" in {location_query}"
        
    querystring = {
        "query": search_query,
        "page": "1",
        "num_pages": "1",
    }
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_JSEARCH_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    
    data = make_request(base_url, headers=headers, params=querystring)
    
    jobs = []
    if data and 'data' in data:
        fetched_count = 0
        for job_entry in data['data']:
            if fetched_count >= limit:
                break
            
            description = job_entry.get('job_description', '')
            cleaned_description = re.sub(r'<[^>]+>', ' ', description)
            cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()

            job_location_parts = [
                job_entry.get('job_city'),
                job_entry.get('job_state'),
                job_entry.get('job_country')
            ]
            job_location = ", ".join(filter(None, job_location_parts)) or "Not specified"

            jobs.append({
                'title': job_entry.get('job_title'),
                'company': job_entry.get('employer_name'),
                'location': job_location,
                'description_text': cleaned_description,
                'extracted_skills': extract_skills_from_text(cleaned_description, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('job_apply_link') or job_entry.get('job_google_link'),
                'publication_date': job_entry.get('job_posted_at_datetime_utc'),
                'source_site': 'JSearch API (RapidAPI)'
            })
            fetched_count +=1
        print(f"Found {len(jobs)} jobs from JSearch (up to limit {limit}).")
    else:
        print("No jobs found or error fetching from JSearch.")
        if data:
            print(f"JSearch response structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    return jobs


# --- Main Orchestrator ---

def scrape_jobs(
    keywords: list[str],
    location: str = "India",
    max_jobs_per_source: int = 3
) -> list[dict]:
    """
    Main function to scrape jobs from all configured API sources.
    """
    if not keywords:
        print("No keywords provided for job scraping.")
        return []

    print(f"--- Starting Job Scraping for keywords: {keywords}, Location: {location} ---")
    all_jobs = []

    remotive_keywords = keywords[:]
    if "remote" not in [kw.lower() for kw in remotive_keywords]:
        remotive_keywords.append("remote")
    all_jobs.extend(fetch_remotive_jobs(keywords=remotive_keywords, limit=max_jobs_per_source))
    all_jobs.extend(fetch_arbeitnow_jobs(keywords=keywords, limit=max_jobs_per_source, location_query=location))
    
    # Adzuna - determine country code from location if possible, or use a default.
    # This is a simple mapping; a more robust solution would be needed for many countries.
    country_code_map = {"india": "in", "usa": "us", "united states": "us", "uk": "gb", "united kingdom": "gb", "germany": "de"}
    adzuna_country_code = country_code_map.get(location.lower(), "gb") # Default to 'gb' if location not mapped
    if location.lower() == "india": adzuna_country_code = "in" # Explicit for India
    
    all_jobs.extend(fetch_adzuna_jobs(keywords=keywords, limit=max_jobs_per_source, location_query=location, country_code=adzuna_country_code))
    
    all_jobs.extend(fetch_jsearch_jobs(keywords=keywords, limit=max_jobs_per_source, location_query=location))
    
    # Conditionally call USAJOBS if location seems US-based or for testing
    # if location.lower() in ["usa", "united states"] or "washington d.c." in location.lower():
    # all_jobs.extend(fetch_usajobs(keywords=keywords, limit=max_jobs_per_source, location_name=location))
    # else:
    # print("Skipping USAJOBS for non-US location in main scrape_jobs call.")
    
    all_jobs.extend(fetch_github_jobs_mirror(keywords=keywords, limit=max_jobs_per_source, location_query=location))


    print(f"\n--- Total jobs fetched before deduplication: {len(all_jobs)} ---")
    
    seen_urls = set()
    unique_jobs = []
    for job in all_jobs:
        job_url = job.get('url')
        if job_url and job_url not in seen_urls:
            unique_jobs.append(job)
            seen_urls.add(job_url)
        elif not job_url: # If no URL, keep it for now
            unique_jobs.append(job)
    
    print(f"--- Total unique jobs (by URL): {len(unique_jobs)} ---")
    return unique_jobs

# --- Example Usage ---
if __name__ == "__main__":
    print("Job Scraper Initializing (API Version)...")

    if not USAJOBS_API_KEY or not USAJOBS_USER_AGENT:
        print("WARNING: USAJOBS_API_KEY or USAJOBS_USER_AGENT is not set. USAJOBS scraping will be skipped if called directly or by main orchestrator for US locations.")
    if not RAPIDAPI_JSEARCH_KEY:
        print("WARNING: RAPIDAPI_JSEARCH_KEY is not set. JSearch scraping will be skipped.")
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        print("WARNING: ADZUNA_APP_ID or ADZUNA_APP_KEY is not set. Adzuna scraping will be skipped.")

    example_search_keywords = ["Python Developer", "Fresher", "Java"]
    example_location = "Mumbai, India"

    scraped_job_data = scrape_jobs(
        keywords=example_search_keywords,
        location=example_location,
        max_jobs_per_source=2
    )

    if scraped_job_data:
        print(f"\n--- Aggregated Scraped Job Data ({len(scraped_job_data)} total unique jobs) ---")
        for i, job in enumerate(scraped_job_data):
            print(f"\nJob {i+1}:")
            print(f"  Title: {job.get('title', 'N/A')}")
            print(f"  Company: {job.get('company', 'N/A')}")
            print(f"  Location: {job.get('location', 'N/A')}")
            print(f"  URL: {job.get('url', 'N/A')}")
            print(f"  Source: {job.get('source_site', 'N/A')}")
            print(f"  Publication Date: {job.get('publication_date', 'N/A')}")
            print(f"  Extracted Skills ({len(job.get('extracted_skills', []))}): {', '.join(job.get('extracted_skills', []))[:150]}...")
    else:
        print("\nNo job data was successfully aggregated in this run.")

