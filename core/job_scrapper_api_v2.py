import os
import requests
import json
import re
from urllib.parse import quote_plus # For URL encoding search terms
from dotenv import load_dotenv # To load .env file for local development

load_dotenv() # This loads all variables from .env into environment variables

# --- Configuration & Constants ---
# API Key Placeholders - these should be set as environment variables
USAJOBS_API_KEY = os.environ.get('USAJOBS_API_KEY')
USAJOBS_USER_AGENT = os.environ.get('USAJOBS_USER_AGENT')
RAPIDAPI_JSEARCH_KEY = os.environ.get('RAPIDAPI_JSEARCH_KEY')
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
ADZUNA_APP_ID = os.environ.get('ADZUNA_APP_ID')
ADZUNA_APP_KEY = os.environ.get('ADZUNA_APP_KEY')

DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 ResumeJobMatcher/1.0'

PREDEFINED_SKILLS_KEYWORDS = [
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
    'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell scripting', 'bash', 'powershell', 'c',
    'objective-c', 'groovy', 'dart', 'lua', 'assembly',
    # Web Development - Frontend
    'html', 'html5', 'css', 'css3', 'react', 'react.js', 'angular', 'angular.js', 'vue', 'vue.js',
    'next.js', 'nuxt.js', 'gatsby', 'jquery', 'bootstrap', 'tailwind css', 'sass', 'less',
    'webpack', 'babel', 'gulp', 'grunt', 'ember.js', 'svelte', 'webassembly', 'restful apis',
    'soap apis', 'graphql', 'ajax', 'json', 'xml', 'jwt', 'websockets', 'ssr', 'csr', 'pwa',
    'responsive design', 'cross-browser compatibility',
    # Web Development - Backend
    'node.js', 'express', 'express.js', 'django', 'flask', 'spring', 'spring boot', 'asp.net', '.net core',
    'laravel', 'ruby on rails', 'phoenix', 'elixir', 'fastapi', 'hapi', 'koa', 'nestJS', 'strapi',
    'serverless framework', 'firebase', 'api development', 'api design',
    # Databases & Data Storage
    'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'mongo', 'redis', 'oracle db', 'sqlite',
    'cassandra', 'dynamodb', 'elasticsearch', 'neo4j', 'couchdb', 'mariadb', 'ms sql server',
    'nosql', 'firebase realtimedb', 'firebase firestore', 'influxdb', 'etcd', 'data warehousing',
    'database design', 'database administration', 'data modeling', 'query optimization', 'sql alchemy',
    'hibernate', 'typeorm', 'prisma',
    # Cloud Platforms & Services
    'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud platform',
    'google cloud', 'heroku', 'digitalocean', 'linode', 'ovh', 'alibaba cloud', 'ibm cloud',
    'oracle cloud infrastructure', 'oci', 'vmware', 'openshift', 'lambda', 'azure functions',
    'google cloud functions', 's3', 'ec2', 'rds', 'azure blob storage', 'azure virtual machines',
    'google cloud storage', 'google compute engine', 'cloudformation', 'azure resource manager',
    'google cloud deployment manager', 'cloudwatch', 'azure monitor', 'stackdriver',
    # DevOps & Infrastructure
    'docker', 'kubernetes', 'k8s', 'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions',
    'circleci', 'travis ci', 'chef', 'puppet', 'vagrant', 'prometheus', 'grafana', 'elk stack',
    'splunk', 'nagios', 'zabbix', 'infrastructure as code', 'iac', 'ci/cd', 'continuous integration',
    'continuous delivery', 'continuous deployment', 'configuration management', 'monitoring',
    'logging', 'alerting', 'site reliability engineering', 'sre', 'devops', 'sysadmin',
    # Operating Systems
    'linux', 'unix', 'windows server', 'macos', 'ubuntu', 'centos', 'debian', 'red hat', 'fedora',
    'coreos', 'alpine linux',
    # Data Science, Machine Learning, AI
    'machine learning', 'ml', 'deep learning', 'dl', 'data analysis', 'data science', 'statistics',
    'natural language processing', 'nlp', 'computer vision', 'cv', 'artificial intelligence', 'ai',
    'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn', 'tensorflow', 'keras', 'pytorch', 'torch',
    'matplotlib', 'seaborn', 'plotly', 'jupyter notebooks', 'rstudio', 'tableau', 'power bi',
    'apache spark', 'spark', 'hadoop', 'kafka', 'apache kafka', 'airflow', 'apache airflow',
    'hive', 'presto', 'dask', 'xgboost', 'lightgbm', 'catboost', 'shap', 'nltk', 'spacy', 'opencv',
    'data mining', 'data visualization', 'big data', 'etl', 'feature engineering', 'model deployment',
    'recommender systems', 'time series analysis', 'a/b testing', 'reinforcement learning', 'mlops',
    # Mobile Development
    'mobile development', 'ios', 'android development', 'react native', 'flutter', 'xamarin',
    'cordova', 'ionic', 'swift', 'objective-c', 'kotlin', 'java (android)', 'swiftui',
    'jetpack compose', 'kotlin multiplatform', 'xcode', 'android studio',
    # Software Engineering Practices & Tools
    'git', 'github', 'gitlab', 'bitbucket', 'svn', 'jira', 'confluence', 'slack',
    'microsoft teams', 'trello', 'asana', 'notion', 'unit testing', 'integration testing',
    'end-to-end testing', 'test driven development', 'tdd', 'behavior driven development', 'bdd',
    'design patterns', 'software architecture', 'microservices architecture', 'agile', 'scrum',
    'kanban', 'waterfall', 'lean', 'six sigma', 'oop', 'object-oriented programming',
    'functional programming', 'rest api design', 'api security', 'oauth', 'saml', 'sso',
    'software development life cycle', 'sdlc', 'code review', 'pair programming', 'version control',
    # Cybersecurity
    'cybersecurity', 'information security', 'network security', 'application security',
    'penetration testing', 'ethical hacking', 'vulnerability assessment', 'siem', 'ids/ips',
    'firewalls', 'cryptography', 'iam', 'identity and access management', 'gdpr', 'hipaa',
    'iso 27001', 'soc2', 'owasp', 'malware analysis', 'digital forensics',
    # Design & UX/UI
    'ui/ux', 'ui design', 'ux design', 'user interface design', 'user experience design',
    'figma', 'adobe xd', 'sketch', 'invision', 'zeplin', 'adobe photoshop', 'photoshop',
    'adobe illustrator', 'illustrator', 'user research', 'wireframing', 'prototyping',
    'usability testing', 'design thinking', 'interaction design', 'visual design', 'design systems',
    # Business & Management
    'project management', 'product management', 'business analysis', 'stakeholder management',
    'requirements gathering', 'business development', 'strategy', 'market research',
    'financial analysis', 'risk management', 'quality assurance', 'qa', 'erp', 'sap', 'salesforce',
    'microsoft dynamics', 'supply chain management', 'logistics',
    # Soft Skills
    'communication', 'verbal communication', 'written communication', 'teamwork', 'collaboration',
    'problem solving', 'analytical skills', 'critical thinking', 'leadership', 'team leadership',
    'time management', 'adaptability', 'flexibility', 'creativity', 'innovation',
    'attention to detail', 'mentoring', 'coaching', 'negotiation', 'conflict resolution',
    'decision making', 'public speaking', 'presentation skills', 'customer service', 'client relations',
    # General/Entry Level Terms
    'fresher', 'entry level', 'trainee', 'intern', 'junior', 'Software Development', 'Design', 'Product',
    'Data Analysis','DevOps','Sysadmin',
    # Domain Specific
    'healthcare IT', 'fintech', 'ecommerce', 'blockchain', 'iot', 'internet of things',
    'bioinformatics', 'gis', 'game development', 'unreal engine', 'unity',
    # Certifications
    'aws certified', 'azure certified', 'gcp certified', 'pmp', 'csm', 'comptia',
    'cissp', 'ccna', 'cisa'
]
PREDEFINED_SKILLS_LOWER = [skill.lower() for skill in PREDEFINED_SKILLS_KEYWORDS]

# --- Helper Functions ---
def make_request(url: str, headers: dict = None, params: dict = None, timeout: int = 15) -> dict | None:
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
    if not text: return []
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
    print(f"\nFetching jobs from Remotive for keywords: {keywords}...")
    if not keywords: print("Remotive: No keywords provided."); return []
    base_url = "https://remotive.com/api/remote-jobs"
    search_query = " ".join(keywords)
    params = {'search': search_query, 'limit': limit}
    data = make_request(base_url, params=params)
    jobs = []
    if data and 'jobs' in data:
        for job_entry in data['jobs']:
            description = job_entry.get('description', '')
            cleaned_description = re.sub(r'<[^>]+>', ' ', description)
            cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()
            jobs.append({
                'title': job_entry.get('title'), 'company': job_entry.get('company_name'),
                'location': job_entry.get('candidate_required_location', 'Remote'),
                'description_text': cleaned_description,
                'extracted_skills': extract_skills_from_text(cleaned_description, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('url'), 'publication_date': job_entry.get('publication_date'),
                'source_site': 'Remotive API'})
        print(f"Found {len(jobs)} jobs from Remotive.")
    else: print("No jobs found or error fetching from Remotive.")
    return jobs

def fetch_arbeitnow_jobs(keywords: list[str], limit: int = 5, location_query: str = None) -> list[dict]:
    print(f"\nFetching jobs from Arbeitnow for keywords: {keywords}, location: {location_query if location_query else 'Global'}...")
    if not keywords: print("Arbeitnow: No keywords provided."); return []
    base_url = "https://arbeitnow.com/api/job-board-api"
    search_query = " ".join(keywords)
    if location_query and location_query.lower() != "any":
        search_query += f" in {location_query}"
    params = {'q': search_query, 'page': 1}
    data = make_request(base_url, params=params)
    jobs = []
    if data and 'data' in data:
        for i, job_entry in enumerate(data['data']):
            if i >= limit: break
            description = job_entry.get('description', '')
            cleaned_description = re.sub(r'<[^>]+>', ' ', description)
            cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()
            jobs.append({
                'title': job_entry.get('title'), 'company': job_entry.get('company_name'),
                'location': job_entry.get('location', location_query if location_query else "Not specified"),
                'description_text': cleaned_description,
                'extracted_skills': extract_skills_from_text(cleaned_description, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('url'), 'publication_date': job_entry.get('created_at'),
                'source_site': 'Arbeitnow API'})
        print(f"Found {len(jobs)} jobs from Arbeitnow (up to limit {limit}).")
    else: print("No jobs found or error fetching from Arbeitnow.")
    return jobs

def fetch_usajobs(keywords: list[str], limit: int = 5, location_name: str = None) -> list[dict]:
    print(f"\nFetching jobs from USAJOBS for keywords: {keywords}, location: {location_name if location_name else 'US Nationwide'}...")
    if not USAJOBS_API_KEY or not USAJOBS_USER_AGENT: print("USAJOBS_API_KEY or USAJOBS_USER_AGENT not set. Skipping USAJOBS."); return []
    if not keywords: print("USAJOBS: No keywords provided."); return []
    base_url = "https://data.usajobs.gov/api/search"
    headers = {'Authorization-Key': USAJOBS_API_KEY, 'User-Agent': USAJOBS_USER_AGENT}
    params = {'Keyword': " ".join(keywords), 'ResultsPerPage': limit}
    if location_name: params['LocationName'] = location_name
    data = make_request(base_url, headers=headers, params=params)
    jobs = []
    if data and data.get('SearchResult', {}).get('SearchResultItems'):
        for item in data['SearchResult']['SearchResultItems']:
            job_entry = item.get('MatchedObjectDescriptor', {})
            desc_parts = [d for d in [job_entry.get('UserArea', {}).get('Details', {}).get('JobSummary'),
                                      job_entry.get('UserArea', {}).get('Details', {}).get('MajorDuties'),
                                      job_entry.get('UserArea', {}).get('Details', {}).get('Requirements')] if d]
            desc = " ".join(str(p) for p in desc_parts if p)
            cleaned_desc = re.sub(r'\s+', ' ', desc).strip()
            jobs.append({
                'title': job_entry.get('PositionTitle'), 'company': job_entry.get('OrganizationName'),
                'location': job_entry.get('PositionLocationDisplay'), 'description_text': cleaned_desc,
                'extracted_skills': extract_skills_from_text(cleaned_desc, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('PositionURI'), 'publication_date': job_entry.get('PublicationStartDate'),
                'source_site': 'USAJOBS API'})
        print(f"Found {len(jobs)} jobs from USAJOBS.")
    else: print("No jobs found or error fetching from USAJOBS.")
    return jobs

def fetch_adzuna_jobs(keywords: list[str], limit: int = 5, location_query: str = None, country_code: str = "gb") -> list[dict]:
    print(f"\nFetching jobs from Adzuna for keywords: {keywords}, country: {country_code}" + (f", location: {location_query}" if location_query else ", location: Global within country") + "...")
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY: print("ADZUNA_APP_ID or ADZUNA_APP_KEY not set. Skipping Adzuna."); return []
    if not keywords: print("Adzuna: No keywords provided."); return []
    base_url = f"http://api.adzuna.com/v1/api/jobs/{country_code.lower()}/search/1"
    params = {
        'app_id': ADZUNA_APP_ID, 'app_key': ADZUNA_APP_KEY,
        'results_per_page': limit, 'what': " ".join(keywords),
        'content-type': 'application/json'}
    if location_query and location_query.lower() != "any": params['where'] = location_query
    data = make_request(base_url, params=params)
    jobs = []
    if data and 'results' in data:
        for job_entry in data['results']:
            desc = job_entry.get('description', '')
            cleaned_desc = re.sub(r'\s+', ' ', desc).strip()
            jobs.append({
                'title': job_entry.get('title'), 'company': job_entry.get('company', {}).get('display_name'),
                'location': job_entry.get('location', {}).get('display_name'), 'description_text': cleaned_desc,
                'extracted_skills': extract_skills_from_text(cleaned_desc, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('redirect_url'), 'publication_date': job_entry.get('created'),
                'source_site': 'Adzuna API'})
        print(f"Found {len(jobs)} jobs from Adzuna.")
    else:
        print("No jobs found or error fetching from Adzuna.")
        if data: print(f"Adzuna response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    return jobs

def fetch_github_jobs_mirror(keywords: list[str], limit: int = 5, location_query: str = None) -> list[dict]:
    print(f"\nFetching jobs from GitHub Jobs Mirror for keywords: {keywords}...")
    if not keywords: print("GitHub Mirror: No keywords provided."); return []
    if "github" in [kw.lower() for kw in keywords] or any(dev_kw in " ".join(keywords).lower() for dev_kw in ["developer", "engineer", "software"]):
        print("Attempting to find GitHub-like jobs via Arbeitnow as a proxy...")
        proxy_keywords = list(set(keywords + ["developer", "engineer", "software"]))
        return fetch_arbeitnow_jobs(proxy_keywords, limit, location_query)
    print("Direct GitHub Jobs API is deprecated. GitHub mirror using specific keywords or broader dev job boards.")
    return []

def fetch_jsearch_jobs(keywords: list[str], limit: int = 5, location_query: str = None) -> list[dict]:
    print(f"\nFetching jobs from JSearch (RapidAPI) for keywords: {keywords}, location: {location_query if location_query else 'Global'}...")
    if not RAPIDAPI_JSEARCH_KEY: print("RAPIDAPI_JSEARCH_KEY not set. Skipping JSearch."); return []
    if not keywords: print("JSearch: No keywords provided."); return []
    base_url = f"https://{RAPIDAPI_HOST}/search"
    search_query_parts = [" ".join(keywords)]
    if location_query and location_query.lower() != "any": search_query_parts.append(f"in {location_query}")

    querystring = {"query": " ".join(search_query_parts), "page": "1", "num_pages": "1"}
    headers = {"X-RapidAPI-Key": RAPIDAPI_JSEARCH_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST}
    data = make_request(base_url, headers=headers, params=querystring)
    jobs = []
    if data and 'data' in data:
        for i, job_entry in enumerate(data['data']):
            if i >= limit: break
            desc = job_entry.get('job_description', '')
            cleaned_desc = re.sub(r'<[^>]+>', ' ', re.sub(r'\s+', ' ', desc)).strip()
            loc_parts = [job_entry.get('job_city'), job_entry.get('job_state'), job_entry.get('job_country')]
            job_loc = ", ".join(filter(None, loc_parts)) or "Not specified"
            jobs.append({
                'title': job_entry.get('job_title'), 'company': job_entry.get('employer_name'),
                'location': job_loc, 'description_text': cleaned_desc,
                'extracted_skills': extract_skills_from_text(cleaned_desc, PREDEFINED_SKILLS_LOWER),
                'url': job_entry.get('job_apply_link') or job_entry.get('job_google_link'),
                'publication_date': job_entry.get('job_posted_at_datetime_utc'),
                'source_site': 'JSearch API (RapidAPI)'})
        print(f"Found {len(jobs)} jobs from JSearch (up to limit {limit}).")
    else:
        print("No jobs found or error fetching from JSearch.")
        if data: print(f"JSearch response structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    return jobs

# --- Main Orchestrator ---
def scrape_jobs(
    keywords: list[str],
    location: str = None,
    max_jobs_per_source: int = 2,
    skills_json_path: str = "extracted_skills.json"
) -> list[dict]:
    final_keywords_to_use = []
    using_skills_from_json = False
    # The skills_file_was_used_and_deleted flag is no longer needed

    try:
        with open(skills_json_path, 'r', encoding='utf-8') as f_skills_json: # Ensure file is closed
            loaded_skills = json.load(f_skills_json)
        # File is closed here due to 'with' statement
        if isinstance(loaded_skills, list) and all(isinstance(s, str) for s in loaded_skills) and loaded_skills:
            print(f"INFO: Using skills from '{skills_json_path}' for job search: {loaded_skills[:10]}...")
            final_keywords_to_use = loaded_skills
            using_skills_from_json = True
            # The file is used but will not be deleted.
            print(f"INFO: Successfully used skills from '{skills_json_path}'. The file has not been deleted.")
        else:
            print(f"INFO: '{skills_json_path}' is empty or not a valid list of skills. Using provided/recommended keywords: {keywords}")
            final_keywords_to_use = keywords
    except FileNotFoundError:
        print(f"INFO: '{skills_json_path}' not found. Using provided/recommended keywords for job search: {keywords}")
        final_keywords_to_use = keywords
    except json.JSONDecodeError:
        print(f"ERROR: Decoding JSON from '{skills_json_path}'. Using provided/recommended keywords: {keywords}")
        final_keywords_to_use = keywords
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading skills from '{skills_json_path}': {e}. Using provided/recommended keywords: {keywords}")
        final_keywords_to_use = keywords

    if not final_keywords_to_use:
        print("WARNING: No keywords available for job scraping. Using default generic keywords.")
        final_keywords_to_use = ["developer", "software", "IT"]

    print(f"--- Starting Job Scraping with effective keywords: {final_keywords_to_use[:10]}..., Location: {location if location else 'Global'} ---")
    all_jobs = []

    current_remotive_keywords = final_keywords_to_use[:]
    if "remote" not in [kw.lower() for kw in current_remotive_keywords]:
        current_remotive_keywords.append("remote")
    all_jobs.extend(fetch_remotive_jobs(keywords=current_remotive_keywords, limit=max_jobs_per_source))

    all_jobs.extend(fetch_arbeitnow_jobs(keywords=final_keywords_to_use, limit=max_jobs_per_source, location_query=location))

    adzuna_search_location_query = location
    adzuna_country_code = "gb"
    if location:
        country_code_map = {"india": "in", "usa": "us", "united states": "us", "uk": "gb", "united kingdom": "gb", "germany": "de", "singapore": "sg"}
        adzuna_country_code = country_code_map.get(location.lower(), "gb")
    all_jobs.extend(fetch_adzuna_jobs(keywords=final_keywords_to_use, limit=max_jobs_per_source, location_query=adzuna_search_location_query, country_code=adzuna_country_code))

    all_jobs.extend(fetch_jsearch_jobs(keywords=final_keywords_to_use, limit=max_jobs_per_source, location_query=location))

    # --- MODIFIED USAJOBS KEYWORD LOGIC ---
    usajobs_search_location_name = None
    if location and (location.lower() in ["usa", "united states"] or "us" in location.lower().split()):
        usajobs_search_location_name = location

    keywords_for_usajobs = []
    if using_skills_from_json: # Personalized search
        keywords_for_usajobs = final_keywords_to_use
        print(f"USAJOBS: Using personalized keywords from JSON for US search: {keywords_for_usajobs[:5]}...")
    else: # Fallback/Recommended search
        keywords_for_usajobs = ["IT Specialist", "Computer Scientist"] # Specific, simpler list
        print(f"USAJOBS: Using specific recommended keywords for US search: {keywords_for_usajobs}")

    if USAJOBS_API_KEY and USAJOBS_USER_AGENT: # Check keys before printing "Attempting..."
        print(f"Attempting USAJOBS search (Targeting US: {usajobs_search_location_name if usajobs_search_location_name else 'Nationwide'})...")
    all_jobs.extend(fetch_usajobs(keywords=keywords_for_usajobs, limit=max_jobs_per_source, location_name=usajobs_search_location_name))
    # --- END OF MODIFIED USAJOBS KEYWORD LOGIC ---

    all_jobs.extend(fetch_github_jobs_mirror(keywords=final_keywords_to_use, limit=max_jobs_per_source, location_query=location))

    print(f"\n--- Total jobs fetched before deduplication: {len(all_jobs)} ---")

    seen_urls = set()
    unique_jobs = []
    for job in all_jobs:
        job_url = job.get('url')
        if job_url and job_url not in seen_urls:
            unique_jobs.append(job)
            seen_urls.add(job_url)
        elif not job_url:
            unique_key = (job.get('title','').lower(), job.get('company','').lower(), job.get('location','').lower())
            if unique_key not in seen_urls:
                 unique_jobs.append(job)
                 seen_urls.add(unique_key)

    print(f"--- Total unique jobs (by URL or content signature): {len(unique_jobs)} ---")
    return unique_jobs

# --- Example Usage ---
if __name__ == "__main__":
    print("Job Scraper Initializing (API Version)...")

    if not USAJOBS_API_KEY or not USAJOBS_USER_AGENT:
        print("WARNING: USAJOBS_API_KEY or USAJOBS_USER_AGENT is not set. USAJOBS scraping might be skipped or fail.")
    if not RAPIDAPI_JSEARCH_KEY:
        print("WARNING: RAPIDAPI_JSEARCH_KEY is not set. JSearch scraping will be skipped.")
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        print("WARNING: ADZUNA_APP_ID or ADZUNA_APP_KEY is not set. Adzuna scraping will be skipped.")

    example_search_keywords = ["Software Developer", "IT Support", "Data Analyst", "Fresher", "Developer", "Python", "Java", "Software", "IT"]
    skills_file_path = "extracted_skills.json"
    test_run_limit = 5

    # Create a dummy extracted_skills.json for testing if it doesn't exist
    if not os.path.exists(skills_file_path):
        print(f"Creating a dummy '{skills_file_path}' for testing Scenario 1.")
        dummy_skills = ["Python", "JavaScript", "Cloud Computing", "React", "Node.js"]
        with open(skills_file_path, 'w', encoding='utf-8') as f_dummy:
            json.dump(dummy_skills, f_dummy)
    else:
        print(f"'{skills_file_path}' already exists for testing Scenario 1.")


    print(f"\n--- SCENARIO 1: Attempting to use '{skills_file_path}' for PERSONALIZED GLOBAL job search ---")
    # This scenario now correctly attempts to use a pre-existing extracted_skills.json
    # and scrape_jobs will attempt to use it but NOT delete it.
    scraped_job_data_personalized = scrape_jobs(
        keywords=example_search_keywords, # These are fallback if skills_file_path is invalid/empty
        location=None,
        max_jobs_per_source=test_run_limit,
        skills_json_path=skills_file_path
    )
    if scraped_job_data_personalized:
        print(f"\n--- Personalized Global Scraped Job Data (using '{skills_file_path}' if found and valid, else fallback global) ---")
        print(f"Total unique jobs: {len(scraped_job_data_personalized)}")
        # for job in scraped_job_data_personalized[:2]: # Print first 2 jobs for brevity
        # print(json.dumps(job, indent=2))
    else:
        print(f"\nNo job data (personalized or fallback global) was successfully aggregated in Scenario 1.")

    # Verify that skills_file_path still exists after Scenario 1
    if os.path.exists(skills_file_path):
        print(f"VERIFICATION: '{skills_file_path}' still exists after Scenario 1 run, as expected.")
    else:
        print(f"VERIFICATION ERROR: '{skills_file_path}' was deleted or not found after Scenario 1 run.")


    print(f"\n--- SCENARIO 2: Testing explicit fallback to recommended keywords for US ---")
    # This scenario uses a non-existent file to ensure fallback to 'keywords' parameter.
    scraped_job_data_recommended = scrape_jobs(
        keywords=example_search_keywords,
        location="United States",
        max_jobs_per_source=test_run_limit,
        skills_json_path="non_existent_skills_file.json" # Ensure this file doesn't exist
    )
    if scraped_job_data_recommended:
        print(f"\n--- Recommended Scraped Job Data (fallback keywords for US) ---")
        print(f"Total unique jobs: {len(scraped_job_data_recommended)}")
        # for job in scraped_job_data_recommended[:2]:
        # print(json.dumps(job, indent=2))
    else:
        print("\nNo recommended job data was successfully aggregated in Scenario 2.")

    # Clean up dummy file if it was created by this test script
    if os.path.exists(skills_file_path) and 'dummy_skills' in locals():
         if input(f"Do you want to delete the dummy '{skills_file_path}' created for this test run? (y/n): ").lower() == 'y':
            os.remove(skills_file_path)
            print(f"Dummy '{skills_file_path}' deleted.")