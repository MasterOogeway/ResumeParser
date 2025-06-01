# AI Resume Parser & Job Matcher

## 🚀 Description

This project is a Flask web application designed to help users analyze their resumes and find relevant job opportunities. Users can upload their resume (PDF format), which is then parsed to extract key information such as skills, experience, and education. The system also calculates a resume score. Based on the extracted skills, it fetches job listings from various APIs. If no specific skills are found or they yield few results, predefined keywords are used to suggest general job opportunities. All processing results, including parsed resume data and job listings, are stored in a MongoDB Atlas database, with options for session-specific data and a cache for recommended jobs.

## ✨ Features

* **Resume Upload:** Supports PDF resume uploads through a web interface.
* **Advanced Resume Parsing:** Extracts contact information, summary, skills (categorized and overall), work experience, education, projects, certifications, and languages.
* **Resume Scoring:** Provides a heuristic-based score for the uploaded resume.
* **Personalized Job Scraping:** Fetches job listings from multiple APIs based on skills extracted from the user's resume.
* **Recommended Job Scraping:** Uses a predefined set of keywords to find general job opportunities if personalized results are insufficient.
* **MongoDB Integration:**
    * Stores personalized search sessions (parsed resume data + jobs found for that resume).
    * Maintains a cache for recommended job listings to optimize and avoid redundant API calls for general searches.
* **Web Interface:** User-friendly interface to upload resumes and view analysis results and job matches.
* **Dynamic Results Display:** Uses JavaScript to asynchronously process resumes and display results without full page reloads.

## 🛠️ Tech Stack & Key Libraries

* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript (Fetch API)
* **Database:** MongoDB Atlas
* **Resume Parsing:**
    * `PyPDF2` (for PDF text extraction)
    * `spaCy` (for NLP tasks, NER, sentence segmentation)
    * `nltk` (for stopwords, potentially other NLP utilities)
    * `fuzzywuzzy` & `python-Levenshtein` (for string matching)
    * `phonenumbers` (for phone number parsing)
    * `email-validator` (for email validation)
    * `python-dateutil` (for date parsing)
* **Job Scraping:**
    * `requests` (for making API calls)
* **Environment Management:**
    * `python-dotenv` (for managing environment variables)
* **MongoDB Interaction:**
    * `pymongo`
    * `dnspython` (for MongoDB+SRV URIs)

## 📁 Project Structure

/your_project_name/
|
|-- .venv/                     # Virtual environment
|
|-- app.py                     # Main Flask application
|
|-- core/                      # Core backend logic modules
|   |-- init.py
|   |-- python_resume_parser_v9.py
|   |-- job_scrapper_api_v2.py
|   |-- database_manager.py
|
|-- static/                    # CSS, client-side JS, images
|   |-- css/
|   |   |-- style.css
|   |-- js/
|   |   |-- main.js
|
|-- templates/                 # HTML templates
|   |-- index.html
|
|-- uploads/                   # Temporary storage for uploaded resumes (add to .gitignore)
|
|-- .env                       # Environment variables (API keys, DB URI, Flask secret key) - DO NOT COMMIT
|-- .gitignore                 # Specifies files for Git to ignore
|-- requirements.txt           # Python package dependencies
|-- requirements-other.txt     # Additional setup notes
|-- README.md                  # This file

## ⚙️ Setup and Installation

1.  **Prerequisites:**
    * Python 3.9+ (preferably 3.12 as noted in your `requirements-other.txt`)
    * `pip` (Python package installer)
    * Git
    * A MongoDB Atlas account and a cluster set up.

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd your_project_name
    ```

3.  **Create and Activate Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    Install all required Python packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download SpaCy NLP Model:**
    The resume parser uses `en_core_web_sm`. Download it using:
    ```bash
    python -m spacy download en_core_web_sm
    ```

6.  **NLTK Data (Stopwords):**
    The parser attempts to download `stopwords` if missing. If you encounter issues, you can pre-download them:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

7.  **Set up Environment Variables (`.env` file):**
    Create a `.env` file in the root directory of the project. Add the following variables with your actual credentials/keys:
    ```env
    # Flask
    FLASK_SECRET_KEY='your_very_strong_and_random_secret_key_here' # Change this!

    # MongoDB Atlas
    MONGO_ATLAS_URI="mongodb+srv://<username>:<password>@<your_cluster_address>/<your_database_name>?retryWrites=true&w=majority"

    # Job Scraper API Keys (from job_scrapper_api_v2.py)
    USAJOBS_API_KEY=your_usajobs_key
    USAJOBS_USER_AGENT=YourAppName/1.0 (your.email@example.com) # Or your actual agent
    RAPIDAPI_JSEARCH_KEY=your_jsearch_key
    ADZUNA_APP_ID=your_adzuna_app_id
    ADZUNA_APP_KEY=your_adzuna_app_key
    ```
    * Replace placeholders with your actual values.
    * Ensure your MongoDB Atlas cluster has network access configured to allow connections from your IP address (for local development) or your server's IP (for deployment).

## ධාවනය Running the Application

1.  Ensure your virtual environment is activated.
2.  Make sure all environment variables in `.env` are correctly set.
3.  Navigate to the project's root directory in your terminal.
4.  Run the Flask development server:
    ```bash
    python app.py
    ```
5.  Open your web browser and go to `http://127.0.0.1:5001` (or the port specified in `app.py`).

## 📋 Usage

1.  Navigate to the home page.
2.  Use the upload area to drag & drop your resume (PDF) or click "Choose File" to select it.
3.  The application will process the resume, displaying a loading indicator.
4.  Once processing is complete, the page will update to show:
    * Resume analysis results (e.g., overall score, key skills).
    * A list of personalized job opportunities based on your resume.
    * A list of general recommended jobs if personalized results are sparse or if specifically fetched.
5.  You can clear the results for a specific session using the "Clear these results" link associated with a search ID (if this feature is fully implemented and linked from the results page).

## 📦 Modules Overview

* **`app.py`**: The main Flask application. Handles web routes, orchestrates the resume parsing and job scraping process, interacts with the database manager, and renders HTML templates.
* **`core/python_resume_parser_v9.py`**: Contains the `AdvancedResumeParser` class responsible for extracting text from PDFs, parsing various sections (contact info, summary, skills, experience, education, etc.), and scoring the resume.
* **`core/job_scrapper_api_v2.py`**: Responsible for fetching job listings from multiple external job APIs using keywords (either extracted from a resume or predefined).
* **`core/database_manager.py`**: Manages all interactions with the MongoDB Atlas database, including connecting to the DB, saving personalized search sessions, and saving/retrieving recommended job listings.
* **`templates/index.html`**: The single HTML page that provides the user interface for uploading resumes and displaying results, using CSS for styling and JavaScript for dynamic interactions.

## 💡 Troubleshooting (General)

* **Import Errors:** Ensure all dependencies from `requirements.txt` are installed in your active virtual environment and that your project structure allows Python to find your custom modules (e.g., `core` directory being a package).
* **API Key Errors:** Double-check that all API keys in your `.env` file are correct and have the necessary permissions/quotas on the respective platforms.
* **MongoDB Connection Issues:**
    * Verify your `MONGO_ATLAS_URI` is correct.
    * Ensure your current IP address is whitelisted in MongoDB Atlas Network Access settings.
    * Check for any error messages from `pymongo` or `dnspython` in the Flask console.
* **Resume Parsing Errors (e.g., "unbalanced parenthesis"):** These often point to issues within the regular expressions in `python_resume_parser_v9.py` when encountering specific resume content. Use the full Python traceback from the Flask console (ensure `debug=True` in Flask) to pinpoint the exact line and regex causing the issue.
* **File Upload Issues:** Ensure the `uploads/` directory exists and your Flask application has permission to write to it.

## 🚀 Future Enhancements (Optional Ideas)

* Support for more resume file formats (e.g., .docx, .txt).
* More sophisticated resume scoring algorithm.
* User accounts to save and manage multiple resumes and job search histories.
* Advanced filtering and sorting for job results.
* Direct application capabilities or links to application portals.
* Skill gap analysis and suggestions for resume improvement.
* Deployment to a cloud platform (e.g., Heroku, AWS, Google Cloud).

---

This `README.md` should give a comprehensive overview of your project. Remember to update it as you make significant changes or add new features!