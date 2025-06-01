import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
import traceback # <--- ADDED THIS IMPORT

# --- Import your custom modules ---
MODULES_LOADED_SUCCESSFULLY = True 

try:
    # Corrected import path and assuming your parser is v9 now
    from core.python_resume_parser_v9 import extract_text_from_pdf, AdvancedResumeParser
except ImportError as e:
    print(f"Error importing Resume Parser module (core.python_resume_parser_v9): {e}")
    MODULES_LOADED_SUCCESSFULLY = False
    def extract_text_from_pdf(path): return "Error: Parser module (core.python_resume_parser_v9) not loaded."
    class AdvancedResumeParser: 
        def parse_resume(self, text): 
            return {"metadata": {"resume_score": 0.0}, "skills": {"all_skills": []}}

try:
    # Corrected import path
    from core.job_scrapper_api_v2 import scrape_jobs, PREDEFINED_SKILLS_KEYWORDS
except ImportError as e:
    print(f"Error importing Job Scrapper module (core.job_scrapper_api_v2): {e}")
    MODULES_LOADED_SUCCESSFULLY = False
    def scrape_jobs(keywords, location, max_jobs_per_source, skills_json_path): return [] 
    PREDEFINED_SKILLS_KEYWORDS = [] 

DB_FUNCTIONS_AVAILABLE = True 
try:
    # Corrected import path
    from core.database_manager import (
        connect_db,
        save_personalized_search_session,
        get_personalized_search_session,
        delete_personalized_search_session,
        save_recommended_job,
        get_recommended_jobs_by_keywords
    )
except ImportError as e:
    print(f"Error importing Database Manager module (core.database_manager): {e}")
    print("Database operations will be skipped.")
    DB_FUNCTIONS_AVAILABLE = False
    MODULES_LOADED_SUCCESSFULLY = False 

    def connect_db(): print("DUMMY DB: connect_db called (core.database_manager not loaded)"); return None
    def save_personalized_search_session(session_id, resume_score, extracted_skills, personalized_job_results, raw_resume_text):
        print("DUMMY DB: save_personalized_search_session called (core.database_manager not loaded)")
    def get_personalized_search_session(session_id):
        print("DUMMY DB: get_personalized_search_session called (core.database_manager not loaded)"); return None
    def delete_personalized_search_session(session_id):
        print("DUMMY DB: delete_personalized_search_session called (core.database_manager not loaded)"); return False
    def save_recommended_job(job_data, source_keywords):
        print("DUMMY DB: save_recommended_job called (core.database_manager not loaded)")
    def get_recommended_jobs_by_keywords(keywords, limit):
        print("DUMMY DB: get_recommended_jobs_by_keywords called (core.database_manager not loaded)"); return []

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_default_strong_random_secret_key_123!')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def process_resume_file_placeholder(pdf_path: str) -> dict:
    print(f"FLASK_APP: Calling resume parser for: {pdf_path}")
    if "AdvancedResumeParser" not in globals() or "extract_text_from_pdf" not in globals():
         return {"raw_resume_text": "Error: Resume parser components not available.", "extracted_skills": [], "resume_score": 0.0}
    try:
        parser_instance = AdvancedResumeParser()
        raw_text = extract_text_from_pdf(pdf_path)
        if "Error: Parser module" in raw_text or "Error: Parser not loaded" in raw_text : 
             raise ValueError(raw_text) 
        if not raw_text or not raw_text.strip():
            if not raw_text and os.path.exists(pdf_path):
                 flash("Could not extract text from the PDF. It might be image-based or corrupted.", "error")
            raise ValueError("No text could be extracted from the resume. The file might be image-based or corrupted.")

        parsed_data_from_parser = parser_instance.parse_resume(raw_text)

        resume_score = parsed_data_from_parser.get('metadata', {}).get('resume_score', 0.0)
        all_extracted_skills = parsed_data_from_parser.get('skills', {}).get('all_skills', [])

        return {"raw_resume_text": raw_text, "extracted_skills": all_extracted_skills, "resume_score": resume_score}
    except Exception as e:
        # --- MODIFICATION: Added traceback printing here ---
        print(f"--- TRACEBACK WITHIN process_resume_file_placeholder ---") 
        traceback.print_exc() 
        print(f"--- END OF TRACEBACK ---")
        # --- END OF MODIFICATION ---
        print(f"Error in process_resume_file_placeholder: {e}") 
        return {"raw_resume_text": f"Error processing resume: {str(e)}", "extracted_skills": [], "resume_score": 0.0}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

db_connection_active = False
if DB_FUNCTIONS_AVAILABLE: 
    db_object_from_connect = connect_db() 
    if db_object_from_connect is not None: 
        db = db_object_from_connect 
        db_connection_active = True
        print("INFO: MongoDB connection established and active.")
    else: 
        print("CRITICAL: Failed to connect to MongoDB (or DB manager not loaded properly). Database operations will be impacted.")
else:
    print("INFO: Database functions are not available due to import errors. DB operations will be skipped.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process_resume', methods=['POST'])
def process_resume_api():
    if not MODULES_LOADED_SUCCESSFULLY: 
        return jsonify({"status": "error", "message": "Core application modules could not be loaded."}), 500

    if 'resume' not in request.files:
        return jsonify({"status": "error", "message": "No resume file part in the request."}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No resume file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(file_path)
            processing_session_id = str(uuid.uuid4())

            parsed_resume_output = process_resume_file_placeholder(file_path)
            raw_text = parsed_resume_output["raw_resume_text"]
            extracted_skills = parsed_resume_output["extracted_skills"]
            resume_score = parsed_resume_output["resume_score"]

            # Check for error messages returned by the placeholder/parser
            if raw_text.startswith("Error processing resume:") or \
               raw_text.startswith("Error: Resume parser components not available.") or \
               raw_text.startswith("Error: Parser module"):
                 return jsonify({"status": "error", "message": f"Resume Parsing Error: {raw_text}"}), 500
            if raw_text == "No text could be extracted from the resume. The file might be image-based or corrupted.": # Specific check
                return jsonify({"status": "error", "message": raw_text}), 400
            
            personalized_job_results = []
            recommended_job_results = []

            if extracted_skills:
                print(f"FLASK_APP: Scraping personalized jobs for skills: {extracted_skills[:5]}...")
                personalized_job_results = scrape_jobs(
                    keywords=extracted_skills,
                    location=None,
                    max_jobs_per_source=5, 
                    skills_json_path=None 
                )
            
            if not personalized_job_results: 
                print(f"FLASK_APP: No personalized jobs found or no skills. Scraping recommended jobs...")
                recommended_job_results = scrape_jobs(
                    keywords=PREDEFINED_SKILLS_KEYWORDS[:10], 
                    location=None,
                    max_jobs_per_source=3, 
                    skills_json_path=None 
                )

            if DB_FUNCTIONS_AVAILABLE and db_connection_active:
                print(f"FLASK_APP: Saving personalized search session to DB: {processing_session_id}")
                save_personalized_search_session(
                    session_id=processing_session_id,
                    resume_score=resume_score,
                    extracted_skills=extracted_skills,
                    personalized_job_results=personalized_job_results,
                    raw_resume_text=raw_text
                )
                if recommended_job_results: 
                    print(f"FLASK_APP: Saving {len(recommended_job_results)} recommended jobs to cache.")
                    for job in recommended_job_results:
                        save_recommended_job(job_data=job, source_keywords=PREDEFINED_SKILLS_KEYWORDS[:10])
            else:
                print("FLASK_APP: DB not available or connection inactive. Results for this AJAX request are not saved persistently.")
                flash("Database not available. Results are temporary and will not be saved.", "warning")
                session['temp_results_' + processing_session_id] = {
                    'search_id': processing_session_id,
                    'resume_data': {
                        'resume_score': resume_score,
                        'extracted_skills': extracted_skills,
                    },
                    'personalized_jobs': personalized_job_results,
                    'recommended_jobs': recommended_job_results
                }

            return jsonify({
                "status": "success",
                "data": {
                    "search_id": processing_session_id,
                    "resume_data": {
                        "resume_score": resume_score,
                        "extracted_skills": extracted_skills,
                    },
                    "personalized_jobs": personalized_job_results,
                    "recommended_jobs": recommended_job_results
                }
            })

        except ValueError as ve: # Catch specific errors like "No text extracted"
             return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as e: # General catch-all for other unexpected errors in this route
            print(f"Error during /api/process_resume: {e}")
            # traceback.print_exc() # This is already here from your previous code
            return jsonify({"status": "error", "message": f"An internal server error occurred in API route: {str(e)}"}), 500
        finally:
            if 'file_path' in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e_remove:
                    print(f"Error removing uploaded file {file_path}: {e_remove}")
    else:
        return jsonify({"status": "error", "message": "Invalid file type. Allowed: PDF, DOC, DOCX."}), 400

@app.route('/results_page/<search_id>')
def show_results_page(search_id):
    search_data = None
    source = "Database"
    if DB_FUNCTIONS_AVAILABLE and db_connection_active:
        search_data = get_personalized_search_session(search_id)
    
    if not search_data and ('temp_results_' + search_id) in session:
        flash("Displaying temporary results as database is unavailable or data not found in DB.", "warning")
        search_data = session['temp_results_' + search_id]
        source = "Temporary Session"

    if not search_data:
        flash('No results found for this search ID, or the session has expired.', 'error')
        recommended_jobs_fallback_display = []
        if DB_FUNCTIONS_AVAILABLE and db_connection_active:
            recommended_jobs_fallback_display = get_recommended_jobs_by_keywords(PREDEFINED_SKILLS_KEYWORDS[:5], limit=10)
        
        return render_template('index.html', 
                               resume_data_display=None, 
                               jobs_display=None, 
                               recommended_jobs_display=recommended_jobs_fallback_display, 
                               search_id_display=search_id,
                               results_source = "Fallback")

    resume_data_display = search_data.get('resume_data', {})
    personalized_jobs_display = search_data.get('personalized_jobs', [])
    recommended_jobs_display = search_data.get('recommended_jobs', []) 

    if not personalized_jobs_display and not recommended_jobs_display and DB_FUNCTIONS_AVAILABLE and db_connection_active:
        skills_for_rec = resume_data_display.get('extracted_skills', PREDEFINED_SKILLS_KEYWORDS[:5])
        recommended_jobs_display = get_recommended_jobs_by_keywords(skills_for_rec, limit=10)
        if recommended_jobs_display:
            source += " (plus fresh recommended)"
    
    return render_template('index.html', 
                           resume_data_display=resume_data_display, 
                           jobs_display=personalized_jobs_display, 
                           recommended_jobs_display=recommended_jobs_display,
                           search_id_display=search_id,
                           results_source = source)

@app.route('/clear_session_results/<search_id>')
def clear_session_data_route(search_id):
    session_cleared_from_db = False
    if DB_FUNCTIONS_AVAILABLE and db_connection_active:
        if delete_personalized_search_session(search_id):
            session_cleared_from_db = True
            flash(f'Search results for ID {search_id} have been cleared from database.', 'success')
        else:
            flash(f'Could not clear results for ID {search_id} from database or already cleared.', 'warning')
    else:
        flash('Database not available. Cannot clear persistent results.', 'error')
    
    if ('temp_results_' + search_id) in session:
        session.pop('temp_results_' + search_id, None)
        flash(f'Temporary search results for ID {search_id} have been cleared from this browser session.', 'success')
    elif not session_cleared_from_db: 
        flash(f'No active temporary results for ID {search_id} to clear from this browser session.', 'info')

    if 'processing_session_id' in session and session['processing_session_id'] == search_id:
         session.pop('processing_session_id', None)

    return redirect(url_for('index'))

if __name__ == '__main__':
    if not MODULES_LOADED_SUCCESSFULLY:
        print("WARNING: One or more core project modules could not be loaded. Application functionality may be limited.")
    if not db_connection_active and DB_FUNCTIONS_AVAILABLE: 
        print("WARNING: Database Manager was loaded, but connection to MongoDB failed.")
    elif not DB_FUNCTIONS_AVAILABLE: 
         print("WARNING: Database Manager module not loaded. Database operations will be skipped.")

    app.run(debug=True, port=5001)