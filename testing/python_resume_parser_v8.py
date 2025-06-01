import spacy
import re
import json
from collections import defaultdict
import dateutil.parser as date_parser
from spacy.matcher import Matcher, PhraseMatcher
from spacy.util import filter_spans
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from PyPDF2 import PdfReader
from typing import Dict, List, Optional, Tuple, Set, Any

# --- START DEBUG FLAGS (DEFINE THESE AT THE VERY TOP OF THE SCRIPT) ---
DEBUG_LINE_PROCESSING = True
DEBUG_FIND_SECTION = True
DEBUG_CONTACT_INFO = True
DEBUG_ENTITY_EXTRACTION = True
DEBUG_NLP_CALLS = True 
DEBUG_JOB_TITLE = True 
# --- END DEBUG FLAGS ---

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    stopwords = None # type: ignore

try:
    from fuzzywuzzy import fuzz, process # type: ignore
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    fuzz = None # type: ignore
    process = None # type: ignore

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts: List[str] = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text: text_parts.append(page_text)
        elif DEBUG_LINE_PROCESSING: print(f"DEBUG: No text extracted from page {page_num + 1}")
    if DEBUG_LINE_PROCESSING and not text_parts: print("DEBUG: No text extracted from any page of the PDF.")
    return "\n\n".join(text_parts)

class AdvancedResumeParser:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Downloading {model_name}..."); spacy.cli.download(model_name); self.nlp = spacy.load(model_name)
        self._setup_entity_ruler()
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._setup_patterns()
        self.skills_db: Dict[str, List[str]] = self._load_skills_database()
        self._setup_skill_matchers()
        self.degree_patterns: List[str] = ["bachelor of technology", "b.tech", "bachelor of engineering", "b.e.", "bachelor of science", "b.s.", "b.sc.", "bachelor of arts", "b.a.", "bachelor of commerce", "b.com.", "master of technology", "m.tech", "master of engineering", "m.e.", "master of science", "m.s.", "m.sc.", "master of arts", "m.a.", "master of commerce", "m.com.", "master of business administration", "m.b.a.", "ph.d.", "doctor of philosophy", "doctorate", "associate degree", "diploma", "post graduate diploma", "pgdm", "certificate", "intermediate", "higher secondary certificate", "hsc", "secondary school certificate", "ssc", "10th", "12th", "xth", "xiith", "class x", "class xii"]
        self.job_titles: List[str] = ["engineer", "developer", "programmer", "analyst", "consultant", "manager", "director", "lead", "specialist", "trainee", "intern", "fellow", "architect", "scientist", "researcher", "executive", "officer", "coordinator", "assistant", "associate", "senior", "junior", "principal", "software engineer", "data scientist", "product manager", "project manager", "business analyst", "qa engineer", "devops engineer", "full stack developer", "frontend developer", "backend developer", "technical lead", "solutions architect", "data analyst", "machine learning engineer", "research intern", "technical trainee", "associate software engineer", "research analyst", "member technical staff"]
        self.non_name_keywords: List[str] = [kw.lower() for kw in ['university', 'inc', 'corp', 'llc', 'ltd', 'school', 'college', 'institute', 'resume', 'cv', 'summary', 'experience', 'education', 'technologies', 'consulting', 'limited', 'solutions', 'coursera', 'udemy', 'infosys', 'springboard', 'nptel', 'profile', 'objective', 'contact', 'details', 'gmail.com', '@', 'http', 'www', 'curriculum vitae', 'biodata', 'linkedin', 'github', 'portfolio', 'address', 'phone', 'email', 'website', 'date of birth', 'nationality', 'technical', 'skills', 'projects', 'internship', 'certification', 'award', 'references', 'declaration', 'page', 'confidential', 'contact number', 'e-mail', 'pvt', 'private']]
        self.non_location_keywords: List[str] = list(set([skill.lower() for cat_skills in self._load_skills_database().values() for skill in cat_skills] + ['logistic regression', 'machine learning', 'data analysis', 'data science', 'remote', 'online', 'various locations', 'multiple cities', 'n/a', 'tbd', 'work from home', 'headquarters'] + self.job_titles))
        if NLTK_AVAILABLE and stopwords:
            try: self.stop_words: Set[str] = set(stopwords.words('english'))
            except LookupError: print("NLTK stopwords not found. Downloading..."); nltk.download('stopwords', quiet=True); self.stop_words = set(stopwords.words('english'))
        else: self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

    def _setup_entity_ruler(self):
        if "custom_entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", name="custom_entity_ruler", before="ner")
            patterns = [
                {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"}}]}, # Added word boundary
                {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"(?:\+91[ -]?)?(?:[6-9]\d{9}|[6-9]\d{2}[ -]?\d{3}[ -]?\d{4})\b"}}]},
                {"label": "LINKEDIN_URL", "pattern": [{"LOWER": {"REGEX": r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w%\.-]+/?$"}}]},
                {"label": "GITHUB_URL", "pattern": [{"LOWER": {"REGEX": r"(?:https?://)?(?:www\.)?github\.com/[\w%\.-]+/?$"}}]},
                {"label": "URL", "pattern": [{"TEXT": {"REGEX": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"}}]}
            ]
            ruler.add_patterns(patterns)
            if DEBUG_ENTITY_EXTRACTION: print("DEBUG: Custom Entity Ruler added to pipeline.")

    def _setup_patterns(self):
        date_patterns: List[List[Dict[str,Any]]] = [
            [{"TEXT": {"REGEX": r"^(19|20)\d{2}$"}}, {"LOWER": {"IN": ["-", "to", "–", "--", "until"]}}, {"TEXT": {"REGEX": r"^(19|20)\d{2}$|present|current|till date"}, "OP": "?"}],
            [{"LOWER": {"IN": ["jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june", "jul", "july", "aug", "august", "sep", "september", "oct", "october", "nov", "november", "dec", "december"]}}, {"TEXT": {"REGEX": r"[\s.,']*(?:19|20)\d{2}"}}],
            [{"TEXT": {"REGEX": r"^\d{1,2}[/\s.-]+\d{4}$"}}], [{"TEXT": {"REGEX": r"^(19|20)\d{2}$"}}]]
        for i, pattern in enumerate(date_patterns): self.matcher.add(f"DATE_PATTERN_{i}", [pattern])

    def _load_skills_database(self) -> Dict[str, List[str]]:
        return {'programming_languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash', 'powershell', 'c', 'objective-c'], 'web_technologies': ['html', 'css', 'react', 'angular', 'vue', 'vue.js', 'node.js', 'express', 'express.js', 'django', 'flask', 'spring', 'spring boot', 'asp.net', 'laravel', 'ruby on rails', 'jquery', 'bootstrap', 'tailwind css', 'sass', 'less', 'webpack', 'gulp', 'grunt', 'next.js', 'nuxt.js', 'gatsby', 'restful apis', 'soap apis', 'graphql', 'ajax', 'json', 'xml', 'jwt'], 'databases': ['sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'mongo', 'redis', 'oracle', 'sqlite', 'cassandra', 'dynamodb', 'elasticsearch', 'neo4j', 'couchdb', 'mariadb', 'firebase', 'ms sql server', 'nosql'], 'cloud_platforms': ['aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud platform', 'google cloud', 'heroku', 'digitalocean', 'linode', 'docker', 'kubernetes', 'k8s', 'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions', 'openshift', 'serverless', 'lambda', 'azure functions', 'google cloud functions', 'ibm cloud'], 'data_science': ['pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'torch', 'keras', 'matplotlib', 'seaborn', 'plotly', 'jupyter', 'jupyter notebook', 'tableau', 'power bi', 'apache spark', 'spark', 'hadoop', 'kafka', 'apache kafka', 'shap', 'nltk', 'spacy', 'opencv', 'excel', 'vba', 'statistics', 'statistical analysis', 'machine learning', 'ml', 'deep learning', 'dl', 'data mining', 'data analysis', 'data visualization', 'nlp', 'natural language processing', 'computer vision', 'big data'], 'mobile_development': ['ios', 'android', 'react native', 'flutter', 'xamarin', 'cordova', 'ionic', 'swiftui', 'jetpack compose', 'kotlin multiplatform'], 'tools_and_software': ['git', 'github', 'gitlab', 'bitbucket', 'svn', 'jira', 'confluence', 'slack', 'microsoft teams', 'trello', 'asana', 'notion', 'adobe photoshop', 'photoshop', 'adobe illustrator', 'illustrator', 'figma', 'sketch', 'invision', 'zeplin', 'intellij idea', 'pycharm', 'vs code', 'visual studio code', 'visual studio', 'android studio', 'xcode', 'mongodb compass', 'oracle sql developer', 'eclipse', 'postman', 'selenium', 'webdriver', 'junit', 'testng', 'maven', 'gradle', 'npm', 'yarn', 'linux', 'unix', 'bash shell', 'powershell script'], 'methodologies': ['agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'continuous integration', 'continuous deployment', 'tdd', 'test driven development', 'bdd', 'behavior driven development', 'microservices', 'rest api', 'graphql api', 'soap', 'oauth', 'saml', 'sso', 'waterfall', 'lean', 'six sigma', 'design patterns'], 'operating_systems': ['linux', 'windows', 'macos', 'mac os x', 'unix', 'ubuntu', 'centos', 'debian', 'red hat', 'fedora', 'ios operating system', 'android operating system'], 'soft_skills': ['leadership', 'team leadership', 'communication', 'verbal communication', 'written communication', 'teamwork', 'collaboration', 'problem solving', 'analytical skills', 'critical thinking', 'project management', 'time management', 'adaptability', 'flexibility', 'creativity', 'innovation', 'analytical thinking', 'mentoring', 'coaching', 'negotiation', 'conflict resolution', 'decision making', 'public speaking', 'presentation skills', 'client relations', 'stakeholder management']}

    def _setup_skill_matchers(self):
        for category, skills in self.skills_db.items():
            try:
                skill_patterns = [self.nlp.make_doc(skill) for skill in skills]
                self.phrase_matcher.add(f"SKILL_{category.upper()}", skill_patterns)
            except Exception as e:
                if DEBUG_LINE_PROCESSING: print(f"Error creating PhraseMatcher pattern for skill in {category}: {e}")

    def _preprocess_contact_text(self, lines: List[str], num_lines_to_check: int = 3) -> List[str]:
        processed_lines = lines[:]
        if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Preprocessing first {num_lines_to_check} lines for contact info spacing (V7)...")

        email_re_str = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        phone_re_str = r'\b(?:\+91[ -]?)?(?:[6-9]\d{9}|[6-9]\d{2}[ -]?\d{3}[ -]?\d{4})\b'
        linkedin_re_str = r'\b(?:https?://)?(?:www\.)?linkedin\.com/in/[\w%\.-]+/?\b' 
        github_re_str = r'\b(?:https?://)?(?:www\.)?github\.com/[\w%\.-]+/?\b'
        leetcode_re_str = r'\b(?:https?://)?(?:www\.)?leetcode\.com/u/[\w%\.-]+/?\b'
        url_re_str = r'\b(?:https?://|www\.)[\w\.-]+(?:\.[a-zA-Z]{2,63})+(?:/[\w%\.\-\=\&\?\!\*~(),$]*)*\b'

        patterns_to_extract_and_space = [
            linkedin_re_str, github_re_str, leetcode_re_str, 
            email_re_str, phone_re_str, url_re_str            
        ]

        for i in range(min(num_lines_to_check, len(processed_lines))):
            line = processed_lines[i]
            original_line_for_debug = line

            is_potential_contact_line = '@' in line or any(char.isdigit() for char in line) or \
                                        any(site in line.lower() for site in ['linkedin', 'github', 'leetcode', 'http', '.com', '.in'])
            all_major_headers_for_check = ['summary', 'experience', 'education', 'skills', 'projects', 'objective', 'profile', 'certification', 'award', 'language']
            is_header = any(hdr_kw.lower() == line.lower().strip(": ") for hdr_kw in all_major_headers_for_check) or \
                        any(line.lower().strip(": ").startswith(hdr_kw.lower()) for hdr_kw in all_major_headers_for_check if len(hdr_kw)>5)
            
            if not is_potential_contact_line or is_header:
                if DEBUG_CONTACT_INFO and i < num_lines_to_check and is_potential_contact_line : print(f"  Skipping line {i+1} for contact preprocessing: '{line[:70]}...' (is header or no indicators)")
                continue
            
            line = line.replace('|', ' ').replace('•', ' ')
            line = re.sub(r'\s+', ' ', line).strip() 

            found_matches = []
            for pattern_str in patterns_to_extract_and_space:
                for match in re.finditer(pattern_str, line, re.IGNORECASE):
                    found_matches.append((match.start(), match.end(), match.group(0)))
            
            found_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            
            unique_matches_final = []
            last_end_pos = -1
            for start, end, text_match in found_matches:
                if start >= last_end_pos:
                    unique_matches_final.append({'start': start, 'end': end, 'text': text_match})
                    last_end_pos = end
            
            if unique_matches_final:
                reconstructed_parts = []
                current_pos = 0
                for match_info in unique_matches_final:
                    if match_info['start'] > current_pos:
                        reconstructed_parts.append(line[current_pos:match_info['start']])
                    reconstructed_parts.append(match_info['text'])
                    current_pos = match_info['end']
                if current_pos < len(line):
                    reconstructed_parts.append(line[current_pos:])
                
                line = " ".join(p.strip() for p in reconstructed_parts if p.strip())
            
            line = re.sub(r'\s+', ' ', line).strip()

            if line != original_line_for_debug and DEBUG_CONTACT_INFO:
                print(f"  Contact Preprocessing (V7): Line {i+1} changed from '{original_line_for_debug}' to '{line}'")
            processed_lines[i] = line
        return processed_lines
        
    def _apply_nlp_preprocessing(self, text_block: str) -> str:
        if not text_block: return ""
        cleaned_text = text_block.replace("•", "\n- ").replace("|", " - ")
        cleaned_text = re.sub(r'\s*\n\s*', '\n', cleaned_text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        return cleaned_text.strip()

    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        if DEBUG_LINE_PROCESSING: print(f"--- RAW PDF TEXT (first 1000 chars) ---\n{resume_text[:1000]}\n--- END RAW PDF TEXT ---")
        initial_lines = resume_text.split('\n')
        lines_after_contact_preprocessing = self._preprocess_contact_text(initial_lines, num_lines_to_check=3)
        self.cleaned_resume_lines: List[str] = [] 
        for line_content in initial_lines: # Use original lines for section finding logic base
            normalized_for_section_finding = re.sub(r'\s+', ' ', line_content).strip()
            if normalized_for_section_finding: self.cleaned_resume_lines.append(normalized_for_section_finding)
        
        final_lines_for_full_doc = []
        for i, original_line_content in enumerate(initial_lines):
            if i < len(lines_after_contact_preprocessing): final_lines_for_full_doc.append(lines_after_contact_preprocessing[i])
            else: final_lines_for_full_doc.append(re.sub(r'\s+', ' ', original_line_content).strip())
        
        text_for_full_doc = "\n".join(final_lines_for_full_doc); text_for_full_doc = re.sub(r'\n{3,}', '\n\n', text_for_full_doc).strip()
        
        if DEBUG_LINE_PROCESSING:
            print(f"--- TEXT FOR FULL NLP DOC (first 1000 chars after contact preprocessing) ---\n{text_for_full_doc[:1000]}\n--- END TEXT ---")
            print(f"--- CLEANED LINES FOR SECTION FINDING (first 20 lines from initial_lines normalized) ---")
            for i, ln in enumerate(self.cleaned_resume_lines[:20]): print(f"  Line {i+1}: {ln}")
            if len(self.cleaned_resume_lines) > 20: print("  ...")
            print("--- END CLEANED LINES ---")
        
        doc: Optional[spacy.tokens.Doc] = None
        if text_for_full_doc:
            if DEBUG_NLP_CALLS: print(f"NLP_CALL_MAIN_DOC_START: Processing FULL DOC ({len(text_for_full_doc)} chars)...")
            try:
                doc = self.nlp(text_for_full_doc)
                if DEBUG_NLP_CALLS: print("NLP_CALL_MAIN_DOC_END: FULL DOC NLP processing complete.")
            except Exception as e_main_nlp: print(f"ERROR: Main NLP processing failed: {e_main_nlp}"); doc = self.nlp("") 
        else: doc = self.nlp("");
        
        if doc is None: # Should not happen if self.nlp("") is used as fallback
            doc = self.nlp("") # Ensure doc is always a spacy.tokens.Doc

        if DEBUG_LINE_PROCESSING and hasattr(doc, 'sents'):
            sents_list = list(doc.sents)
            print(f"--- TOTAL SENTENCES FOUND BY SPACY on FULL DOC: {len(sents_list)} ---")
            for i, sent in enumerate(sents_list[:10]): print(f"  Full Doc Sentence {i+1}: {sent.text[:100].strip()}...")
            if len(sents_list) > 10: print(f"  ... and {len(sents_list) - 10} more sentences.")
        
        if DEBUG_ENTITY_EXTRACTION:
            print("\n--- DETECTED ENTITIES IN FULL DOC ---")
            for ent_idx, ent in enumerate(doc.ents):
                if ent_idx < 40 : print(f"  Entity: '{ent.text}', Label: '{ent.label_}' ({ent.start_char}-{ent.end_char})")
            if len(doc.ents) > 40: print(f"  ... and {len(doc.ents) - 40} more entities.")
            print("--- END DETECTED ENTITIES ---\n")
        
        parsed_data: Dict[str, Any] = {
            'contact_info': self.extract_contact_info_advanced(doc),
            'summary': self.extract_summary_advanced(),
            'skills': self.extract_skills_advanced(doc),
            'experience': self.extract_experience_advanced(),
            'education': self.extract_education_advanced(),
            'certifications': self.extract_certifications_advanced(),
            'languages': self.extract_languages(),
            'projects': self.extract_projects(),
            'awards': self.extract_awards()}
        
        parsed_data['metadata'] = {
            'total_lines_for_sections': len(self.cleaned_resume_lines),
            'total_tokens_in_full_doc': len(doc),
            'parsing_confidence': self.calculate_parsing_confidence(parsed_data),
            'resume_score': self.calculate_parsing_confidence(parsed_data) # Added resume_score, same as confidence for now
        }
        return parsed_data

    def extract_contact_info_advanced(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        contact_info: Dict[str, Any] = {'name': None, 'email': None, 'phone': None, 'location': None, 'urls': [], 'social_profiles': {}}
        if DEBUG_CONTACT_INFO: print("DEBUG_CONTACT: Starting contact info extraction...")
        if hasattr(self, 'cleaned_resume_lines') and self.cleaned_resume_lines:
            first_line_text = self.cleaned_resume_lines[0]
            if DEBUG_CONTACT_INFO:
                print(f"DEBUG_CONTACT: First line candidate for name: '{first_line_text}'")
                print(f"DEBUG_CONTACT: Word count: {len(first_line_text.split())}, istitle: {first_line_text.istitle()}, isupper: {first_line_text.isupper()}")
            if 1 < len(first_line_text.split()) <= 4 and (first_line_text.istitle() or (first_line_text.isupper() and len(first_line_text.split())<=2) ):
                failing_keyword = None; is_likely_name = True
                for kw_meta in self.non_name_keywords:
                    if kw_meta.lower() in first_line_text.lower(): failing_keyword = kw_meta; is_likely_name = False; break
                if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: First line keyword check for name (is_likely_name): {is_likely_name}" + (f", Failing Keyword: '{failing_keyword}'" if failing_keyword else ""))
                if is_likely_name: contact_info['name'] = first_line_text;
        
        if not contact_info['name']:
            potential_names_ner: List[Dict[str, Any]] = []
            for ent in doc.ents:
                if ent.label_ == "PERSON" and ent.start_char < 300: # Only consider PERSON entities near the top
                    name_text = ent.text.strip()
                    # Further heuristics for PERSON entities to be considered names
                    if len(name_text) > 3 and 1 < len(name_text.split()) <= 4 and not any(kw.lower() in name_text.lower() for kw in self.non_name_keywords):
                        potential_names_ner.append({'text': name_text, 'start': ent.start_char})
            if potential_names_ner:
                potential_names_ner.sort(key=lambda x: x['start']); contact_info['name'] = potential_names_ner[0]['text']
                if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Name from NER PERSON entity: '{contact_info['name']}'")
        
        extracted_emails, extracted_phones, extracted_linkedin, extracted_github = set(), set(), set(), set()
        for ent in doc.ents:
            ent_text = ent.text.strip()
            try:
                if ent.label_ == "EMAIL" and not contact_info['email'] and ent_text not in extracted_emails:
                    if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Trying email entity: '{ent_text}'")
                    validated_email = validate_email(ent_text, check_deliverability=False); contact_info['email'] = validated_email.email; extracted_emails.add(ent_text)
                    if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Email found and set: {contact_info['email']}")
                elif ent.label_ == "PHONE" and not contact_info['phone'] and ent_text not in extracted_phones:
                    if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Trying phone entity: '{ent_text}'")
                    parsed_phone = None
                    try: parsed_phone = phonenumbers.parse(ent_text, "IN")
                    except phonenumbers.phonenumberutil.NumberParseException: 
                        try: parsed_phone = phonenumbers.parse(ent_text, None) # Try without region if IN fails
                        except phonenumbers.phonenumberutil.NumberParseException: pass
                    if parsed_phone and phonenumbers.is_valid_number(parsed_phone): contact_info['phone'] = phonenumbers.format_number(parsed_phone, phonenumbers.PhoneNumberFormat.E164); extracted_phones.add(ent_text)
                    elif not contact_info['phone'] and re.match(r'(?:\+91[ -]?)?(?:[6-9]\d{9}|[6-9]\d{2}[ -]?\d{3}[ -]?\d{4})\b', ent_text): contact_info['phone'] = ent_text; extracted_phones.add(ent_text) # Fallback regex if lib fails
                    if contact_info['phone'] and DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Phone found and set: {contact_info['phone']}")
                elif ent.label_ == "LINKEDIN_URL" and ent_text not in extracted_linkedin:
                    url = ent_text if ent_text.lower().startswith("http") else "https://" + ent_text.lower().replace("www.","")
                    if "linkedin.com/in/" in url:
                        contact_info['social_profiles']['linkedin'] = url
                        if url not in contact_info['urls']: contact_info['urls'].append(url); extracted_linkedin.add(ent_text)
                        if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: LinkedIn URL found: {url}")
                elif ent.label_ == "GITHUB_URL" and ent_text not in extracted_github:
                    url = ent_text if ent_text.lower().startswith("http") else "https://" + ent_text.lower().replace("www.","")
                    if "github.com/" in url:
                        contact_info['social_profiles']['github'] = url
                        if url not in contact_info['urls']: contact_info['urls'].append(url); extracted_github.add(ent_text)
                        if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: GitHub URL found: {url}")
                elif ent.label_ == "URL" and ent_text.startswith("http") and ent_text not in contact_info['urls'] and not any(known_url_part in ent_text for known_url_part in ["linkedin.com", "github.com"]):
                    contact_info['urls'].append(ent_text)
                    if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: General URL found: {ent_text}")
                elif ent.label_ in ["GPE", "LOC"] and not contact_info['location'] and ent.start_char < 500: # Location usually near top
                    loc_text = ent.text.strip()
                    if 2 < len(loc_text) < 35 and len(loc_text.split()) <= 4 and not any(kw_loc.lower() in loc_text.lower() for kw_loc in self.non_location_keywords):
                        contact_info['location'] = loc_text
                        if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Location from NER: {loc_text}")
            except EmailNotValidError:
                if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Invalid email format for '{ent_text}'")
            except Exception as e_contact_inner: 
                 if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Inner error processing entity '{ent_text}' for {ent.label_}: {e_contact_inner}")
        if DEBUG_CONTACT_INFO: print(f"DEBUG_CONTACT: Final Contact Info: {contact_info}")
        return contact_info

    def find_section(self, section_keywords: List[str]) -> Optional[spacy.tokens.Doc]:
        if not hasattr(self, 'cleaned_resume_lines') or not self.cleaned_resume_lines:
            if DEBUG_FIND_SECTION: print(f"DEBUG_FIND_SECTION: `cleaned_resume_lines` unavailable for: {section_keywords}")
            return None
        section_start_line_idx = -1; header_line_text_found = ""; fuzzy_threshold = 80
        if DEBUG_FIND_SECTION: print(f"\nDEBUG_FIND_SECTION: Searching for section (line-based): {section_keywords}")
        for i, line_text in enumerate(self.cleaned_resume_lines):
            line_text_lower = line_text.lower()
            if not line_text_lower or len(line_text.split()) > 7: continue # Skip empty or very long lines for headers
            for keyword in section_keywords:
                kw_lower = keyword.lower(); is_match = False
                if kw_lower == line_text_lower: is_match = True
                elif line_text.upper() == kw_lower.upper() and len(line_text.split()) == len(kw_lower.split()): is_match = True # Case-insensitive exact match
                elif line_text_lower.startswith(kw_lower) and (len(line_text.split()) - len(kw_lower.split()) <= 2): is_match = True # Starts with keyword, few extra words
                elif FUZZY_AVAILABLE and fuzz and fuzz.ratio(line_text_lower, kw_lower) > fuzzy_threshold and \
                     (line_text.istitle() or line_text.isupper() or len(line_text.split()) <=3 ): is_match = True # Fuzzy match for short, capitalized lines
                if is_match: section_start_line_idx = i; header_line_text_found = line_text; break
            if section_start_line_idx != -1: break
        
        if section_start_line_idx == -1:
            if DEBUG_FIND_SECTION: print(f"  Section header NOT FOUND for: {section_keywords}")
            return None
        
        if DEBUG_FIND_SECTION: print(f"  FOUND header '{header_line_text_found}' at line index {section_start_line_idx} for {section_keywords}")
        
        section_content_lines: List[str] = []
        # Check if content is on the same line as the header
        content_on_header_line = header_line_text_found
        for kw_strip in section_keywords: # Try to strip the keyword that matched
            if header_line_text_found.lower().startswith(kw_strip.lower()):
                # More robust stripping of the keyword prefix
                pattern = r'^\s*' + re.escape(kw_strip) + r'[:\s\-]*'; stripped_content = re.sub(pattern, '', header_line_text_found, flags=re.IGNORECASE).strip()
                if stripped_content and stripped_content.lower() != header_line_text_found.lower().replace(kw_strip.lower(),"").strip(":- "): # Ensure something was actually stripped
                    content_on_header_line = stripped_content; break 
        
        if content_on_header_line and content_on_header_line.lower() != header_line_text_found.lower(): # If content was indeed on the header line
             section_content_lines.append(content_on_header_line)
             if DEBUG_FIND_SECTION: print(f"    Content found on header line: '{content_on_header_line}'")

        # Collect subsequent lines until another known section header is found
        all_known_section_starters_lower: List[str] = list(set([kw.lower() for kw in ['summary', 'profile', 'objective', 'overview', 'experience', 'employment', 'internship', 'project', 'portfolio', 'education', 'academic', 'qualification', 'scholastic', 'skills', 'technical skills', 'technologies', 'certification', 'certificate', 'license', 'credential', 'award', 'honor', 'recognition', 'scholarship', 'language', 'publication', 'reference', 'contact', 'declaration', 'personal detail', 'activity', 'extracurricular', 'achievement']] + [jt.lower() for jt in self.job_titles if len(jt.split()) <=2 and len(jt)>3]))
        
        for j in range(section_start_line_idx + 1, len(self.cleaned_resume_lines)):
            current_line_text = self.cleaned_resume_lines[j]; current_line_lower = current_line_text.lower(); is_next_section_header = False
            # Heuristic for a new section header: short line, often title case or all caps
            if len(current_line_text.split()) < 5 and len(current_line_text) < 40: # Arbitrary limits, can be tuned
                for other_header_kw in all_known_section_starters_lower:
                    # Avoid stopping if the "other_header_kw" is just a variation of the current section we are looking for
                    is_variation_of_current = any(current_search_kw.lower() == other_header_kw for current_search_kw in section_keywords)
                    if is_variation_of_current: continue

                    if other_header_kw == current_line_lower or \
                       (FUZZY_AVAILABLE and fuzz and fuzz.ratio(current_line_lower, other_header_kw) > 85 and \
                        (current_line_text.istitle() or current_line_text.isupper() or len(current_line_text.split())<=2)):
                        if DEBUG_FIND_SECTION: print(f"    Stopping section '{section_keywords[0]}' (header: '{header_line_text_found}') due to new header line '{current_line_text}' matching '{other_header_kw}'")
                        is_next_section_header = True; break
            if is_next_section_header: break
            section_content_lines.append(current_line_text)
        
        if not section_content_lines:
            if DEBUG_FIND_SECTION: print(f"  No content lines collected for section {section_keywords} (after header line check).")
            return None
            
        full_section_text = "\n".join(section_content_lines).strip()
        if not full_section_text:
             if DEBUG_FIND_SECTION: print(f"  Collected section text is empty for {section_keywords}.")
             return None

        text_to_process_nlp = self._apply_nlp_preprocessing(full_section_text)
        if DEBUG_FIND_SECTION:
            print(f"  Successfully extracted text for section '{section_keywords[0]}', original length: {len(full_section_text)}, processed for NLP length: {len(text_to_process_nlp)}.")
            if len(text_to_process_nlp) != len(full_section_text) or "•" in full_section_text or "|" in full_section_text: # If cleaning changed the text
                 print(f"--- DEBUG: Text about to be NLP'd for section '{section_keywords[0]}' (after specific cleaning) ---")
                 print(f"'''{text_to_process_nlp[:500].strip()}...'''"); print(f"--- END DEBUG TEXT ---")
        
        section_doc: Optional[spacy.tokens.Doc] = None
        try:
            if text_to_process_nlp: 
                if DEBUG_NLP_CALLS: print(f"NLP_CALL_FIND_SECTION_START: Processing section '{section_keywords[0]}' ({len(text_to_process_nlp)} chars)...")
                section_doc = self.nlp(text_to_process_nlp)
                if DEBUG_NLP_CALLS: print(f"NLP_CALL_FIND_SECTION_END: COMPLETED section '{section_keywords[0]}'.")
            else: 
                if DEBUG_FIND_SECTION: print(f"  Skipping NLP for section '{section_keywords[0]}' as preprocessed text is empty."); return None
        except Exception as e_nlp_section:
            if DEBUG_FIND_SECTION: print(f"ERROR: NLP processing failed for section '{section_keywords[0]}': {e_nlp_section}"); return None
        
        return section_doc

    def extract_summary_advanced(self) -> Optional[str]:
        summary_keywords = ['summary', 'professional summary', 'objective', 'profile', 'about me', 'career objective', 'overview', 'professional profile']
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract summary...")
        summary_section_doc = self.find_section(summary_keywords)
        if summary_section_doc and summary_section_doc.text.strip():
            summary_text = summary_section_doc.text.strip()
            # Attempt to remove the header keyword itself from the beginning of the summary
            for kw in summary_keywords:
                if summary_text.lower().startswith(kw.lower()):
                    pattern = r'^\s*' + re.escape(kw) + r'[:\s]*'; summary_text = re.sub(pattern, '', summary_text, flags=re.IGNORECASE).strip(); break
            return summary_text if len(summary_text) > 20 else None # Only return if substantial
        return None

    def extract_skills_advanced(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract skills...")
        skills_found_global: Dict[str, List[str]] = defaultdict(list)
        skill_contexts: List[Dict[str,str]] = [] # Store skill and its context

        # 1. Extract skills from the entire document using PhraseMatcher
        if doc and doc.text.strip():
            matches = self.phrase_matcher(doc)
            # Filter overlapping spans, keeping the longest ones
            filtered_matches = filter_spans([doc[s:e] for _, s, e in matches])
            for span in filtered_matches:
                skill_text_original = span.text; skill_text_lower = skill_text_original.lower()
                category = self.categorize_skill(skill_text_lower)
                if skill_text_lower not in [s.lower() for s in skills_found_global[category]]: # Avoid duplicates within category
                    skills_found_global[category].append(skill_text_original)
                
                # Capture context around the skill
                context_start = max(0, span.start_char - 60); context_end = min(len(doc.text), span.end_char + 60)
                skill_contexts.append({'skill': skill_text_original, 'category': category, 'context': doc.text[context_start:context_end].replace("\n", " ")})

        # 2. Extract skills specifically from a "Skills" section (if found)
        skills_section_doc = self.find_section(['skills', 'technical skills', 'competencies', 'technologies', 'technical proficiency', 'key skills'])
        if skills_section_doc:
            additional_skills = self.extract_skills_from_section_doc(skills_section_doc)
            for cat, skills_list in additional_skills.items():
                for skill in skills_list:
                    if skill.lower() not in [s.lower() for s in skills_found_global[cat]]: # Avoid duplicates
                        skills_found_global[cat].append(skill)
        
        # Sort skills within each category and create a flat list of all unique skills
        for category in skills_found_global: skills_found_global[category] = sorted(list(set(skills_found_global[category])), key=lambda x: x.lower())
        all_s = [skill for skills_list in skills_found_global.values() for skill in skills_list]
        
        return {'skills_by_category': dict(skills_found_global), 'all_skills': sorted(list(set(all_s)), key=lambda x: x.lower()), 'skill_contexts': skill_contexts[:20]} # Limit contexts for brevity

    def extract_skills_from_section_doc(self, section_doc: spacy.tokens.Doc) -> Dict[str, list]:
        skills: Dict[str, List[str]] = defaultdict(list)
        # Use PhraseMatcher on the dedicated skills section
        matches_in_section = self.phrase_matcher(section_doc)
        for _, start, end in matches_in_section:
            skill_text = section_doc[start:end].text; cat = self.categorize_skill(skill_text.lower())
            if skill_text.lower() not in [s.lower() for s in skills[cat]]: skills[cat].append(skill_text)

        # Fallback: If PhraseMatcher finds few skills, try line-based splitting and categorization
        # This helps with comma-separated lists or skills not perfectly in the phrase_matcher's DB
        if not matches_in_section or sum(len(v) for v in skills.values()) < 3 : # Arbitrary threshold
            for line in section_doc.text.split('\n'):
                line_clean = re.sub(r'^[•\-\*\s]+|[✓❖➢]\s*', '', line.strip()).strip() # Remove common bullets
                if not line_clean or len(line_clean.split()) > 7 : continue # Skip empty or very long lines
                
                potential_skills = [p.strip() for p in re.split(r'[,;/()]+', line_clean) if p.strip()]
                for skill_candidate in potential_skills:
                    if 1 < len(skill_candidate) < 35 and skill_candidate.lower() not in self.stop_words : # Basic validation
                        cat = self.categorize_skill(skill_candidate.lower())
                        # Add if categorized and not already present (case-insensitive)
                        if cat != 'other' and skill_candidate.lower() not in [s.lower() for s in skills[cat]]:
                            skills[cat].append(skill_candidate)
        return skills

    def categorize_skill(self, skill_lower: str) -> str:
        for category, known_skills in self.skills_db.items():
            if skill_lower in [ks.lower() for ks in known_skills]: return category
        
        if not FUZZY_AVAILABLE or not process: return 'other' # Fallback if fuzzywuzzy not available
        
        best_category = 'other'; highest_score = 78 # Threshold for fuzzy match
        for category, known_skills in self.skills_db.items():
            if not known_skills: continue
            match_result = process.extractOne(skill_lower, known_skills, scorer=fuzz.ratio) # type: ignore
            if match_result and match_result[1] > highest_score:
                highest_score = match_result[1]; best_category = category
        
        # Additional heuristic: if skill contains a category name part
        if best_category == 'other':
             for cat_key in self.skills_db.keys():
                 cat_display_parts = cat_key.replace('_', ' ').lower().split()
                 if any(part in skill_lower for part in cat_display_parts if len(part)>3): # e.g. "programming" in "programming_languages"
                     return cat_key
        return best_category

    def extract_experience_advanced(self) -> List[Dict[str, Any]]:
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract experience...")
        experiences: List[Dict[str, Any]] = []
        experience_section_doc = self.find_section(['experience', 'work experience', 'professional experience', 'employment history', 'internship experience', 'internships', 'work history', 'career summary', 'career progression'])
        
        if not experience_section_doc or not experience_section_doc.text.strip(): return experiences
        
        job_entries_docs = self.split_experience_entries(experience_section_doc)
        
        for i, entry_doc in enumerate(job_entries_docs):
            if not entry_doc or not entry_doc.text.strip(): continue
            if DEBUG_FIND_SECTION: print(f"DEBUG_EXPERIENCE_ENTRY: Parsing job entry {i+1} text: '''{entry_doc.text[:100].strip()}...'''")
            job_info = self.parse_job_entry_advanced(entry_doc)
            if job_info and (job_info.get('position') or job_info.get('company')): # Must have at least position or company
                experiences.append(job_info)
        
        if experiences: # Sort by end date (descending), with "Present" being the most recent
            try:
                experiences.sort(key=lambda x: (date_parser.parse(x['end_date'], fuzzy=True, default=date_parser.parse("1900-01-01")) if x.get('end_date') and x['end_date'].lower() != "present" else date_parser.parse("2999-12-31")), reverse=True)
            except (date_parser.ParserError, TypeError): # Handle potential parsing errors or None values
                if DEBUG_FIND_SECTION: print("DEBUG_EXPERIENCE: Could not sort experience entries by date due to parsing error or None dates.")
                pass # Keep unsorted if dates are problematic
        return experiences

    def split_experience_entries(self, experience_section_doc: spacy.tokens.Doc) -> List[spacy.tokens.Doc]:
        entries_docs: List[spacy.tokens.Doc] = []; current_entry_lines: List[str] = []
        lines = experience_section_doc.text.split('\n')
        # Regex for common date line patterns (e.g., "Month YYYY - Month YYYY", "YYYY - Present")
        date_line_pattern = r'\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2}\s*[-–to]+\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2}|Present|Current|Till Date)\b'
        year_pattern = r'\b(19|20)\d{2}\b'

        for i, line_text_orig in enumerate(lines):
            line_text = line_text_orig.strip()
            if not line_text: # Blank line often separates entries
                if current_entry_lines: # If we have content, process it as an entry
                    entry_text_block = self._apply_nlp_preprocessing("\n".join(current_entry_lines).strip())
                    if entry_text_block: 
                        if DEBUG_NLP_CALLS: print(f"NLP_CALL_SPLIT_EXP_START: Processing experience entry block ({len(entry_text_block)} chars): '''{entry_text_block[:100]}...'''")
                        try:
                            entries_docs.append(self.nlp(entry_text_block))
                            if DEBUG_NLP_CALLS: print(f"NLP_CALL_SPLIT_EXP_END: Completed experience entry block.")
                        except Exception as e: print(f"ERROR during NLP in split_experience_entries: {e}")
                    current_entry_lines = []
                continue # Move to next line

            starts_new_entry = False
            if current_entry_lines: # Only consider splitting if there's an active entry being built
                line_is_short_cap = len(line_text.split()) <= 5 and (line_text.istitle() or line_text.isupper())
                is_bullet = line_text.startswith(('-', '•', '*', '+', '➢'))

                if not is_bullet and line_is_short_cap: # Potential new job title or company line
                    # Check if it looks like a job title
                    if any(jt.lower() in line_text.lower() for jt in self.job_titles):
                        starts_new_entry = True
                    # Check if it looks like a company name (and previous line was a bullet or long)
                    elif re.search(r'\b(?:Inc\.?|Ltd\.?|LLC|Corp\.?|Solutions|Technologies|Consulting|Group|Limited|University|College|School|Institute|Pvt)\b', line_text, re.IGNORECASE) or \
                         (len(line_text.split()) <=4 and line_text.istitle() and not any(jt.lower() in line_text.lower() for jt in self.job_titles)): # Likely a company name
                         if i > 0 and (lines[i-1].strip().startswith(('-', '•', '*', '+', '➢')) or len(lines[i-1].strip().split()) > 6 or not lines[i-1].strip()): # If previous line was descriptive or blank
                             starts_new_entry = True
                
                # Check for date lines if not already starting a new entry
                if not starts_new_entry and not is_bullet and len(line_text.split()) < 7: # Date lines are usually short
                    if re.search(date_line_pattern, line_text, re.IGNORECASE) or \
                       re.fullmatch(year_pattern + r'\s*[-–to]+\s*' + year_pattern, line_text) or \
                       re.fullmatch(year_pattern + r'\s*[-–to]+\s*(?:Present|Current)', line_text, re.IGNORECASE):
                        # If this date line is significantly different from the first line of current_entry_lines (if it also had a date)
                        # This heuristic is tricky; for now, assume a date line after some text starts a new entry.
                        if len(current_entry_lines) > 1 : # If there's already some content for the current entry
                            starts_new_entry = True
            
            if starts_new_entry and current_entry_lines:
                entry_text_block = self._apply_nlp_preprocessing("\n".join(current_entry_lines).strip())
                if entry_text_block: 
                    if DEBUG_NLP_CALLS: print(f"NLP_CALL_SPLIT_EXP_START: Processing experience entry block (split) ({len(entry_text_block)} chars): '''{entry_text_block[:100]}...'''")
                    try:
                        entries_docs.append(self.nlp(entry_text_block))
                        if DEBUG_NLP_CALLS: print(f"NLP_CALL_SPLIT_EXP_END: Completed experience entry block (split).")
                    except Exception as e: print(f"ERROR during NLP in split_experience_entries (split): {e}")
                current_entry_lines = [line_text_orig] # Start new entry with current line
            else:
                current_entry_lines.append(line_text_orig)
        
        # Process any remaining lines as the last entry
        if current_entry_lines:
            entry_text_block = self._apply_nlp_preprocessing("\n".join(current_entry_lines).strip())
            if entry_text_block: 
                if DEBUG_NLP_CALLS: print(f"NLP_CALL_SPLIT_EXP_START: Processing final experience entry block ({len(entry_text_block)} chars): '''{entry_text_block[:100]}...'''")
                try:
                    entries_docs.append(self.nlp(entry_text_block))
                    if DEBUG_NLP_CALLS: print(f"NLP_CALL_SPLIT_EXP_END: Completed final experience entry block.")
                except Exception as e: print(f"ERROR during NLP in split_experience_entries (final): {e}")
        return entries_docs

    def parse_job_entry_advanced(self, entry_doc: spacy.tokens.Doc) -> Optional[Dict[str, Any]]:
        job_info: Dict[str, Any] = {'position': None, 'company': None, 'location': None, 'start_date': None, 'end_date': None, 'duration_text': None, 'description': [], 'achievements': [], 'technologies_used': []}
        if not entry_doc or not entry_doc.text.strip(): return None
        
        job_info['position'] = self.extract_job_title(entry_doc)
        
        potential_companies: List[str] = []
        first_line_of_entry = entry_doc.text.split('\n')[0].strip()

        for ent in entry_doc.ents:
            if ent.label_ == "ORG":
                # Avoid matching parts of the job title as company, unless it's a clear company name
                if not (job_info['position'] and ent.text.strip().lower() in job_info['position'].lower()) and \
                   ent.text.lower() not in ['department', 'team', 'group', 'ltd', 'inc', 'llc', 'pvt ltd', 'pvt. ltd.', 'limited', 'consulting', 'university', 'college', 'school', 'institute']: # Common non-company ORG labels
                    potential_companies.append(ent.text.strip())
        
        if not potential_companies: 
            if job_info['position'] and job_info['position'] in first_line_of_entry:
                company_candidate_text = re.sub(re.escape(job_info['position']), "", first_line_of_entry, count=1, flags=re.IGNORECASE).strip(" \t,-|@at")
                company_match = re.match(r"((?:[A-Z][\w.&'-]+(?:\s+|$)){1,4})", company_candidate_text) # Up to 4 capitalized words
                if company_match and len(company_match.group(1).strip().split()) <=4 :
                    potential_companies.append(company_match.group(1).strip())
            elif (first_line_of_entry.istitle() or first_line_of_entry.isupper()) and not any(jt.lower() in first_line_of_entry.lower() for jt in self.job_titles):
                 if 0 < len(first_line_of_entry.split()) <= 4 and first_line_of_entry.lower() not in self.non_name_keywords:
                     potential_companies.append(first_line_of_entry)
        
        if potential_companies: 
            job_info['company'] = potential_companies[0] # Simplistic choice, could be refined

        # Location extraction, trying to avoid picking up parts of company/position
        if job_info['company'] and job_info['company'] in first_line_of_entry and not job_info['location']:
            line_for_loc = first_line_of_entry
            if job_info['position']: line_for_loc = re.sub(re.escape(job_info['position']), '', line_for_loc, flags=re.IGNORECASE, count=1)
            if job_info['company']: line_for_loc = re.sub(re.escape(job_info['company']), '', line_for_loc, flags=re.IGNORECASE, count=1)
            loc_candidate = line_for_loc.strip(' ,|-@at').strip()
            if loc_candidate and 0 < len(loc_candidate.split()) <= 3 and not any(kw.lower() in loc_candidate.lower() for kw in self.non_location_keywords): job_info['location'] = loc_candidate
        
        if not job_info['location']: # Fallback to NER for location if not found on first line
            for ent in entry_doc.ents:
                if ent.label_ in ["GPE", "LOC"] and not job_info['location']:
                    loc_text = ent.text.strip()
                    if 2 < len(loc_text) < 25 and len(loc_text.split()) <= 3 and not any(kw.lower() in loc_text.lower() for kw in self.non_location_keywords):
                        job_info['location'] = loc_text; break
        
        # Date extraction
        date_range_match = re.search(r'((?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2})\s*[-–to]+\s*((?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2}|Present|Current|Till Date)\b', entry_doc.text, re.IGNORECASE)
        if date_range_match:
            job_info['duration_text'] = date_range_match.group(0); job_info['start_date'] = self.parse_date_entity(date_range_match.group(1)); job_info['end_date'] = self.parse_date_entity(date_range_match.group(2))
        else: # Fallback if specific range pattern not found
            date_texts = [entry_doc[s:e].text for _,s,e in self.matcher(entry_doc) if "DATE_PATTERN" in self.nlp.vocab.strings[_]] or [ent.text for ent in entry_doc.ents if ent.label_ == "DATE"]
            if date_texts:
                parsed_dates = sorted(list(set(d for d in [self.parse_date_entity(dt) for dt in date_texts] if d)))
                if len(parsed_dates) >= 2: job_info['start_date'], job_info['end_date'], job_info['duration_text'] = parsed_dates[0], parsed_dates[-1], f"{parsed_dates[0]} - {parsed_dates[-1]}"
                elif len(parsed_dates) == 1:
                    job_info['start_date'] = parsed_dates[0]; job_info['duration_text'] = parsed_dates[0]
                    if re.search(r'\b(Present|Current|Till Date)\b', entry_doc.text, re.IGNORECASE): job_info['end_date'] = 'Present'; job_info['duration_text'] += " - Present"
        
        # Description and Achievements
        header_info_texts = {str(v).lower() for v in [job_info['position'], job_info['company'], job_info['location'], job_info['duration_text']] if v and len(str(v)) > 2}
        if job_info['duration_text']: 
            for part in re.split(r'\s*[-–to]+\s*', job_info['duration_text'].lower()):
                if part.strip() and len(part.strip()) > 2 : header_info_texts.add(part.strip())
        
        for line_text_orig in entry_doc.text.split('\n'):
            sent_text = self._apply_nlp_preprocessing(line_text_orig).strip() 
            if not sent_text or len(sent_text) < 10: continue

            is_header_line_content = False
            if len(sent_text.split()) < 10: 
                normalized_sent_text_lower = sent_text.lower()
                for header_kw_part in header_info_texts:
                    if header_kw_part in normalized_sent_text_lower:
                        if len(normalized_sent_text_lower) <= len(header_kw_part) + 10:
                            is_header_line_content = True; break
            if is_header_line_content: continue
            
            if sent_text.startswith(('-', '•', '*', '+', '➢')) or any(ind.lower() in sent_text.lower() for ind in ['achieved', 'improved', 'developed', 'led', 'managed', 'responsible for', 'key achievement', 'contributed to', 'implemented', 'designed', 'created', 'launched', 'executed', 'optimized', 'streamlined']):
                job_info['achievements'].append(sent_text.lstrip('-•*+➢ ').strip())
            elif len(sent_text.split()) > 3 : job_info['description'].append(sent_text)
        
        # Technologies used
        entry_techs = set()
        text_for_skills_in_job = (job_info['position'] or "") + "\n" + "\n".join(job_info['description']) + "\n" + "\n".join(job_info['achievements'])
        if text_for_skills_in_job.strip():
            cleaned_text_for_skills = self._apply_nlp_preprocessing(text_for_skills_in_job)
            if DEBUG_NLP_CALLS: print(f"NLP_CALL_JOB_SKILLS_START: Processing skills text for job '{job_info['position'] if job_info['position'] else 'Unknown'}' ({len(cleaned_text_for_skills)} chars). Snippet: '''{cleaned_text_for_skills[:100].replace(chr(10), ' ')}...'''")
            try:
                job_skills_doc = self.nlp(cleaned_text_for_skills)
                if DEBUG_NLP_CALLS: print(f"NLP_CALL_JOB_SKILLS_END: Completed skills text for job '{job_info['position'] if job_info['position'] else 'Unknown'}'.")
                for _,s,e in self.phrase_matcher(job_skills_doc):
                    tech_text = job_skills_doc[s:e].text
                    # Avoid adding parts of position/company as tech unless it's a clear tech skill
                    is_in_header = (job_info['position'] and tech_text.lower() in job_info['position'].lower()) or \
                                   (job_info['company'] and tech_text.lower() in job_info['company'].lower())
                    if not is_in_header or self.categorize_skill(tech_text.lower()) != 'other': # If it's a known skill category, allow it even if in header
                        entry_techs.add(tech_text.lower())
            except Exception as e:
                if DEBUG_NLP_CALLS: print(f"ERROR in NLP_CALL_JOB_SKILLS: {e}")
        job_info['technologies_used'] = sorted(list(entry_techs))
        return job_info

    def extract_job_title(self, doc_entry: spacy.tokens.Doc) -> Optional[str]:
        if not doc_entry or not doc_entry.text.strip(): return None
        lines_to_check = doc_entry.text.split('\n')[:2]; text_to_search_in = "\n".join(lines_to_check)
        if DEBUG_JOB_TITLE: print(f"DEBUG_JOB_TITLE: Text for title extraction: '''{text_to_search_in}'''")
        
        sorted_job_titles = sorted(self.job_titles, key=len, reverse=True) # Prioritize longer, more specific titles
        for title_keyword_idx, title_keyword in enumerate(sorted_job_titles):
            prefix_pattern = r"(?:[A-Z][a-z]+(?:[- ](?:of|and|&|for))?\s+){0,2}" 
            suffix_pattern = r"(?:\s+(?:[A-Z][a-zA-Z0-9.&-]*|[IVXLCDM]+)){0,2}(?:\s*\([\w\s.&-]+\))?" 
            pattern_str = r"\b(" + prefix_pattern + re.escape(title_keyword) + suffix_pattern + r")\b"
            
            if DEBUG_JOB_TITLE and title_keyword_idx < 5 : print(f"  Trying title pattern for '{title_keyword}': {pattern_str[:100]}...")
            try:
                match = re.search(pattern_str, text_to_search_in, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip(" .,|-")
                    if DEBUG_JOB_TITLE: print(f"    Regex candidate: '{candidate}' from keyword '{title_keyword}'")
                    # Basic validation for the candidate
                    if 2 < len(candidate) < 70 and len(candidate.split()) <= 7 and \
                       candidate.lower() not in self.non_name_keywords and \
                       candidate.lower() not in ['experience', 'education', 'skills', 'project', 'summary', 'objective', 'company', 'location', 'date', 'duration', 'role', 'profile', 'responsibilities', 'achievements', 'description']:
                        # Check if it's an ORG entity (less likely to be a title unless it's a common job word)
                        is_org_entity = any(ent.text.strip().lower() == candidate.lower() and ent.label_ == "ORG" for ent in doc_entry.ents)
                        is_common_title_word = any(common_jt.lower() in candidate.lower() for common_jt in ["manager", "director", "lead", "engineer", "developer", "analyst", "consultant", "specialist", "intern", "trainee", "architect"])
                        if not is_org_entity or is_common_title_word: # If not an ORG or contains common job term
                            if DEBUG_JOB_TITLE: print(f"  RETURNING job title (regex match): '{candidate}'")
                            return candidate
            except re.error as e_re:
                if DEBUG_JOB_TITLE: print(f"    Regex error for {title_keyword}: {e_re}")
                continue
        
        # Fallback to noun chunks if regex fails
        first_line = lines_to_check[0].strip() if lines_to_check else ""
        if 0 < len(first_line.split()) < 10: 
            cleaned_first_line = self._apply_nlp_preprocessing(first_line)
            if DEBUG_NLP_CALLS: print(f"NLP_CALL_JOB_TITLE_FALLBACK_START: Processing first line for title: '''{cleaned_first_line}'''")
            try:
                first_line_doc = self.nlp(cleaned_first_line)
                if DEBUG_NLP_CALLS: print("NLP_CALL_JOB_TITLE_FALLBACK_END: Completed first line.")
                for chunk in first_line_doc.noun_chunks:
                    # Check if noun chunk is near the beginning and contains job-like terms
                    if chunk.start_char < 20 and len(chunk.text.split()) <= 5 and any(jt_part.lower() in chunk.text.lower() for jt_part in ['engineer', 'developer', 'analyst', 'manager', 'specialist', 'intern', 'trainee', 'lead']):
                        # Avoid if it's identified as an ORG by NER
                        if not any(ent.text.strip().lower() == chunk.text.lower() and ent.label_ == "ORG" for ent in doc_entry.ents):
                             if DEBUG_JOB_TITLE: print(f"  RETURNING job title (noun chunk fallback): '{chunk.text.strip()}'")
                             return chunk.text.strip()
            except Exception as e:
                 if DEBUG_NLP_CALLS: print(f"ERROR in NLP_CALL_JOB_TITLE_FALLBACK: {e}")
        
        # Simplest fallback: if the first line is short, title-cased, and not a known non-title keyword
        if 1 < len(first_line.split()) <= 4 and first_line.istitle():
            if first_line.lower() not in self.non_name_keywords and not any(jt_part.lower() in first_line.lower() for jt_part in self.job_titles[:5]): # Avoid very common job titles here if already missed
                if not any(ent.text.strip().lower() == first_line.lower() and ent.label_ == "ORG" for ent in doc_entry.ents):
                    if DEBUG_JOB_TITLE: print(f"  RETURNING job title (simple first line fallback): '{first_line}'")
                    return first_line
        if DEBUG_JOB_TITLE: print(f"  No job title found by extract_job_title for: '''{text_to_search_in[:100].strip()}...'''")
        return None
    
    def _looks_like_new_education_entry(self, line_text_lower: str) -> bool:
        if not line_text_lower.strip() or len(line_text_lower.split()) > 15: return False # Too long for a typical header/start
        # Check for year (common in education entries)
        if re.search(r'\b(19|20)\d{2}\b', line_text_lower): return True
        # Check for degree patterns at the beginning of the line
        first_few_words = " ".join(line_text_lower.split()[:4]) # Check first few words
        for pattern in self.degree_patterns:
            if pattern in first_few_words: return True
        # Check for "Class X/XII" patterns
        if re.match(r'(?:class\s+|[XV Ixvi]+th)\b', line_text_lower, re.IGNORECASE): return True
        # Check for title case + keywords if it's a short line
        if line_text_lower.istitle() and len(line_text_lower.split()) <= 5 and any(ed_kw in line_text_lower for ed_kw in ['degree', 'diploma', 'certificate', 'major', 'minor', 'program']): return True
        return False

    def extract_education_advanced(self) -> List[Dict[str, Any]]:
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract education...")
        education_entries: List[Dict[str, Any]] = []
        education_section_doc = self.find_section(['education', 'academic background', 'qualifications', 'academic qualifications', 'scholastics', 'education background'])
        
        if not education_section_doc or not education_section_doc.text.strip(): return education_entries
        
        entry_texts_blocks: List[str] = []; current_entry_lines: List[str] = []; lines = education_section_doc.text.split('\n')
        
        for i, line_content in enumerate(lines):
            line = line_content.strip()
            if not line: # Blank line usually separates entries
                if current_entry_lines: entry_texts_blocks.append("\n".join(current_entry_lines)); current_entry_lines = []
                continue

            prev_line_ended_item = False # Heuristic: did the previous line look like it completed an item?
            if i > 0 and current_entry_lines:
                last_of_current = current_entry_lines[-1].lower()
                # If last line ended with a year, or GPA/percentage, or "passed/completed"
                if re.search(r'\b(?:19|20)\d{2}\b$', last_of_current) or \
                   re.search(r'(?:cgpa|gpa|percentage|marks|score)\s*[:\s]*[\d.%]+', last_of_current) or \
                   re.search(r'\b(?:pass|passed|completed)\b', last_of_current):
                    prev_line_ended_item = True
            
            is_new_edu_marker_line = self._looks_like_new_education_entry(line.lower())

            if current_entry_lines and (is_new_edu_marker_line or prev_line_ended_item):
                # If the current line looks like a new entry OR the previous line seemed to complete an entry
                # AND the current line is not just a continuation bullet (unless we have substantial content already)
                if not line.startswith(('-', '•', '*', '+', '➢')) or len(current_entry_lines) > 2 : # Start new if not a bullet, or if current entry is already multi-line
                    entry_texts_blocks.append("\n".join(current_entry_lines)); current_entry_lines = [line_content]
                else: # Likely a continuation bullet of the current item
                    current_entry_lines.append(line_content)
            else:
                current_entry_lines.append(line_content)
        
        if current_entry_lines: entry_texts_blocks.append("\n".join(current_entry_lines)) # Add the last entry

        for entry_text in entry_texts_blocks:
            entry_text_stripped = self._apply_nlp_preprocessing(entry_text.strip())
            if entry_text_stripped and len(entry_text_stripped.split()) > 1: # Basic check for meaningful content
                if DEBUG_NLP_CALLS: print(f"NLP_CALL_EDU_ENTRY_START: Processing education entry block ({len(entry_text_stripped)} chars): '''{entry_text_stripped[:100]}...'''")
                try:
                    entry_doc_for_parsing = self.nlp(entry_text_stripped)
                    if DEBUG_NLP_CALLS: print(f"NLP_CALL_EDU_ENTRY_END: Completed education entry block.")
                    parsed_entry = self.parse_single_education_entry(entry_doc_for_parsing)
                    if parsed_entry.get('degree') or parsed_entry.get('institution'): # Must have degree or institution
                        education_entries.append(parsed_entry)
                except Exception as e:
                    if DEBUG_NLP_CALLS: print(f"ERROR in NLP_CALL_EDU_ENTRY: {e}")
        return education_entries

    def parse_single_education_entry(self, entry_doc: spacy.tokens.Doc) -> Dict[str, Any]:
        edu_info: Dict[str, Any] = {'degree': None, 'institution': None, 'location': None, 'graduation_date': None, 'field_of_study': None, 'gpa_score': None}
        if not entry_doc or not entry_doc.text.strip(): return edu_info

        sorted_degree_patterns = sorted(self.degree_patterns, key=len, reverse=True); found_degree_candidate_text = None
        
        # Search for degree patterns in the first few lines of the entry
        for line_in_entry in entry_doc.text.split('\n')[:3]: # Check first 3 lines
            line_lower = line_in_entry.lower().strip()
            for pattern in sorted_degree_patterns:
                if pattern in line_lower:
                    # Try to extract the full phrase containing the pattern
                    match_phrase = re.search(r'((?:[A-Z][a-z.,\s]+)*?' + re.escape(pattern) + r'[A-Za-z\s.,&()\-\/\d\' ]*)', line_in_entry, re.IGNORECASE)
                    if match_phrase: found_degree_candidate_text = match_phrase.group(1).strip().rstrip(',.-'); break
            if found_degree_candidate_text: break
        
        # If not found in first lines, search in all sentences
        if not found_degree_candidate_text:
            for sent in entry_doc.sents:
                sent_text_lower = sent.text.lower()
                for pattern in sorted_degree_patterns:
                    if pattern in sent_text_lower:
                        match_phrase = re.search(r'((?:[A-Z][a-z.,\s]+)*?' + re.escape(pattern) + r'[A-Za-z\s.,&()\-\/\d\' ]*)', sent.text, re.IGNORECASE)
                        if match_phrase: found_degree_candidate_text = match_phrase.group(1).strip().rstrip(',.-'); break
                if found_degree_candidate_text: break
        
        if found_degree_candidate_text:
            # Clean up common institutional phrases from degree text
            cleaned_degree_text = re.sub(r'\s*\(?(?:CBSE|ISC|State Board|University of .*|Institute of .*)\)?\s*$', '', found_degree_candidate_text, flags=re.IGNORECASE).strip()
            
            # Try to extract field of study / major
            major_match = re.search(r'(?:in|with major in|specializ(?:ation|ing) in)\s+((?:[A-Za-z\s&()]+(?:Engineering|Science|Arts|Commerce|Management|Studies|Applications|Technology|Computer Science))[\w\s]*)', cleaned_degree_text, re.IGNORECASE)
            if major_match:
                edu_info['field_of_study'] = major_match.group(1).strip().rstrip(',.-')
                degree_text_only = cleaned_degree_text[:major_match.start()].strip().rstrip(',-').strip()
                edu_info['degree'] = degree_text_only if degree_text_only else cleaned_degree_text # Fallback if stripping leaves nothing
            else:
                edu_info['degree'] = cleaned_degree_text
        
        # Extract institution, location, graduation date using NER
        for ent in entry_doc.ents:
            if ent.label_ == "ORG" and not edu_info['institution']:
                # Avoid generic ORG labels unless it's a multi-word name
                if ent.text.lower() not in ['university', 'college', 'institute', 'school', 'board', 'department'] or len(ent.text.split()) > 1:
                    # Avoid if the ORG is part of the already identified degree
                    if not (edu_info['degree'] and ent.text.lower() in edu_info['degree'].lower()):
                        edu_info['institution'] = ent.text.strip()
            elif ent.label_ in ["GPE", "LOC"] and not edu_info['location']:
                loc_text = ent.text.strip()
                if 2 < len(loc_text) < 25 and len(loc_text.split()) <= 3 and not any(kw.lower() in loc_text.lower() for kw in self.non_location_keywords):
                    edu_info['location'] = loc_text
            elif ent.label_ == "DATE" and not edu_info['graduation_date']:
                edu_info['graduation_date'] = self.parse_date_entity(ent.text)
        
        # Fallback for graduation date if NER missed it
        if not edu_info['graduation_date']:
            # Look for a year, possibly preceded by a month
            date_match = re.search(r'\b(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\s,.\']*)?((?:19|20)\d{2})\b', entry_doc.text, re.IGNORECASE)
            if date_match: edu_info['graduation_date'] = self.parse_date_entity(date_match.group(0).strip())

        # Extract GPA/Score
        gpa_match = re.search(r'(?:CGPA|GPA|Percentage|Score|Marks)\s*[:\-]?\s*([\d.]+\s*(?:%|\/\s*\d{1,2}(?:\.\d+)?))|([\d.]+\s*%)', entry_doc.text, re.IGNORECASE)
        if gpa_match:
            score = gpa_match.group(1) or gpa_match.group(2); edu_info['gpa_score'] = score.strip() if score else None
        
        return edu_info

    def extract_certifications_advanced(self) -> List[Dict[str, Any]]:
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract certifications...")
        certs: List[Dict[str, Any]] = []
        section_doc = self.find_section(['certifications', 'certificates', 'credentials', 'licenses', 'training', 'courses', 'online courses', 'professional development'])
        if not section_doc or not section_doc.text.strip(): return certs

        for line_text_orig in section_doc.text.split('\n'):
            line_text = line_text_orig.strip()
            if not line_text or len(line_text) < 5: continue # Skip very short lines

            cleaned_name = re.sub(r'^[•\-\*\s]+|[✓❖➢]\s*', '', line_text).strip() # Remove bullets
            cert_info: Dict[str, Any] = {'name': cleaned_name, 'issuer': None, 'date': None}
            
            # Use NLP on the line to find ORG (issuer) and DATE
            cleaned_line_for_nlp = self._apply_nlp_preprocessing(line_text)
            if DEBUG_NLP_CALLS: print(f"NLP_CALL_CERT_LINE_START: Processing cert line ({len(cleaned_line_for_nlp)} chars): '''{cleaned_line_for_nlp[:100]}...'''")
            try:
                line_doc_for_cert = self.nlp(cleaned_line_for_nlp)
                if DEBUG_NLP_CALLS: print(f"NLP_CALL_CERT_LINE_END: Completed cert line.")
                found_org_ner, found_date_ner = None, None
                for ent in line_doc_for_cert.ents:
                    if ent.label_ == "ORG" and not found_org_ner:
                        # Check against a list of common issuers or if it's a multi-word ORG
                        if ent.text.lower() in ['coursera', 'udemy', 'edx', 'linkedin learning', 'google', 'microsoft', 'aws', 'ibm', 'nptel', 'cisco', 'oracle'] or len(ent.text.split()) >= 1:
                            found_org_ner = ent.text.strip()
                    elif ent.label_ == "DATE" and not found_date_ner:
                        found_date_ner = self.parse_date_entity(ent.text)
                cert_info['issuer'] = found_org_ner; cert_info['date'] = found_date_ner
            except Exception as e:
                if DEBUG_NLP_CALLS: print(f"ERROR in NLP_CALL_CERT_LINE: {e}")

            # Fallback regex if NER fails for issuer/date
            if not cert_info['issuer']:
                issuer_match = re.search(r'(?:by|from|issued by|on|at)\s+([A-Z][A-Za-z0-9\s.&-]+)(?:\s*\(|,|$|\s+-\s+\d{4})', line_text) # Capture capitalized issuer names
                if issuer_match: cert_info['issuer'] = issuer_match.group(1).strip()
            
            if not cert_info['date']:
                found_date_str = None
                # Date in parentheses: (Month YYYY) or (YYYY)
                m1 = re.search(r'\(((?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2})\)', line_text, re.IGNORECASE)
                if m1: found_date_str = m1.group(1)
                else: # Date at the end of the line: , Month YYYY or , YYYY
                    m2 = re.search(r'(?:,|\s|^)\s*((?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2})$', line_text, re.IGNORECASE)
                    if m2: found_date_str = m2.group(1)
                if found_date_str: cert_info['date'] = self.parse_date_entity(found_date_str.strip())

            # Clean up name by removing issuer/date if they were part of it
            if cert_info['issuer'] and cert_info['issuer'] in cert_info['name']:
                cert_info['name'] = cert_info['name'].replace(cert_info['issuer'], '').strip(' ,-byfrom().').strip()
            if cert_info['date'] and cert_info['date'] in cert_info['name']:
                cert_info['name'] = cert_info['name'].replace(cert_info['date'], '').strip(' ,-().').strip()
            
            cert_info['name'] = re.sub(r'\s*\((?:Online|Virtual|Course)\)$', '', cert_info['name'], flags=re.IGNORECASE).strip() # Remove common suffixes

            if cert_info['name']: certs.append(cert_info)
        return certs

    def extract_languages(self) -> List[Dict[str, Any]]:
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract languages...")
        langs: List[Dict[str, Any]] = []
        section_doc = self.find_section(['languages', 'language skills', 'linguistic proficiency', 'language proficiency'])
        if not section_doc or not section_doc.text.strip(): return langs

        language_names_db = ['english', 'spanish', 'french', 'german', 'chinese', 'mandarin', 'japanese', 'korean', 'italian', 'portuguese', 'arabic', 'hindi', 'russian', 'punjabi', 'telugu', 'tamil', 'marathi', 'bengali', 'gujarati', 'urdu', 'kannada', 'malayalam']
        proficiency_levels_db = ['native', 'mother tongue', 'bilingual', 'fluent', 'proficient', 'professional working proficiency', 'full professional proficiency', 'professional', 'conversational', 'intermediate', 'upper intermediate', 'lower intermediate', 'basic', 'beginner', 'advanced', 'working proficiency', 'limited working proficiency', 'elementary proficiency', 'good', 'fair', 'excellent']

        for line in section_doc.text.split('\n'):
            line_text_orig, line_text_lower = line.strip(), line.strip().lower()
            if not line_text_lower or len(line_text_lower) < 3: continue

            line_clean = re.sub(r'^[•\-\*\s]+|[✓❖➢]\s*', '', line_text_orig).strip()
            found_lang_name = None

            for lang_name in sorted(language_names_db, key=len, reverse=True): # Match longer names first
                match_lang_re = re.search(r'\b(' + re.escape(lang_name) + r')\b', line_clean, re.IGNORECASE)
                if match_lang_re:
                    found_lang_name = match_lang_re.group(1); break
            
            if found_lang_name:
                lang_info: Dict[str, Optional[str]] = {'language': found_lang_name, 'proficiency': None}
                # Try to find proficiency in the rest of the line
                rest_of_line = re.sub(r'\b' + re.escape(found_lang_name) + r'\b', '', line_clean, flags=re.IGNORECASE).strip(' ():,-')
                for level in sorted(proficiency_levels_db, key=len, reverse=True):
                    if re.search(r'\b' + re.escape(level) + r'\b', rest_of_line, re.IGNORECASE):
                        lang_info['proficiency'] = level.title(); break
                langs.append(lang_info)
        return langs

    def extract_projects(self) -> List[Dict[str, Any]]:
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract projects...")
        projects: List[Dict[str, Any]] = []
        section_doc = self.find_section(['projects', 'personal projects', 'academic projects', 'portfolio', 'key projects', 'github projects'])
        if not section_doc or not section_doc.text.strip() : return projects
        
        current_project_lines: List[str] = []
        project_title_line: Optional[str] = None

        def save_current_project(title_line: Optional[str], desc_lines: List[str]):
            if not title_line and not desc_lines: return # Nothing to save
            
            project_name = title_line if title_line else "Unnamed Project"
            project_link: Optional[str] = None
            
            if title_line:
                url_match_title = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', title_line)
                if url_match_title:
                    project_link = url_match_title.group(0)
                    project_name = project_name.replace(project_link, "").strip(" |-•@(),")


            full_desc_text = "\n".join(desc_line.lstrip("•✔❖➢- ").strip() for desc_line in desc_lines if desc_line.strip())
            
            tech_used_in_project: Set[str] = set()
            # Combine title and description for skill extraction from project context
            text_for_project_skills = (project_name if project_name != "Unnamed Project" else "") + "\n" + full_desc_text
            
            if text_for_project_skills.strip():
                if DEBUG_NLP_CALLS: print(f"NLP_CALL_PROJECT_SKILLS_START: Processing project text ({len(text_for_project_skills)} chars) for '{project_name[:30]}...' : '''{text_for_project_skills[:100]}...'''")
                try:
                    project_skills_doc = self.nlp(self._apply_nlp_preprocessing(text_for_project_skills))
                    if DEBUG_NLP_CALLS: print(f"NLP_CALL_PROJECT_SKILLS_END: Completed project text for '{project_name[:30]}...'.")
                    for _,s,e in self.phrase_matcher(project_skills_doc):
                        tech_used_in_project.add(project_skills_doc[s:e].text.lower())
                except Exception as e:
                    if DEBUG_NLP_CALLS: print(f"ERROR in NLP_CALL_PROJECT_SKILLS: {e}")

            # Explicitly look for "Tech Stack:" or "Technologies:" lines if NLP missed some
            tech_stack_keywords = ["tech stack:", "technologies used:", "technologies:", "tools:"]
            for desc_line_idx, d_line in enumerate(desc_lines):
                d_line_lower = d_line.lower()
                for ts_kw in tech_stack_keywords:
                    if ts_kw in d_line_lower:
                        tech_text_from_stack = d_line_lower.split(ts_kw, 1)[-1].strip()
                        if DEBUG_NLP_CALLS: print(f"NLP_CALL_PROJECT_TECH_STACK_EXPLICIT_START: Processing explicit tech line ({len(tech_text_from_stack)} chars): '''{tech_text_from_stack[:100]}...'''")
                        try:
                            tech_doc = self.nlp(self._apply_nlp_preprocessing(tech_text_from_stack))
                            if DEBUG_NLP_CALLS: print(f"NLP_CALL_PROJECT_TECH_STACK_EXPLICIT_END: Completed explicit tech line.")
                            for _,s,e in self.phrase_matcher(tech_doc):
                                tech_used_in_project.add(tech_doc[s:e].text.lower())
                            # Also add comma/slash separated items from this line if PhraseMatcher missed them
                            raw_techs = [t.strip() for t in re.split(r'[,;/]+', tech_text_from_stack) if t.strip() and len(t.strip()) > 1 and len(t.strip()) < 25]
                            for rt in raw_techs:
                                if self.categorize_skill(rt) != 'other': tech_used_in_project.add(rt)

                        except Exception as e:
                             if DEBUG_NLP_CALLS: print(f"ERROR in NLP_CALL_PROJECT_TECH_STACK_EXPLICIT: {e}")
                        # Once "Tech Stack:" is processed for a line, assume subsequent lines are not part of it unless explicitly stated
                        # This part might need refinement if tech stack spans multiple lines without clear markers.
                        break # Stop checking other tech_stack_keywords for this line
                # else: continue # If no tech_stack_keyword found in this line, continue to next line
                # break # If a tech_stack_keyword was found, break from outer loop (desc_lines) - This logic seems flawed, should process all lines.
                        # Corrected: Should not break from desc_lines loop here.

            projects.append({
                'name': project_name,
                'description': full_desc_text if full_desc_text else None,
                'technologies': sorted(list(tech_used_in_project)),
                'link': project_link
            })

        lines = section_doc.text.split('\n')
        i=0
        while i < len(lines):
            line_text = lines[i].strip()
            i+=1 # Increment here to avoid issues with continue
            if not line_text: continue

            is_title_line = False
            # Heuristic for a project title:
            # - Not starting with a common bullet/list marker
            # - Relatively short (e.g., < 15 words, < 150 chars)
            # - Often title-cased, all caps, or contains keywords like 'project', 'study of', 'app'
            # - OR if the *next* line clearly starts with a bullet (suggesting current line is a title)
            if not line_text.startswith(('-', '•', '*', '+', '➢')) and len(line_text.split()) < 15 and len(line_text) < 150:
                 if line_text.istitle() or line_text.isupper() or \
                    any(ptk.lower() in line_text.lower() for ptk in ['project', 'study of', 'implementation of', 'development of', 'webapp', 'app', 'tool', 'analysis of']):
                    is_title_line = True
                 elif i < len(lines) and lines[i].strip().startswith(('-', '•', '*', '+', '➢')): 
                    is_title_line = True
            
            if is_title_line:
                if project_title_line or current_project_lines: # Save previous project if a new title is found or if there was content
                    save_current_project(project_title_line, current_project_lines)
                project_title_line = line_text
                current_project_lines = []
            elif project_title_line: # If we are under an active project title, append line to its description
                current_project_lines.append(line_text)
            # else: # Line is not a title and no active project title (e.g., preamble text before first project) - ignore for now.

        if project_title_line or current_project_lines: # Save the last collected project
            save_current_project(project_title_line, current_project_lines)
            
        return projects

    def extract_awards(self) -> List[Dict[str, Any]]:
        if DEBUG_FIND_SECTION: print("DEBUG_MAIN_EXTRACTION: Attempting to extract awards...")
        awards_list: List[Dict[str, Any]] = []
        section_doc = self.find_section(['awards', 'honors', 'achievements', 'recognition', 'scholarships', 'accomplishments', 'grants'])
        if not section_doc or not section_doc.text.strip(): return awards_list

        for line in section_doc.text.split('\n'):
            line_text = line.strip()
            if line_text and len(line_text) > 5: # Basic check for meaningful content
                cleaned_line = re.sub(r'^[•\-\*\s]+|[✓❖➢]\s*', '', line_text).strip() # Remove bullets
                award_info: Dict[str, Optional[str]] = {'name': cleaned_line, 'date': None, 'issuer': None}
                
                # Try to extract year first
                year_match = re.search(r'\b((?:19|20)\d{2})\b', cleaned_line)
                if year_match:
                    award_info['date'] = year_match.group(1)
                    # Remove the year from the line to avoid it being part of name/issuer
                    cleaned_line = re.sub(r'\s*\(?' + re.escape(year_match.group(1)) + r'\)?\s*$', '', cleaned_line).strip()

                # Try to extract issuer using keywords or NER
                issuer_match = re.search(r'(?:by|from|at|issued by)\s+([A-Z][A-Za-z0-9\s.&-]+)', cleaned_line) # Look for capitalized issuer
                if issuer_match:
                    award_info['issuer'] = issuer_match.group(1).strip()
                    cleaned_line = cleaned_line[:issuer_match.start()].strip(' ,-') # Remove issuer part from name
                elif FUZZY_AVAILABLE and process : # Fallback to NER if regex fails
                    cleaned_line_for_nlp = self._apply_nlp_preprocessing(cleaned_line) # Use the potentially year-stripped line
                    if DEBUG_NLP_CALLS: print(f"NLP_CALL_AWARD_LINE_START: Processing award line ({len(cleaned_line_for_nlp)} chars): '''{cleaned_line_for_nlp[:100]}...'''")
                    try:
                        line_doc_for_award = self.nlp(cleaned_line_for_nlp)
                        if DEBUG_NLP_CALLS: print(f"NLP_CALL_AWARD_LINE_END: Completed award line.")
                        for ent in line_doc_for_award.ents:
                            if ent.label_ == "ORG" and (len(ent.text.split()) > 1 or ent.text.lower() in ['university', 'college', 'school', 'institute', 'foundation', 'society']): # More specific ORG check
                                award_info['issuer'] = ent.text; break 
                    except Exception as e:
                         if DEBUG_NLP_CALLS: print(f"ERROR in NLP_CALL_AWARD_LINE: {e}")
                
                award_info['name'] = cleaned_line.strip(',').strip() # Whatever remains is the name
                if award_info['name']: awards_list.append(award_info)
        return awards_list
    
    def parse_date_entity(self, date_text: str) -> Optional[str]:
        date_text_lower = date_text.lower().strip(); date_text_original = date_text.strip()
        if any(word in date_text_lower for word in ['present', 'current', 'now', 'today', 'till date']): return 'Present'
        try:
            # Handle YYYY format directly
            if re.fullmatch(r'(19|20)\d{2}', date_text_lower): return date_text_lower
            
            # Handle "Month YYYY" or "Mon YYYY"
            month_year_match = re.match(r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\.?\s*\'?((?:19|20)\d{2})', date_text_lower)
            if month_year_match:
                try: return date_parser.parse(f"{month_year_match.group(1)} {month_year_match.group(2)}").strftime('%Y-%m')
                except: pass # Fall through if specific parse fails

            # Handle MM/YYYY or MM-YYYY or MM.YYYY
            mm_yyyy_match = re.match(r'(\d{1,2})[/\s.-]+((?:19|20)\d{2})', date_text_lower)
            if mm_yyyy_match:
                try: return date_parser.parse(f"{mm_yyyy_match.group(1)}-{mm_yyyy_match.group(2)}").strftime('%Y-%m')
                except: pass

            # General fuzzy parsing as a last resort for structured dates
            parsed_date = date_parser.parse(date_text_lower, fuzzy=True, default=date_parser.parse("1900-01-01"))
            if parsed_date.year != 1900 : # If fuzzy parsing yielded a plausible year
                # If only year was likely intended (e.g., "2023" parsed to "2023-01-01" by default)
                if parsed_date.month == 1 and parsed_date.day == 1 and not (re.search(r'\b(jan|1st|01)\b', date_text_lower)): # No explicit month/day
                    year_only_in_text = re.fullmatch(r'(19|20)\d{2}', date_text_original) # Check original text
                    if year_only_in_text: return year_only_in_text.group(0)
                return parsed_date.strftime('%Y-%m') # Return YYYY-MM
            else: # If fuzzy parsing defaulted to 1900, it likely didn't find a good date, try just year
                year_only_match = re.search(r'\b(19|20)\d{2}\b', date_text_lower)
                if year_only_match: return year_only_match.group(0)
        except (ValueError, TypeError, OverflowError): # Catch various parsing errors
            pass
        
        # Final fallback: if it's just a year, return that.
        year_only_match_final = re.search(r'\b(19|20)\d{2}\b', date_text_lower)
        if year_only_match_final: return year_only_match_final.group(0)
        
        return date_text_original # Return original if all else fails

    def parse_date_range_text(self, text: str) -> List[str]:
        parts = re.split(r'\s*[-–to]+\s*', text.strip(), 1) # Split on common range separators
        parsed_dates = [self.parse_date_entity(p.strip()) for p in parts if p.strip()] # Parse each part
        return [d for d in parsed_dates if d] # Filter out None results

    def calculate_parsing_confidence(self, parsed_data: Dict[str, Any]) -> float:
        # This is a heuristic score. Can be refined.
        score, max_score = 0.0, 0.0
        
        # Contact Info (Max 25)
        max_score += 25
        if parsed_data['contact_info'].get('name'): score += 10
        if parsed_data['contact_info'].get('email'): score += 8
        if parsed_data['contact_info'].get('phone'): score += 7
        
        # Skills (Max 15)
        max_score += 15
        if parsed_data['skills'].get('all_skills') and len(parsed_data['skills']['all_skills']) >= 5: score += 15
        elif parsed_data['skills'].get('all_skills'): score += len(parsed_data['skills']['all_skills']) * 2 # Proportional for fewer skills
        
        # Summary (Max 5)
        max_score += 5
        if parsed_data.get('summary'): score += 5
        
        # Experience (Max 30)
        max_score += 30
        num_exp = len(parsed_data.get('experience', []))
        if num_exp > 0: score += 10 # Base score for having experience section
        score += min(num_exp * 5, 20) # Score for number of entries, capped
            
        # Education (Max 20)
        max_score += 20
        num_edu = len(parsed_data.get('education', []))
        if num_edu > 0: score += 8 # Base score
        score += min(num_edu * 4, 12) # Score for number of entries, capped
        
        # Projects (Max 5) - Optional, so lower weight
        max_score += 5
        if parsed_data.get('projects'): score += min(len(parsed_data['projects']) * 2.5, 5)

        # Certifications (Max 5) - Optional
        max_score += 5
        if parsed_data.get('certifications'): score += min(len(parsed_data['certifications']) * 2.5, 5)
        
        return round(score / max_score, 2) if max_score > 0 else 0.0

    def format_output_advanced(self, resume_data: Dict[str, Any]) -> str:
        output = ["="*80, "ADVANCED RESUME PARSER RESULTS", "="*80]
        metadata = resume_data.get('metadata', {})
        
        # Use 'resume_score' if available, else 'parsing_confidence'
        score_to_display = metadata.get('resume_score', metadata.get('parsing_confidence', 0))
        output.append(f"Resume Score (Parsing Confidence): {score_to_display:.0%}")
        
        output.append(f"Document Stats: Lines for Sections: {metadata.get('total_lines_for_sections', 0)}, Tokens in Full Doc: {metadata.get('total_tokens_in_full_doc', 0)}\n")
        
        contact = resume_data.get('contact_info', {})
        if any(contact.values()): # Check if any contact info field has a value
            output.append("CONTACT INFORMATION:\n" + "-"*40)
            if contact.get('name'): output.append(f"Name: {contact['name']}")
            if contact.get('email'): output.append(f"Email: {contact['email']}")
            if contact.get('phone'): output.append(f"Phone: {contact['phone']}")
            if contact.get('location'): output.append(f"Location: {contact['location']}")
            if contact.get('social_profiles'):
                output.append("Social Profiles:")
                for platform, url_val in contact['social_profiles'].items(): output.append(f"  {platform.title()}: {url_val}")
            # Filter out social profile URLs from the general 'urls' list to avoid duplicates
            other_urls_to_show = [u for u in contact.get('urls', []) if u not in contact.get('social_profiles', {}).values()]
            if other_urls_to_show : output.append(f"Other URLs: {', '.join(other_urls_to_show)}")
            output.append("") # Newline after contact info
            
        if resume_data.get('summary'): output.append("PROFESSIONAL SUMMARY:\n" + "-"*40 + f"\n{resume_data['summary']}\n")
        
        skills_data = resume_data.get('skills', {})
        if skills_data.get('all_skills'):
            output.append("TECHNICAL SKILLS:\n" + "-"*40)
            if skills_data.get('skills_by_category'):
                for cat, s_list in skills_data['skills_by_category'].items():
                    if s_list: output.append(f"{cat.replace('_', ' ').title()}: {', '.join(s_list)}")
            else: # Fallback if no categories, just show all skills
                output.append(f"All Skills: {', '.join(skills_data['all_skills'])}")
            output.append(f"\nTotal Unique Skills: {len(skills_data.get('all_skills', []))}\n")

        # Loop for other sections
        for section_name_display, section_key in [
            ("WORK EXPERIENCE", "experience"), 
            ("EDUCATION", "education"), 
            ("PROJECTS", "projects"), 
            ("CERTIFICATIONS", "certifications"), 
            ("AWARDS & HONORS", "awards"), 
            ("LANGUAGES", "languages")
        ]:
            items = resume_data.get(section_key, [])
            if items:
                output.append(f"{section_name_display}:\n" + "-" * 40)
                for item_idx, item in enumerate(items):
                    output.append(f"  --- Entry {item_idx + 1} ---")
                    if isinstance(item, dict):
                        for k,v_item in item.items():
                            if v_item: # Only print if value exists
                                if isinstance(v_item, list) and v_item: # If it's a non-empty list
                                    output.append(f"    {k.replace('_',' ').title()}:")
                                    for sub_item in v_item[:3]: # Print first 3 items of a list (e.g., description lines)
                                        output.append(f"      • {str(sub_item)[:100]}{'...' if len(str(sub_item))>100 else ''}")
                                elif isinstance(v_item, str) and v_item.strip(): # If it's a non-empty string
                                    output.append(f"    {k.replace('_',' ').title()}: {v_item[:150]}{'...' if len(v_item)>150 else ''}")
                                elif not isinstance(v_item, list): # For other non-list, non-empty items (like dates)
                                     output.append(f"    {k.replace('_',' ').title()}: {v_item}")
                    output.append("") # Newline between entries within a section
                output.append("") # Newline after the whole section
        return '\n'.join(output)


def main():
    print("Initializing Advanced Resume Parser...")
    if NLTK_AVAILABLE:
        try: nltk.data.find('corpora/stopwords')
        except LookupError : nltk.download('stopwords', quiet=True)

    parser = AdvancedResumeParser()
    # pdf_path = "Resume1.pdf" 
    pdf_path = "Resume1.pdf" # Make sure this file exists in the same directory or provide full path
    print(f"Extracting text from '{pdf_path}'...")
    try:
        resume_text = extract_text_from_pdf(pdf_path)
    except FileNotFoundError: print(f"Error: File '{pdf_path}' not found."); return
    except Exception as e: print(f"Error during PDF text extraction: {e}"); import traceback; traceback.print_exc(); return

    if not resume_text or not resume_text.strip(): print(f"Error: No text extracted from '{pdf_path}'."); return

    print("Parsing resume...")
    try:
        parsed_data = parser.parse_resume(resume_text)
    except Exception as e: print(f"Error during resume parsing: {e}"); import traceback; traceback.print_exc(); return
    
    print("\nPARSED DATA (JSON):")
    print("="*60)
    try: print(json.dumps(parsed_data, indent=2, default=str)) # Use default=str for non-serializable objects like datetime
    except TypeError as e:
        print(f"Error serializing to JSON: {e}. Partial data:")
        for k, v in parsed_data.items():
            try: print(f"--- {k} ---"); print(json.dumps({k: v}, indent=2, default=str))
            except TypeError: print(f"Could not serialize key: {k}")
    
    print("\n\nFORMATTED OUTPUT (RECTIFIED RUN V8 - Contact Preproc V7 - JobTitle Regex V8):") # Update version markers as needed
    print("="*60)
    print(parser.format_output_advanced(parsed_data))
    
    # --- MODIFICATION: Clearer Resume Score Display ---
    print("\nRESUME SCORE & PARSING STATISTICS:")
    print("="*60)
    ci = parsed_data.get('contact_info', {}); 
    sk = parsed_data.get('skills', {}); 
    md = parsed_data.get('metadata', {})
    
    resume_score = md.get('resume_score', md.get('parsing_confidence', 0)) # Get resume_score, fallback to parsing_confidence
    print(f"Resume Score: {resume_score:.0%}") # Display resume_score prominently

    print(f"\nName: {ci.get('name', 'N/A')}")
    print(f"Email: {ci.get('email', 'N/A')}")
    print(f"Phone: {ci.get('phone', 'N/A')}")
    print(f"Location: {ci.get('location', 'N/A')}")
    print(f"Summary found: {'Yes' if parsed_data.get('summary') else 'No'}")
    all_extracted_skills = sk.get('all_skills', [])
    print(f"Skills identified: {len(all_extracted_skills)}")
    for sec_key in ["experience", "education", "projects", "certifications", "awards", "languages"]: print(f"{sec_key.title()} entries: {len(parsed_data.get(sec_key, []))}")
    # print(f"Parsing confidence: {md.get('parsing_confidence', 0):.0%}") # Already displayed as Resume Score
    print(f"Lines Processed for Sections: {md.get('total_lines_for_sections', 'N/A')}")
    print(f"Tokens in Full Doc: {md.get('total_tokens_in_full_doc', 'N/A')}")

    # --- MODIFICATION: Save extracted skills to a separate JSON file ---
    if all_extracted_skills:
        skills_output_path = "extracted_skills.json"
        try:
            with open(skills_output_path, 'w', encoding='utf-8') as f_skills:
                json.dump(all_extracted_skills, f_skills, indent=2)
            print(f"\nSuccessfully saved extracted skills to: {skills_output_path}")
        except IOError as e:
            print(f"\nError saving extracted skills to file: {e}")
    else:
        print("\nNo skills were extracted to save.")


if __name__ == "__main__":
    main()