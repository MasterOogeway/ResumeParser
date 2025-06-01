import importlib

packages = {
    "flask": "Flask",
    "flask_wtf": "Flask-WTF",
    "flask_login": "Flask-Login",
    "nltk": "NLTK",
    "spacy": "spaCy",
    "sklearn": "scikit-learn",
    "bs4": "BeautifulSoup4",
    "selenium": "Selenium",
    "pymongo": "PyMongo",
    "dnspython": "dnspython",
    "PyPDF2": "PyPDF2",
    "docx": "python-docx",
    "pdfminer": "pdfminer.six",
    "sqlalchemy": "SQLAlchemy",
    "dotenv": "python-dotenv",
    "flask_cors": "Flask-CORS"
}

print("\n🔍 Checking Installed Packages:\n")

for module_name, display_name in packages.items():
    try:
        importlib.import_module(module_name)
        print(f"✅ {display_name} is installed.")
    except ImportError:
        print(f"❌ {display_name} is NOT installed.")

print("\n💡 Use `pip install <package-name>` to install any missing ones.")
