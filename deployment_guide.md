# Σύστημα Κατανομής Μαθητών - Deployment Guide

## Δομή Φακέλων

```
student-assignment-system/
├── app.py                                    # Streamlit interface (ΒΕΛΤΙΩΜΕΝΟ)
├── complete_student_assignment_FIXED.py     # Πυρήνας συστήματος κατανομής  
├── requirements.txt                          # Dependencies (ΕΝΗΜΕΡΩΜΕΝΟ)
├── DEPLOYMENT_GUIDE.md                      # Αυτό το αρχείο
├── .gitignore                               # Git ignore rules
├── .streamlit/                              # Streamlit configuration
│   └── config.toml                          # App configuration
└── tests/                                   # Test files (προαιρετικό)
    └── test_basic.py
```

## Γρήγορη Εκκίνηση

### Βήμα 1: Κλωνοποίηση/Δημιουργία Φακέλου
```bash
mkdir student-assignment-system
cd student-assignment-system

# Κατέβασε τα αρχεία από την Claude συνομιλία
# Ή δημιούργησε τα χειροκίνητα
```

### Βήμα 2: Virtual Environment (Προτεινόμενο)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ή 
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Βήμα 3: Τοπική Εκτέλεση
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

### Στα Γρήγορα:
1. **GitHub**: Ανέβασε τα αρχεία στο GitHub
2. **Streamlit Cloud**: Πήγαινε στο [share.streamlit.io](https://share.streamlit.io)
3. **Deploy**: Επιλογή repository και `app.py`

### Λεπτομερώς:

#### Α. Προετοιμασία GitHub Repository
```bash
git init
git add .
git commit -m "Initial student assignment system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/student-assignment.git
git push -u origin main
```

#### Β. Streamlit Cloud Setup
1. Πήγαινε στο [share.streamlit.io](https://share.streamlit.io)
2. Sign in με GitHub account
3. Click "New app"
4. Επιλογή:
   - **Repository**: `YOUR_USERNAME/student-assignment`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy!"

#### Γ. Advanced Configuration (προαιρετικό)

Δημιούργησε `.streamlit/config.toml`:
```toml
[global]
dataFrameSerialization = "legacy"

[server]
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## Εναλλακτικές Πλατφόρμες Deployment

### 1. Heroku
```bash
# Procfile
web: sh setup.sh && streamlit run app.py

# setup.sh
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

### 2. Railway
1. Connect GitHub repository
2. Deploying αυτόματα
3. Environment variables setup

### 3. Google Cloud Run
```bash
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

## Configuration Files

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Streamlit
.streamlit/secrets.toml

# Data files
*.xlsx
*.csv
output/
temp_*

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### .streamlit/secrets.toml (για secrets)
```toml
# Αν χρειάζονται API keys ή databases
[database]
username = "your_username"
password = "your_password"

[api_keys]
some_service = "your_api_key"
```

## Βελτιστοποιήσεις Performance

### Memory Management
```python
# Προσθήκη στο app.py
import streamlit as st

# Cache για μεγάλα datasets
@st.cache_data(max_entries=3)
def load_large_dataset(file):
    return pd.read_excel(file)

# Session state cleanup
if 'large_data' in st.session_state:
    if st.button("Clear Cache"):
        del st.session_state['large_data']
        st.rerun()
```

### Resource Limits
```python
# Στο αρχείο του app.py
MAX_STUDENTS = 300  # Όριο για production
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if len(df) > MAX_STUDENTS:
    st.error(f"Πάρα πολλοί μαθητές. Μέγιστο: {MAX_STUDENTS}")
    st.stop()
```

## Monitoring & Logging

### Basic Logging
```python
# Στο app.py
import logging
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log user actions
logger.info(f"User processed {len(df)} students")
```

### Error Tracking (προαιρετικό)
```python
# Για production apps
try:
    import sentry_sdk
    sentry_sdk.init(
        dsn="YOUR_SENTRY_DSN",
        traces_sample_rate=1.0
    )
except ImportError:
    pass  # Sentry optional
```

## Testing

### Βασικό test αρχείο (`tests/test_basic.py`)
```python
import pytest
import pandas as pd
import sys
sys.path.append('..')

from complete_student_assignment_FIXED import create_sample_data, StudentAssignmentSystem

def test_sample_data_creation():
    df = create_sample_data(30)
    assert len(df) == 30
    assert "ΟΝΟΜΑ" in df.columns
    
def test_system_initialization():
    system = StudentAssignmentSystem()
    assert system is not None

def test_basic_assignment():
    df = create_sample_data(20)
    system = StudentAssignmentSystem()
    
    results = system.process_complete_assignment(df, num_classes=2, max_scenarios=1)
    assert results["status"] == "SUCCESS"
```

### Εκτέλεση tests
```bash
pip install pytest
pytest tests/
```

## Troubleshooting

### Συνήθη προβλήματα:

1. **Module not found error**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Memory errors στο Streamlit Cloud**
   - Μείωση max_scenarios σε 2
   - Όριο μαθητών < 150
   - Χρήση st.cache_data για optimization

3. **Greek encoding issues**
   ```python
   # Στο app.py
   import locale
   try:
       locale.setlocale(locale.LC_ALL, 'el_GR.UTF-8')
   except locale.Error:
       pass  # Fallback to default
   ```

4. **Excel export issues**
   ```bash
   pip install openpyxl xlsxwriter --upgrade
   ```

5. **Slow performance**
   - Ενεργοποίηση caching
   - Μείωση dataset size
   - Χρήση st.fragment για partial reruns

### Debug Mode
```python
# Στο app.py για development
DEBUG = True  # Set to False for production

if DEBUG:
    st.write("Debug info:", df.dtypes)
    st.write("Memory usage:", df.memory_usage(deep=True).sum() / 1024**2, "MB")
```

## Security Considerations

### File Upload Security
```python
# Στο app.py
ALLOWED_EXTENSIONS = {'.xlsx', '.xls', '.csv'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def secure_file_upload(uploaded_file):
    if uploaded_file is None:
        return None
        
    # Check file extension
    file_ext = pathlib.Path(uploaded_file.name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        st.error(f"Μη επιτρεπόμενος τύπος αρχείου: {file_ext}")
        return None
    
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("Πολύ μεγάλο αρχείο")
        return None
        
    return uploaded_file
```

### Data Sanitization
```python
def sanitize_dataframe(df):
    # Remove potential script injections
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace(r'<script.*?</script>', '', regex=True)
    
    return df
```

## Παραγωγή (Production) Checklist

- [ ] Dependencies pinned σε requirements.txt
- [ ] Error handling για όλες τις user actions
- [ ] File size και memory limits
- [ ] Greek encoding support
- [ ] Backup/download functionality
- [ ] User input validation
- [ ] Performance monitoring
- [ ] Security checks για file uploads
- [ ] Mobile-friendly interface
- [ ] Documentation και help text

## Support & Updates

Για updates και support:
1. Έλεγχος GitHub repository για νέες versions
2. Monitoring των Streamlit Cloud logs
3. User feedback collection μέσω της εφαρμογής

---

**Σημείωση**: Αυτή η έκδοση περιλαμβάνει σημαντικές βελτιώσεις στο error handling, UX, και performance optimization σε σχέση με την αρχική έκδοση.