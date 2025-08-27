# Σύστημα Κατανομής Μαθητών - Streamlit Deployment

## Αρχεία που χρειάζεστε

```
your-project/
├── app.py                                    # Streamlit interface
├── complete_student_assignment_FIXED.py     # Πλήρες σύστημα κατανομής  
├── requirements.txt                          # Dependencies
└── README.md                                 # Οδηγίες (αυτό το αρχείο)
```

## Streamlit Cloud Deployment

### Βήμα 1: Ανέβασμα στο GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/student-assignment.git
git push -u origin main
```

### Βήμα 2: Deploy στο Streamlit Cloud
1. Πηγαίνετε στο [share.streamlit.io](https://share.streamlit.io)
2. Συνδεθείτε με GitHub
3. Κάντε κλικ "New app"
4. Επιλέξτε το repository σας
5. Main file path: `app.py`
6. Κάντε κλικ "Deploy"

## Τοπική εκτέλεση

```bash
# Εγκατάσταση dependencies
pip install -r requirements.txt

# Εκτέλεση Streamlit app
streamlit run app.py
```

## Χαρακτηριστικά

- **Πλήρης 7-βημη κατανομή**: Όλα τα βήματα του αρχικού συστήματος
- **Interactive UI**: Progress bars, file upload, download buttons
- **Validation**: Έλεγχος δεδομένων πριν την επεξεργασία
- **Export**: CSV και Excel λήψη αποτελεσμάτων
- **Δειγματικά δεδομένα**: Για δοκιμή χωρίς αρχείο

## Απαιτήσεις δεδομένων

### Υποχρεωτικές στήλες:
- `ΟΝΟΜΑ`: Ονοματεπώνυμο μαθητή
- `ΦΥΛΟ`: Α (Αγόρι) ή Κ (Κορίτσι) 
- `ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ`: True/False ή Ν/Ο
- `ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ`: True/False ή Ν/Ο

### Προαιρετικές στήλες:
- `ΦΙΛΟΙ`: Λίστα ονομάτων φίλων
- `ΖΩΗΡΟΣ`: True/False ή Ν/Ο
- `ΙΔΙΑΙΤΕΡΟΤΗΤΑ`: True/False ή Ν/Ο
- `ΣΥΓΚΡΟΥΣΗ`: Λίστα συγκρουόμενων μαθητών

## Troubleshooting

### Σφάλμα dependencies:
```bash
pip install --upgrade streamlit pandas numpy openpyxl
```

### Memory issues στο Streamlit Cloud:
- Μειώστε τον αριθμό μαθητών < 100
- Μειώστε max_scenarios σε 2

### File upload issues:
- Ελέγξτε ότι το Excel έχει τις σωστές στήλες
- CSV files πρέπει να είναι UTF-8 encoded