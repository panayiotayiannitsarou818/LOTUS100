#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App για Σύστημα Κατανομής Μαθητών
==========================================
Βελτιωμένη έκδοση με καλύτερο error handling και UX
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import io
from pathlib import Path
import sys
import traceback
import logging
from typing import Optional, Dict, Any

# Import του πλήρους συστήματος
try:
    from complete_student_assignment_FIXED import (
        StudentAssignmentSystem,
        SystemDebugger, 
        create_sample_data,
        normalize_dataframe,
        validate_required_columns
    )
except ImportError as e:
    st.error(f"Σφάλμα import: {e}")
    st.stop()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="Σύστημα Κατανομής Μαθητών",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS για καλύτερη εμφάνιση
st.markdown("""
<style>
.main-header {
    padding: 1rem 0;
    border-bottom: 2px solid #f0f2f6;
    margin-bottom: 1rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}
.success-box {
    padding: 1rem;
    background-color: #d4edda;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}
.warning-box {
    padding: 1rem;
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
.error-box {
    padding: 1rem;
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data_cached(num_students: int) -> pd.DataFrame:
    """Cached sample data generation"""
    return create_sample_data(num_students)

def safe_file_upload(uploaded_file) -> Optional[pd.DataFrame]:
    """Ασφαλής φόρτωση αρχείου με error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Δοκιμή διαφορετικών encodings
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file)
        
        if df.empty:
            st.error("Το αρχείο είναι κενό")
            return None
            
        df = normalize_dataframe(df)
        return df
        
    except Exception as e:
        st.error(f"Σφάλμα φόρτωσης αρχείου: {str(e)}")
        logger.error(f"File upload error: {e}", exc_info=True)
        return None

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Επικύρωση DataFrame με λεπτομερείς πληροφορίες"""
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    try:
        # Βασικοί έλεγχοι
        validation["stats"]["total_rows"] = len(df)
        validation["stats"]["total_columns"] = len(df.columns)
        
        # Έλεγχος απαιτούμενων στηλών
        is_valid, missing_cols = validate_required_columns(df)
        if not is_valid:
            validation["errors"].extend([f"Λείπει στήλη: {col}" for col in missing_cols])
            validation["is_valid"] = False
        
        # Έλεγχος δεδομένων
        if "ΟΝΟΜΑ" in df.columns:
            empty_names = df["ΟΝΟΜΑ"].isna().sum()
            duplicate_names = df["ΟΝΟΜΑ"].duplicated().sum()
            
            if empty_names > 0:
                validation["errors"].append(f"Κενά ονόματα: {empty_names}")
                validation["is_valid"] = False
                
            if duplicate_names > 0:
                validation["warnings"].append(f"Διπλότυπα ονόματα: {duplicate_names}")
        
        # Στατιστικά
        if "ΦΥΛΟ" in df.columns:
            boys = int((df["ΦΥΛΟ"] == "Α").sum())
            girls = int((df["ΦΥΛΟ"] == "Κ").sum())
            validation["stats"]["boys"] = boys
            validation["stats"]["girls"] = girls
            
            invalid_gender = len(df) - boys - girls
            if invalid_gender > 0:
                validation["warnings"].append(f"Άγνωστο φύλο: {invalid_gender}")
        
        if "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ" in df.columns:
            teacher_kids = int((df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True).sum())
            validation["stats"]["teacher_kids"] = teacher_kids
            
        return validation
        
    except Exception as e:
        validation["errors"].append(f"Σφάλμα επικύρωσης: {str(e)}")
        validation["is_valid"] = False
        return validation

def display_validation_results(validation: Dict[str, Any]):
    """Εμφάνιση αποτελεσμάτων επικύρωσης"""
    if validation["is_valid"]:
        st.markdown('<div class="success-box">✅ Δεδομένα έγκυρα</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">❌ Προβλήματα δεδομένων</div>', unsafe_allow_html=True)
        for error in validation["errors"]:
            st.error(f"• {error}")
    
    for warning in validation["warnings"]:
        st.markdown(f'<div class="warning-box">⚠️ {warning}</div>', unsafe_allow_html=True)
    
    # Στατιστικά
    if validation["stats"]:
        st.write("**Στατιστικά:**")
        stats = validation["stats"]
        for key, value in stats.items():
            if key not in ["total_columns"]:  # Αποφυγή εμφάνισης μη-σημαντικών
                st.write(f"• {key.replace('_', ' ').title()}: {value}")

def execute_assignment_safely(system: StudentAssignmentSystem, df: pd.DataFrame, 
                            num_classes: int, max_scenarios: int) -> Dict[str, Any]:
    """Ασφαλής εκτέλεση κατανομής με error handling"""
    try:
        # Memory check για μεγάλα datasets
        if len(df) > 200:
            st.warning("Μεγάλο dataset - η επεξεργασία μπορεί να διαρκέσει αρκετά λεπτά")
            
        results = system.process_complete_assignment(
            df, 
            num_classes=num_classes,
            max_scenarios=max_scenarios
        )
        
        return results
        
    except MemoryError:
        st.error("Ανεπαρκής μνήμη - δοκιμάστε με λιγότερους μαθητές")
        return {"status": "ERROR", "error": "Memory error"}
        
    except Exception as e:
        st.error(f"Σφάλμα εκτέλεσης: {str(e)}")
        logger.error(f"Assignment execution error: {e}", exc_info=True)
        return {"status": "ERROR", "error": str(e)}

def create_download_section(results: Dict[str, Any]):
    """Δημιουργία section λήψης αρχείων"""
    if "final_df" not in results:
        return
    
    final_df = results["final_df"]
    
    st.subheader("💾 Λήψη Αποτελεσμάτων")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv_data = final_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📄 Λήψη CSV",
            data=csv_data,
            file_name=f"student_assignment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel export
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name='Κατανομή', index=False)
                
                # Προσθήκη στατιστικών
                final_col = None
                for col in final_df.columns:
                    if "ΒΗΜΑ6" in col and "ΤΜΗΜΑ" in col:
                        final_col = col
                        break
                
                if final_col:
                    stats_df = generate_class_statistics(final_df, final_col)
                    stats_df.to_excel(writer, sheet_name='Στατιστικά', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="📊 Λήψη Excel",
                data=output.getvalue(),
                file_name=f"assignment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Σφάλμα δημιουργίας Excel: {e}")
    
    with col3:
        # JSON export για developers
        json_data = final_df.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            label="🔧 Λήψη JSON",
            data=json_data,
            file_name=f"assignment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )

def generate_class_statistics(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    """Δημιουργία στατιστικών ανά τμήμα"""
    stats_data = []
    for class_name in sorted(df[class_col].dropna().unique()):
        class_df = df[df[class_col] == class_name]
        
        stats_data.append({
            "Τμήμα": class_name,
            "Σύνολο": len(class_df),
            "Αγόρια": int((class_df["ΦΥΛΟ"] == "Α").sum()) if "ΦΥΛΟ" in class_df.columns else 0,
            "Κορίτσια": int((class_df["ΦΥΛΟ"] == "Κ").sum()) if "ΦΥΛΟ" in class_df.columns else 0,
            "Εκπαιδευτικοί": int((class_df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True).sum()) if "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ" in class_df.columns else 0,
            "Καλά Ελληνικά": int((class_df["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"] == True).sum()) if "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ" in class_df.columns else 0
        })
    
    return pd.DataFrame(stats_data)

def display_results_section(results: Dict[str, Any]):
    """Εμφάνιση αποτελεσμάτων με καλύτερο formatting"""
    if "final_df" not in results:
        st.error("Δεν βρέθηκαν τελικά αποτελέσματα")
        return
    
    final_df = results["final_df"]
    
    # Εύρεση τελικής στήλης
    final_col = None
    for col in final_df.columns:
        if "ΒΗΜΑ6" in col and "ΤΜΗΜΑ" in col:
            final_col = col
            break
    
    if not final_col:
        final_col = [col for col in final_df.columns if "ΒΗΜΑ" in col][-1] if any("ΒΗΜΑ" in col for col in final_df.columns) else None
    
    if final_col and final_col in final_df.columns:
        st.subheader("📈 Αποτελέσματα Κατανομής")
        
        # Στατιστικές μετρικές
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_students = len(final_df)
            st.metric("Συνολικοί Μαθητές", total_students)
        
        with col2:
            class_sizes = final_df[final_col].value_counts()
            pop_diff = class_sizes.max() - class_sizes.min() if len(class_sizes) > 1 else 0
            st.metric("Διαφορά Πληθυσμού", pop_diff, delta=f"{'🟢' if pop_diff <= 2 else '🔴'}")
        
        with col3:
            if "step7" in results and results["step7"]["best"]:
                best_score = results["step7"]["best"]["total_score"]
                st.metric("Τελικό Score", best_score)
            else:
                st.metric("Τελικό Score", "N/A")
        
        with col4:
            num_classes = len(class_sizes)
            st.metric("Αριθμός Τμημάτων", num_classes)
        
        # Πίνακας στατιστικών
        st.write("**Λεπτομερή στατιστικά ανά τμήμα:**")
        stats_table = generate_class_statistics(final_df, final_col)
        st.dataframe(stats_table, use_container_width=True, hide_index=True)
        
        # Visualization με bar chart
        st.write("**Κατανομή πληθυσμού:**")
        class_counts = final_df[final_col].value_counts().sort_index()
        st.bar_chart(class_counts)
    
    # Εμφάνιση τελικών δεδομένων (περιορισμένη)
    st.write("**Δείγμα τελικών αποτελεσμάτων (πρώτες 20 γραμμές):**")
    display_df = final_df.head(20)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    if len(final_df) > 20:
        st.info(f"Εμφανίζονται οι πρώτες 20 από {len(final_df)} συνολικές εγγραφές. Κατεβάστε το πλήρες αρχείο παρακάτω.")

def main():
    """Κύρια συνάρτηση εφαρμογής"""
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🎓 Σύστημα Κατανομής Μαθητών")
    st.markdown("*Ολοκληρωμένο σύστημα 7 βημάτων με βελτιωμένο error handling*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar για ρυθμίσεις
    st.sidebar.header("⚙️ Ρυθμίσεις")
    
    # Επιλογή τρόπου εισαγωγής δεδομένων
    data_source = st.sidebar.radio(
        "Πηγή δεδομένων:",
        ["Ανέβασμα αρχείου", "Δειγματικά δεδομένα"],
        help="Επιλέξτε πώς θα εισάγετε τα δεδομένα"
    )
    
    df = None
    
    if data_source == "Ανέβασμα αρχείου":
        st.sidebar.subheader("📁 Ανέβασμα αρχείου")
        uploaded_file = st.sidebar.file_uploader(
            "Επιλέξτε Excel ή CSV:",
            type=['xlsx', 'xls', 'csv'],
            help="Ανεβάστε αρχείο με στοιχεία μαθητών"
        )
        
        if uploaded_file is not None:
            with st.spinner("Φόρτωση αρχείου..."):
                df = safe_file_upload(uploaded_file)
                
            if df is not None:
                st.success(f"✅ Φορτώθηκαν {len(df)} εγγραφές από {uploaded_file.name}")
    
    else:  # Δειγματικά δεδομένα
        st.sidebar.subheader("🔬 Δειγματικά δεδομένα")
        num_students = st.sidebar.slider("Αριθμός μαθητών:", 20, 150, 50, help="Περισσότεροι από 100 μαθητές μπορεί να είναι αργοί")
        
        if st.sidebar.button("🎲 Δημιουργία δειγματικών δεδομένων", type="primary"):
            with st.spinner("Δημιουργία δεδομένων..."):
                try:
                    df = load_sample_data_cached(num_students)
                    st.success(f"✅ Δημιουργήθηκαν {len(df)} δειγματικές εγγραφές")
                except Exception as e:
                    st.error(f"Σφάλμα δημιουργίας δεδομένων: {e}")
    
    # Κύριο περιεχόμενο
    if df is not None:
        # Παράμετροι επεξεργασίας
        st.sidebar.header("🔧 Παράμετροι Εκτέλεσης")
        
        num_classes = st.sidebar.number_input(
            "Αριθμός τμημάτων:", 
            min_value=2, 
            max_value=8, 
            value=min(4, max(2, len(df) // 20)),  # Intelligent default
            help="Προτεινόμενος αριθμός βάσει πληθυσμού"
        )
        
        max_scenarios = st.sidebar.slider(
            "Μέγιστα σενάρια ανά βήμα:", 
            1, 5, 3,
            help="Περισσότερα σενάρια = καλύτερα αποτελέσματα αλλά πιο αργή εκτέλεση"
        )
        
        # Επισκόπηση δεδομένων
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Επισκόπηση Δεδομένων")
            
            # Βασικές μετρικές
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.metric("Συνολικοί Μαθητές", len(df))
            
            with metrics_cols[1]:
                boys = int((df["ΦΥΛΟ"] == "Α").sum()) if "ΦΥΛΟ" in df.columns else 0
                st.metric("Αγόρια", boys)
            
            with metrics_cols[2]:
                girls = int((df["ΦΥΛΟ"] == "Κ").sum()) if "ΦΥΛΟ" in df.columns else 0
                st.metric("Κορίτσια", girls)
            
            with metrics_cols[3]:
                teacher_kids = int((df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True).sum()) if "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ" in df.columns else 0
                st.metric("Παιδιά Εκπαιδευτικών", teacher_kids)
            
            # Δείγμα δεδομένων
            st.write("**Δείγμα δεδομένων (πρώτες 10 γραμμές):**")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔍 Επικύρωση")
            
            with st.spinner("Επικύρωση δεδομένων..."):
                validation = validate_dataframe(df)
            
            display_validation_results(validation)
        
        # Κουμπί εκτέλεσης
        st.header("🚀 Εκτέλεση Κατανομής")
        
        # Warning για μεγάλα datasets
        if len(df) > 100:
            st.warning("⚠️ Μεγάλο dataset - η επεξεργασία μπορεί να διαρκέσει 2-5 λεπτά")
        
        execute_button = st.button(
            "🎯 Εκκίνηση Ανάθεσης", 
            type="primary", 
            use_container_width=True,
            disabled=not validation["is_valid"]
        )
        
        if execute_button:
            if validation["is_valid"]:
                # Progress container
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Εκτέλεση με progress updates
                    with st.spinner("Εκτέλεση 7-βημάτων κατανομής..."):
                        try:
                            system = StudentAssignmentSystem()
                            
                            # Simulate progress updates
                            status_text.text("Εκκίνηση συστήματος...")
                            progress_bar.progress(0.1)
                            
                            results = execute_assignment_safely(
                                system, df, num_classes, max_scenarios
                            )
                            
                            progress_bar.progress(1.0)
                            status_text.text("Ολοκληρώθηκε!")
                            
                            if results["status"] == "SUCCESS":
                                st.balloons()  # Celebration!
                                st.success("🎉 Κατανομή ολοκληρώθηκε επιτυχώς!")
                                
                                # Εμφάνιση αποτελεσμάτων
                                display_results_section(results)
                                
                                # Download buttons
                                create_download_section(results)
                                
                            else:
                                st.error(f"❌ Σφάλμα κατανομής: {results.get('error', 'Άγνωστο σφάλμα')}")
                                
                        except Exception as e:
                            st.error(f"❌ Κρίσιμο σφάλμα: {e}")
                            logger.error(f"Critical error in main execution: {e}", exc_info=True)
            else:
                st.error("❌ Παρακαλώ διορθώστε τα προβλήματα δεδομένων πρώτα")
    
    else:
        # Οδηγίες χωρίς δεδομένα
        st.info("👆 Παρακαλώ επιλέξτε πηγή δεδομένων από την πλαϊνή μπάρα")
        
        with st.expander("📖 Οδηγίες χρήσης", expanded=True):
            st.markdown("""
            ### Απαιτούμενες στήλες:
            - **ΟΝΟΜΑ**: Ονοματεπώνυμο μαθητή
            - **ΦΥΛΟ**: Α (Αγόρι) ή Κ (Κορίτσι)
            - **ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ**: True/False ή Ν/Ο
            - **ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ**: True/False ή Ν/Ο
            
            ### Προαιρετικές στήλες:
            - **ΦΙΛΟΙ**: Λίστα ονομάτων φίλων
            - **ΖΩΗΡΟΣ**: True/False ή Ν/Ο
            - **ΙΔΙΑΙΤΕΡΟΤΗΤΑ**: True/False ή Ν/Ο
            - **ΣΥΓΚΡΟΥΣΗ**: Λίστα συγκρουόμενων μαθητών
            
            ### Τα 7 Βήματα του συστήματος:
            1. **Παιδιά εκπαιδευτικών** (immutable τοποθέτηση)
            2. **Ζωηροί & ιδιαιτερότητες** (παιδαγωγική ισορροπία)
            3. **Αμοιβαίες φιλίες** (τοποθέτηση δυάδων)
            4. **Φιλικές ομάδες** (ομαδοποίηση)
            5. **Υπόλοιποι μαθητές** (συμπλήρωση)
            6. **Τελικός έλεγχος** (εξισορρόπηση)
            7. **Βαθμολόγηση** (επιλογή βέλτιστου)
            
            ### Συμβουλές για καλύτερα αποτελέσματα:
            - Χρησιμοποιήστε συνεπή ονοματολογία
            - Ελέγξτε ότι τα φύλα είναι μόνο Α ή Κ
            - Τα ονόματα φίλων πρέπει να ταιριάζουν ακριβώς
            - Για μεγάλα datasets (>100), αυξήστε τον αριθμό τμημάτων
            """)
        
        # Δείγμα template
        with st.expander("📋 Λήψη Template"):
            template_df = pd.DataFrame({
                "ΟΝΟΜΑ": ["Γιάννης Παπαδόπουλος", "Μαρία Κωνσταντίνου", "Νίκος Γεωργίου"],
                "ΦΥΛΟ": ["Α", "Κ", "Α"],
                "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ": [True, False, False],
                "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ": [True, True, False],
                "ΦΙΛΟΙ": ["Νίκος Γεωργίου", "Γιάννης Παπαδόπουλος", "Γιάννης Παπαδόπουλος"],
                "ΖΩΗΡΟΣ": [False, True, False],
                "ΙΔΙΑΙΤΕΡΟΤΗΤΑ": [False, False, True],
                "ΣΥΓΚΡΟΥΣΗ": ["", "", ""]
            })
            
            st.dataframe(template_df, use_container_width=True, hide_index=True)
            
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                "📥 Λήψη Template CSV",
                csv_template,
                "student_template.csv",
                "text/csv",
                help="Κατεβάστε αυτό το template και συμπληρώστε με τα δικά σας δεδομένα"
            )

if __name__ == "__main__":
    main()