#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App για Σύστημα Κατανομής Μαθητών
==========================================
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import io
from pathlib import Path
import sys

# Import του πλήρους συστήματος
from complete_student_assignment_FIXED import (
    StudentAssignmentSystem,
    SystemDebugger, 
    create_sample_data,
    normalize_dataframe,
    validate_required_columns
)

# Streamlit page config
st.set_page_config(
    page_title="Σύστημα Κατανομής Μαθητών",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎓 Σύστημα Κατανομής Μαθητών")
    st.markdown("*Ολοκληρωμένο σύστημα 7 βημάτων*")
    
    # Sidebar για ρυθμίσεις
    st.sidebar.header("⚙️ Ρυθμίσεις")
    
    # Επιλογή τρόπου εισαγωγής δεδομένων
    data_source = st.sidebar.radio(
        "Πηγή δεδομένων:",
        ["Ανέβασμα αρχείου", "Δειγματικά δεδομένα"]
    )
    
    df = None
    
    if data_source == "Ανέβασμα αρχείου":
        uploaded_file = st.sidebar.file_uploader(
            "Επιλέξτε Excel ή CSV:",
            type=['xlsx', 'xls', 'csv'],
            help="Ανεβάστε αρχείο με στοιχεία μαθητών"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = normalize_dataframe(df)
                st.success(f"✅ Φορτώθηκαν {len(df)} εγγραφές")
                
            except Exception as e:
                st.error(f"❌ Σφάλμα φόρτωσης: {e}")
    
    else:  # Δειγματικά δεδομένα
        num_students = st.sidebar.slider("Αριθμός μαθητών:", 20, 100, 50)
        if st.sidebar.button("Δημιουργία δειγματικών δεδομένων"):
            with st.spinner("Δημιουργία δεδομένων..."):
                df = create_sample_data(num_students)
                st.success(f"✅ Δημιουργήθηκαν {len(df)} δειγματικές εγγραφές")
    
    if df is not None:
        # Παράμετροι επεξεργασίας
        st.sidebar.header("🔧 Παράμετροι")
        
        num_classes = st.sidebar.number_input(
            "Αριθμός τμημάτων:", 
            min_value=2, 
            max_value=8, 
            value=3,
            help="Αριθμός τμημάτων για κατανομή"
        )
        
        max_scenarios = st.sidebar.slider("Μέγιστα σενάρια ανά βήμα:", 1, 5, 3)
        
        # Εμφάνιση βασικών στοιχείων
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Επισκόπηση Δεδομένων")
            
            # Βασικές μετρικές
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.metric("Συνολικοί Μαθητές", len(df))
            
            with metrics_cols[1]:
                boys = (df["ΦΥΛΟ"] == "Α").sum() if "ΦΥΛΟ" in df.columns else 0
                st.metric("Αγόρια", boys)
            
            with metrics_cols[2]:
                girls = (df["ΦΥΛΟ"] == "Κ").sum() if "ΦΥΛΟ" in df.columns else 0
                st.metric("Κορίτσια", girls)
            
            with metrics_cols[3]:
                teacher_kids = (df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True).sum() if "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ" in df.columns else 0
                st.metric("Παιδιά Εκπαιδευτικών", teacher_kids)
            
            # Δείγμα δεδομένων
            st.write("**Δείγμα δεδομένων:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("🔍 Επικύρωση")
            
            # Δημιουργία debugger για validation
            system = StudentAssignmentSystem()
            debugger = SystemDebugger(system)
            
            with st.spinner("Επικύρωση δεδομένων..."):
                validation = debugger.validate_input_data(df)
            
            if validation["is_valid"]:
                st.success("✅ Δεδομένα έγκυρα")
            else:
                st.error("❌ Προβλήματα δεδομένων")
                
                for error in validation["errors"]:
                    st.error(f"• {error}")
            
            for warning in validation["warnings"]:
                st.warning(f"⚠️ {warning}")
            
            # Στατιστικά validation
            st.write("**Στατιστικά:**")
            for key, value in validation["stats"].items():
                if key != "columns":  # Αποφυγή εμφάνισης μεγάλης λίστας
                    st.write(f"• {key}: {value}")
        
        # Κουμπί εκτέλεσης
        st.header("🚀 Εκτέλεση Κατανομής")
        
        if st.button("Εκκίνηση Ανάθεσης", type="primary", use_container_width=True):
            if validation["is_valid"]:
                
                # Progress indicators
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    step_info = st.empty()
                
                try:
                    status_text.text("Εκκίνηση συστήματος κατανομής...")
                    progress_bar.progress(0.1)
                    
                    # Εκτέλεση κατανομής με progress updates
                    with st.spinner("Εκτέλεση 7 βημάτων κατανομής..."):
                        results = execute_assignment_with_progress(
                            system, df, num_classes, max_scenarios,
                            progress_bar, status_text, step_info
                        )
                    
                    progress_bar.progress(1.0)
                    
                    if results["status"] == "SUCCESS":
                        st.success("🎉 Κατανομή ολοκληρώθηκε επιτυχώς!")
                        
                        # Εμφάνιση αποτελεσμάτων
                        display_results(results)
                        
                        # Download buttons
                        create_download_buttons(results)
                    
                    else:
                        st.error(f"❌ Σφάλμα κατανομής: {results.get('error', 'Άγνωστο σφάλμα')}")
                    
                    status_text.text("Ολοκληρώθηκε!")
                    
                except Exception as e:
                    st.error(f"❌ Σφάλμα εκτέλεσης: {e}")
                    progress_bar.progress(0)
                    status_text.text("Σφάλμα!")
            
            else:
                st.error("❌ Παρακαλώ διορθώστε τα προβλήματα δεδομένων πρώτα")
    
    else:
        # Οδηγίες χωρίς δεδομένα
        st.info("👆 Παρακαλώ επιλέξτε πηγή δεδομένων από την πλαϊνή μπάρα")
        
        with st.expander("📖 Οδηγίες χρήσης"):
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
            
            ### Τα 7 Βήματα:
            1. **Παιδιά εκπαιδευτικών** (immutable τοποθέτηση)
            2. **Ζωηροί & ιδιαιτερότητες** (παιδαγωγική ισορροπία)
            3. **Αμοιβαίες φιλίες** (τοποθέτηση δυάδων)
            4. **Φιλικές ομάδες** (ομαδοποίηση)
            5. **Υπόλοιποι μαθητές** (συμπλήρωση)
            6. **Τελικός έλεγχος** (εξισορρόπηση)
            7. **Βαθμολόγηση** (επιλογή βέλτιστου)
            """)

def execute_assignment_with_progress(system, df, num_classes, max_scenarios, 
                                   progress_bar, status_text, step_info):
    """Εκτέλεση κατανομής με progress indicators."""
    
    # Override print functions για Streamlit
    import builtins
    original_print = builtins.print
    
    def streamlit_print(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        if "===" in message:
            step_info.info(message.replace("===", "").strip())
        elif "ΒΗΜΑ" in message:
            step_info.success(f"🔄 {message}")
        elif "✅" in message:
            step_info.success(message)
        elif "❌" in message:
            step_info.error(message)
    
    # Temporarily replace print
    builtins.print = streamlit_print
    
    try:
        # Εκτέλεση με progress updates
        results = system.process_complete_assignment(
            df, 
            num_classes=num_classes,
            max_scenarios=max_scenarios
        )
        
        # Update progress based on steps
        steps_completed = 0
        if "step1" in results: 
            steps_completed += 1
            progress_bar.progress(0.15)
            status_text.text("Βήμα 1: Παιδιά εκπαιδευτικών ✓")
            
        if "step2" in results: 
            steps_completed += 1
            progress_bar.progress(0.30)
            status_text.text("Βήμα 2: Ζωηροί & ιδιαιτερότητες ✓")
            
        if "step3" in results: 
            steps_completed += 1
            progress_bar.progress(0.45)
            status_text.text("Βήμα 3: Αμοιβαίες φιλίες ✓")
            
        if "step4" in results: 
            steps_completed += 1
            progress_bar.progress(0.60)
            status_text.text("Βήμα 4: Φιλικές ομάδες ✓")
            
        if "step5" in results: 
            steps_completed += 1
            progress_bar.progress(0.75)
            status_text.text("Βήμα 5: Υπόλοιποι μαθητές ✓")
            
        if "step6" in results: 
            steps_completed += 1
            progress_bar.progress(0.90)
            status_text.text("Βήμα 6: Τελικός έλεγχος ✓")
            
        if "step7" in results: 
            steps_completed += 1
            progress_bar.progress(0.95)
            status_text.text("Βήμα 7: Βαθμολόγηση ✓")
        
        return results
        
    finally:
        # Restore original print
        builtins.print = original_print

def display_results(results):
    """Εμφάνιση αποτελεσμάτων κατανομής."""
    
    if "final_df" not in results:
        st.error("Δεν βρέθηκαν τελικά αποτελέσματα")
        return
    
    final_df = results["final_df"]
    
    st.subheader("📈 Αποτελέσματα")
    
    # Βρες την τελική στήλη τμήματος
    final_col = None
    for col in final_df.columns:
        if "ΒΗΜΑ6" in col and "ΤΜΗΜΑ" in col:
            final_col = col
            break
    
    if not final_col:
        final_col = [col for col in final_df.columns if "ΒΗΜΑ" in col][-1]
    
    if final_col and final_col in final_df.columns:
        # Στατιστικά κατανομής
        stats_table = generate_class_statistics(final_df, final_col)
        st.write("**Κατανομή ανά τμήμα:**")
        st.dataframe(stats_table, use_container_width=True)
        
        # Στατιστικές μετρικές
        col1, col2, col3 = st.columns(3)
        
        with col1:
            class_sizes = final_df[final_col].value_counts()
            pop_diff = class_sizes.max() - class_sizes.min() if len(class_sizes) > 1 else 0
            st.metric("Διαφορά Πληθυσμού", pop_diff)
        
        with col2:
            if "step7" in results and results["step7"]["best"]:
                best_score = results["step7"]["best"]["total_score"]
                st.metric("Τελικό Score", best_score)
            else:
                st.metric("Τελικό Score", "N/A")
        
        with col3:
            if "step6" in results:
                iterations = results["step6"]["summary"]["iterations"]
                st.metric("Βελτιώσεις Βήμα 6", iterations)
    
    # Τελικά δεδομένα με τμήματα
    st.write("**Τελικά δεδομένα με τμήματα:**")
    st.dataframe(final_df, use_container_width=True)

def generate_class_statistics(df, class_col):
    """Δημιουργία στατιστικών ανά τμήμα."""
    
    stats_data = []
    for class_name in sorted(df[class_col].dropna().unique()):
        class_df = df[df[class_col] == class_name]
        
        stats_data.append({
            "Τμήμα": class_name,
            "Σύνολο": len(class_df),
            "Αγόρια": (class_df["ΦΥΛΟ"] == "Α").sum() if "ΦΥΛΟ" in class_df.columns else 0,
            "Κορίτσια": (class_df["ΦΥΛΟ"] == "Κ").sum() if "ΦΥΛΟ" in class_df.columns else 0,
            "Εκπαιδευτικοί": (class_df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True).sum() if "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ" in class_df.columns else 0,
            "Καλά Ελληνικά": (class_df["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"] == True).sum() if "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ" in class_df.columns else 0
        })
    
    return pd.DataFrame(stats_data)

def create_download_buttons(results):
    """Δημιουργία κουμπιών λήψης."""
    
    if "final_df" not in results:
        return
    
    final_df = results["final_df"]
    
    st.subheader("💾 Λήψη Αποτελεσμάτων")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_data = final_df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📄 Λήψη CSV",
            data=csv_data,
            file_name="student_assignment_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name='Κατανομή', index=False)
            
            # Προσθήκη στατιστικών αν υπάρχει τελική στήλη
            final_col = None
            for col in final_df.columns:
                if "ΒΗΜΑ6" in col:
                    final_col = col
                    break
            
            if final_col:
                stats_df = generate_class_statistics(final_df, final_col)
                stats_df.to_excel(writer, sheet_name='Στατιστικά', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📊 Λήψη Excel",
            data=output.getvalue(),
            file_name="assignment_complete.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

if __name__ == "__main__":
    main()