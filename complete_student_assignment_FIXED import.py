#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ολοκληρωμένο Σύστημα Κατανομής Μαθητών - 7 ΒΗΜΑΤΑ
=================================================

ΒΗΜΑΤΑ:
1. Παιδιά εκπαιδευτικών (immutable)
2. Ζωηροί & Ιδιαιτερότητες 
3. Αμοιβαίες φιλίες (δυάδες)
4. Φιλικές ομάδες
5. Υπόλοιποι μαθητές
6. Τελικός έλεγχος & διορθώσεις
7. Βαθμολόγηση & επιλογή βέλτιστου

ΔΙΟΡΘΩΣΕΙΣ:
- Fix syntax bug στο SystemDebugger (γραμμή 292)
- Διόρθωση YES_VALUES (αφαίρεση "ΤΑΙ")
- Βελτιωμένο Βήμα 6 με κανόνα-βασισμένες ανταλλαγές
"""

import os
import sys
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, FrozenSet
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import re
import ast

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

RANDOM_SEED = 42
MAX_STUDENTS_PER_CLASS = 25
MIN_CLASSES = 2
random.seed(RANDOM_SEED)

# Column mappings for backward compatibility
COLUMN_MAPPINGS = {
    "ΟΝΟΜΑ": ["ΟΝΟΜΑ", "NAME", "ΌΝΟΜΑ", "ΟΝΟΜΑΤΕΠΩΝΥΜΟ"],
    "ΦΥΛΟ": ["ΦΥΛΟ", "GENDER", "ΦΏΛΟ"],
    "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ": ["ΓΝΩΣΗ", "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ", "ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"],
    "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ": ["ΕΚΠΑΙΔ", "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ", "ΠΑΙΔΙ ΕΚΠΑΙΔΕΥΤΙΚΟΥ"],
    "ΖΩΗΡΟΣ": ["ΖΩΗΡΟΣ", "ΕΝΕΡΓΗΤΙΚΟΣ", "ΖΩΗΡΌ"],
    "ΙΔΙΑΙΤΕΡΟΤΗΤΑ": ["ΙΔΙΑΙΤ", "ΙΔΙΑΙΤΕΡΟΤΗΤΑ", "ΕΙΔΙΚΕΣ_ΑΝΑΓΚΕΣ"],
    "ΦΙΛΟΙ": ["ΦΙΛΟΙ", "ΦΊΛΟΙ", "FRIENDS"],
    "ΣΥΓΚΡΟΥΣΗ": ["ΣΥΓΚΡΟΥΣΗ", "CONFLICT", "ΣΥΓΚΡΟΥΣΕΙΣ"]
}

# ΔΙΟΡΘΩΣΗ: Αφαίρεση "ΤΑΙ" typo
YES_VALUES = {"Ν", "ΝΑΙ", "YES", "TRUE", "1", "Y"}
NO_VALUES = {"Ο", "ΟΧΙ", "NO", "FALSE", "0", "N"}

# Penalty weights
PENALTY_WEIGHTS = {
    "population_imbalance": 3,
    "gender_imbalance": 2, 
    "greek_knowledge_imbalance": 1,
    "broken_friendship": 5,
    "pedagogical_conflict": 3
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_yes_no(value) -> bool:
    """Κανονικοποίηση Ν/Ο τιμών σε boolean."""
    if pd.isna(value):
        return False
    
    str_val = str(value).strip().upper()
    return str_val in YES_VALUES

def parse_friends_list(friends_value) -> List[str]:
    """Parse φίλων από διάφορα formats με βελτιωμένη ασφάλεια."""
    if isinstance(friends_value, list):
        return [str(f).strip() for f in friends_value if str(f).strip()]
    
    if pd.isna(friends_value):
        return []
    
    friends_str = str(friends_value).strip()
    if not friends_str or friends_str.upper() in ["NAN", "NONE", ""]:
        return []
    
    # Προσπάθεια parsing ως Python literal (βελτιωμένο)
    try:
        parsed = ast.literal_eval(friends_str)
        if isinstance(parsed, list):
            return [str(f).strip() for f in parsed if str(f).strip()]
    except (ValueError, SyntaxError, TypeError):
        pass
    
    # Split με ασφαλές regex (βελτιωμένο από friendship_filters_fixed)
    parts = re.split(r"[,\|\;/·\n]+", friends_str)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]

def are_mutual_friends_safe(df: pd.DataFrame, name1: str, name2: str, 
                           name_col: str = "ΟΝΟΜΑ", friends_col: str = "ΦΙΛΟΙ") -> bool:
    """Ασφαλής έλεγχος αμοιβαίας φιλίας (χωρίς substring pitfalls)."""
    
    row1 = df[df[name_col].astype(str) == str(name1)]
    row2 = df[df[name_col].astype(str) == str(name2)]
    
    if row1.empty or row2.empty:
        return False
    
    friends1 = set(parse_friends_list(row1.iloc[0].get(friends_col, "")))
    friends2 = set(parse_friends_list(row2.iloc[0].get(friends_col, "")))
    
    # Exact token comparison (όχι substring)
    name1_str = str(name1).strip()
    name2_str = str(name2).strip()
    
    return name2_str in friends1 and name1_str in friends2

def count_broken_friendships_advanced(df: pd.DataFrame, assignment_col: str,
                                    names: Optional[List[str]] = None,
                                    name_col: str = "ΟΝΟΜΑ", 
                                    friends_col: str = "ΦΙΛΟΙ") -> int:
    """
    Μέτρηση σπασμένων φιλιών χωρίς διπλομέτρηση.
    Βασισμένο στο friendship_filters_fixed.py με βελτιώσεις.
    """
    
    if assignment_col not in df.columns:
        return 0
    
    if names is None:
        names = df[name_col].astype(str).tolist()
    
    # Δημιουργία mapping: όνομα -> τμήμα
    name_to_class = {}
    for _, row in df.iterrows():
        student_name = str(row[name_col]).strip()
        assigned_class = row[assignment_col]
        if pd.notna(assigned_class):
            name_to_class[student_name] = str(assigned_class)
    
    # Μέτρηση με αποφυγή διπλομέτρησης
    broken = 0
    checked_pairs = set()
    
    for i, name_a in enumerate(names):
        for name_b in names[i+1:]:  # Αποφεύγει διπλομέτρηση
            pair_key = frozenset([name_a, name_b])
            
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)
            
            # Έλεγχος αμοιβαίας φιλίας
            if are_mutual_friends_safe(df, name_a, name_b, name_col, friends_col):
                class_a = name_to_class.get(name_a)
                class_b = name_to_class.get(name_b)
                
                # Σπασμένη φιλία αν διαφορετικά τμήματα ή κάποιος μη-τοποθετημένος
                if class_a is None or class_b is None or class_a != class_b:
                    broken += 1
    
    return broken

def filter_scenarios_by_friendships(scenarios: List[Tuple[str, pd.DataFrame, Dict[str, Any]]], 
                                  assignment_col: str,
                                  names: Optional[List[str]] = None,
                                  top_k: int = 5) -> List[Tuple[str, pd.DataFrame, Dict[str, Any]]]:
    """
    Φιλτράρισμα σεναρίων βάσει σπασμένων φιλιών.
    Βασισμένο στο friendship_filters_fixed.py.
    """
    
    # Υπολογισμός σπασμένων φιλιών για κάθε σενάριο
    scored_scenarios = []
    
    for scenario_name, scenario_df, scenario_metrics in scenarios:
        try:
            broken_count = count_broken_friendships_advanced(
                scenario_df, assignment_col, names
            )
            
            # Προσθήκη broken friendships στα metrics
            updated_metrics = scenario_metrics.copy()
            updated_metrics["broken_friendships"] = broken_count
            
            scored_scenarios.append((broken_count, scenario_name, scenario_df, updated_metrics))
            
        except Exception as e:
            print(f"Σφάλμα υπολογισμού φιλιών για {scenario_name}: {e}")
            # Θεωρούμε χειρότερο σενάριο
            updated_metrics = scenario_metrics.copy()
            updated_metrics["broken_friendships"] = float("inf")
            scored_scenarios.append((float("inf"), scenario_name, scenario_df, updated_metrics))
    
    # Στρατηγική επιλογής (από friendship_filters_fixed)
    zero_broken = [(score, name, df, metrics) for score, name, df, metrics in scored_scenarios if score == 0]
    
    if len(zero_broken) >= top_k:
        # Αν έχουμε αρκετά με 0 σπασμένες φιλίες, κρατάμε τα πρώτα
        selected = zero_broken[:top_k]
    else:
        # Ταξινόμηση κατά αύξοντα αριθμό σπασμένων φιλιών
        scored_scenarios.sort(key=lambda x: x[0])
        selected = scored_scenarios[:top_k]
    
    # Μετατροπή σε αρχικό format
    result = []
    for _, scenario_name, scenario_df, updated_metrics in selected:
        result.append((scenario_name, scenario_df, updated_metrics))
    
    return result

def infer_assignment_column_smart(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """
    Έξυπνος εντοπισμός στήλης ανάθεσης.
    Βασισμένο στο friendship_filters_fixed.py.
    """
    
    if preferred and preferred in df.columns:
        return preferred
    
    # Αναζήτηση ΒΗΜΑ* στηλών με regex
    pattern = re.compile(r"^ΒΗΜΑ\d+_ΣΕΝΑΡΙΟ_\d+$", re.IGNORECASE)
    step_columns = []
    
    for col in df.columns:
        if pattern.match(str(col)):
            step_columns.append(col)
    
    if step_columns:
        # Επιστροφή της "υψηλότερης" στήλης βήματος
        step_columns.sort(reverse=True)
        return step_columns[0]
    
    # Fallback options
    fallback_options = [
        "ΠΡΟΤΕΙΝΟΜΕΝΟ_ΤΜΗΜΑ", "ΤΜΗΜΑ", "ΒΗΜΑ6_ΤΜΗΜΑ", "ΤΕΛΙΚΟ_ΤΜΗΜΑ"
    ]
    
    for option in fallback_options:
        if option in df.columns:
            return option
    
    # Τελευταία επιλογή: τελευταία στήλη
    return df.columns[-1] if len(df.columns) > 0 else None

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Κανονικοποίηση DataFrame."""
    df = df.copy()
    
    # Κανονικοποίηση ονομάτων στηλών
    rename_map = {}
    for target_col, possible_names in COLUMN_MAPPINGS.items():
        for col in df.columns:
            col_upper = str(col).strip().upper()
            if any(name.upper() in col_upper for name in possible_names):
                rename_map[col] = target_col
                break
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Κανονικοποίηση τιμών για boolean-like στήλες
    bool_cols = ["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ", "ΖΩΗΡΟΣ", "ΙΔΙΑΙΤΕΡΟΤΗΤΑ", "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(normalize_yes_no)
    
    if "ΟΝΟΜΑ" in df.columns:
        df["ΟΝΟΜΑ"] = df["ΟΝΟΜΑ"].astype(str).str.strip()
    
    return df

def compute_optimal_classes(num_students: int, max_per_class: int = 25, min_classes: int = 2) -> int:
    """Υπολογισμός βέλτιστου αριθμού τμημάτων."""
    try:
        num_students = int(num_students)
    except (ValueError, TypeError):
        num_students = 0
    
    return max(min_classes, math.ceil((num_students or 0) / max_per_class))

def make_class_labels(num_classes: int, prefix: str = 'Α') -> List[str]:
    """Δημιουργία ετικετών τμημάτων (π.χ. ['Α1', 'Α2', ...])."""
    try:
        k = int(num_classes)
    except (ValueError, TypeError):
        k = 2
    
    return [f"{prefix}{i+1}" for i in range(k)]

# Backward compatibility aliases
compute_num_classes = compute_optimal_classes  # Alias για συμβατότητα

def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Επικύρωση ότι υπάρχουν απαιτούμενες στήλες."""
    required = ["ΟΝΟΜΑ", "ΦΥΛΟ", "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ", "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"]
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing

# =============================================================================
# ΒΗΜΑ 1: ΠΑΙΔΙΑ ΕΚΠΑΙΔΕΥΤΙΚΩΝ (IMMUTABLE)
# =============================================================================

@dataclass(frozen=True)
class Step1Scenario:
    """Immutable σενάριο βήματος 1."""
    id: int
    column_name: str
    assignments: Dict[str, str]
    description: str
    broken_friendships: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class Step1ImmutableProcessor:
    """Επεξεργαστής που εξασφαλίζει immutability του Βήματος 1."""
    
    def __init__(self):
        self._scenarios: Optional[List[Step1Scenario]] = None
        self._is_locked: bool = False
    
    def create_scenarios(self, df: pd.DataFrame, num_classes: Optional[int] = None) -> List[Step1Scenario]:
        """Δημιουργία immutable σεναρίων."""
        if self._is_locked:
            raise RuntimeError("Step1 είναι ήδη κλειδωμένο")
        
        df_norm = normalize_dataframe(df)
        
        if num_classes is None:
            num_classes = compute_optimal_classes(len(df_norm))
        
        # Εντοπισμός παιδιών εκπαιδευτικών
        teacher_kids = self._get_teacher_kids(df_norm)
        if not teacher_kids:
            return []
        
        # Εξαγωγή φιλιών
        friendships = self._extract_friendships(df_norm, teacher_kids)
        
        # Δημιουργία σεναρίων
        scenarios = self._generate_scenarios(teacher_kids, num_classes, friendships)
        
        self._scenarios = scenarios
        return scenarios
    
    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Εφαρμόζει τα σενάρια στο DataFrame και το κλειδώνει."""
        if not self._scenarios:
            raise RuntimeError("Δεν έχουν δημιουργηθεί σενάρια ακόμη")
        
        result_df = df.copy()
        
        for scenario in self._scenarios:
            col_name = scenario.column_name
            result_df[col_name] = ""
            
            for student_name, class_assigned in scenario.assignments.items():
                mask = result_df["ΟΝΟΜΑ"] == student_name
                if mask.any():
                    result_df.loc[mask, col_name] = class_assigned
        
        self._is_locked = True
        return result_df
    
    def _get_teacher_kids(self, df: pd.DataFrame) -> List[str]:
        """Εντοπισμός παιδιών εκπαιδευτικών."""
        teacher_kids_df = df[df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True]
        return teacher_kids_df["ΟΝΟΜΑ"].astype(str).str.strip().tolist()
    
    def _extract_friendships(self, df: pd.DataFrame, teacher_kids: List[str]) -> FrozenSet[Tuple[str, str]]:
        """Εξαγωγή αμοιβαίων φιλιών."""
        teacher_kids_set = set(teacher_kids)
        student_friends = {}
        
        # Εύρεση φιλιών από ΦΙΛΟΙ στήλη
        if "ΦΙΛΟΙ" in df.columns:
            for _, row in df.iterrows():
                student_name = str(row["ΟΝΟΜΑ"]).strip()
                if student_name in teacher_kids_set:
                    friends_list = parse_friends_list(row["ΦΙΛΟΙ"])
                    valid_friends = [f for f in friends_list if f in teacher_kids_set and f != student_name]
                    if valid_friends:
                        student_friends[student_name] = set(valid_friends)
        
        # Έλεγχος αμοιβαιότητας
        friendships = set()
        for student_a in student_friends:
            friends_of_a = student_friends[student_a]
            for student_b in friends_of_a:
                if student_b in student_friends and student_a in student_friends[student_b]:
                    pair = tuple(sorted([student_a, student_b]))
                    friendships.add(pair)
        
        return frozenset(friendships)
    
    def _generate_scenarios(self, teacher_kids: List[str], num_classes: int, 
                          friendships: FrozenSet[Tuple[str, str]]) -> List[Step1Scenario]:
        """Δημιουργία σεναρίων."""
        class_labels = [f"Α{i+1}" for i in range(num_classes)]
        scenarios = []
        
        if len(teacher_kids) <= num_classes:
            # Σειριακή κατανομή
            assignments = {}
            for i, name in enumerate(teacher_kids):
                assignments[name] = class_labels[i % num_classes]
            
            scenario = Step1Scenario(
                id=1,
                column_name="ΒΗΜΑ1_ΣΕΝΑΡΙΟ_1",
                assignments=assignments,
                description="Σειριακή κατανομή ≤1/τμήμα",
                broken_friendships=0
            )
            scenarios.append(scenario)
        else:
            # Εξαντλητική παραγωγή (απλουστευμένη)
            for i in range(min(5, len(teacher_kids))):
                assignments = {}
                for j, name in enumerate(teacher_kids):
                    assignments[name] = class_labels[(j + i) % num_classes]
                
                broken = sum(1 for f1, f2 in friendships 
                           if assignments.get(f1) != assignments.get(f2))
                
                scenario = Step1Scenario(
                    id=i+1,
                    column_name=f"ΒΗΜΑ1_ΣΕΝΑΡΙΟ_{i+1}",
                    assignments=assignments,
                    description="Ισόροπη κατανομή",
                    broken_friendships=broken
                )
                scenarios.append(scenario)
        
        return scenarios

# =============================================================================
# ΒΗΜΑ 2: ΖΩΗΡΟΙ & ΙΔΙΑΙΤΕΡΟΤΗΤΕΣ
# =============================================================================

class Step2Processor:
    """Επεξεργαστής ζωηρών και ιδιαιτεροτήτων."""
    
    def process_energetic_and_special(self, df: pd.DataFrame, step1_col: str, 
                                    num_classes: Optional[int] = None, 
                                    max_results: int = 5) -> List[Tuple[str, pd.DataFrame, Dict[str, Any]]]:
        """Επεξεργασία ζωηρών και ιδιαιτεροτήτων."""
        
        if num_classes is None:
            num_classes = compute_optimal_classes(len(df))
        
        class_labels = [f"Α{i+1}" for i in range(num_classes)]
        
        # Εντοπισμός μαθητών για βήμα 2
        scope = self._identify_step2_scope(df, step1_col)
        if not scope:
            # Fallback: αντιγραφή αποτελεσμάτων step1
            result_df = df.copy()
            step2_col = step1_col.replace("ΒΗΜΑ1", "ΒΗΜΑ2")
            result_df[step2_col] = result_df[step1_col]
            return [("default", result_df, {"ped_conflicts": 0, "broken": 0, "penalty": 0})]
        
        # Παραγωγή λύσεων
        results = self._generate_step2_solutions(df, step1_col, scope, class_labels, max_results)
        
        return results[:max_results]
    
    def _identify_step2_scope(self, df: pd.DataFrame, step1_col: str) -> Set[str]:
        """Εντοπισμός μαθητών για επεξεργασία στο βήμα 2."""
        scope = set()
        for _, row in df.iterrows():
            placed = pd.notna(row.get(step1_col))
            is_energetic = normalize_yes_no(row.get("ΖΩΗΡΟΣ", False))
            is_special = normalize_yes_no(row.get("ΙΔΙΑΙΤΕΡΟΤΗΤΑ", False))
            is_teacher_child = normalize_yes_no(row.get("ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ", False))
            
            # Κριτήρια βήματος 2 - ΔΙΟΡΘΩΣΗ: αφαίρεση ήδη τοποθετημένων παιδιών εκπαιδευτικών
            if not placed and (is_energetic or is_special):
                scope.add(str(row.get("ΟΝΟΜΑ", "")).strip())
        
        return scope
    
    def _generate_step2_solutions(self, df: pd.DataFrame, step1_col: str, scope: Set[str], 
                                class_labels: List[str], max_results: int) -> List[Tuple[str, pd.DataFrame, Dict[str, Any]]]:
        """Παραγωγή λύσεων βήματος 2."""
        
        # Φιλτράρισμα μόνο ζωηρών/ιδιαιτεροτήτων προς τοποθέτηση
        to_place = []
        for name in scope:
            name_row = df[df["ΟΝΟΜΑ"] == name]
            if name_row.empty:
                continue
            
            placed = pd.notna(name_row[step1_col].iloc[0])
            is_energetic = normalize_yes_no(name_row["ΖΩΗΡΟΣ"].iloc[0])
            is_special = normalize_yes_no(name_row["ΙΔΙΑΙΤΕΡΟΤΗΤΑ"].iloc[0])
            
            if not placed and (is_energetic or is_special):
                to_place.append(name)
        
        if not to_place:
            result_df = df.copy()
            step2_col = step1_col.replace("ΒΗΜΑ1", "ΒΗΜΑ2")
            result_df[step2_col] = result_df[step1_col]
            return [("default", result_df, {"ped_conflicts": 0, "broken": 0, "penalty": 0})]
        
        # Απλή στρατηγική τοποθέτησης
        solutions = []
        for attempt in range(max_results):
            result_df = df.copy()
            step2_col = step1_col.replace("ΒΗΜΑ1", "ΒΗΜΑ2")
            result_df[step2_col] = result_df[step1_col]
            
            # Τοποθέτηση με round-robin
            for i, name in enumerate(to_place):
                target_class = class_labels[i % len(class_labels)]
                result_df.loc[result_df["ΟΝΟΜΑ"] == name, step2_col] = target_class
            
            # Υπολογισμός μετρικών
            ped_conflicts = self._count_pedagogical_conflicts(result_df, step2_col)
            penalty = self._calculate_penalty(result_df, step2_col, len(class_labels))
            
            solutions.append((f"solution_{attempt+1}", result_df, {
                "ped_conflicts": ped_conflicts,
                "broken": 0,
                "penalty": penalty
            }))
        
        return solutions
    
    def _count_pedagogical_conflicts(self, df: pd.DataFrame, class_col: str) -> int:
        """Μέτρηση παιδαγωγικών συγκρούσεων."""
        conflicts = 0
        for class_name in df[class_col].dropna().unique():
            class_students = df[df[class_col] == class_name]
            
            for i, row1 in class_students.iterrows():
                for j, row2 in class_students.iterrows():
                    if i >= j:
                        continue
                    
                    e1 = normalize_yes_no(row1.get("ΖΩΗΡΟΣ", False))
                    s1 = normalize_yes_no(row1.get("ΙΔΙΑΙΤΕΡΟΤΗΤΑ", False))
                    e2 = normalize_yes_no(row2.get("ΖΩΗΡΟΣ", False))
                    s2 = normalize_yes_no(row2.get("ΙΔΙΑΙΤΕΡΟΤΗΤΑ", False))
                    
                    if self._calculate_conflict_penalty(e1, s1, e2, s2) > 0:
                        conflicts += 1
        
        return conflicts
    
    def _calculate_conflict_penalty(self, a_energetic: bool, a_special: bool, 
                                   b_energetic: bool, b_special: bool) -> int:
        """Υπολογισμός ποινής παιδαγωγικής σύγκρουσης."""
        if a_special and b_special:
            return 5
        if (a_special and b_energetic) or (b_special and a_energetic):
            return 4
        if a_energetic and b_energetic:
            return 3
        return 0
    
    def _calculate_penalty(self, df: pd.DataFrame, class_col: str, num_classes: int) -> int:
        """Υπολογισμός συνολικής ποινής."""
        penalty = 0
        
        # Ποινή ανισορροπίας πληθυσμού
        class_sizes = []
        for i in range(num_classes):
            class_name = f"Α{i+1}"
            size = int((df[class_col] == class_name).sum())
            class_sizes.append(size)
        
        if class_sizes:
            pop_diff = max(class_sizes) - min(class_sizes)
            penalty += max(0, pop_diff - 1) * PENALTY_WEIGHTS["population_imbalance"]
        
        return penalty

# =============================================================================
# ΒΗΜΑ 3: ΑΜΟΙΒΑΙΕΣ ΦΙΛΙΕΣ (ΔΥΑΔΕΣ)
# =============================================================================

class Step3Processor:
    """Επεξεργαστής αμοιβαίων φιλιών."""
    
    def process_mutual_friendships(self, df: pd.DataFrame, step2_col: str,
                                 num_classes: Optional[int] = None, 
                                 max_results: int = 5) -> List[Tuple[str, pd.DataFrame, Dict]]:
        """Τοποθέτηση μαθητών με αμοιβαίες φιλίες."""
        
        if num_classes is None:
            num_classes = compute_optimal_classes(len(df))
        
        # Εντοπισμός αμοιβαίων δυάδων
        mutual_pairs = self._find_mutual_pairs(df)
        if not mutual_pairs:
            result_df = df.copy()
            step3_col = step2_col.replace("ΒΗΜΑ2", "ΒΗΜΑ3")
            result_df[step3_col] = result_df[step2_col]
            return [("no_pairs", result_df, {"broken": 0, "penalty": 0})]
        
        # Παραγωγή λύσεων
        solutions = self._generate_friendship_solutions(df, step2_col, mutual_pairs, num_classes, max_results)
        
        return solutions
    
    def _find_mutual_pairs(self, df: pd.DataFrame) -> Set[Tuple[str, str]]:
        """Εντοπισμός αμοιβαίων φιλιών."""
        if "ΦΙΛΟΙ" not in df.columns:
            return set()
        
        pairs = set()
        names = df["ΟΝΟΜΑ"].astype(str).tolist()
        
        for i, name_a in enumerate(names):
            for name_b in names[i+1:]:
                if self._are_mutual_friends(df, name_a, name_b):
                    pairs.add(tuple(sorted([name_a, name_b])))
        
        return pairs
    
    def _are_mutual_friends(self, df: pd.DataFrame, name_a: str, name_b: str) -> bool:
        """Έλεγχος αμοιβαίας φιλίας."""
        row_a = df[df["ΟΝΟΜΑ"] == name_a]
        row_b = df[df["ΟΝΟΜΑ"] == name_b]
        
        if row_a.empty or row_b.empty:
            return False
        
        friends_a = set(parse_friends_list(row_a.iloc[0].get("ΦΙΛΟΙ", "")))
        friends_b = set(parse_friends_list(row_b.iloc[0].get("ΦΙΛΟΙ", "")))
        
        return name_b in friends_a and name_a in friends_b
    
    def _generate_friendship_solutions(self, df: pd.DataFrame, step2_col: str, 
                                     mutual_pairs: Set[Tuple[str, str]], num_classes: int, 
                                     max_results: int) -> List[Tuple[str, pd.DataFrame, Dict]]:
        """Παραγωγή λύσεων φιλιών."""
        
        solutions = []
        for attempt in range(max_results):
            candidate_df = df.copy()
            step3_col = step2_col.replace("ΒΗΜΑ2", "ΒΗΜΑ3")
            candidate_df[step3_col] = candidate_df[step2_col]
            
            # Προσπάθεια τοποθέτησης φίλων μαζί
            placed_friends = 0
            for friend_a, friend_b in mutual_pairs:
                if self._try_place_friends_together(candidate_df, step3_col, friend_a, friend_b, num_classes):
                    placed_friends += 1
            
            # Υπολογισμός μετρικών
            broken = self._count_broken_friendships(candidate_df, step3_col, mutual_pairs)
            penalty = self._calculate_penalty(candidate_df, step3_col, num_classes)
            
            solutions.append((f"attempt_{attempt+1}", candidate_df, {
                "broken": broken,
                "penalty": penalty,
                "placed_friends": placed_friends
            }))
        
        # Ταξινόμηση λύσεων
        solutions.sort(key=lambda x: (x[2]["broken"], x[2]["penalty"]))
        
        return solutions[:max_results]
    
    def _try_place_friends_together(self, df: pd.DataFrame, step3_col: str, 
                                   friend_a: str, friend_b: str, num_classes: int) -> bool:
        """Προσπάθεια τοποθέτησης φίλων μαζί."""
        
        # Έλεγχος τρέχουσας κατάστασης
        a_class = df[df["ΟΝΟΜΑ"] == friend_a][step3_col].iloc[0] if not df[df["ΟΝΟΜΑ"] == friend_a].empty else None
        b_class = df[df["ΟΝΟΜΑ"] == friend_b][step3_col].iloc[0] if not df[df["ΟΝΟΜΑ"] == friend_b].empty else None
        
        # Αν και οι δυο είναι ήδη τοποθετημένοι
        if pd.notna(a_class) and pd.notna(b_class):
            return a_class == b_class
        
        # Αν ένας είναι τοποθετημένος, τοποθέτησε τον άλλον στο ίδιο τμήμα
        if pd.notna(a_class) and pd.isna(b_class):
            if self._can_place_in_class(df, step3_col, friend_b, a_class):
                df.loc[df["ΟΝΟΜΑ"] == friend_b, step3_col] = a_class
                return True
        
        if pd.notna(b_class) and pd.isna(a_class):
            if self._can_place_in_class(df, step3_col, friend_a, b_class):
                df.loc[df["ΟΝΟΜΑ"] == friend_a, step3_col] = b_class
                return True
        
        # Αν κανένας δεν είναι τοποθετημένος, βρες το καλύτερο τμήμα
        if pd.isna(a_class) and pd.isna(b_class):
            for i in range(num_classes):
                class_name = f"Α{i+1}"
                if (self._can_place_in_class(df, step3_col, friend_a, class_name) and 
                    self._can_place_in_class(df, step3_col, friend_b, class_name)):
                    df.loc[df["ΟΝΟΜΑ"] == friend_a, step3_col] = class_name
                    df.loc[df["ΟΝΟΜΑ"] == friend_b, step3_col] = class_name
                    return True
        
        return False
    
    def _can_place_in_class(self, df: pd.DataFrame, step3_col: str, student_name: str, class_name: str) -> bool:
        """Έλεγχος αν μπορεί να τοποθετηθεί σε τμήμα."""
        current_size = (df[step3_col] == class_name).sum()
        return current_size < MAX_STUDENTS_PER_CLASS
    
    def _count_broken_friendships(self, df: pd.DataFrame, step3_col: str, 
                                mutual_pairs: Set[Tuple[str, str]]) -> int:
        """Μέτρηση σπασμένων φιλιών."""
        broken = 0
        name_to_class = {row["ΟΝΟΜΑ"]: row[step3_col] for _, row in df.iterrows() if pd.notna(row[step3_col])}
        
        for friend_a, friend_b in mutual_pairs:
            class_a = name_to_class.get(friend_a)
            class_b = name_to_class.get(friend_b)
            
            if class_a is None or class_b is None or class_a != class_b:
                broken += 1
        
        return broken
    
    def _calculate_penalty(self, df: pd.DataFrame, step3_col: str, num_classes: int) -> int:
        """Υπολογισμός ποινής για βήμα 3."""
        penalty = 0
        
        # Βασικές ποινές ισορροπίας
        pop_counts = []
        for i in range(num_classes):
            class_name = f"Α{i+1}"
            pop_counts.append((df[step3_col] == class_name).sum())
        
        if pop_counts:
            penalty += max(0, max(pop_counts) - min(pop_counts) - 2)
        
        return penalty

# =============================================================================
# ΒΗΜΑ 4: ΦΙΛΙΚΕΣ ΟΜΑΔΕΣ
# =============================================================================

class Step4Processor:
    """Επεξεργαστής φιλικών ομάδων."""
    
    def process_friendship_groups(self, df: pd.DataFrame, step3_col: str, 
                                num_classes: Optional[int] = None, 
                                max_results: int = 5) -> List[Tuple[str, pd.DataFrame, Dict]]:
        """Τοποθέτηση φιλικών ομάδων."""
        
        if num_classes is None:
            num_classes = compute_optimal_classes(len(df))
        
        # Δημιουργία φιλικών ομάδων (μόνο δυάδες)
        groups = self._create_friendship_groups(df, step3_col)
        if not groups:
            result_df = df.copy()
            step4_col = step3_col.replace("ΒΗΜΑ3", "ΒΗΜΑ4")
            result_df[step4_col] = result_df[step3_col]
            return [("no_groups", result_df, {"penalty": 0})]
        
        # Εφαρμογή στρατηγικής τοποθέτησης
        results = self._apply_group_strategy(df, step3_col, groups, num_classes, max_results)
        
        return results[:max_results]
    
    def _create_friendship_groups(self, df: pd.DataFrame, step3_col: str) -> List[List[str]]:
        """Δημιουργία φιλικών ομάδων (μόνο δυάδες)."""
        if "ΦΙΛΟΙ" not in df.columns:
            return []
        
        unassigned = df[df[step3_col].isna()].copy()
        
        # Φιλτράρισμα μαθητών με φίλους
        unassigned = unassigned[
            unassigned['ΦΙΛΟΙ'].map(lambda x: len(parse_friends_list(x)) > 0)
        ]
        
        if len(unassigned) == 0:
            return []
        
        names = list(unassigned['ΟΝΟΜΑ'].astype(str).unique())
        
        # Δημιουργία μόνο δυάδων
        used = set()
        groups = []
        
        for pair in itertools.combinations(names, 2):
            if set(pair) & used:
                continue
            if self._is_fully_mutual_pair(df, pair[0], pair[1]):
                groups.append(list(pair))
                used |= set(pair)
        
        return groups
    
    def _is_fully_mutual_pair(self, df: pd.DataFrame, name_a: str, name_b: str) -> bool:
        """Έλεγχος πλήρους αμοιβαίας φιλίας δυάδας."""
        try:
            row_a = df[df['ΟΝΟΜΑ'] == name_a]
            row_b = df[df['ΟΝΟΜΑ'] == name_b]
            
            if row_a.empty or row_b.empty:
                return False
            
            friends_a = set(parse_friends_list(row_a.iloc[0].get('ΦΙΛΟΙ', '')))
            friends_b = set(parse_friends_list(row_b.iloc[0].get('ΦΙΛΟΙ', '')))
            
            return name_b in friends_a and name_a in friends_b
            
        except Exception:
            return False
    
    def _apply_group_strategy(self, df: pd.DataFrame, step3_col: str, groups: List[List[str]], 
                            num_classes: int, max_results: int) -> List[Tuple[str, pd.DataFrame, Dict]]:
        """Εφαρμογή στρατηγικής ομάδων."""
        
        class_labels = [f"Α{i+1}" for i in range(num_classes)]
        solutions = []
        
        # Απλή στρατηγική - τοποθέτηση ομάδων
        for attempt in range(max_results):
            result_df = self._apply_group_placement(df, step3_col, groups, class_labels, attempt)
            penalty = self._calculate_group_penalty(result_df, step3_col.replace("ΒΗΜΑ3", "ΒΗΜΑ4"), num_classes)
            
            solutions.append((f"solution_{attempt+1}", result_df, {"penalty": penalty}))
        
        # Ταξινόμηση κατά penalty
        solutions.sort(key=lambda x: x[2]["penalty"])
        
        return solutions
    
    def _apply_group_placement(self, df: pd.DataFrame, step3_col: str, groups: List[List[str]], 
                             class_labels: List[str], seed: int) -> pd.DataFrame:
        """Εφαρμογή τοποθέτησης ομάδων."""
        result_df = df.copy()
        step4_col = step3_col.replace("ΒΗΜΑ3", "ΒΗΜΑ4")
        result_df[step4_col] = result_df[step3_col]
        
        # Τοποθέτηση ομάδων με rotation
        for i, group in enumerate(groups):
            target_class = class_labels[(i + seed) % len(class_labels)]
            
            # Έλεγχος χωρητικότητας
            current_size = (result_df[step4_col] == target_class).sum()
            if current_size + len(group) <= MAX_STUDENTS_PER_CLASS:
                mask = result_df['ΟΝΟΜΑ'].isin(group)
                result_df.loc[mask, step4_col] = target_class
        
        return result_df
    
    def _calculate_group_penalty(self, df: pd.DataFrame, step4_col: str, num_classes: int) -> int:
        """Υπολογισμός ποινής ομάδων."""
        penalty = 0
        
        # Ποινές ισορροπίας
        for i in range(num_classes):
            class_name = f"Α{i+1}"
            class_df = df[df[step4_col] == class_name]
            
            total = len(class_df)
            
            # Ποινή για υπερχείλιση
            if total > MAX_STUDENTS_PER_CLASS:
                penalty += (total - MAX_STUDENTS_PER_CLASS) * 10
        
        return penalty

# =============================================================================
# ΒΗΜΑ 5: ΥΠΟΛΟΙΠΟΙ ΜΑΘΗΤΕΣ
# =============================================================================

class Step5Processor:
    """Επεξεργαστής υπόλοιπων μαθητών."""
    
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
    
    def process_remaining_students(self, df: pd.DataFrame, step4_col: str, 
                                 num_classes: Optional[int] = None) -> Tuple[pd.DataFrame, int]:
        """Τοποθέτηση υπόλοιπων μη-τοποθετημένων μαθητών."""
        df = df.copy()
        
        if num_classes is None:
            num_classes = compute_optimal_classes(len(df))
        
        class_labels = [f"Α{i+1}" for i in range(num_classes)]
        
        # Εντοπισμός μαθητών για βήμα 5
        step5_students = self._identify_step5_students(df, step4_col)
        
        # Τοποθέτηση με στρατηγική ισοκατανομής
        for student_name in step5_students:
            self._assign_remaining_student(df, student_name, step4_col, class_labels)
        
        # Δημιουργία στήλης βήματος 5
        step5_col = step4_col.replace("ΒΗΜΑ4", "ΒΗΜΑ5")
        df[step5_col] = df[step4_col]
        
        # Υπολογισμός penalty
        penalty = self._calculate_step5_penalty(df, step5_col, num_classes)
        
        return df, penalty
    
    def _identify_step5_students(self, df: pd.DataFrame, step4_col: str) -> List[str]:
        """Εντοπισμός μαθητών για επεξεργασία στο βήμα 5."""
        step5_mask = df[step4_col].isna()
        return df[step5_mask]["ΟΝΟΜΑ"].astype(str).tolist()
    
    def _assign_remaining_student(self, df: pd.DataFrame, student_name: str, 
                                step4_col: str, class_labels: List[str]) -> None:
        """Κατανομή μεμονωμένου μαθητή."""
        student_row = df[df["ΟΝΟΜΑ"] == student_name]
        if student_row.empty:
            return
        
        # Στρατηγική: ελάχιστος πληθυσμός
        class_populations = {label: (df[step4_col] == label).sum() for label in class_labels}
        
        min_population = min(class_populations.values())
        candidate_classes = [
            label for label, pop in class_populations.items() 
            if pop == min_population and pop < MAX_STUDENTS_PER_CLASS
        ]
        
        if candidate_classes:
            chosen_class = random.choice(candidate_classes)
            df.loc[df["ΟΝΟΜΑ"] == student_name, step4_col] = chosen_class
    
    def _calculate_step5_penalty(self, df: pd.DataFrame, step5_col: str, num_classes: int) -> int:
        """Υπολογισμός penalty βήματος 5."""
        penalty = 0
        
        # Ποινή ανισορροπίας πληθυσμού
        class_sizes = []
        for i in range(num_classes):
            class_name = f"Α{i+1}"
            size = int((df[step5_col] == class_name).sum())
            class_sizes.append(size)
        
        if class_sizes:
            pop_diff = max(class_sizes) - min(class_sizes)
            penalty += max(0, pop_diff - 1) * PENALTY_WEIGHTS["population_imbalance"]
        
        return penalty

# =============================================================================
# ΒΗΜΑ 6: ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ & ΔΙΟΡΘΩΣΕΙΣ (ΒΕΛΤΙΩΜΕΝΟ)
# =============================================================================

class Step6Processor:
    """Επεξεργαστής τελικού ελέγχου και διορθώσεων."""
    
    def apply_final_check(self, df: pd.DataFrame, step5_col: str, 
                         num_classes: Optional[int] = None, max_iter: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Εφαρμογή τελικού ελέγχου και διορθώσεων."""
        
        if num_classes is None:
            num_classes = compute_optimal_classes(len(df))
        
        result_df = df.copy()
        step6_col = step5_col.replace("ΒΗΜΑ5", "ΒΗΜΑ6")
        result_df[step6_col] = result_df[step5_col]
        
        # Audit στήλες
        result_df["ΒΗΜΑ6_ΚΙΝΗΣΗ"] = None
        result_df["ΑΙΤΙΑ_ΑΛΛΑΓΗΣ"] = None
        
        iterations = 0
        status = "VALID"
        
        # Κύριος αλγόριθμος βελτίωσης
        while iterations < max_iter:
            iterations += 1
            metrics = self._calculate_metrics(result_df, step6_col)
            
            # Έλεγχος στόχων
            within_targets = (
                metrics["pop_diff"] <= 2 and
                metrics["gender_diff"] <= 3 and 
                metrics["lang_diff"] <= 3
            )
            
            if within_targets:
                break
            
            # ΒΕΛΤΙΩΜΕΝΗ προσπάθεια βελτίωσης
            improved = self._attempt_improvement(result_df, step6_col, metrics, iterations)
            
            if not improved:
                break
        
        # Τελικός έλεγχος
        final_metrics = self._calculate_metrics(result_df, step6_col)
        final_penalty = self._calculate_final_penalty(result_df, step6_col)
        
        summary = {
            "iterations": iterations,
            "final_metrics": final_metrics,
            "final_penalty": final_penalty,
            "status": status
        }
        
        return result_df, summary
    
    def _calculate_metrics(self, df: pd.DataFrame, step6_col: str) -> Dict[str, int]:
        """Υπολογισμός μετρικών."""
        classes = df[step6_col].dropna().unique()
        
        # Πληθυσμιακές διαφορές
        class_sizes = [int((df[step6_col] == c).sum()) for c in classes]
        pop_diff = max(class_sizes) - min(class_sizes) if class_sizes else 0
        
        # Διαφορές φύλου
        boys_counts = [int(((df[step6_col] == c) & (df["ΦΥΛΟ"] == "Α")).sum()) for c in classes]
        girls_counts = [int(((df[step6_col] == c) & (df["ΦΥΛΟ"] == "Κ")).sum()) for c in classes]
        
        boys_diff = max(boys_counts) - min(boys_counts) if boys_counts else 0
        girls_diff = max(girls_counts) - min(girls_counts) if girls_counts else 0
        gender_diff = max(boys_diff, girls_diff)
        
        # Διαφορές γλώσσας
        lang_counts = [int(((df[step6_col] == c) & (df["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"] == True)).sum()) for c in classes]
        lang_diff = max(lang_counts) - min(lang_counts) if lang_counts else 0
        
        return {
            "pop_diff": pop_diff,
            "gender_diff": gender_diff,
            "lang_diff": lang_diff
        }
    
    def _attempt_improvement(self, df: pd.DataFrame, step6_col: str, metrics: Dict[str, int], iteration: int) -> bool:
        """ΒΕΛΤΙΩΜΕΝΗ στρατηγική βελτίωσης με ιεραρχημένα κριτήρια."""
        
        # Ιεράρχηση προτεραιότητας: πληθυσμός (≤2) → φύλο → γλώσσα
        if metrics["pop_diff"] > 2:
            return self._balance_population(df, step6_col, iteration)
        elif metrics["gender_diff"] > 3:
            return self._balance_gender(df, step6_col, iteration)
        elif metrics["lang_diff"] > 3:
            return self._balance_language(df, step6_col, iteration)
        
        return False

    def _balance_population(self, df: pd.DataFrame, step6_col: str, iteration: int) -> bool:
        """Εξισορρόπηση πληθυσμού με προστασία φιλιών."""
        
        classes = df[step6_col].dropna().unique()
        class_sizes = {cls: (df[step6_col] == cls).sum() for cls in classes}
        
        # Βρες μεγαλύτερη και μικρότερη τάξη
        max_class = max(class_sizes.keys(), key=lambda x: class_sizes[x])
        min_class = min(class_sizes.keys(), key=lambda x: class_sizes[x])
        
        if class_sizes[max_class] - class_sizes[min_class] <= 2:
            return False
        
        # Βρες κατάλληλο μαθητή για μετακίνηση (χωρίς σπάσιμο φιλιών)
        candidates = df[df[step6_col] == max_class]["ΟΝΟΜΑ"].tolist()
        
        for student in candidates:
            if not self._would_break_friendships(df, student, min_class, step6_col):
                df.loc[df["ΟΝΟΜΑ"] == student, step6_col] = min_class
                df.loc[df["ΟΝΟΜΑ"] == student, "ΒΗΜΑ6_ΚΙΝΗΣΗ"] = f"POP_BALANCE_{iteration}"
                df.loc[df["ΟΝΟΜΑ"] == student, "ΑΙΤΙΑ_ΑΛΛΑΓΗΣ"] = "Population Balance"
                return True
        
        return False

    def _balance_gender(self, df: pd.DataFrame, step6_col: str, iteration: int) -> bool:
        """Εξισορρόπηση φύλου."""
        
        classes = df[step6_col].dropna().unique()
        
        for gender in ["Α", "Κ"]:
            gender_counts = {}
            for cls in classes:
                count = ((df[step6_col] == cls) & (df["ΦΥΛΟ"] == gender)).sum()
                gender_counts[cls] = count
            
            if max(gender_counts.values()) - min(gender_counts.values()) > 3:
                max_class = max(gender_counts.keys(), key=lambda x: gender_counts[x])
                min_class = min(gender_counts.keys(), key=lambda x: gender_counts[x])
                
                # Βρες μαθητή του συγκεκριμένου φύλου
                candidates = df[(df[step6_col] == max_class) & (df["ΦΥΛΟ"] == gender)]["ΟΝΟΜΑ"].tolist()
                
                for student in candidates:
                    if not self._would_break_friendships(df, student, min_class, step6_col):
                        df.loc[df["ΟΝΟΜΑ"] == student, step6_col] = min_class
                        df.loc[df["ΟΝΟΜΑ"] == student, "ΒΗΜΑ6_ΚΙΝΗΣΗ"] = f"GENDER_BALANCE_{iteration}"
                        df.loc[df["ΟΝΟΜΑ"] == student, "ΑΙΤΙΑ_ΑΛΛΑΓΗΣ"] = f"Gender Balance ({gender})"
                        return True
        
        return False

    def _balance_language(self, df: pd.DataFrame, step6_col: str, iteration: int) -> bool:
        """Εξισορρόπηση γλωσσικής γνώσης."""
        
        classes = df[step6_col].dropna().unique()
        lang_counts = {}
        
        for cls in classes:
            count = ((df[step6_col] == cls) & (df["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"] == True)).sum()
            lang_counts[cls] = count
        
        if max(lang_counts.values()) - min(lang_counts.values()) > 3:
            max_class = max(lang_counts.keys(), key=lambda x: lang_counts[x])
            min_class = min(lang_counts.keys(), key=lambda x: lang_counts[x])
            
            # Βρες μαθητή με καλή γνώση ελληνικών
            candidates = df[(df[step6_col] == max_class) & 
                           (df["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"] == True)]["ΟΝΟΜΑ"].tolist()
            
            for student in candidates:
                if not self._would_break_friendships(df, student, min_class, step6_col):
                    df.loc[df["ΟΝΟΜΑ"] == student, step6_col] = min_class
                    df.loc[df["ΟΝΟΜΑ"] == student, "ΒΗΜΑ6_ΚΙΝΗΣΗ"] = f"LANG_BALANCE_{iteration}"
                    df.loc[df["ΟΝΟΜΑ"] == student, "ΑΙΤΙΑ_ΑΛΛΑΓΗΣ"] = "Language Balance"
                    return True
        
        return False

    def _would_break_friendships(self, df: pd.DataFrame, student_name: str, target_class: str, class_col: str) -> bool:
        """Έλεγχος αν η μετακίνηση θα σπάσει φιλίες."""
        
        if "ΦΙΛΟΙ" not in df.columns:
            return False
        
        student_row = df[df["ΟΝΟΜΑ"] == student_name]
        if student_row.empty:
            return False
        
        friends = parse_friends_list(student_row.iloc[0].get("ΦΙΛΟΙ", ""))
        
        for friend in friends:
            friend_row = df[df["ΟΝΟΜΑ"] == friend]
            if not friend_row.empty:
                friend_class = friend_row.iloc[0].get(class_col)
                if pd.notna(friend_class) and friend_class != target_class:
                    # Έλεγχος αμοιβαιότητας φιλίας
                    if are_mutual_friends_safe(df, student_name, friend):
                        return True  # Θα σπάσει αμοιβαία φιλία
        
        return False
    
    def _calculate_final_penalty(self, df: pd.DataFrame, step6_col: str) -> int:
        """Υπολογισμός τελικής ποινής."""
        penalty = 0
        classes = df[step6_col].dropna().unique()
        
        # Ποινή πληθυσμού
        class_sizes = [int((df[step6_col] == c).sum()) for c in classes]
        if class_sizes:
            pop_diff = max(class_sizes) - min(class_sizes)
            penalty += max(0, pop_diff - 1) * 3
        
        return penalty

# =============================================================================
# ΒΗΜΑ 7: ΒΑΘΜΟΛΟΓΗΣΗ & ΕΠΙΛΟΓΗ ΒΕΛΤΙΣΤΟΥ
# =============================================================================

class Step7Processor:
    """Επεξεργαστής βαθμολόγησης και επιλογής βέλτιστου σεναρίου."""
    
    def score_scenarios(self, df: pd.DataFrame, scenario_cols: List[str]) -> Dict[str, Any]:
        """Βαθμολόγηση σεναρίων."""
        scores = []
        
        for col in scenario_cols:
            if col not in df.columns:
                continue
            
            score = self._score_one_scenario(df, col)
            scores.append(score)
        
        if not scores:
            return {"best": None, "scores": []}
        
        # Ταξινόμηση και επιλογή καλύτερου
        scores_sorted = sorted(scores, key=lambda s: s["total_score"])
        best = scores_sorted[0]
        
        return {"best": best, "scores": scores_sorted}
    
    def _score_one_scenario(self, df: pd.DataFrame, scenario_col: str) -> Dict[str, Any]:
        """Βαθμολόγηση ενός σεναρίου."""
        
        # Μέτρηση ανά τμήμα
        classes = df[scenario_col].dropna().unique()
        pop_counts = {}
        boys_counts = {}
        girls_counts = {}
        good_greek_counts = {}
        
        for c in classes:
            class_df = df[df[scenario_col] == c]
            pop_counts[c] = len(class_df)
            boys_counts[c] = int((class_df["ΦΥΛΟ"] == "Α").sum())
            girls_counts[c] = int((class_df["ΦΥΛΟ"] == "Κ").sum())
            good_greek_counts[c] = int((class_df["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"] == True).sum())
        
        # Υπολογισμός penalties
        population_penalty = self._calculate_population_penalty(pop_counts)
        gender_penalty = self._calculate_gender_penalty(boys_counts, girls_counts)
        greek_penalty = self._calculate_greek_penalty(good_greek_counts)
        
        total_score = population_penalty + gender_penalty + greek_penalty
        
        return {
            "scenario_col": scenario_col,
            "population_counts": pop_counts,
            "boys_counts": boys_counts,
            "girls_counts": girls_counts,
            "good_greek_counts": good_greek_counts,
            "population_penalty": population_penalty,
            "gender_penalty": gender_penalty,
            "greek_penalty": greek_penalty,
            "total_score": total_score
        }
    
    def _calculate_population_penalty(self, counts: Dict[str, int]) -> int:
        """Υπολογισμός ποινής πληθυσμού."""
        if not counts:
            return 0
        
        values = list(counts.values())
        penalty = 0
        
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                diff = abs(values[i] - values[j])
                if diff > 1:
                    penalty += (diff - 1) * 3
        
        return penalty
    
    def _calculate_gender_penalty(self, boys_counts: Dict[str, int], girls_counts: Dict[str, int]) -> int:
        """Υπολογισμός ποινής φύλου."""
        penalty = 0
        
        # Ποινή αγοριών
        if boys_counts:
            boys_values = list(boys_counts.values())
            for i in range(len(boys_values)):
                for j in range(i+1, len(boys_values)):
                    diff = abs(boys_values[i] - boys_values[j])
                    if diff > 1:
                        penalty += (diff - 1) * 2
        
        # Ποινή κοριτσιών
        if girls_counts:
            girls_values = list(girls_counts.values())
            for i in range(len(girls_values)):
                for j in range(i+1, len(girls_values)):
                    diff = abs(girls_values[i] - girls_values[j])
                    if diff > 1:
                        penalty += (diff - 1) * 2
        
        return penalty
    
    def _calculate_greek_penalty(self, counts: Dict[str, int]) -> int:
        """Υπολογισμός ποινής γνώσης ελληνικών."""
        if not counts:
            return 0
        
        values = list(counts.values())
        penalty = 0
        
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                diff = abs(values[i] - values[j])
                if diff > 2:
                    penalty += (diff - 2) * 1
        
        return penalty

# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================

class StudentAssignmentSystem:
    """Κύριο σύστημα κατανομής μαθητών."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        
        # Processors για κάθε βήμα
        self.step1_processor = Step1ImmutableProcessor()
        self.step2_processor = Step2Processor()
        self.step3_processor = Step3Processor()
        self.step4_processor = Step4Processor()
        self.step5_processor = Step5Processor(random_seed)
        self.step6_processor = Step6Processor()
        self.step7_processor = Step7Processor()
        
        # Statistics generator
        self.stats_generator = StatisticsGenerator()
    
    def process_complete_assignment(self, df: pd.DataFrame, 
                                  num_classes: Optional[int] = None,
                                  max_scenarios: int = 3) -> Dict[str, Any]:
        """Εκτέλεση πλήρους κατανομής 7 βημάτων."""
        
        print("=== ΕΚΚΙΝΗΣΗ ΣΥΣΤΗΜΑΤΟΣ ΚΑΤΑΝΟΜΗΣ ΜΑΘΗΤΩΝ ===")
        
        # Προετοιμασία
        df_processed = normalize_dataframe(df)
        is_valid, missing_cols = validate_required_columns(df_processed)
        
        if not is_valid:
            raise ValueError(f"Λείπουν απαιτούμενες στήλες: {missing_cols}")
        
        if num_classes is None:
            num_classes = compute_optimal_classes(len(df_processed))
        
        print(f"Επεξεργασία {len(df_processed)} μαθητών σε {num_classes} τμήματα")
        
        results = {"original_df": df_processed}
        
        try:
            # ΒΗΜΑ 1: Παιδιά εκπαιδευτικών (immutable)
            print("\n🔄 ΒΗΜΑ 1: Παιδιά εκπαιδευτικών")
            step1_scenarios = self.step1_processor.create_scenarios(df_processed, num_classes)
            df_step1 = self.step1_processor.apply_to_dataframe(df_processed)
            results["step1"] = {"df": df_step1, "scenarios": step1_scenarios}
            print(f"✅ Δημιουργήθηκαν {len(step1_scenarios)} σενάρια")
            
            # ΒΗΜΑ 2: Ζωηροί & Ιδιαιτερότητες
            print("\n🔄 ΒΗΜΑ 2: Ζωηροί & Ιδιαιτερότητες")
            step1_col = "ΒΗΜΑ1_ΣΕΝΑΡΙΟ_1" if step1_scenarios else "ΒΗΜΑ1_ΣΕΝΑΡΙΟ_1"
            step2_results = self.step2_processor.process_energetic_and_special(
                df_step1, step1_col, num_classes, max_scenarios
            )
            results["step2"] = step2_results
            print(f"✅ Παράχθηκαν {len(step2_results)} λύσεις")
            
            # Επιλογή καλύτερης λύσης από βήμα 2
            best_step2 = min(step2_results, key=lambda x: x[2]["penalty"])
            df_step2 = best_step2[1]
            step2_col = [col for col in df_step2.columns if col.startswith("ΒΗΜΑ2_")][0]
            
            # ΒΗΜΑ 3: Αμοιβαίες φιλίες
            print("\n🔄 ΒΗΜΑ 3: Αμοιβαίες φιλίες")
            step3_results = self.step3_processor.process_mutual_friendships(
                df_step2, step2_col, num_classes, max_scenarios
            )
            results["step3"] = step3_results
            print(f"✅ Παράχθηκαν {len(step3_results)} λύσεις")
            
            # Επιλογή καλύτερης λύσης από βήμα 3
            best_step3 = min(step3_results, key=lambda x: x[2]["broken"])
            df_step3 = best_step3[1]
            step3_col = [col for col in df_step3.columns if col.startswith("ΒΗΜΑ3_")][0]
            
            # ΒΗΜΑ 4: Φιλικές ομάδες
            print("\n🔄 ΒΗΜΑ 4: Φιλικές ομάδες")
            step4_results = self.step4_processor.process_friendship_groups(
                df_step3, step3_col, num_classes, max_scenarios
            )
            results["step4"] = step4_results
            print(f"✅ Παράχθηκαν {len(step4_results)} λύσεις")
            
            # Επιλογή καλύτερης λύσης από βήμα 4
            best_step4 = min(step4_results, key=lambda x: x[2]["penalty"])
            df_step4 = best_step4[1]
            step4_col = [col for col in df_step4.columns if col.startswith("ΒΗΜΑ4_")][0]
            
            # ΒΗΜΑ 5: Υπόλοιποι μαθητές
            print("\n🔄 ΒΗΜΑ 5: Υπόλοιποι μαθητές")
            df_step5, step5_penalty = self.step5_processor.process_remaining_students(
                df_step4, step4_col, num_classes
            )
            results["step5"] = {"df": df_step5, "penalty": step5_penalty}
            print(f"✅ Ολοκληρώθηκε με penalty: {step5_penalty}")
            
            step5_col = [col for col in df_step5.columns if col.startswith("ΒΗΜΑ5_")][0]
            
            # ΒΗΜΑ 6: Τελικός έλεγχος & διορθώσεις
            print("\n🔄 ΒΗΜΑ 6: Τελικός έλεγχος & διορθώσεις")
            df_step6, step6_summary = self.step6_processor.apply_final_check(
                df_step5, step5_col, num_classes
            )
            results["step6"] = {"df": df_step6, "summary": step6_summary}
            print(f"✅ Ολοκληρώθηκε σε {step6_summary['iterations']} επαναλήψεις")
            
            step6_col = [col for col in df_step6.columns if col.startswith("ΒΗΜΑ6_")][0]
            
            # ΒΗΜΑ 7: Βαθμολόγηση & επιλογή βέλτιστου
            print("\n🔄 ΒΗΜΑ 7: Βαθμολόγηση & επιλογή βέλτιστου")
            scenario_cols = [col for col in df_step6.columns if "ΒΗΜΑ" in col and "ΣΕΝΑΡΙΟ" in col]
            step7_results = self.step7_processor.score_scenarios(df_step6, scenario_cols)
            results["step7"] = step7_results
            
            if step7_results["best"]:
                print(f"✅ Βέλτιστο σενάριο: {step7_results['best']['scenario_col']} "
                      f"(Score: {step7_results['best']['total_score']})")
            
            results["final_df"] = df_step6
            results["status"] = "SUCCESS"
            
        except Exception as e:
            print(f"❌ Σφάλμα κατά την επεξεργασία: {e}")
            results["status"] = "ERROR"
            results["error"] = str(e)
        
        print("\n=== ΟΛΟΚΛΗΡΩΣΗ ΣΥΣΤΗΜΑΤΟΣ ===")
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "output") -> None:
        """Αποθήκευση αποτελεσμάτων σε Excel αρχεία."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if "final_df" in results:
            # Κύριο αρχείο με όλα τα βήματα
            main_file = output_path / "student_assignment_complete.xlsx"
            results["final_df"].to_excel(main_file, index=False)
            print(f"💾 Κύριο αρχείο: {main_file}")
            
            # Summary αρχείο
            self._save_summary(results, output_path / "assignment_summary.txt")
        
        print(f"📁 Όλα τα αρχεία αποθηκεύτηκαν στο: {output_path}")
    
    def _save_summary(self, results: Dict[str, Any], summary_file: Path) -> None:
        """Αποθήκευση συνοπτικών αποτελεσμάτων."""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ΣΥΝΟΠΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΚΑΤΑΝΟΜΗΣ ΜΑΘΗΤΩΝ\n")
            f.write("=" * 50 + "\n\n")
            
            if "original_df" in results:
                f.write(f"Συνολικοί μαθητές: {len(results['original_df'])}\n")
            
            if "step1" in results:
                scenarios = results["step1"]["scenarios"]
                f.write(f"Βήμα 1 σενάρια: {len(scenarios)}\n")
                for scenario in scenarios:
                    f.write(f"  - {scenario.column_name}: {scenario.description}\n")
            
            if "step7" in results and results["step7"]["best"]:
                best = results["step7"]["best"]
                f.write(f"\nΒέλτιστο σενάριο: {best['scenario_col']}\n")
                f.write(f"Συνολικό score: {best['total_score']}\n")
                f.write(f"Population penalty: {best['population_penalty']}\n")
                f.write(f"Gender penalty: {best['gender_penalty']}\n")
                f.write(f"Greek penalty: {best['greek_penalty']}\n")
            
            f.write(f"\nΚατάσταση: {results.get('status', 'UNKNOWN')}\n")


# =============================================================================
# ΣΤΑΤΙΣΤΙΚΕΣ & ΑΝΑΦΟΡΕΣ
# =============================================================================

class StatisticsGenerator:
    """Γεννήτρια στατιστικών και αναφορών."""
    
    def generate_statistics_table(self, df: pd.DataFrame, class_col: str = "ΤΜΗΜΑ") -> pd.DataFrame:
        """Δημιουργεί ενιαίο πίνακα στατιστικών ανά τμήμα."""
        
        if class_col not in df.columns:
            raise ValueError(f"Η στήλη {class_col} δεν υπάρχει")
        
        df_work = df.copy()
        
        # Κανονικοποίηση για στατιστικές (μόνο True values)
        boys = df_work[df_work["ΦΥΛΟ"] == "Α"].groupby(class_col).size()
        girls = df_work[df_work["ΦΥΛΟ"] == "Κ"].groupby(class_col).size()
        
        # Μόνο Ν (True) values για characteristics
        educators = df_work[df_work["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True].groupby(class_col).size()
        energetic = df_work[df_work["ΖΩΗΡΟΣ"] == True].groupby(class_col).size()
        special = df_work[df_work["ΙΔΙΑΙΤΕΡΟΤΗΤΑ"] == True].groupby(class_col).size()
        greek = df_work[df_work["ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"] == True].groupby(class_col).size()
        total = df_work.groupby(class_col).size()
        
        # Ενοποίηση πίνακα
        stats = pd.DataFrame({
            "ΑΓΟΡΙΑ": boys,
            "ΚΟΡΙΤΣΙΑ": girls,
            "ΕΚΠΑΙΔΕΥΤΙΚΟΙ": educators,
            "ΖΩΗΡΟΙ": energetic,
            "ΙΔΙΑΙΤΕΡΟΤΗΤΑ": special,
            "ΓΝΩΣΗ_ΕΛΛ": greek,
            "ΣΥΝΟΛΟ": total
        }).fillna(0).astype(int)
        
        # Ταξινόμηση τμημάτων (Α1, Α2, ...)
        if stats.index.astype(str).str.match(r'^Α\d+).all():
            stats = stats.sort_index(key=lambda x: x.str.extract(r'(\d+)', expand=False).astype(int))
        else:
            stats = stats.sort_index()
        
        return stats

# =============================================================================
# DEBUGGING & TESTING UTILITIES (ΔΙΟΡΘΩΜΕΝΕΣ)
# =============================================================================

class SystemDebugger:
    """Εργαλεία debugging για το σύστημα κατανομής."""
    
    def __init__(self, system: 'StudentAssignmentSystem'):
        self.system = system
    
    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Επικύρωση εισερχόμενων δεδομένων."""
        
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
            validation["stats"]["columns"] = list(df.columns)
            
            # Έλεγχος απαιτούμενων στηλών
            required_cols = ["ΟΝΟΜΑ", "ΦΥΛΟ", "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ", "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                validation["errors"].append(f"Λείπουν απαιτούμενες στήλες: {missing_cols}")
                validation["is_valid"] = False
            
            # Έλεγχος δεδομένων
            if "ΟΝΟΜΑ" in df.columns:
                duplicate_names = df["ΟΝΟΜΑ"].duplicated().sum()
                if duplicate_names > 0:
                    validation["warnings"].append(f"Βρέθηκαν {duplicate_names} διπλότυπα ονόματα")
                
                empty_names = df["ΟΝΟΜΑ"].isna().sum()
                if empty_names > 0:
                    validation["errors"].append(f"Βρέθηκαν {empty_names} κενά ονόματα")
                    validation["is_valid"] = False
            
            # Έλεγχος φύλου
            if "ΦΥΛΟ" in df.columns:
                boys = int((df["ΦΥΛΟ"] == "Α").sum())
                girls = int((df["ΦΥΛΟ"] == "Κ").sum())
                invalid_gender = len(df) - boys - girls
                
                validation["stats"]["boys"] = boys
                validation["stats"]["girls"] = girls
                
                if invalid_gender > 0:
                    validation["warnings"].append(f"Βρέθηκαν {invalid_gender} μη έγκυρες τιμές φύλου")
            
            # Έλεγχος παιδιών εκπαιδευτικών
            if "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ" in df.columns:
                teacher_kids = int((df["ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ"] == True).sum())
                validation["stats"]["teacher_kids"] = teacher_kids
                
                if teacher_kids == 0:
                    validation["warnings"].append("Δεν βρέθηκαν παιδιά εκπαιδευτικών")
            
            # Προτεινόμενος αριθμός τμημάτων
            optimal_classes = compute_optimal_classes(len(df))
            validation["stats"]["suggested_classes"] = optimal_classes
            
        except Exception as e:
            validation["errors"].append(f"Σφάλμα επικύρωσης: {e}")
            validation["is_valid"] = False
        
        return validation

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_student_data(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Φόρτωση δεδομένων μαθητών από Excel/CSV."""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Το αρχείο {file_path} δεν βρέθηκε")
    
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(f"Μη υποστηριζόμενος τύπος αρχείου: {file_path.suffix}")
        
        print(f"✅ Φορτώθηκαν {len(df)} εγγραφές από {file_path}")
        return df
        
    except Exception as e:
        raise Exception(f"Σφάλμα φόρτωσης αρχείου: {e}")

def create_sample_data(num_students: int = 50) -> pd.DataFrame:
    """Δημιουργία δειγματικών δεδομένων για δοκιμή."""
    
    random.seed(RANDOM_SEED)
    
    # Βασικά στοιχεία
    names = [f"Μαθητής_{i+1}" for i in range(num_students)]
    genders = [random.choice(["Α", "Κ"]) for _ in range(num_students)]
    
    # Χαρακτηριστικά
    teacher_kids = [random.choice([True, False]) for _ in range(num_students)]
    energetic = [random.choice([True, False]) for _ in range(num_students)]
    special_needs = [random.choice([True, False]) for _ in range(num_students)]
    good_greek = [random.choice([True, False]) for _ in range(num_students)]
    
    # Φίλοι (απλουστευμένα)
    friends = []
    for i in range(num_students):
        friend_count = random.randint(0, 3)
        student_friends = random.sample([n for j, n in enumerate(names) if j != i], 
                                      min(friend_count, len(names)-1))
        friends.append(student_friends if student_friends else "")
    
    df = pd.DataFrame({
        "ΟΝΟΜΑ": names,
        "ΦΥΛΟ": genders,
        "ΠΑΙΔΙ_ΕΚΠΑΙΔΕΥΤΙΚΟΥ": teacher_kids,
        "ΖΩΗΡΟΣ": energetic,
        "ΙΔΙΑΙΤΕΡΟΤΗΤΑ": special_needs,
        "ΚΑΛΗ_ΓΝΩΣΗ_ΕΛΛΗΝΙΚΩΝ": good_greek,
        "ΦΙΛΟΙ": friends,
        "ΣΥΓΚΡΟΥΣΗ": [""] * num_students
    })
    
    return df

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Κύρια συνάρτηση εκτέλεσης."""
    
    try:
        # Δημιουργία δειγματικών δεδομένων
        print("Δημιουργία δειγματικών δεδομένων...")
        sample_df = create_sample_data(60)
        
        # Εκκίνηση συστήματος
        system = StudentAssignmentSystem(random_seed=RANDOM_SEED)
        
        # Εκτέλεση πλήρους κατανομής
        results = system.process_complete_assignment(sample_df, num_classes=3, max_scenarios=3)
        
        # Αποθήκευση αποτελεσμάτων
        system.save_results(results, "student_assignment_output")
        
        # Εμφάνιση συνοπτικών στοιχείων
        if results["status"] == "SUCCESS":
            final_df = results["final_df"]
            print(f"\n📊 ΤΕΛΙΚΑ ΣΤΟΙΧΕΙΑ:")
            print(f"Συνολικοί μαθητές: {len(final_df)}")
            
            # Στατιστικά ανά τμήμα
            if "ΒΗΜΑ6_ΤΜΗΜΑ" in final_df.columns:
                class_stats = final_df["ΒΗΜΑ6_ΤΜΗΜΑ"].value_counts()
                print("Κατανομή ανά τμήμα:")
                for class_name, count in class_stats.items():
                    print(f"  {class_name}: {count} μαθητές")
        
        return results
        
    except Exception as e:
        print(f"❌ Σφάλμα εκτέλεσης: {e}")
        return None


if __name__ == "__main__":
    main()