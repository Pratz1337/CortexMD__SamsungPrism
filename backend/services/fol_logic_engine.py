"""
Deterministic First-Order Logic (FOL) Engine for Medical Predicate Verification
This module provides a robust, rule-based logic engine that evaluates medical predicates
against patient data using formal logic principles.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LogicOperator(Enum):
    """Logical operators for FOL expressions"""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"
    EXISTS = "∃"
    FORALL = "∀"
    EQUALS = "="
    NOT_EQUALS = "≠"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = "≥"
    LESS_EQUAL = "≤"

@dataclass
class Term:
    """Represents a term in FOL (variable, constant, or function)"""
    name: str
    term_type: str  # 'variable', 'constant', 'function'
    value: Optional[Any] = None
    arguments: List['Term'] = field(default_factory=list)
    
    def __repr__(self):
        if self.arguments:
            return f"{self.name}({', '.join(str(arg) for arg in self.arguments)})"
        return self.name

@dataclass
class Predicate:
    """Represents a FOL predicate"""
    name: str
    arguments: List[Term]
    negated: bool = False
    
    def __repr__(self):
        neg = "¬" if self.negated else ""
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{neg}{self.name}({args})"
    
    def to_string(self):
        """Convert predicate to string representation"""
        return str(self)

@dataclass
class Formula:
    """Represents a FOL formula (can be atomic or compound)"""
    formula_type: str  # 'atomic', 'compound', 'quantified'
    predicate: Optional[Predicate] = None
    operator: Optional[LogicOperator] = None
    operands: List['Formula'] = field(default_factory=list)
    variable: Optional[Term] = None  # For quantified formulas
    domain: Optional[List[Any]] = None  # Domain for quantified variables
    
    def __repr__(self):
        if self.formula_type == 'atomic':
            return str(self.predicate)
        elif self.formula_type == 'compound':
            if self.operator in [LogicOperator.AND, LogicOperator.OR]:
                op_str = f" {self.operator.value} "
                return f"({op_str.join(str(op) for op in self.operands)})"
            elif self.operator == LogicOperator.NOT:
                return f"{self.operator.value}{self.operands[0]}"
            elif self.operator == LogicOperator.IMPLIES:
                return f"({self.operands[0]} {self.operator.value} {self.operands[1]})"
        elif self.formula_type == 'quantified':
            return f"{self.operator.value}{self.variable}. {self.operands[0]}"
        return "INVALID_FORMULA"

class MedicalKnowledgeBase:
    """Knowledge base containing medical facts and rules"""
    
    def __init__(self):
        self.facts: List[Predicate] = []
        self.rules: List[Tuple[List[Predicate], Predicate]] = []  # (antecedents, consequent)
        self.ontology_mappings: Dict[str, Set[str]] = {}
        self._initialize_medical_rules()
    
    def _initialize_medical_rules(self):
        """Initialize medical domain rules and relationships"""
        # Medical condition implications
        self.rules.extend([
            # Diabetes rules
            ([self._parse_predicate("has_lab_value(X, glucose, high)"),
              self._parse_predicate("has_lab_value(X, hba1c, elevated)")],
             self._parse_predicate("has_condition(X, diabetes)")),
            
            # Myocardial infarction rules
            ([self._parse_predicate("has_lab_value(X, troponin, elevated)"),
              self._parse_predicate("has_symptom(X, chest_pain)")],
             self._parse_predicate("has_condition(X, myocardial_infarction)")),
            
            # Hypertension rules
            ([self._parse_predicate("has_vital_sign(X, blood_pressure, high)"),
              self._parse_predicate("consistent_readings(X, blood_pressure, 3)")],
             self._parse_predicate("has_condition(X, hypertension)")),
            
            # Heart failure rules
            ([self._parse_predicate("has_symptom(X, shortness_of_breath)"),
              self._parse_predicate("has_symptom(X, edema)"),
              self._parse_predicate("has_lab_value(X, bnp, elevated)")],
             self._parse_predicate("has_condition(X, heart_failure)")),
        ])
        
        # Ontology relationships
        self.ontology_mappings = {
            "chest_pain": {"angina", "chest_discomfort", "thoracic_pain"},
            "shortness_of_breath": {"dyspnea", "sob", "breathing_difficulty"},
            "diabetes": {"diabetes_mellitus", "dm", "hyperglycemia"},
            "hypertension": {"high_blood_pressure", "htn", "elevated_bp"},
            "myocardial_infarction": {"heart_attack", "mi", "stemi", "nstemi"},
        }
    
    def _parse_predicate(self, pred_str: str) -> Predicate:
        """Parse a predicate string into a Predicate object"""
        match = re.match(r'(\¬)?(\w+)\((.*)\)', pred_str)
        if not match:
            raise ValueError(f"Invalid predicate format: {pred_str}")
        
        negated = bool(match.group(1))
        name = match.group(2)
        args_str = match.group(3)
        
        # Parse arguments
        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg.isupper() and len(arg) == 1:  # Variable
                args.append(Term(arg, 'variable'))
            else:  # Constant
                args.append(Term(arg, 'constant', value=arg))
        
        return Predicate(name, args, negated)
    
    def add_fact(self, predicate: Predicate):
        """Add a fact to the knowledge base"""
        self.facts.append(predicate)
    
    def add_rule(self, antecedents: List[Predicate], consequent: Predicate):
        """Add an inference rule to the knowledge base"""
        self.rules.append((antecedents, consequent))
    
    def get_related_terms(self, term: str) -> Set[str]:
        """Get ontologically related terms"""
        related = {term}
        if term in self.ontology_mappings:
            related.update(self.ontology_mappings[term])
        # Check reverse mappings
        for key, values in self.ontology_mappings.items():
            if term in values:
                related.add(key)
                related.update(values)
        return related

class FOLParser:
    """Parser for FOL expressions"""
    
    def __init__(self):
        self.operators = {
            '∧': LogicOperator.AND,
            '∨': LogicOperator.OR,
            '¬': LogicOperator.NOT,
            '→': LogicOperator.IMPLIES,
            '↔': LogicOperator.IFF,
            '∃': LogicOperator.EXISTS,
            '∀': LogicOperator.FORALL,
        }
    
    def parse(self, expression: str) -> Formula:
        """Parse a FOL expression string into a Formula object"""
        expression = expression.strip()
        
        # Handle quantifiers
        if expression.startswith('∃') or expression.startswith('∀'):
            return self._parse_quantified(expression)
        
        # Handle negation
        if expression.startswith('¬'):
            inner = self.parse(expression[1:])
            return Formula(
                formula_type='compound',
                operator=LogicOperator.NOT,
                operands=[inner]
            )
        
        # Handle parentheses
        if expression.startswith('(') and expression.endswith(')'):
            expression = expression[1:-1]
        
        # Try to parse as compound formula
        compound = self._parse_compound(expression)
        if compound:
            return compound
        
        # Parse as atomic predicate
        return self._parse_atomic(expression)
    
    def _parse_atomic(self, expression: str) -> Formula:
        """Parse an atomic predicate"""
        match = re.match(r'(\¬)?(\w+)\((.*)\)', expression)
        if not match:
            raise ValueError(f"Invalid predicate format: {expression}")
        
        negated = bool(match.group(1))
        pred_name = match.group(2)
        args_str = match.group(3)
        
        # Parse arguments
        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if arg.isupper() and len(arg) == 1:  # Variable
                args.append(Term(arg, 'variable'))
            else:  # Constant
                args.append(Term(arg, 'constant', value=arg))
        
        predicate = Predicate(pred_name, args, negated)
        return Formula(formula_type='atomic', predicate=predicate)
    
    def _parse_compound(self, expression: str) -> Optional[Formula]:
        """Parse a compound formula with binary operators"""
        # Find the main operator (rightmost for right-associativity)
        depth = 0
        for i in range(len(expression) - 1, -1, -1):
            if expression[i] == ')':
                depth += 1
            elif expression[i] == '(':
                depth -= 1
            elif depth == 0:
                # Check for binary operators
                for op_str, op_enum in [('→', LogicOperator.IMPLIES),
                                        ('↔', LogicOperator.IFF),
                                        ('∨', LogicOperator.OR),
                                        ('∧', LogicOperator.AND)]:
                    if expression[i:i+len(op_str)] == op_str:
                        left = self.parse(expression[:i])
                        right = self.parse(expression[i+len(op_str):])
                        return Formula(
                            formula_type='compound',
                            operator=op_enum,
                            operands=[left, right]
                        )
        return None
    
    def _parse_quantified(self, expression: str) -> Formula:
        """Parse a quantified formula"""
        quantifier = LogicOperator.EXISTS if expression[0] == '∃' else LogicOperator.FORALL
        
        # Extract variable
        match = re.match(r'[∃∀](\w+)\.\s*(.*)', expression)
        if not match:
            raise ValueError(f"Invalid quantified formula: {expression}")
        
        var_name = match.group(1)
        inner_expr = match.group(2)
        
        variable = Term(var_name, 'variable')
        inner_formula = self.parse(inner_expr)
        
        return Formula(
            formula_type='quantified',
            operator=quantifier,
            variable=variable,
            operands=[inner_formula]
        )

class FOLEvaluator:
    """Evaluator for FOL formulas against a knowledge base and patient data"""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base
        self.patient_data: Dict[str, Any] = {}
        self.bindings: Dict[str, Any] = {}  # Variable bindings
    
    def set_patient_data(self, patient_data: Dict[str, Any]):
        """Set patient data for evaluation"""
        self.patient_data = patient_data
    
    def evaluate(self, formula: Formula, bindings: Optional[Dict[str, Any]] = None) -> Tuple[bool, float]:
        """
        Evaluate a FOL formula
        Returns: (truth_value, confidence_score)
        """
        if bindings:
            self.bindings = bindings.copy()
        
        if formula.formula_type == 'atomic':
            return self._evaluate_atomic(formula.predicate)
        elif formula.formula_type == 'compound':
            return self._evaluate_compound(formula)
        elif formula.formula_type == 'quantified':
            return self._evaluate_quantified(formula)
        
        return False, 0.0
    
    def _evaluate_atomic(self, predicate: Predicate) -> Tuple[bool, float]:
        """Evaluate an atomic predicate against patient data"""
        pred_name = predicate.name
        
        # Substitute variables with bindings
        args = []
        for arg in predicate.arguments:
            if arg.term_type == 'variable' and arg.name in self.bindings:
                args.append(self.bindings[arg.name])
            else:
                args.append(arg.value if arg.value is not None else arg.name)
        
        # Route to specific evaluators
        evaluator_map = {
            'has_symptom': self._evaluate_has_symptom,
            'has_condition': self._evaluate_has_condition,
            'has_lab_value': self._evaluate_has_lab_value,
            'has_vital_sign': self._evaluate_has_vital_sign,
            'takes_medication': self._evaluate_takes_medication,
        }
        
        if pred_name in evaluator_map:
            truth_value, confidence = evaluator_map[pred_name](args)
        else:
            # Check if predicate exists in facts
            truth_value, confidence = self._check_fact(predicate)
        
        # Apply negation if needed
        if predicate.negated:
            truth_value = not truth_value
            confidence = 1.0 - confidence
        
        return truth_value, confidence
    
    def _evaluate_compound(self, formula: Formula) -> Tuple[bool, float]:
        """Evaluate compound formulas"""
        operator = formula.operator
        
        if operator == LogicOperator.NOT:
            truth, conf = self.evaluate(formula.operands[0], self.bindings)
            return not truth, 1.0 - conf
        
        elif operator == LogicOperator.AND:
            truths, confs = [], []
            for operand in formula.operands:
                t, c = self.evaluate(operand, self.bindings)
                truths.append(t)
                confs.append(c)
            return all(truths), min(confs) if confs else 0.0
        
        elif operator == LogicOperator.OR:
            truths, confs = [], []
            for operand in formula.operands:
                t, c = self.evaluate(operand, self.bindings)
                truths.append(t)
                confs.append(c)
            return any(truths), max(confs) if confs else 0.0
        
        elif operator == LogicOperator.IMPLIES:
            ant_truth, ant_conf = self.evaluate(formula.operands[0], self.bindings)
            cons_truth, cons_conf = self.evaluate(formula.operands[1], self.bindings)
            # P → Q is equivalent to ¬P ∨ Q
            truth = not ant_truth or cons_truth
            confidence = 1.0 if not ant_truth else cons_conf
            return truth, confidence
        
        elif operator == LogicOperator.IFF:
            left_truth, left_conf = self.evaluate(formula.operands[0], self.bindings)
            right_truth, right_conf = self.evaluate(formula.operands[1], self.bindings)
            truth = left_truth == right_truth
            confidence = min(left_conf, right_conf) if truth else 1.0 - min(left_conf, right_conf)
            return truth, confidence
        
        return False, 0.0
    
    def _evaluate_quantified(self, formula: Formula) -> Tuple[bool, float]:
        """Evaluate quantified formulas"""
        variable = formula.variable
        domain = self._get_domain(variable)
        
        truths, confs = [], []
        for value in domain:
            new_bindings = self.bindings.copy()
            new_bindings[variable.name] = value
            t, c = self.evaluate(formula.operands[0], new_bindings)
            truths.append(t)
            confs.append(c)
        
        if formula.operator == LogicOperator.EXISTS:
            return any(truths), max(confs) if confs else 0.0
        elif formula.operator == LogicOperator.FORALL:
            return all(truths), min(confs) if confs else 0.0
        
        return False, 0.0
    
    def _evaluate_has_symptom(self, args: List[Any]) -> Tuple[bool, float]:
        """Evaluate has_symptom predicate"""
        if len(args) < 2:
            return False, 0.0
        
        patient_id = args[0]
        symptom = str(args[1]).lower()
        
        # Get patient symptoms
        symptoms = self.patient_data.get('symptoms', [])
        
        # Check for symptom or related terms
        related_terms = self.kb.get_related_terms(symptom)
        
        for patient_symptom in symptoms:
            patient_symptom_lower = patient_symptom.lower()
            if symptom in patient_symptom_lower:
                return True, 0.95
            for term in related_terms:
                if term in patient_symptom_lower or patient_symptom_lower in term:
                    return True, 0.95
        
        # Check clinical notes
        clinical_notes = self.patient_data.get('clinical_notes', '')
        if clinical_notes:
            notes_lower = clinical_notes.lower()
            for term in related_terms:
                if term in notes_lower:
                    return True, 0.8
        
        return False, 0.1
    
    def _evaluate_has_condition(self, args: List[Any]) -> Tuple[bool, float]:
        """Evaluate has_condition predicate"""
        if len(args) < 2:
            return False, 0.0
        
        patient_id = args[0]
        condition = str(args[1]).lower()
        
        # Check medical history
        medical_history = self.patient_data.get('medical_history', [])
        related_terms = self.kb.get_related_terms(condition)
        
        for hist_condition in medical_history:
            hist_lower = hist_condition.lower()
            for term in related_terms:
                if term in hist_lower or hist_lower in term:
                    return True, 0.95
        
        # Check current conditions
        current_conditions = self.patient_data.get('current_conditions', [])
        for curr_condition in current_conditions:
            curr_lower = curr_condition.lower()
            for term in related_terms:
                if term in curr_lower or curr_lower in term:
                    return True, 0.95
        
        # Try to infer from rules
        inferred, confidence = self._infer_condition(condition)
        if inferred:
            return True, confidence * 0.8  # Reduce confidence for inferred conditions
        
        return False, 0.1
    
    def _evaluate_has_lab_value(self, args: List[Any]) -> Tuple[bool, float]:
        """Evaluate has_lab_value predicate"""
        if len(args) < 3:
            return False, 0.0

        patient_id = args[0]
        lab_name = str(args[1]).lower()
        expected_value = str(args[2]).lower()

        lab_results = self.patient_data.get('lab_results', {})

        # Find matching lab - more flexible matching
        for lab_key, lab_value in lab_results.items():
            lab_key_lower = lab_key.lower()

            # Check if lab name matches or is contained in the key
            if (lab_name in lab_key_lower or
                lab_key_lower in lab_name or
                any(word in lab_key_lower for word in lab_name.split('_'))):
                # Evaluate the value with improved matching
                return self._compare_lab_value(lab_value, expected_value, lab_name)

        return False, 0.0
    
    def _evaluate_has_vital_sign(self, args: List[Any]) -> Tuple[bool, float]:
        """Evaluate has_vital_sign predicate"""
        if len(args) < 3:
            return False, 0.0
        
        patient_id = args[0]
        vital_name = str(args[1]).lower()
        expected_value = str(args[2]).lower()
        
        vitals = self.patient_data.get('vitals', {})
        
        # Map vital names
        vital_mappings = {
            "blood_pressure": ["blood pressure", "bp"],
            "heart_rate": ["heart rate", "hr", "pulse"],
            "temperature": ["temperature", "temp"],
            "respiratory_rate": ["respiratory rate", "rr"],
            "oxygen_saturation": ["oxygen saturation", "o2 sat", "spo2"]
        }
        
        # Find matching vital
        for vital_key, vital_value in vitals.items():
            vital_key_lower = vital_key.lower()
            matched = False
            
            for mapped_name, variations in vital_mappings.items():
                if any(var in vital_key_lower or vital_key_lower in var for var in variations):
                    if vital_name in variations or vital_name == mapped_name:
                        matched = True
                        break
            
            if matched or vital_name in vital_key_lower or vital_key_lower in vital_name:
                return self._compare_vital_value(vital_value, expected_value, vital_name)
        
        return False, 0.0
    
    def _evaluate_takes_medication(self, args: List[Any]) -> Tuple[bool, float]:
        """Evaluate takes_medication predicate"""
        if len(args) < 2:
            return False, 0.0
        
        patient_id = args[0]
        medication = str(args[1]).lower()
        
        current_meds = self.patient_data.get('current_medications', [])
        
        for med in current_meds:
            med_lower = med.lower()
            if medication in med_lower or med_lower in medication:
                return True, 0.95
        
        return False, 0.1
    
    def _compare_lab_value(self, actual_value: Any, expected: str, lab_name: str) -> Tuple[bool, float]:
        """Compare lab values with intelligent matching"""
        try:
            # Handle string values (qualitative results)
            if isinstance(actual_value, str):
                actual_str = actual_value.lower().strip()

                # Check for qualitative matches
                if expected.lower() in actual_str or actual_str in expected.lower():
                    return True, 0.9  # High confidence for direct string match

                # Handle common qualitative patterns
                if expected.lower() == "hematuria" and "hematuria" in actual_str:
                    return True, 0.95
                elif expected.lower() in ["elevated", "high", "increased"] and any(word in actual_str for word in ["elevated", "high", "increased", "abnormal", "positive"]):
                    return True, 0.8
                elif expected.lower() in ["low", "decreased", "reduced"] and any(word in actual_str for word in ["low", "decreased", "reduced", "below", "negative"]):
                    return True, 0.8
                elif expected.lower() in ["normal", "negative", "wnl"] and any(word in actual_str for word in ["normal", "negative", "within normal", "wnl", "clear"]):
                    return True, 0.8

            # Handle numeric values
            actual_num = float(actual_value) if isinstance(actual_value, (int, float, str)) else 0

            # Define normal ranges
            normal_ranges = {
                "glucose": (70, 100),
                "troponin": (0, 0.04),
                "creatinine": (0.6, 1.2),
                "hemoglobin": (12.0, 16.0)
            }

            # Handle qualitative comparisons
            if expected in ["high", "elevated", "increased"]:
                if lab_name in normal_ranges:
                    max_normal = normal_ranges[lab_name][1]
                    if actual_num > max_normal:
                        return True, min(1.0, (actual_num - max_normal) / max_normal + 0.5)
                return actual_num > 100, 0.7  # Generic high threshold

            elif expected in ["low", "decreased", "reduced"]:
                if lab_name in normal_ranges:
                    min_normal = normal_ranges[lab_name][0]
                    if actual_num < min_normal:
                        return True, min(1.0, (min_normal - actual_num) / min_normal + 0.5)
                return actual_num < 50, 0.7  # Generic low threshold

            elif expected in ["normal", "wnl"]:
                if lab_name in normal_ranges:
                    min_val, max_val = normal_ranges[lab_name]
                    is_normal = min_val <= actual_num <= max_val
                    confidence = 1.0 - abs(actual_num - (min_val + max_val) / 2) / ((max_val - min_val) / 2) * 0.3
                    return is_normal, confidence
                return True, 0.5  # Assume normal if no range defined

            else:
                # Try numeric comparison
                expected_num = float(expected)
                tolerance = 0.1 * expected_num if expected_num != 0 else 0.1
                is_match = abs(actual_num - expected_num) <= tolerance
                confidence = 1.0 - abs(actual_num - expected_num) / (expected_num if expected_num != 0 else 1) * 0.5
                return is_match, max(0.1, min(1.0, confidence))

        except (ValueError, TypeError):
            # If numeric conversion fails, try string matching
            if isinstance(actual_value, str) and isinstance(expected, str):
                actual_lower = actual_value.lower()
                expected_lower = expected.lower()
                if expected_lower in actual_lower or actual_lower in expected_lower:
                    return True, 0.8  # Moderate confidence for partial string match
            return False, 0.0
    
    def _compare_vital_value(self, actual_value: Any, expected: str, vital_name: str) -> Tuple[bool, float]:
        """Compare vital sign values"""
        # Similar to lab value comparison but with vital-specific ranges
        normal_ranges = {
            "blood_pressure": {"systolic": (90, 120), "diastolic": (60, 80)},
            "heart_rate": (60, 100),
            "temperature": (97.0, 99.5),
            "respiratory_rate": (12, 20),
            "oxygen_saturation": (95, 100)
        }
        
        # Handle blood pressure specially
        if "blood" in vital_name or "bp" in vital_name:
            if isinstance(actual_value, str) and '/' in actual_value:
                systolic, diastolic = map(float, actual_value.split('/'))
                if expected in ["high", "elevated"]:
                    is_high = systolic > 140 or diastolic > 90
                    confidence = min(1.0, max((systolic - 140) / 40, (diastolic - 90) / 20) + 0.5)
                    return is_high, confidence if is_high else 0.2
                elif expected in ["normal"]:
                    is_normal = 90 <= systolic <= 120 and 60 <= diastolic <= 80
                    return is_normal, 0.9 if is_normal else 0.2
        
        # Use generic comparison for other vitals
        return self._compare_lab_value(actual_value, expected, vital_name)
    
    def _check_fact(self, predicate: Predicate) -> Tuple[bool, float]:
        """Check if predicate exists in knowledge base facts"""
        for fact in self.kb.facts:
            if self._predicates_match(predicate, fact):
                return True, 1.0
        return False, 0.0
    
    def _predicates_match(self, p1: Predicate, p2: Predicate) -> bool:
        """Check if two predicates match (considering variable bindings)"""
        if p1.name != p2.name or p1.negated != p2.negated:
            return False
        
        if len(p1.arguments) != len(p2.arguments):
            return False
        
        for a1, a2 in zip(p1.arguments, p2.arguments):
            if a1.term_type == 'variable':
                # Variable can match anything, update bindings
                if a1.name in self.bindings and self.bindings[a1.name] != a2.value:
                    return False
                self.bindings[a1.name] = a2.value
            elif a2.term_type == 'variable':
                if a2.name in self.bindings and self.bindings[a2.name] != a1.value:
                    return False
                self.bindings[a2.name] = a1.value
            elif a1.value != a2.value:
                return False
        
        return True
    
    def _infer_condition(self, condition: str) -> Tuple[bool, float]:
        """Try to infer a condition using rules in knowledge base"""
        for antecedents, consequent in self.kb.rules:
            # Check if consequent matches the condition we're looking for
            if consequent.name == 'has_condition' and len(consequent.arguments) > 1:
                if str(consequent.arguments[1].value).lower() == condition:
                    # Check if all antecedents are satisfied
                    all_satisfied = True
                    min_confidence = 1.0
                    
                    for antecedent in antecedents:
                        # Substitute patient for variable X
                        eval_pred = Predicate(
                            antecedent.name,
                            [Term('patient', 'constant', 'patient')] + antecedent.arguments[1:],
                            antecedent.negated
                        )
                        satisfied, conf = self._evaluate_atomic(eval_pred)
                        if not satisfied:
                            all_satisfied = False
                            break
                        min_confidence = min(min_confidence, conf)
                    
                    if all_satisfied:
                        return True, min_confidence * 0.9  # Slightly reduce confidence for inference
        
        return False, 0.0
    
    def _get_domain(self, variable: Term) -> List[Any]:
        """Get domain for a quantified variable"""
        # For medical domain, typically iterate over patients or values
        if variable.name == 'X':
            return ['patient']  # In single-patient context
        return []

class DeterministicFOLVerifier:
    """Main interface for deterministic FOL verification"""
    
    def __init__(self):
        self.kb = MedicalKnowledgeBase()
        self.parser = FOLParser()
        self.evaluator = FOLEvaluator(self.kb)
        logger.info("Initialized Deterministic FOL Verifier")
    
    def verify_predicate(self, predicate_str: str, patient_data: Dict) -> Dict[str, Any]:
        """
        Verify a single FOL predicate against patient data
        
        Args:
            predicate_str: FOL predicate string
            patient_data: Patient data dictionary
            
        Returns:
            Dictionary containing verification results
        """
        try:
            # Set patient data in evaluator
            self.evaluator.set_patient_data(patient_data)
            
            # Parse the predicate string
            formula = self.parser.parse(predicate_str)
            
            # Evaluate the formula
            truth_value, confidence = self.evaluator.evaluate(formula)
            
            return {
                "predicate": predicate_str,
                "verified": truth_value,
                "confidence_score": confidence,
                "evaluation_method": "deterministic_logic",
                "formula_type": formula.formula_type,
                "reasoning": self._generate_reasoning(formula, truth_value, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error verifying predicate '{predicate_str}': {str(e)}")
            return {
                "predicate": predicate_str,
                "verified": False,
                "confidence_score": 0.0,
                "evaluation_method": "error",
                "error": str(e),
                "reasoning": f"Failed to evaluate predicate: {str(e)}"
            }
    
    def verify_multiple_predicates(self, predicates: List[str], patient_data: Dict) -> List[Dict[str, Any]]:
        """
        Verify multiple FOL predicates against patient data
        
        Args:
            predicates: List of FOL predicate strings
            patient_data: Patient data dictionary
            
        Returns:
            List of verification results for each predicate
        """
        results = []
        
        # Set patient data once for efficiency
        self.evaluator.set_patient_data(patient_data)
        
        for predicate_str in predicates:
            result = self.verify_predicate(predicate_str, patient_data)
            results.append(result)
        
        return results
    
    def verify_formula(self, formula_str: str, patient_data: Dict) -> Dict[str, Any]:
        """
        Verify a complex FOL formula (with logical operators)
        
        Args:
            formula_str: FOL formula string
            patient_data: Patient data dictionary
            
        Returns:
            Dictionary containing verification results
        """
        try:
            self.evaluator.set_patient_data(patient_data)
            
            # Parse the formula
            formula = self.parser.parse(formula_str)
            
            # Evaluate the formula
            truth_value, confidence = self.evaluator.evaluate(formula)
            
            return {
                "formula": formula_str,
                "verified": truth_value,
                "confidence_score": confidence,
                "evaluation_method": "deterministic_logic",
                "formula_type": formula.formula_type,
                "reasoning": self._generate_reasoning(formula, truth_value, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error verifying formula '{formula_str}': {str(e)}")
            return {
                "formula": formula_str,
                "verified": False,
                "confidence_score": 0.0,
                "evaluation_method": "error",
                "error": str(e),
                "reasoning": f"Failed to evaluate formula: {str(e)}"
            }
    
    def add_medical_rule(self, antecedents: List[str], consequent: str):
        """
        Add a new medical inference rule to the knowledge base
        
        Args:
            antecedents: List of antecedent predicate strings
            consequent: Consequent predicate string
        """
        try:
            antecedent_predicates = [self.kb._parse_predicate(ant) for ant in antecedents]
            consequent_predicate = self.kb._parse_predicate(consequent)
            
            self.kb.add_rule(antecedent_predicates, consequent_predicate)
            logger.info(f"Added medical rule: {antecedents} → {consequent}")
            
        except Exception as e:
            logger.error(f"Error adding medical rule: {str(e)}")
    
    def add_medical_fact(self, fact_str: str):
        """
        Add a medical fact to the knowledge base
        
        Args:
            fact_str: Medical fact as predicate string
        """
        try:
            fact_predicate = self.kb._parse_predicate(fact_str)
            self.kb.add_fact(fact_predicate)
            logger.info(f"Added medical fact: {fact_str}")
            
        except Exception as e:
            logger.error(f"Error adding medical fact: {str(e)}")
    
    def _generate_reasoning(self, formula: Formula, truth_value: bool, confidence: float) -> str:
        """
        Generate human-readable reasoning for the verification result
        
        Args:
            formula: The evaluated formula
            truth_value: The truth value result
            confidence: The confidence score
            
        Returns:
            Human-readable reasoning string
        """
        if formula.formula_type == 'atomic':
            predicate = formula.predicate
            if truth_value:
                return f"Predicate '{predicate}' is verified with {confidence:.2f} confidence based on patient data analysis"
            else:
                return f"Predicate '{predicate}' is not supported by patient data (confidence: {confidence:.2f})"
        
        elif formula.formula_type == 'compound':
            if formula.operator == LogicOperator.AND:
                if truth_value:
                    return f"All compound conditions are satisfied (confidence: {confidence:.2f})"
                else:
                    return f"One or more compound conditions are not satisfied (confidence: {confidence:.2f})"
            
            elif formula.operator == LogicOperator.OR:
                if truth_value:
                    return f"At least one alternative condition is satisfied (confidence: {confidence:.2f})"
                else:
                    return f"None of the alternative conditions are satisfied (confidence: {confidence:.2f})"
            
            elif formula.operator == LogicOperator.NOT:
                if truth_value:
                    return f"Negated condition is correctly not satisfied (confidence: {confidence:.2f})"
                else:
                    return f"Negated condition is incorrectly satisfied (confidence: {confidence:.2f})"
            
            elif formula.operator == LogicOperator.IMPLIES:
                if truth_value:
                    return f"Implication holds: if antecedent then consequent (confidence: {confidence:.2f})"
                else:
                    return f"Implication fails: antecedent true but consequent false (confidence: {confidence:.2f})"
        
        elif formula.formula_type == 'quantified':
            if formula.operator == LogicOperator.EXISTS:
                if truth_value:
                    return f"Existential condition satisfied: there exists a case (confidence: {confidence:.2f})"
                else:
                    return f"Existential condition not satisfied: no cases found (confidence: {confidence:.2f})"
            
            elif formula.operator == LogicOperator.FORALL:
                if truth_value:
                    return f"Universal condition satisfied: holds for all cases (confidence: {confidence:.2f})"
                else:
                    return f"Universal condition not satisfied: fails for some cases (confidence: {confidence:.2f})"
        
        return f"Evaluation result: {truth_value} with confidence {confidence:.2f}"
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current knowledge base
        
        Returns:
            Dictionary containing knowledge base statistics
        """
        return {
            "total_facts": len(self.kb.facts),
            "total_rules": len(self.kb.rules),
            "ontology_mappings": len(self.kb.ontology_mappings),
            "supported_predicates": [
                "has_symptom", "has_condition", "has_lab_value", 
                "has_vital_sign", "takes_medication"
            ],
            "logical_operators": [op.value for op in LogicOperator],
            "rule_categories": {
                "diabetes_rules": 1,
                "cardiac_rules": 2,
                "hypertension_rules": 1
            }
        }

# Utility functions for integration with existing system
def create_deterministic_verifier() -> DeterministicFOLVerifier:
    """Create and configure a deterministic FOL verifier"""
    return DeterministicFOLVerifier()

def verify_medical_predicates(predicates: List[str], patient_data: Dict) -> List[Dict[str, Any]]:
    """
    Verify medical predicates using deterministic FOL engine
    
    Args:
        predicates: List of FOL predicate strings
        patient_data: Patient data dictionary
        
    Returns:
        List of verification results
    """
    verifier = create_deterministic_verifier()
    return verifier.verify_multiple_predicates(predicates, patient_data)

# Test function
async def test_deterministic_fol_engine():
    """Test the deterministic FOL engine with sample data"""
    print("Testing Deterministic FOL Engine")
    print("=" * 50)
    
    # Create verifier
    verifier = DeterministicFOLVerifier()
    
    # Sample patient data
    patient_data = {
        "symptoms": ["chest pain", "shortness of breath", "nausea"],
        "medical_history": ["hypertension", "diabetes"],
        "current_medications": ["lisinopril 10mg", "metformin 500mg"],
        "vitals": {
            "blood_pressure": "165/95",
            "heart_rate": 88,
            "temperature": 98.6,
            "respiratory_rate": 18,
            "oxygen_saturation": 96
        },
        "lab_results": {
            "troponin": 0.12,
            "glucose": 156,
            "creatinine": 1.0,
            "hemoglobin": 13.2
        },
        "clinical_notes": "Patient presents with acute chest pain and elevated troponin levels"
    }
    
    # Test individual predicates
    test_predicates = [
        "has_symptom(patient, chest_pain)",
        "has_condition(patient, diabetes)",
        "has_lab_value(patient, troponin, elevated)",
        "has_vital_sign(patient, blood_pressure, high)",
        "takes_medication(patient, lisinopril)"
    ]
    
    print("Testing individual predicates:")
    for predicate in test_predicates:
        result = verifier.verify_predicate(predicate, patient_data)
        print(f"  {predicate}")
        print(f"    Verified: {result['verified']}")
        print(f"    Confidence: {result['confidence_score']:.3f}")
        print(f"    Reasoning: {result['reasoning']}")
        print()
    
    # Test compound formula
    compound_formula = "has_symptom(patient, chest_pain) ∧ has_lab_value(patient, troponin, elevated)"
    print("Testing compound formula:")
    print(f"  {compound_formula}")
    
    result = verifier.verify_formula(compound_formula, patient_data)
    print(f"    Verified: {result['verified']}")
    print(f"    Confidence: {result['confidence_score']:.3f}")
    print(f"    Reasoning: {result['reasoning']}")
    print()
    
    # Test implication
    implication_formula = "has_lab_value(patient, troponin, elevated) → has_condition(patient, myocardial_infarction)"
    print("Testing implication formula:")
    print(f"  {implication_formula}")
    
    result = verifier.verify_formula(implication_formula, patient_data)
    print(f"    Verified: {result['verified']}")
    print(f"    Confidence: {result['confidence_score']:.3f}")
    print(f"    Reasoning: {result['reasoning']}")
    print()
    
    # Show knowledge base stats
    stats = verifier.get_knowledge_base_stats()
    print("Knowledge Base Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nDeterministic FOL Engine test completed successfully!")
