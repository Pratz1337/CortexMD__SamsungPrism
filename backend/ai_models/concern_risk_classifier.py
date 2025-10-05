#!/usr/bin/env python3
"""
Simple LLM-based risk classifier for CONCERN Early Warning System
Uses LLM to analyze diagnosis and classify patient risk level
"""

import json
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConcernRiskClassifier:
    """Simple risk classifier using LLM for CONCERN system"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
    
    def classify_patient_risk(self, diagnosis_data: Dict[str, Any], patient_history: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify patient risk using LLM based on diagnosis results
        Returns: {
            'risk_level': 'low' | 'medium' | 'high' | 'critical',
            'confidence': float,
            'reasoning': str,
            'recommendations': list
        }
        """
        try:
            # Extract key diagnosis information
            primary_diagnosis = diagnosis_data.get('primary_diagnosis', 'Unknown')
            confidence_score = diagnosis_data.get('confidence_score', 0.0)
            symptoms = diagnosis_data.get('symptoms', [])
            clinical_recommendations = diagnosis_data.get('clinical_recommendations', [])
            differential_diagnoses = diagnosis_data.get('differential_diagnoses', [])
            
            # Build prompt for LLM
            prompt = f"""You are a clinical risk assessment AI. Based on the following diagnosis information, classify the patient's risk level.

DIAGNOSIS INFORMATION:
- Primary Diagnosis: {primary_diagnosis}
- Confidence Score: {confidence_score:.2%}
- Symptoms: {', '.join(symptoms) if symptoms else 'None reported'}
- Clinical Recommendations: {json.dumps(clinical_recommendations, indent=2) if clinical_recommendations else 'None'}
- Differential Diagnoses: {json.dumps(differential_diagnoses, indent=2) if differential_diagnoses else 'None'}

{f"PATIENT HISTORY: Previous risk levels: {patient_history.get('previous_risks', [])}" if patient_history else ""}

TASK: Classify the patient's current risk level based on this diagnosis.

IMPORTANT GUIDELINES:
1. CRITICAL: Life-threatening conditions, immediate intervention needed (e.g., heart attack, stroke, severe trauma)
2. HIGH: Serious conditions requiring urgent attention (e.g., pneumonia, unstable angina, severe infections)
3. MEDIUM: Conditions needing medical attention but not immediately life-threatening (e.g., moderate infections, controlled chronic conditions)
4. LOW: Minor conditions or stable chronic conditions (e.g., common cold, well-controlled diabetes)

Respond with a JSON object containing:
{{
    "risk_level": "low|medium|high|critical",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of risk classification",
    "recommendations": ["recommendation 1", "recommendation 2"]
}}
"""

            # Call LLM service if available
            if self.llm_service:
                try:
                    response = self.llm_service.generate_response(prompt, temperature=0.3, max_tokens=500)
                    
                    # Parse LLM response
                    if '```json' in response:
                        json_str = response.split('```json')[1].split('```')[0].strip()
                    else:
                        # Try to find JSON in response
                        import re
                        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                        json_str = json_match.group(0) if json_match else '{}'
                    
                    result = json.loads(json_str)
                    
                    # Validate result
                    risk_level = result.get('risk_level', 'medium').lower()
                    if risk_level not in ['low', 'medium', 'high', 'critical']:
                        risk_level = 'medium'
                    
                    return {
                        'risk_level': risk_level,
                        'confidence': float(result.get('confidence', 0.7)),
                        'reasoning': result.get('reasoning', 'Risk assessment based on diagnosis'),
                        'recommendations': result.get('recommendations', []),
                        'timestamp': datetime.now().isoformat(),
                        'method': 'llm_classification'
                    }
                    
                except Exception as e:
                    logger.error(f"LLM classification error: {e}")
                    # Fall back to rule-based classification
            
            # Fallback: Simple rule-based classification if LLM not available
            return self._rule_based_classification(diagnosis_data)
            
        except Exception as e:
            logger.error(f"Risk classification error: {e}")
            return {
                'risk_level': 'medium',
                'confidence': 0.5,
                'reasoning': 'Default classification due to error',
                'recommendations': ['Monitor patient closely'],
                'timestamp': datetime.now().isoformat(),
                'method': 'fallback'
            }
    
    def _rule_based_classification(self, diagnosis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based fallback classification"""
        confidence_score = diagnosis_data.get('confidence_score', 0.0)
        primary_diagnosis = diagnosis_data.get('primary_diagnosis', '').lower()
        
        # High-risk keywords
        critical_keywords = ['heart attack', 'stroke', 'cardiac arrest', 'myocardial', 'sepsis', 'shock']
        high_keywords = ['pneumonia', 'infection', 'fracture', 'bleeding', 'chest pain']
        
        risk_level = 'low'
        reasoning = 'Based on diagnosis analysis'
        
        # Check for critical conditions
        if any(keyword in primary_diagnosis for keyword in critical_keywords):
            risk_level = 'critical'
            reasoning = 'Critical condition detected requiring immediate intervention'
        elif any(keyword in primary_diagnosis for keyword in high_keywords):
            risk_level = 'high'
            reasoning = 'Serious condition requiring urgent medical attention'
        elif confidence_score > 0.8:
            risk_level = 'medium'
            reasoning = 'High confidence diagnosis of moderate severity'
        
        return {
            'risk_level': risk_level,
            'confidence': confidence_score,
            'reasoning': reasoning,
            'recommendations': self._get_default_recommendations(risk_level),
            'timestamp': datetime.now().isoformat(),
            'method': 'rule_based'
        }
    
    def _get_default_recommendations(self, risk_level: str) -> list:
        """Get default recommendations based on risk level"""
        recommendations = {
            'critical': [
                'Immediate medical intervention required',
                'Continuous vital signs monitoring',
                'Alert medical team immediately'
            ],
            'high': [
                'Urgent medical assessment needed',
                'Monitor vital signs every hour',
                'Prepare for potential intervention'
            ],
            'medium': [
                'Schedule follow-up within 24-48 hours',
                'Monitor symptoms progression',
                'Administer prescribed medications'
            ],
            'low': [
                'Continue current treatment plan',
                'Schedule routine follow-up',
                'Monitor for any changes'
            ]
        }
        return recommendations.get(risk_level, ['Monitor patient condition'])

# Simple usage example
if __name__ == "__main__":
    classifier = ConcernRiskClassifier()
    
    # Test diagnosis data
    test_diagnosis = {
        'primary_diagnosis': 'Acute Myocardial Infarction',
        'confidence_score': 0.92,
        'symptoms': ['Chest pain', 'Shortness of breath', 'Sweating'],
        'clinical_recommendations': ['Immediate cardiac catheterization', 'Aspirin administration']
    }
    
    result = classifier.classify_patient_risk(test_diagnosis)
    print(json.dumps(result, indent=2))
