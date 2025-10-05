from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from datetime import datetime 
class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FHIR = "fhir"

class DiagnosisItem(BaseModel):
    """Model for individual diagnosis items"""
    model_config = ConfigDict(extra="forbid")
    
    diagnosis: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: Optional[str] = None

class ValidationIssue(BaseModel):
    """Model for data validation issues"""
    severity: str  # "error", "warning", "info"
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None

class ProcessingMetadata(BaseModel):
    """Model for data processing metadata"""
    processing_timestamp: str
    data_quality_score: float
    validation_issues: List[ValidationIssue]
    preprocessing_applied: List[str]
    phi_detected: bool = False
    anonymized: bool = False

class PatientInput(BaseModel):
    """Enhanced model for patient input data"""
    text_data: Optional[str] = None
    image_paths: Optional[List[str]] = None
    fhir_data: Optional[Dict[str, Any]] = None
    patient_id: Optional[str] = None
    clinical_context: Optional[Dict[str, Any]] = None
    processing_metadata: Optional[ProcessingMetadata] = None
    
class DiagnosisResult(BaseModel):
    """Enhanced model for diagnosis results"""
    primary_diagnosis: str
    confidence_score: float
    top_diagnoses: List[DiagnosisItem]
    reasoning_paths: List[str]
    verification_status: Optional[str] = None
    clinical_impression: Optional[str] = None
    data_quality_assessment: Optional[Dict[str, Any]] = None
    clinical_recommendations: Optional[List[str]] = None
    data_utilization: Optional[List[str]] = None
    # Error handling fields
    error: Optional[bool] = False
    error_message: Optional[str] = None
    errors: Optional[List[str]] = None
    fol_verification: Optional[Dict[str, Any]] = None
    
class MedicalExplanation(BaseModel):
    """Enhanced model for medical explanations"""
    id: Optional[str] = None  # Add ID field for FOL verification
    explanation: str
    confidence: float
    verified: bool = False
    fol_predicates: Optional[List[str]] = None
    supporting_evidence: Optional[List[str]] = None
    category: Optional[str] = None  # "symptom_based", "vital_signs", "imaging", etc.

class ImageAnalysisResult(BaseModel):
    """Model for medical image analysis results"""
    image_path: str
    modality: Optional[str] = None
    body_part: Optional[str] = None
    quality_score: float
    preprocessing_applied: List[str]
    clinical_findings: List[str]
    ar_compatible: bool = False

# CONCERN Early Warning System Models
class ClinicalNote(BaseModel):
    """Model for clinical notes from nurses"""
    note_id: str
    patient_id: str
    nurse_id: Optional[str] = None
    note_content: str
    note_type: str = "nursing"  # nursing, physician, other
    timestamp: datetime
    location: Optional[str] = None
    shift: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PatientVisit(BaseModel):
    """Model for patient visit/check metadata"""
    visit_id: str
    patient_id: str
    nurse_id: Optional[str] = None
    visit_timestamp: datetime
    visit_duration: Optional[int] = None  # in minutes
    visit_type: str = "routine"  # routine, urgent, emergency
    location: str
    notes: Optional[str] = None

class ConcernScore(BaseModel):
    """Model for CONCERN risk scores"""
    patient_id: str
    concern_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str  # low, medium, high, critical
    contributing_factors: List[str]
    metadata_patterns: Dict[str, Any]
    timestamp: datetime
    alert_triggered: bool = False

class PatientMonitoring(BaseModel):
    """Model for patient monitoring data"""
    patient_id: str
    admission_date: datetime
    current_status: str = "stable"  # stable, concerning, critical
    latest_concern_score: Optional[float] = None
    trend_data: Optional[List[Dict[str, Any]]] = None
    alerts_history: Optional[List[Dict[str, Any]]] = None
    nurse_visit_frequency: Optional[float] = None  # visits per hour
    last_updated: datetime