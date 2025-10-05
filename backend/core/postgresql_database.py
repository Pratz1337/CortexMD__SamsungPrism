"""
PostgreSQL Database Manager for CortexMD
Using SQLAlchemy ORM with PostgreSQL
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

logger = logging.getLogger(__name__)

Base = declarative_base()

# ===== POSTGRESQL MODELS =====

class Patient(Base):
    __tablename__ = 'patients'
    
    patient_id = Column(String(50), primary_key=True)
    patient_name = Column(String(200), nullable=False)
    date_of_birth = Column(String(20))
    gender = Column(String(20))
    admission_date = Column(DateTime, default=datetime.now)
    current_status = Column(String(50), default='active')
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    diagnosis_sessions = relationship("DiagnosisSession", back_populates="patient")
    concern_scores = relationship("ConcernScore", back_populates="patient")
    concern_severity = relationship("ConcernSeverityTracking", back_populates="patient", uselist=False)
    chat_messages = relationship("ChatMessage", back_populates="patient")
    clinical_notes = relationship("ClinicalNote", back_populates="patient")
    patient_visits = relationship("PatientVisit", back_populates="patient")

class DiagnosisSession(Base):
    __tablename__ = 'diagnosis_sessions'
    
    session_id = Column(String(50), primary_key=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    status = Column(String(50), default='pending')
    
    # Diagnosis data
    patient_input = Column(JSON)
    diagnosis_result = Column(JSON)
    confidence_score = Column(Float)
    primary_diagnosis = Column(Text)
    
    # Processing metadata
    processing_time = Column(Float)
    ai_model_used = Column(String(100))
    verification_status = Column(String(50))
    
    # Relationships
    patient = relationship("Patient", back_populates="diagnosis_sessions")

class ConcernScore(Base):
    __tablename__ = 'concern_scores'
    
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    concern_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    risk_factors = Column(JSON)
    metadata_patterns = Column(JSON)
    created_at = Column(DateTime, default=datetime.now, index=True)
    alert_triggered = Column(Boolean, default=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="concern_scores")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    message_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    message = Column(Text, nullable=False)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    message_metadata = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Relationships
    patient = relationship("Patient", back_populates="chat_messages")

class ClinicalNote(Base):
    __tablename__ = 'clinical_notes'
    
    note_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    nurse_id = Column(String(50))
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    location = Column(String(100))
    shift = Column(String(20))
    note_type = Column(String(50), default='nursing')
    note_metadata = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Relationships
    patient = relationship("Patient", back_populates="clinical_notes")

class PatientVisit(Base):
    __tablename__ = 'patient_visits'
    
    visit_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    nurse_id = Column(String(50))
    timestamp = Column(DateTime, default=datetime.now, index=True)
    location = Column(String(100), nullable=False)
    visit_type = Column(String(50), default='routine')
    duration_minutes = Column(Integer, default=5)
    notes = Column(Text)
    
    # Relationships
    patient = relationship("Patient", back_populates="patient_visits")

class ConcernSeverityTracking(Base):
    __tablename__ = 'concern_severity_tracking'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, unique=True, index=True)
    
    # Cumulative severity tracking
    cumulative_severity = Column(Float, default=0.0)
    total_diagnoses = Column(Integer, default=0)
    average_severity = Column(Float, default=0.0)
    
    # Last diagnosis severity components
    last_diagnosis_confidence = Column(Float)
    last_fol_verification = Column(Float)
    last_enhanced_verification = Column(Float)
    last_explainability_score = Column(Float)
    last_imaging_present = Column(Boolean, default=False)
    last_computed_severity = Column(Float)
    
    # Current risk assessment
    current_risk_level = Column(String(20), default='low')  # low, medium, high, critical
    current_risk_score = Column(Float, default=0.0, index=True)
    
    # Historical tracking
    max_severity_reached = Column(Float, default=0.0)
    severity_history = Column(JSON, default=list)  # Array of {timestamp, severity, diagnosis_id}
    
    # Metadata
    last_diagnosis_timestamp = Column(DateTime, default=datetime.now, index=True)
    first_diagnosis_timestamp = Column(DateTime, default=datetime.now)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    patient = relationship("Patient", back_populates="concern_severity")

class GradCAMImage(Base):
    __tablename__ = 'gradcam_images'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey('diagnosis_sessions.session_id'), nullable=False, index=True)
    patient_id = Column(String(255), ForeignKey('patients.patient_id'), nullable=False, index=True)
    original_image_path = Column(Text, nullable=False)
    image_filename = Column(String(255), nullable=False)
    heatmap_image = Column(Text)  # Base64 encoded
    overlay_image = Column(Text)   # Base64 encoded
    volume_image = Column(Text)    # Base64 encoded
    analysis_data = Column(JSON)
    predictions = Column(JSON)
    activation_regions = Column(JSON)
    medical_interpretation = Column(JSON)
    processing_successful = Column(Boolean, default=False)
    processing_time = Column(Float)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.now, index=True)

# ===== POSTGRESQL DATABASE MANAGER =====

class PostgreSQLDatabase:
    """PostgreSQL database manager using SQLAlchemy ORM"""
    
    def __init__(self, database_url: str = None):
        """Initialize PostgreSQL connection"""
        if not database_url:
            # Default PostgreSQL connection
            database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql://postgres:password@localhost:5432/cortexmd'
            )
        
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        
        self._connect()
    
    def _connect(self):
        """Connect to PostgreSQL database with optimized pooling"""
        try:
            # Optimize connection pooling for better performance
            self.engine = create_engine(
                self.database_url,
                pool_size=20,  # Increased from 10
                max_overflow=40,  # Increased from 20
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,  # Disable SQL logging for performance
                connect_args={
                    "connect_timeout": 10,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 5,
                    "keepalives_count": 5,
                }
            )
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            logger.info(f"✅ Connected to PostgreSQL: {self.database_url.split('@')[-1]}")
            
            # Insert sample data if needed
            self._ensure_sample_data()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def _ensure_sample_data(self):
        """Ensure sample data exists"""
        try:
            with self.get_session() as session:
                # Check if sample patient exists
                existing_patient = session.query(Patient).filter_by(patient_id="PATIENT_001").first()
                
                if not existing_patient:
                    # Create sample patient
                    sample_patient = Patient(
                        patient_id="PATIENT_001",
                        patient_name="John Doe",
                        date_of_birth="1985-06-15",
                        gender="Male",
                        admission_date=datetime.now(),
                        current_status="active"
                    )
                    session.add(sample_patient)
                    session.commit()
                    logger.info("✅ Created sample patient PATIENT_001")
                
        except Exception as e:
            logger.error(f"❌ Failed to ensure sample data: {e}")
    
    # ===== PATIENT OPERATIONS =====
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient by ID"""
        try:
            with self.get_session() as session:
                patient = session.query(Patient).filter_by(patient_id=patient_id).first()
                if patient:
                    return {
                        'patient_id': patient.patient_id,
                        'patient_name': patient.patient_name,
                        'date_of_birth': patient.date_of_birth,
                        'gender': patient.gender,
                        'admission_date': patient.admission_date.isoformat() if patient.admission_date else None,
                        'current_status': patient.current_status,
                        'created_at': patient.created_at.isoformat() if patient.created_at else None
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get patient {patient_id}: {e}")
            return None
    
    def get_all_patients(self) -> List[Dict[str, Any]]:
        """Get all patients"""
        try:
            with self.get_session() as session:
                # Use yield_per for large result sets and select only required columns
                patients_q = session.query(
                    Patient.patient_id,
                    Patient.patient_name,
                    Patient.date_of_birth,
                    Patient.gender,
                    Patient.admission_date,
                    Patient.current_status,
                    Patient.created_at
                ).yield_per(1000)

                return [
                    {
                        'patient_id': p.patient_id,
                        'patient_name': p.patient_name,
                        'date_of_birth': p.date_of_birth,
                        'gender': p.gender,
                        'admission_date': p.admission_date.isoformat() if p.admission_date else None,
                        'current_status': p.current_status,
                        'created_at': p.created_at.isoformat() if p.created_at else None
                    }
                    for p in patients_q
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get all patients: {e}")
            return []
    
    def create_patient(self, patient_data: Dict[str, Any]) -> bool:
        """Create a new patient"""
        try:
            with self.get_session() as session:
                patient = Patient(
                    patient_id=patient_data['patient_id'],
                    patient_name=patient_data.get('patient_name', ''),
                    date_of_birth=patient_data.get('date_of_birth'),
                    gender=patient_data.get('gender'),
                    admission_date=datetime.fromisoformat(patient_data.get('admission_date', datetime.now().isoformat())),
                    current_status=patient_data.get('current_status', 'active')
                )
                session.add(patient)
                session.commit()
                logger.info(f"✅ Created patient {patient_data['patient_id']}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to create patient: {e}")
            return False
    
    # ===== DIAGNOSIS OPERATIONS =====
    
    def save_diagnosis_session(self, session_id: str, patient_id: str, diagnosis_data: Dict[str, Any]) -> bool:
        """Save diagnosis session - FIXED to auto-create patients if they don't exist"""
        try:
            with self.get_session() as session:
                # CRITICAL FIX: Ensure patient exists before saving diagnosis (foreign key constraint)
                patient = session.query(Patient).filter_by(patient_id=patient_id).first()
                if not patient:
                    # Auto-create patient record to prevent foreign key violations
                    print(f"🏥 AUTO-CREATING patient record for {patient_id}")
                    
                    # Extract patient info from diagnosis_data if available
                    patient_input = diagnosis_data.get('patient_input', {})
                    if isinstance(patient_input, dict):
                        patient_name = patient_input.get('patient_name', f"Patient {patient_id}")
                        gender = patient_input.get('gender', 'unknown')
                        date_of_birth = patient_input.get('date_of_birth')
                    elif hasattr(patient_input, '__dict__'):
                        patient_name = getattr(patient_input, 'patient_name', f"Patient {patient_id}")
                        gender = getattr(patient_input, 'gender', 'unknown')
                        date_of_birth = getattr(patient_input, 'date_of_birth', None)
                    else:
                        patient_name = f"Patient {patient_id}"
                        gender = "unknown"
                        date_of_birth = None
                    
                    patient = Patient(
                        patient_id=patient_id,
                        patient_name=patient_name,
                        gender=gender,
                        date_of_birth=date_of_birth,
                        current_status="active",
                        admission_date=datetime.now()
                    )
                    session.add(patient)
                    session.flush()  # Ensure patient is created before diagnosis session
                    print(f"✅ Patient {patient_id} AUTO-CREATED: {patient_name}, {gender}")
                else:
                    print(f"✅ Patient {patient_id} already exists, proceeding with diagnosis save")
                
                existing_session = session.query(DiagnosisSession).filter_by(session_id=session_id).first()
                
                if existing_session:
                    # Update existing session
                    existing_session.status = diagnosis_data.get('status', 'completed')
                    existing_session.patient_input = diagnosis_data.get('patient_input', {})
                    
                    # Handle DiagnosisResult object properly
                    diagnosis_result = diagnosis_data.get('diagnosis_result')
                    if diagnosis_result:
                        if hasattr(diagnosis_result, 'dict'):  # Pydantic model
                            existing_session.diagnosis_result = diagnosis_result.dict()
                            existing_session.confidence_score = getattr(diagnosis_result, 'confidence_score', 0.0)
                            existing_session.primary_diagnosis = getattr(diagnosis_result, 'primary_diagnosis', '')
                        elif isinstance(diagnosis_result, dict):  # Already a dict
                            existing_session.diagnosis_result = diagnosis_result
                            existing_session.confidence_score = diagnosis_result.get('confidence_score', 0.0)
                            existing_session.primary_diagnosis = diagnosis_result.get('primary_diagnosis', '')
                        else:  # Fallback
                            existing_session.diagnosis_result = {}
                            existing_session.confidence_score = 0.0
                            existing_session.primary_diagnosis = str(diagnosis_result) if diagnosis_result else ''
                    
                    existing_session.processing_time = diagnosis_data.get('processing_time', 0.0)
                    existing_session.ai_model_used = diagnosis_data.get('ai_model_used', '')
                    existing_session.verification_status = diagnosis_data.get('verification_status', '')
                    existing_session.updated_at = datetime.now()
                else:
                    # Create new session - handle DiagnosisResult object properly
                    diagnosis_result = diagnosis_data.get('diagnosis_result')
                    diagnosis_result_dict = {}
                    confidence_score = 0.0
                    primary_diagnosis = ''
                    
                    if diagnosis_result:
                        if hasattr(diagnosis_result, 'dict'):  # Pydantic model
                            diagnosis_result_dict = diagnosis_result.dict()
                            confidence_score = getattr(diagnosis_result, 'confidence_score', 0.0)
                            primary_diagnosis = getattr(diagnosis_result, 'primary_diagnosis', '')
                        elif isinstance(diagnosis_result, dict):  # Already a dict
                            diagnosis_result_dict = diagnosis_result
                            confidence_score = diagnosis_result.get('confidence_score', 0.0)
                            primary_diagnosis = diagnosis_result.get('primary_diagnosis', '')
                        else:  # Fallback
                            primary_diagnosis = str(diagnosis_result) if diagnosis_result else ''
                    
                    new_session = DiagnosisSession(
                        session_id=session_id,
                        patient_id=patient_id,
                        status=diagnosis_data.get('status', 'completed'),
                        patient_input=diagnosis_data.get('patient_input', {}),
                        diagnosis_result=diagnosis_result_dict,
                        confidence_score=confidence_score,
                        primary_diagnosis=primary_diagnosis,
                        processing_time=diagnosis_data.get('processing_time', 0.0),
                        ai_model_used=diagnosis_data.get('ai_model_used', ''),
                        verification_status=diagnosis_data.get('verification_status', '')
                    )
                    session.add(new_session)
                
                session.commit()
                logger.info(f"✅ Saved diagnosis session {session_id} for patient {patient_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save diagnosis session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_diagnosis_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get diagnosis session by ID"""
        try:
            with self.get_session() as session:
                diagnosis = session.query(DiagnosisSession).filter_by(session_id=session_id).first()
                if diagnosis:
                    return {
                        'session_id': diagnosis.session_id,
                        'patient_id': diagnosis.patient_id,
                        'created_at': diagnosis.created_at.isoformat() if diagnosis.created_at else None,
                        'status': diagnosis.status,
                        'patient_input': diagnosis.patient_input,
                        'diagnosis_result': diagnosis.diagnosis_result,
                        'confidence_score': diagnosis.confidence_score,
                        'primary_diagnosis': diagnosis.primary_diagnosis,
                        'processing_time': diagnosis.processing_time,
                        'ai_model_used': diagnosis.ai_model_used,
                        'verification_status': diagnosis.verification_status
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get diagnosis session {session_id}: {e}")
            return None
    
    def get_patient_diagnosis_sessions(self, patient_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get all diagnosis sessions for a patient - with FULL details including patient_input and diagnosis_result"""
        try:
            with self.get_session() as session:
                # Set timeout on query execution to prevent hanging
                session.connection(execution_options={'statement_timeout': 5000})  # 5 second timeout
                
                # Select ALL columns including patient_input and diagnosis_result for complete data
                query = session.query(DiagnosisSession).filter(DiagnosisSession.patient_id == patient_id).order_by(DiagnosisSession.created_at.desc())
                if limit:
                    query = query.limit(limit)
                else:
                    # Set a reasonable default limit to prevent large queries
                    query = query.limit(50)
                    
                sessions = query.all()
                return [
                    {
                        'session_id': s.session_id,
                        'created_at': s.created_at.isoformat() if s.created_at else None,
                        'updated_at': s.updated_at.isoformat() if s.updated_at else None,
                        'status': s.status,
                        'primary_diagnosis': s.primary_diagnosis,
                        'confidence_score': s.confidence_score,
                        'processing_time': s.processing_time,
                        'ai_model_used': s.ai_model_used,
                        'verification_status': s.verification_status,
                        'patient_input': s.patient_input,  # Include patient_input for symptoms
                        'diagnosis_result': s.diagnosis_result  # Include diagnosis_result for complete details
                    }
                    for s in sessions
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get diagnosis sessions for {patient_id}: {e}")
            return []
    
    # ===== CONCERN SCORE OPERATIONS =====
    
    def add_concern_score(self, patient_id: str, concern_score: float, risk_level: str, 
                         risk_factors: List[str], metadata_patterns: Dict[str, Any]) -> bool:
        """Add CONCERN score for a patient"""
        try:
            with self.get_session() as session:
                concern = ConcernScore(
                    patient_id=patient_id,
                    concern_score=concern_score,
                    risk_level=risk_level,
                    risk_factors=risk_factors,
                    metadata_patterns=metadata_patterns,
                    alert_triggered=risk_level in ['high', 'critical']
                )
                session.add(concern)
                session.commit()
                logger.info(f"✅ Added CONCERN score for {patient_id}: {risk_level} ({concern_score:.2f})")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to add CONCERN score for {patient_id}: {e}")
            return False
    
    def get_latest_concern_score(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get latest CONCERN score for a patient"""
        try:
            with self.get_session() as session:
                concern = session.query(ConcernScore).filter_by(patient_id=patient_id).order_by(ConcernScore.created_at.desc()).first()
                if concern:
                    return {
                        'patient_id': concern.patient_id,
                        'concern_score': concern.concern_score,
                        'risk_level': concern.risk_level,
                        'risk_factors': concern.risk_factors,
                        'metadata_patterns': concern.metadata_patterns,
                        'created_at': concern.created_at.isoformat() if concern.created_at else None,
                        'alert_triggered': concern.alert_triggered
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get CONCERN score for {patient_id}: {e}")
            return None
    
    # ===== CHAT OPERATIONS =====
    
    def save_chat_message(self, patient_id: str, message: str, response: str = None) -> bool:
        """Save chat message"""
        try:
            with self.get_session() as session:
                chat_msg = ChatMessage(
                    patient_id=patient_id,
                    message=message,
                    response=response,
                    message_metadata={}
                )
                session.add(chat_msg)
                session.commit()
                logger.info(f"✅ Saved chat message for {patient_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save chat message for {patient_id}: {e}")
            return False
    
    def get_patient_chat_history(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a patient"""
        try:
            with self.get_session() as session:
                messages = session.query(ChatMessage).filter_by(patient_id=patient_id).order_by(ChatMessage.timestamp.desc()).limit(limit).all()
                return [
                    {
                        'message_id': msg.message_id,
                        'patient_id': msg.patient_id,
                        'message': msg.message,
                        'response': msg.response,
                        'timestamp': msg.timestamp.isoformat() if msg.timestamp else None
                    }
                    for msg in messages
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get chat history for {patient_id}: {e}")
            return []
    
    # ===== CLINICAL NOTES OPERATIONS =====
    
    def add_clinical_note(self, patient_id: str, nurse_id: str, content: str, 
                         location: str = None, shift: str = None, note_type: str = "nursing") -> bool:
        """Add a clinical note"""
        try:
            with self.get_session() as session:
                note = ClinicalNote(
                    patient_id=patient_id,
                    nurse_id=nurse_id,
                    content=content,
                    location=location,
                    shift=shift,
                    note_type=note_type,
                    timestamp=datetime.now()
                )
                session.add(note)
                session.commit()
                logger.info(f"✅ Added clinical note for patient {patient_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to add clinical note: {e}")
            return False
    
    def get_patient_clinical_notes(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get clinical notes for a patient"""
        try:
            with self.get_session() as session:
                notes_q = session.query(
                    ClinicalNote.note_id,
                    ClinicalNote.patient_id,
                    ClinicalNote.nurse_id,
                    ClinicalNote.content,
                    ClinicalNote.location,
                    ClinicalNote.shift,
                    ClinicalNote.note_type,
                    ClinicalNote.timestamp
                ).filter(ClinicalNote.patient_id == patient_id).order_by(ClinicalNote.timestamp.desc()).limit(limit).yield_per(200)

                return [
                    {
                        'note_id': note.note_id,
                        'patient_id': note.patient_id,
                        'nurse_id': note.nurse_id,
                        'content': note.content,
                        'location': note.location,
                        'shift': note.shift,
                        'note_type': note.note_type,
                        'timestamp': note.timestamp.isoformat() if note.timestamp else None
                    }
                    for note in notes_q
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get clinical notes for {patient_id}: {e}")
            return []
    
    # ===== PATIENT VISITS OPERATIONS =====
    
    def add_patient_visit(self, patient_id: str, nurse_id: str, location: str,
                         visit_type: str = "routine", duration_minutes: int = 5, notes: str = None) -> bool:
        """Add a patient visit record"""
        try:
            with self.get_session() as session:
                visit = PatientVisit(
                    patient_id=patient_id,
                    nurse_id=nurse_id,
                    location=location,
                    visit_type=visit_type,
                    duration_minutes=duration_minutes,
                    notes=notes,
                    timestamp=datetime.now()
                )
                session.add(visit)
                session.commit()
                logger.info(f"✅ Added patient visit for patient {patient_id}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to add patient visit: {e}")
            return False
    
    def get_patient_visits(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get patient visits for a patient"""
        try:
            with self.get_session() as session:
                visits_q = session.query(
                    PatientVisit.visit_id,
                    PatientVisit.patient_id,
                    PatientVisit.nurse_id,
                    PatientVisit.location,
                    PatientVisit.visit_type,
                    PatientVisit.duration_minutes,
                    PatientVisit.notes,
                    PatientVisit.timestamp
                ).filter(PatientVisit.patient_id == patient_id).order_by(PatientVisit.timestamp.desc()).limit(limit).yield_per(200)

                return [
                    {
                        'visit_id': visit.visit_id,
                        'patient_id': visit.patient_id,
                        'nurse_id': visit.nurse_id,
                        'location': visit.location,
                        'visit_type': visit.visit_type,
                        'duration_minutes': visit.duration_minutes,
                        'notes': visit.notes,
                        'timestamp': visit.timestamp.isoformat() if visit.timestamp else None
                    }
                    for visit in visits_q
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get patient visits for {patient_id}: {e}")
            return []

    # ===== GRADCAM OPERATIONS =====
    
    async def save_gradcam_image(self, gradcam_record: Dict[str, Any]) -> bool:
        """Save a single GradCAM image to PostgreSQL database"""
        try:
            with self.get_session() as session:
                gradcam_image = GradCAMImage(
                    session_id=gradcam_record.get('session_id', ''),
                    patient_id=gradcam_record.get('patient_id', ''),
                    original_image_path=gradcam_record.get('original_image_path', ''),
                    image_filename=gradcam_record.get('image_filename', ''),
                    heatmap_image=gradcam_record.get('heatmap_image', ''),
                    overlay_image=gradcam_record.get('overlay_image', ''),
                    volume_image=gradcam_record.get('volume_image', ''),
                    analysis_data=gradcam_record.get('analysis_data', {}),
                    predictions=gradcam_record.get('predictions', []),
                    activation_regions=gradcam_record.get('activation_regions', []),
                    medical_interpretation=gradcam_record.get('medical_interpretation', {}),
                    processing_successful=gradcam_record.get('processing_successful', False),
                    processing_time=gradcam_record.get('processing_time', 0.0),
                    error_message=gradcam_record.get('error_message')
                )
                session.add(gradcam_image)
                session.commit()
                logger.info(f"✅ Saved GradCAM image for session {gradcam_record.get('session_id', '')}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to save GradCAM image: {e}")
            return False
    
    async def save_gradcam_images(self, session_id: str, patient_id: str, gradcam_data: list) -> bool:
        """Save GradCAM images to PostgreSQL database"""
        try:
            with self.get_session() as session:
                for item in gradcam_data:
                    if not item.get('success', False):
                        continue
                    
                    # Convert visualizations to base64 strings for storage
                    heatmap_b64 = None
                    overlay_b64 = None
                    volume_b64 = None
                    
                    # Handle both base64_images format and visualizations format
                    if item.get('base64_images'):
                        base64_imgs = item['base64_images']
                        heatmap_b64 = base64_imgs.get('heatmap')
                        overlay_b64 = base64_imgs.get('overlay')
                        volume_b64 = base64_imgs.get('volume')
                    elif item.get('visualizations'):
                        vis = item['visualizations']
                        heatmap_b64 = vis.get('heatmap_image')
                        overlay_b64 = vis.get('overlay_image')
                        volume_b64 = vis.get('volume_image')
                    
                    gradcam_image = GradCAMImage(
                        session_id=session_id,
                        patient_id=patient_id,
                        original_image_path=item.get('original_file', item.get('image_file', '')),
                        image_filename=(item.get('original_file', item.get('image_file', '')) or '').split('/')[-1],
                        heatmap_image=heatmap_b64,
                        overlay_image=overlay_b64,
                        volume_image=volume_b64,
                        analysis_data=item.get('analysis', {}),
                        predictions=item.get('predictions', []),
                        activation_regions=item.get('activation_regions', []),
                        medical_interpretation=item.get('medical_interpretation', {}),
                        processing_successful=item.get('success', False),
                        processing_time=item.get('analysis', {}).get('processing_time', 0.0),
                        error_message=item.get('error')
                    )
                    session.add(gradcam_image)
                
                session.commit()
                logger.info(f"✅ Saved GradCAM images for session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save GradCAM images to PostgreSQL: {e}")
            return False

    def get_gradcam_images_sync(self, session_id: str) -> list:
        """Retrieve GradCAM images from PostgreSQL database (synchronous)"""
        try:
            with self.get_session() as session:
                gradcam_images = session.query(GradCAMImage).filter_by(session_id=session_id)\
                                      .order_by(GradCAMImage.created_at.asc()).all()
                
                result = []
                for img in gradcam_images:
                    visualizations = {}
                    if img.heatmap_image:
                        visualizations['heatmap_image'] = img.heatmap_image
                    if img.overlay_image:
                        visualizations['overlay_image'] = img.overlay_image
                    if img.volume_image:
                        visualizations['volume_image'] = img.volume_image
                    
                    # Create the structure that the frontend expects
                    item = {
                        'success': img.processing_successful,
                        'image_file': img.image_filename or img.original_image_path.split('/')[-1],
                        'original_file': img.original_image_path,
                        'visualizations': visualizations if visualizations else None,
                        'analysis': {
                            'predicted_class': (img.analysis_data or {}).get('predicted_class', 'Unknown'),
                            'confidence_score': (img.analysis_data or {}).get('confidence_score', 0.0),
                            'processing_time': img.processing_time or 0.0,
                            'activation_regions_count': len(img.activation_regions or [])
                        },
                        'predictions': img.predictions or [],
                        'activation_regions': img.activation_regions or [],
                        'medical_interpretation': img.medical_interpretation or {},
                        'error': img.error_message
                    }
                    result.append(item)
                
                logger.info(f"✅ Retrieved {len(result)} GradCAM images for session {session_id}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Error retrieving GradCAM images: {e}")
            return []

    # ===== CONCERN SEVERITY TRACKING OPERATIONS =====
    
    def update_patient_severity(self, patient_id: str, diagnosis_confidence: float, 
                               fol_verification: float = None, enhanced_verification: float = None,
                               explainability_score: float = None, imaging_present: bool = False,
                               diagnosis_id: str = None) -> Dict[str, Any]:
        """Update patient severity tracking with cumulative scoring"""
        try:
            with self.get_session() as session:
                # Ensure the patient exists first
                patient = session.query(Patient).filter_by(patient_id=patient_id).first()
                if not patient:
                    # Create a basic patient record if it doesn't exist
                    patient = Patient(
                        patient_id=patient_id,
                        patient_name=f"Patient {patient_id}",
                        current_status="active"
                    )
                    session.add(patient)
                    session.commit()  # Commit patient creation first
                
                # Get or create severity tracking record
                severity_record = session.query(ConcernSeverityTracking).filter_by(patient_id=patient_id).first()
                
                if not severity_record:
                    # First diagnosis - create new record
                    severity_record = ConcernSeverityTracking(
                        patient_id=patient_id,
                        cumulative_severity=0.0,
                        total_diagnoses=0,
                        average_severity=0.0,
                        max_severity_reached=0.0,
                        first_diagnosis_timestamp=datetime.now()
                    )
                    session.add(severity_record)
                
                # Compute current severity score from components
                current_severity = self._compute_severity_score(
                    diagnosis_confidence, fol_verification, enhanced_verification,
                    explainability_score, imaging_present
                )
                
                # Update severity tracking with cumulative approach
                severity_record.total_diagnoses += 1
                severity_record.cumulative_severity += current_severity
                severity_record.average_severity = severity_record.cumulative_severity / severity_record.total_diagnoses
                
                # Store last diagnosis components
                severity_record.last_diagnosis_confidence = diagnosis_confidence
                severity_record.last_fol_verification = fol_verification
                severity_record.last_enhanced_verification = enhanced_verification
                severity_record.last_explainability_score = explainability_score
                severity_record.last_imaging_present = imaging_present
                severity_record.last_computed_severity = current_severity
                severity_record.last_diagnosis_timestamp = datetime.now()
                
                # Update max severity if needed
                if current_severity > severity_record.max_severity_reached:
                    severity_record.max_severity_reached = current_severity
                
                # Calculate final risk score (weighted combination of factors)
                risk_score = self._calculate_risk_score(severity_record)
                severity_record.current_risk_score = risk_score
                severity_record.current_risk_level = self._get_risk_level(risk_score)
                
                # Calculate temporary risk score for this entry
                temp_risk_score = self._calculate_risk_score(severity_record)
                temp_risk_level = self._get_risk_level(temp_risk_score)
                
                # Add to severity history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'severity': current_severity,
                    'risk_level': temp_risk_level,
                    'diagnosis_id': diagnosis_id,
                    'cumulative_severity': severity_record.cumulative_severity,
                    'total_diagnoses': severity_record.total_diagnoses
                }
                
                # Ensure history is properly handled for PostgreSQL JSON column
                if severity_record.severity_history is None:
                    severity_record.severity_history = []
                
                # Create a new list to ensure PostgreSQL detects the change
                updated_history = list(severity_record.severity_history)
                updated_history.append(history_entry)
                severity_record.severity_history = updated_history
                
                # Mark JSON column as modified for SQLAlchemy tracking
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(severity_record, 'severity_history')
                
                # Keep only last 100 entries to prevent excessive growth
                if len(severity_record.severity_history) > 100:
                    severity_record.severity_history = severity_record.severity_history[-100:]
                    flag_modified(severity_record, 'severity_history')
                
                session.commit()
                
                result = {
                    'patient_id': patient_id,
                    'current_severity': current_severity,
                    'cumulative_severity': severity_record.cumulative_severity,
                    'total_diagnoses': severity_record.total_diagnoses,
                    'average_severity': severity_record.average_severity,
                    'risk_level': severity_record.current_risk_level,
                    'risk_score': severity_record.current_risk_score,
                    'max_severity_reached': severity_record.max_severity_reached
                }
                
                logger.info(f"✅ Updated severity for patient {patient_id}: {severity_record.current_risk_level} ({risk_score:.3f})")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to update patient severity for {patient_id}: {e}")
            return None
    
    def get_patient_severity(self, patient_id: str) -> Dict[str, Any]:
        """Get current severity tracking for a patient"""
        try:
            with self.get_session() as session:
                severity_record = session.query(ConcernSeverityTracking).filter_by(patient_id=patient_id).first()
                
                if not severity_record:
                    # Return default/baseline severity
                    return {
                        'patient_id': patient_id,
                        'current_severity': 0.0,
                        'cumulative_severity': 0.0,
                        'total_diagnoses': 0,
                        'average_severity': 0.0,
                        'risk_level': 'low',
                        'risk_score': 0.0,
                        'max_severity_reached': 0.0,
                        'last_diagnosis_timestamp': None,
                        'severity_history': []
                    }
                
                return {
                    'patient_id': severity_record.patient_id,
                    'current_severity': severity_record.last_computed_severity or 0.0,
                    'cumulative_severity': severity_record.cumulative_severity,
                    'total_diagnoses': severity_record.total_diagnoses,
                    'average_severity': severity_record.average_severity,
                    'risk_level': severity_record.current_risk_level,
                    'risk_score': severity_record.current_risk_score,
                    'max_severity_reached': severity_record.max_severity_reached,
                    'last_diagnosis_timestamp': severity_record.last_diagnosis_timestamp.isoformat() if severity_record.last_diagnosis_timestamp else None,
                    'first_diagnosis_timestamp': severity_record.first_diagnosis_timestamp.isoformat() if severity_record.first_diagnosis_timestamp else None,
                    'severity_history': severity_record.severity_history or []
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get patient severity for {patient_id}: {e}")
            return None
    
    def get_all_patient_severities(self) -> List[Dict[str, Any]]:
        """Get severity tracking for all patients"""
        try:
            with self.get_session() as session:
                severity_records = session.query(ConcernSeverityTracking)\
                    .order_by(ConcernSeverityTracking.current_risk_score.desc()).all()
                
                return [
                    {
                        'patient_id': record.patient_id,
                        'cumulative_severity': record.cumulative_severity,
                        'total_diagnoses': record.total_diagnoses,
                        'average_severity': record.average_severity,
                        'risk_level': record.current_risk_level,
                        'risk_score': record.current_risk_score,
                        'max_severity_reached': record.max_severity_reached,
                        'last_diagnosis_timestamp': record.last_diagnosis_timestamp.isoformat() if record.last_diagnosis_timestamp else None
                    }
                    for record in severity_records
                ]
                
        except Exception as e:
            logger.error(f"❌ Failed to get all patient severities: {e}")
            return []
    
    def _compute_severity_score(self, diagnosis_confidence: float, 
                               fol_verification: float = None, enhanced_verification: float = None,
                               explainability_score: float = None, imaging_present: bool = False) -> float:
        """Compute severity score from diagnosis components"""
        # Base severity from diagnosis confidence
        severity = diagnosis_confidence or 0.0
        
        # Add bonus for verification methods
        if fol_verification is not None:
            severity += fol_verification * 0.2  # FOL adds up to 20% 
        
        if enhanced_verification is not None:
            severity += enhanced_verification * 0.15  # Enhanced verification adds up to 15%
        
        if explainability_score is not None:
            severity += explainability_score * 0.1  # Explainability adds up to 10%
        
        if imaging_present:
            severity += 0.05  # Imaging presence adds 5%
        
        # Cap at 1.0 maximum
        return min(severity, 1.0)
    
    def _calculate_risk_score(self, severity_record: ConcernSeverityTracking) -> float:
        """Calculate overall risk score from severity tracking record"""
        # Weighted combination of factors:
        # - 40% current average severity
        # - 30% max severity reached (historical peak)
        # - 20% total diagnoses count (normalized)
        # - 10% recency factor
        
        avg_weight = 0.4
        max_weight = 0.3
        count_weight = 0.2
        recency_weight = 0.1
        
        # Average severity component
        avg_component = severity_record.average_severity * avg_weight
        
        # Max severity component
        max_component = severity_record.max_severity_reached * max_weight
        
        # Diagnosis count component (normalize by expected maximum of ~10 diagnoses)
        normalized_count = min(severity_record.total_diagnoses / 10.0, 1.0)
        count_component = normalized_count * count_weight
        
        # Recency component (boost if diagnosed recently)
        if severity_record.last_diagnosis_timestamp:
            hours_since = (datetime.now() - severity_record.last_diagnosis_timestamp).total_seconds() / 3600
            # Decay over 24 hours
            recency_factor = max(0, 1 - (hours_since / 24.0))
            recency_component = recency_factor * recency_weight
        else:
            recency_component = 0
        
        total_risk_score = avg_component + max_component + count_component + recency_component
        return min(total_risk_score, 1.0)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def update_patient_risk_from_llm(self, patient_id: str, risk_level: str, 
                                     risk_confidence: float, reasoning: str,
                                     recommendations: List[str], diagnosis_id: str = None) -> Dict[str, Any]:
        """Update patient risk based on LLM classification"""
        try:
            with self.get_session() as session:
                # Ensure patient exists
                patient = session.query(Patient).filter_by(patient_id=patient_id).first()
                if not patient:
                    # Create minimal patient record
                    patient = Patient(
                        patient_id=patient_id,
                        patient_name=f"Patient {patient_id}",
                        gender="unknown"
                    )
                    session.add(patient)
                    session.commit()
                
                # Get or create severity tracking record
                severity_record = session.query(ConcernSeverityTracking).filter_by(patient_id=patient_id).first()
                
                if not severity_record:
                    # First diagnosis - create new record
                    severity_record = ConcernSeverityTracking(
                        patient_id=patient_id,
                        cumulative_severity=0.0,
                        total_diagnoses=0,
                        average_severity=0.0,
                        max_severity_reached=0.0,
                        first_diagnosis_timestamp=datetime.now()
                    )
                    session.add(severity_record)
                
                # Update with LLM results
                severity_record.total_diagnoses += 1
                severity_record.current_risk_level = risk_level
                severity_record.current_risk_score = risk_confidence  # Use confidence as risk score
                severity_record.last_diagnosis_timestamp = datetime.now()
                
                # Convert risk level to numeric severity for backwards compatibility
                severity_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
                current_severity = severity_map.get(risk_level, 0.5)
                
                # Update cumulative metrics
                severity_record.cumulative_severity += current_severity
                severity_record.average_severity = severity_record.cumulative_severity / severity_record.total_diagnoses
                severity_record.last_computed_severity = current_severity
                
                if current_severity > severity_record.max_severity_reached:
                    severity_record.max_severity_reached = current_severity
                
                # Store LLM-specific data in JSON fields (we'll store in severity_history)
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'severity': current_severity,
                    'risk_level': risk_level,
                    'risk_confidence': risk_confidence,
                    'reasoning': reasoning,
                    'recommendations': recommendations,
                    'diagnosis_id': diagnosis_id,
                    'method': 'llm_classification'
                }
                
                # Update history
                if severity_record.severity_history is None:
                    severity_record.severity_history = []
                
                updated_history = list(severity_record.severity_history)
                updated_history.append(history_entry)
                severity_record.severity_history = updated_history
                
                # Mark as modified
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(severity_record, 'severity_history')
                
                # Keep only last 100 entries
                if len(severity_record.severity_history) > 100:
                    severity_record.severity_history = severity_record.severity_history[-100:]
                    flag_modified(severity_record, 'severity_history')
                
                session.commit()
                
                result = {
                    'patient_id': patient_id,
                    'risk_level': risk_level,
                    'risk_confidence': risk_confidence,
                    'reasoning': reasoning,
                    'recommendations': recommendations,
                    'total_diagnoses': severity_record.total_diagnoses,
                    'current_severity': current_severity
                }
                
                logger.info(f"✅ Updated LLM risk for patient {patient_id}: {risk_level} (confidence: {risk_confidence:.2f})")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to update LLM risk for {patient_id}: {e}")
            return None
    
    # ===== UTILITY OPERATIONS =====
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            with self.get_session() as session:
                # Test query
                patient_count = session.query(Patient).count()
                return {
                    'status': 'healthy',
                    'database_type': 'postgresql',
                    'patient_count': patient_count,
                    'connection_url': self.database_url.split('@')[-1] if '@' in self.database_url else 'localhost'
                }
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'database_type': 'postgresql',
                'error': str(e)
            }

# Global instance
postgresql_db = None

def get_postgresql_database(database_url: str = None) -> PostgreSQLDatabase:
    """Get PostgreSQL database instance"""
    global postgresql_db
    if postgresql_db is None:
        postgresql_db = PostgreSQLDatabase(database_url)
    return postgresql_db
