"""
Optimized Database Manager for CortexMD
High-Performance PostgreSQL Implementation with Caching
"""

import os
import logging
import redis
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, joinedload, selectinload
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.pool import QueuePool
from functools import wraps
import uuid

logger = logging.getLogger(__name__)

# Redis cache configuration
REDIS_CACHE_ENABLED = os.getenv('REDIS_CACHE_ENABLED', 'true').lower() == 'true'
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# Cache TTL settings (in seconds)
CACHE_TTL_PATIENT = 300  # 5 minutes
CACHE_TTL_DIAGNOSIS = 180  # 3 minutes
CACHE_TTL_DASHBOARD = 60  # 1 minute

Base = declarative_base()

# ===== OPTIMIZED POSTGRESQL MODELS =====

class Patient(Base):
    __tablename__ = 'patients'
    
    patient_id = Column(String(50), primary_key=True, index=True)  # Added index
    patient_name = Column(String(200), nullable=False, index=True)  # Added index for searches
    date_of_birth = Column(String(20))
    gender = Column(String(20), index=True)  # Added index
    admission_date = Column(DateTime, default=datetime.now, index=True)  # Added index
    current_status = Column(String(50), default='active', index=True)  # Added index
    created_at = Column(DateTime, default=datetime.now, index=True)  # Added index
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships with lazy loading optimization
    diagnosis_sessions = relationship("DiagnosisSession", back_populates="patient", lazy='select')
    concern_scores = relationship("ConcernScore", back_populates="patient", lazy='select')
    concern_severity = relationship("ConcernSeverityTracking", back_populates="patient", uselist=False, lazy='select')
    chat_messages = relationship("ChatMessage", back_populates="patient", lazy='select')
    clinical_notes = relationship("ClinicalNote", back_populates="patient", lazy='select')
    patient_visits = relationship("PatientVisit", back_populates="patient", lazy='select')

class DiagnosisSession(Base):
    __tablename__ = 'diagnosis_sessions'
    
    session_id = Column(String(50), primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)  # Added index
    created_at = Column(DateTime, default=datetime.now, index=True)  # Added index
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    status = Column(String(50), default='pending', index=True)  # Added index
    
    # Diagnosis data - optimized storage
    patient_input_summary = Column(Text)  # Store summary instead of full JSON
    diagnosis_summary = Column(Text)  # Store summary for quick access
    confidence_score = Column(Float, index=True)  # Added index
    primary_diagnosis = Column(Text, index=True)  # Added index for searches
    
    # Full data stored separately for when needed
    patient_input = Column(JSON)
    diagnosis_result = Column(JSON)
    
    # Processing metadata
    processing_time = Column(Float)
    ai_model_used = Column(String(100), index=True)  # Added index
    verification_status = Column(String(50), index=True)  # Added index
    
    # Relationships
    patient = relationship("Patient", back_populates="diagnosis_sessions")
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_patient_created', 'patient_id', 'created_at'),
        Index('idx_patient_status', 'patient_id', 'status'),
        Index('idx_status_created', 'status', 'created_at'),
    )

class ConcernScore(Base):
    __tablename__ = 'concern_scores'
    
    id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    concern_score = Column(Float, nullable=False, index=True)  # Added index
    risk_level = Column(String(20), nullable=False, index=True)  # Added index
    risk_factors = Column(JSON)
    metadata_patterns = Column(JSON)
    created_at = Column(DateTime, default=datetime.now, index=True)  # Added index
    alert_triggered = Column(Boolean, default=False, index=True)  # Added index
    
    # Relationships
    patient = relationship("Patient", back_populates="concern_scores")
    
    # Composite indexes
    __table_args__ = (
        Index('idx_patient_created_concern', 'patient_id', 'created_at'),
        Index('idx_risk_created', 'risk_level', 'created_at'),
    )

class ConcernSeverityTracking(Base):
    __tablename__ = 'concern_severity_tracking'
    
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), primary_key=True, index=True)
    cumulative_concern_score = Column(Float, default=0.0, index=True)  # Added index
    risk_level = Column(String(20), default='low', index=True)  # Added index
    total_diagnoses = Column(Integer, default=0)
    total_severity_events = Column(Integer, default=0)
    average_severity = Column(Float, default=0.0)
    trend_direction = Column(String(20), default='stable')
    last_concern_score = Column(Float, default=0.0)
    
    # Enhanced tracking fields
    highest_risk_achieved = Column(String(20), default='low')
    risk_escalation_count = Column(Integer, default=0)
    days_since_last_critical = Column(Integer, default=0)
    
    # Metadata
    last_diagnosis_timestamp = Column(DateTime, default=datetime.now, index=True)  # Added index
    first_diagnosis_timestamp = Column(DateTime, default=datetime.now)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    patient = relationship("Patient", back_populates="concern_severity")

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    message_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    message = Column(Text, nullable=False)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.now, index=True)  # Added index
    message_metadata = Column(JSON)
    
    # Relationships
    patient = relationship("Patient", back_populates="chat_messages")
    
    # Composite index
    __table_args__ = (
        Index('idx_patient_timestamp', 'patient_id', 'timestamp'),
    )

class ClinicalNote(Base):
    __tablename__ = 'clinical_notes'
    
    note_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    nurse_id = Column(String(50), index=True)  # Added index
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)  # Added index
    note_type = Column(String(50), default='general', index=True)  # Added field and index
    
    # Relationships
    patient = relationship("Patient", back_populates="clinical_notes")
    
    # Composite indexes
    __table_args__ = (
        Index('idx_patient_timestamp_notes', 'patient_id', 'timestamp'),
        Index('idx_nurse_timestamp', 'nurse_id', 'timestamp'),
    )

class PatientVisit(Base):
    __tablename__ = 'patient_visits'
    
    visit_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    patient_id = Column(String(50), ForeignKey('patients.patient_id'), nullable=False, index=True)
    visit_date = Column(DateTime, default=datetime.now, index=True)  # Added index
    visit_type = Column(String(50), default='routine', index=True)  # Added index
    diagnosis = Column(Text)
    treatment = Column(Text)
    notes = Column(Text)
    
    # Relationships
    patient = relationship("Patient", back_populates="patient_visits")
    
    # Composite index
    __table_args__ = (
        Index('idx_patient_visit_date', 'patient_id', 'visit_date'),
    )

# ===== CACHE DECORATOR =====

def cache_result(cache_key_prefix: str, ttl: int = 300):
    """Decorator to cache function results in Redis"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not REDIS_CACHE_ENABLED or not hasattr(self, 'redis_client'):
                return func(self, *args, **kwargs)
            
            # Create cache key
            cache_key = f"{cache_key_prefix}:{':'.join(map(str, args))}"
            
            try:
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache HIT for {cache_key}")
                    return json.loads(cached_result)
                
                # Execute function and cache result
                logger.debug(f"Cache MISS for {cache_key}")
                result = func(self, *args, **kwargs)
                
                # Cache the result
                self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(result, default=str)
                )
                
                return result
                
            except Exception as e:
                logger.warning(f"Cache error for {cache_key}: {e}")
                return func(self, *args, **kwargs)
        
        return wrapper
    return decorator

# ===== OPTIMIZED POSTGRESQL DATABASE MANAGER =====

class OptimizedPostgreSQLDatabase:
    """High-performance PostgreSQL database manager with caching"""
    
    def __init__(self, database_url: str = None):
        """Initialize optimized PostgreSQL connection"""
        if not database_url:
            database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql://postgres:password@localhost:5432/cortexmd'
            )
        
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.redis_client = None
        
        self._connect()
        self._setup_cache()
    
    def _connect(self):
        """Connect with optimized connection pooling"""
        try:
            # Highly optimized connection pool
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=50,          # Increased pool size
                max_overflow=100,      # Increased overflow
                pool_pre_ping=True,
                pool_recycle=1800,     # 30 minutes
                pool_timeout=30,       # Connection timeout
                echo=False,            # No SQL logging for performance
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "CortexMD_Optimized",
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 5,
                    "keepalives_count": 5,
                    # Note: work_mem, shared_buffers, effective_cache_size are server-side configs
                    # They should be set in postgresql.conf, not in connection args
                }
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine,
                expire_on_commit=False  # Prevent lazy loading issues
            )
            
            # Create tables and indexes
            Base.metadata.create_all(bind=self.engine)
            
            logger.info(f"✅ Optimized PostgreSQL connected: {self.database_url.split('@')[-1]}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    def _setup_cache(self):
        """Setup Redis cache if enabled"""
        if REDIS_CACHE_ENABLED:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, 
                    port=REDIS_PORT, 
                    db=REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Redis cache connected: {REDIS_HOST}:{REDIS_PORT}")
                
            except Exception as e:
                logger.warning(f"⚠️ Redis cache unavailable: {e}")
                self.redis_client = None
        else:
            logger.info("📋 Redis caching disabled")
    
    def get_session(self) -> Session:
        """Get an optimized database session"""
        return self.SessionLocal()
    
    def invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if self.redis_client:
            try:
                keys = self.redis_client.keys(f"*{pattern}*")
                if keys:
                    self.redis_client.delete(*keys)
                    logger.debug(f"Invalidated {len(keys)} cache entries for pattern: {pattern}")
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")
    
    # ===== OPTIMIZED PATIENT OPERATIONS =====
    
    @cache_result("patient", CACHE_TTL_PATIENT)
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient by ID with caching"""
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
    
    @cache_result("all_patients", CACHE_TTL_PATIENT)
    def get_all_patients(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all patients with pagination and caching"""
        try:
            with self.get_session() as session:
                patients = session.query(Patient)\
                    .order_by(Patient.created_at.desc())\
                    .limit(limit)\
                    .offset(offset)\
                    .all()
                
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
                    for p in patients
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get all patients: {e}")
            return []
    
    @cache_result("patient_summary", CACHE_TTL_DIAGNOSIS)
    def get_patient_diagnosis_sessions_summary(self, patient_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get diagnosis sessions summary (fast) - without full JSON data"""
        try:
            with self.get_session() as session:
                sessions = session.query(DiagnosisSession)\
                    .filter_by(patient_id=patient_id)\
                    .order_by(DiagnosisSession.created_at.desc())\
                    .limit(limit)\
                    .all()
                
                return [
                    {
                        'session_id': s.session_id,
                        'created_at': s.created_at.isoformat() if s.created_at else None,
                        'status': s.status,
                        'primary_diagnosis': s.primary_diagnosis,
                        'confidence_score': s.confidence_score,
                        'processing_time': s.processing_time,
                        'ai_model_used': s.ai_model_used,
                        'verification_status': s.verification_status,
                        'diagnosis_summary': s.diagnosis_summary,
                        'patient_input_summary': s.patient_input_summary
                    }
                    for s in sessions
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get diagnosis sessions summary for {patient_id}: {e}")
            return []
    
    def get_patient_diagnosis_session_full(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full diagnosis session details (only when needed)"""
        try:
            with self.get_session() as session:
                s = session.query(DiagnosisSession).filter_by(session_id=session_id).first()
                if s:
                    return {
                        'session_id': s.session_id,
                        'patient_id': s.patient_id,
                        'created_at': s.created_at.isoformat() if s.created_at else None,
                        'updated_at': s.updated_at.isoformat() if s.updated_at else None,
                        'status': s.status,
                        'primary_diagnosis': s.primary_diagnosis,
                        'confidence_score': s.confidence_score,
                        'patient_input': s.patient_input,      # Full data
                        'diagnosis_result': s.diagnosis_result,  # Full data
                        'processing_time': s.processing_time,
                        'ai_model_used': s.ai_model_used,
                        'verification_status': s.verification_status
                    }
                return None
        except Exception as e:
            logger.error(f"❌ Failed to get full diagnosis session {session_id}: {e}")
            return None
    
    @cache_result("patient_dashboard", CACHE_TTL_DASHBOARD)
    def get_patient_dashboard_optimized(self, patient_id: str) -> Dict[str, Any]:
        """Get optimized patient dashboard with single query and caching"""
        try:
            with self.get_session() as session:
                # Single query with joins to get all related data efficiently
                patient = session.query(Patient)\
                    .options(
                        selectinload(Patient.concern_severity),
                        selectinload(Patient.concern_scores).limit(5),
                    )\
                    .filter_by(patient_id=patient_id)\
                    .first()
                
                if not patient:
                    return {'error': 'Patient not found'}
                
                # Get recent diagnosis sessions (summary only)
                recent_sessions = session.query(DiagnosisSession)\
                    .filter_by(patient_id=patient_id)\
                    .order_by(DiagnosisSession.created_at.desc())\
                    .limit(5)\
                    .all()
                
                # Get recent concern scores
                concern_scores = [
                    {
                        'concern_score': cs.concern_score,
                        'risk_level': cs.risk_level,
                        'created_at': cs.created_at.isoformat(),
                        'alert_triggered': cs.alert_triggered
                    }
                    for cs in patient.concern_scores[:5]
                ]
                
                # Build dashboard response
                dashboard = {
                    'patient_info': {
                        'patient_id': patient.patient_id,
                        'patient_name': patient.patient_name,
                        'current_status': patient.current_status,
                        'admission_date': patient.admission_date.isoformat() if patient.admission_date else None
                    },
                    'recent_diagnoses': [
                        {
                            'session_id': s.session_id,
                            'created_at': s.created_at.isoformat(),
                            'primary_diagnosis': s.primary_diagnosis,
                            'confidence_score': s.confidence_score,
                            'status': s.status
                        }
                        for s in recent_sessions
                    ],
                    'concern_data': {
                        'current_risk_level': patient.concern_severity.risk_level if patient.concern_severity else 'low',
                        'cumulative_score': patient.concern_severity.cumulative_concern_score if patient.concern_severity else 0.0,
                        'total_diagnoses': patient.concern_severity.total_diagnoses if patient.concern_severity else 0,
                        'trend_direction': patient.concern_severity.trend_direction if patient.concern_severity else 'stable'
                    },
                    'recent_concern_scores': concern_scores,
                    'stats': {
                        'total_diagnoses': len(recent_sessions),
                        'total_concern_entries': len(concern_scores)
                    }
                }
                
                return dashboard
                
        except Exception as e:
            logger.error(f"❌ Failed to get optimized patient dashboard for {patient_id}: {e}")
            return {'error': str(e)}
    
    def create_patient(self, patient_data: Dict[str, Any]) -> bool:
        """Create patient and invalidate cache"""
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
                
                # Invalidate related caches
                self.invalidate_cache("all_patients")
                self.invalidate_cache(f"patient:{patient_data['patient_id']}")
                
                logger.info(f"✅ Created patient {patient_data['patient_id']}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to create patient: {e}")
            return False
    
    def update_patient_status(self, patient_id: str, status: str) -> bool:
        """Update patient status and invalidate cache"""
        try:
            with self.get_session() as session:
                patient = session.query(Patient).filter_by(patient_id=patient_id).first()
                if patient:
                    patient.current_status = status
                    patient.updated_at = datetime.now()
                    session.commit()
                    
                    # Invalidate caches
                    self.invalidate_cache(f"patient:{patient_id}")
                    self.invalidate_cache("all_patients")
                    self.invalidate_cache(f"patient_dashboard:{patient_id}")
                    
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update patient status: {e}")
            return False
    
    def get_patients_filtered(self, search: str = "", status: str = "", limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get patients with filtering (not cached due to dynamic nature)"""
        try:
            with self.get_session() as session:
                query = session.query(Patient)
                
                # Apply filters
                if search:
                    search_pattern = f"%{search}%"
                    query = query.filter(
                        (Patient.patient_name.ilike(search_pattern)) |
                        (Patient.patient_id.ilike(search_pattern))
                    )
                
                if status:
                    query = query.filter(Patient.current_status == status)
                
                # Apply pagination and ordering
                patients = query.order_by(Patient.created_at.desc())\
                    .limit(limit)\
                    .offset(offset)\
                    .all()
                
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
                    for p in patients
                ]
                
        except Exception as e:
            logger.error(f"❌ Failed to get filtered patients: {e}")
            return []
    
    def get_patients_bulk(self, patient_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple patients efficiently"""
        try:
            with self.get_session() as session:
                patients = session.query(Patient)\
                    .filter(Patient.patient_id.in_(patient_ids))\
                    .all()
                
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
                    for p in patients
                ]
                
        except Exception as e:
            logger.error(f"❌ Failed to get bulk patients: {e}")
            return []
    
    @cache_result("patient_concern", CACHE_TTL_DASHBOARD)
    def get_patient_concern_data_optimized(self, patient_id: str) -> Dict[str, Any]:
        """Get patient CONCERN data with caching"""
        try:
            with self.get_session() as session:
                # Get concern severity tracking
                severity = session.query(ConcernSeverityTracking)\
                    .filter_by(patient_id=patient_id)\
                    .first()
                
                # Get recent concern scores
                recent_scores = session.query(ConcernScore)\
                    .filter_by(patient_id=patient_id)\
                    .order_by(ConcernScore.created_at.desc())\
                    .limit(10)\
                    .all()
                
                if not severity:
                    return {
                        'patient_id': patient_id,
                        'risk_level': 'low',
                        'cumulative_score': 0.0,
                        'total_diagnoses': 0,
                        'trend_direction': 'stable',
                        'recent_scores': []
                    }
                
                return {
                    'patient_id': patient_id,
                    'risk_level': severity.risk_level,
                    'cumulative_score': severity.cumulative_concern_score,
                    'total_diagnoses': severity.total_diagnoses,
                    'trend_direction': severity.trend_direction,
                    'last_concern_score': severity.last_concern_score,
                    'highest_risk_achieved': severity.highest_risk_achieved,
                    'risk_escalation_count': severity.risk_escalation_count,
                    'days_since_last_critical': severity.days_since_last_critical,
                    'recent_scores': [
                        {
                            'concern_score': cs.concern_score,
                            'risk_level': cs.risk_level,
                            'created_at': cs.created_at.isoformat(),
                            'alert_triggered': cs.alert_triggered
                        }
                        for cs in recent_scores
                    ]
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get patient concern data for {patient_id}: {e}")
            return {'error': str(e)}
    
    # ===== COMPATIBILITY METHODS =====
    
    def get_patient_diagnosis_sessions(self, patient_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Compatibility method - returns full diagnosis sessions (slower)"""
        try:
            with self.get_session() as session:
                query = session.query(DiagnosisSession).filter_by(patient_id=patient_id).order_by(DiagnosisSession.created_at.desc())
                if limit:
                    query = query.limit(limit)
                sessions = query.all()
                return [
                    {
                        'session_id': s.session_id,
                        'created_at': s.created_at.isoformat() if s.created_at else None,
                        'updated_at': s.updated_at.isoformat() if s.updated_at else None,
                        'status': s.status,
                        'primary_diagnosis': s.primary_diagnosis,
                        'confidence_score': s.confidence_score,
                        'patient_input': s.patient_input,
                        'diagnosis_result': s.diagnosis_result,  # Full diagnosis details
                        'processing_time': s.processing_time,
                        'ai_model_used': s.ai_model_used,
                        'verification_status': s.verification_status
                    }
                    for s in sessions
                ]
        except Exception as e:
            logger.error(f"❌ Failed to get diagnosis sessions for {patient_id}: {e}")
            return []

# ===== SINGLETON INSTANCE =====

_optimized_db_instance = None

def get_optimized_database(database_url: str = None) -> OptimizedPostgreSQLDatabase:
    """Get singleton optimized database instance"""
    global _optimized_db_instance
    
    if _optimized_db_instance is None:
        _optimized_db_instance = OptimizedPostgreSQLDatabase(database_url)
    
    return _optimized_db_instance
