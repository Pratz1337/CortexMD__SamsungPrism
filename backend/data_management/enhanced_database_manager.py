import logging
import uuid
import base64
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, Boolean, LargeBinary, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import BYTEA
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
logger = logging.getLogger(__name__)


class SimpleCache:
    """Simple in-memory cache with TTL for image data"""
    
    def __init__(self, max_size: int = 50, ttl_seconds: int = 300):  # 5 minutes TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return data
                else:
                    # Expired, remove it
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """Set item in cache with current timestamp"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry (simple LRU approximation)
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()


# Global cache instance
image_cache = SimpleCache(max_size=500, ttl_seconds=1800)  # 30 minutes TTL, max 500 images


class ScannedNote(Base):
    """Model for scanned medical notes with images stored in PostgreSQL"""
    __tablename__ = 'scanned_notes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    note_id = Column(String(255), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(255), nullable=False)
    nurse_id = Column(String(255), default='AR_SCANNER')
    
    # Image storage
    image_data = Column(BYTEA, nullable=False)  # Store the actual image
    image_mime_type = Column(String(50), default='image/png')
    image_size = Column(Integer)
    thumbnail_data = Column(BYTEA)  # Small preview image
    
    # OCR and text extraction
    ocr_text = Column(Text)
    ocr_confidence = Column(Float)
    ocr_metadata = Column(JSON)
    
    # Parsed structured data
    parsed_data = Column(JSON)
    
    # AI-generated content
    ai_summary = Column(Text)
    ai_extracted_entities = Column(JSON)
    ai_confidence_score = Column(Float)
    ai_model_used = Column(String(100))
    
    # Clinical note link
    clinical_note_id = Column(String(255))
    
    # Metadata
    scan_location = Column(String(100))
    scan_shift = Column(String(20))
    scan_timestamp = Column(DateTime, default=datetime.now)
    processing_status = Column(String(50), default='pending')
    processing_error = Column(Text)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)


class EnhancedDatabaseManager:
    """Enhanced database manager for scanned notes and clinical data"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/cortexmd')
        # Optimized connection pool settings for better performance
        self.engine = create_engine(
            self.database_url,
            pool_size=10,          # Maintain 10 connections in pool
            max_overflow=20,       # Allow up to 20 additional connections
            pool_timeout=30,       # Wait up to 30 seconds for connection
            pool_recycle=3600,     # Recycle connections after 1 hour
            pool_pre_ping=True,    # Test connections before use
            echo=False             # Disable SQL logging for performance
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_scanned_note(self, 
                         patient_id: str,
                         image_data: bytes,
                         image_mime_type: str = 'image/png',
                         thumbnail_data: bytes = None,
                         ocr_text: str = None,
                         ocr_confidence: float = None,
                         ocr_metadata: Dict = None,
                         parsed_data: Dict = None,
                         ai_summary: str = None,
                         ai_extracted_entities: Dict = None,
                         ai_confidence_score: float = None,
                         ai_model_used: str = None,
                         nurse_id: str = 'AR_SCANNER',
                         scan_location: str = None,
                         scan_shift: str = None,
                         clinical_note_id: str = None) -> Optional[str]:
        """
        Save a scanned note with all associated data to the database.
        Returns the note_id if successful, None if failed.
        """
        try:
            with self.get_session() as session:
                note = ScannedNote(
                    patient_id=patient_id,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    image_size=len(image_data),
                    thumbnail_data=thumbnail_data,
                    ocr_text=ocr_text,
                    ocr_confidence=ocr_confidence,
                    ocr_metadata=ocr_metadata or {},
                    parsed_data=parsed_data or {},
                    ai_summary=ai_summary,
                    ai_extracted_entities=ai_extracted_entities or {},
                    ai_confidence_score=ai_confidence_score,
                    ai_model_used=ai_model_used,
                    nurse_id=nurse_id,
                    scan_location=scan_location,
                    scan_shift=scan_shift,
                    clinical_note_id=clinical_note_id,
                    processing_status='completed',
                    scan_timestamp=datetime.now()
                )
                
                session.add(note)
                session.commit()
                
                logger.info(f"✅ Saved scanned note {note.note_id} for patient {patient_id}")
                return note.note_id
                
        except Exception as e:
            logger.error(f"❌ Failed to save scanned note: {e}")
            return None
    
    def get_scanned_note(self, note_id: str, include_image_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get a scanned note by ID with optional image data loading"""
        try:
            with self.get_session() as session:
                if include_image_data:
                    # Full query with image data for image retrieval endpoints
                    note = session.query(ScannedNote).filter_by(note_id=note_id).first()
                    if not note:
                        return None
                    
                    return {
                        'note_id': note.note_id,
                        'patient_id': note.patient_id,
                        'nurse_id': note.nurse_id,
                        'image_data': note.image_data,
                        'image_mime_type': note.image_mime_type,
                        'image_size': note.image_size,
                        'thumbnail_data': note.thumbnail_data,
                        'ocr_text': note.ocr_text,
                        'ocr_confidence': note.ocr_confidence,
                        'ocr_metadata': note.ocr_metadata,
                        'parsed_data': note.parsed_data,
                        'ai_summary': note.ai_summary,
                        'ai_extracted_entities': note.ai_extracted_entities,
                        'ai_confidence_score': note.ai_confidence_score,
                        'ai_model_used': note.ai_model_used,
                        'clinical_note_id': note.clinical_note_id,
                        'scan_location': note.scan_location,
                        'scan_shift': note.scan_shift,
                        'scan_timestamp': note.scan_timestamp.isoformat() if note.scan_timestamp else None,
                        'processing_status': note.processing_status,
                        'processing_error': note.processing_error,
                        'created_at': note.created_at.isoformat() if note.created_at else None,
                        'updated_at': note.updated_at.isoformat() if note.updated_at else None
                    }
                else:
                    # Lightweight query without image data for metadata-only requests
                    from sqlalchemy import text
                    result = session.execute(text("""
                        SELECT note_id, patient_id, nurse_id, image_mime_type, image_size, 
                               ocr_text, ocr_confidence, ocr_metadata, parsed_data, 
                               ai_summary, ai_extracted_entities, ai_confidence_score, ai_model_used,
                               clinical_note_id, scan_location, scan_shift, scan_timestamp,
                               processing_status, processing_error, created_at, updated_at
                        FROM scanned_notes 
                        WHERE note_id = :note_id
                    """), {'note_id': note_id}).first()
                    
                    if not result:
                        return None
                    
                    return {
                        'note_id': result[0],
                        'patient_id': result[1],
                        'nurse_id': result[2],
                        'image_mime_type': result[3],
                        'image_size': result[4],
                        'thumbnail_data': None,  # Not loaded
                        'ocr_text': result[5],
                        'ocr_confidence': result[6],
                        'ocr_metadata': result[7],
                        'parsed_data': result[8],
                        'ai_summary': result[9],
                        'ai_extracted_entities': result[10],
                        'ai_confidence_score': result[11],
                        'ai_model_used': result[12],
                        'clinical_note_id': result[13],
                        'scan_location': result[14],
                        'scan_shift': result[15],
                        'scan_timestamp': result[16].isoformat() if result[16] else None,
                        'processing_status': result[17],
                        'processing_error': result[18],
                        'created_at': result[19].isoformat() if result[19] else None,
                        'updated_at': result[20].isoformat() if result[20] else None
                    }
        except Exception as e:
            logger.error(f"❌ Failed to get scanned note {note_id}: {e}")
            return None
    
    def get_patient_scanned_notes(self, patient_id: str, limit: int = 50, offset: int = 0, include_thumbnails: bool = False) -> List[Dict[str, Any]]:
        """Get scanned notes for a patient with pagination and optional thumbnail loading"""
        try:
            with self.get_session() as session:
                if include_thumbnails:
                    # Include thumbnails for gallery view
                    notes = session.query(ScannedNote).filter_by(patient_id=patient_id)\
                        .order_by(ScannedNote.scan_timestamp.desc())\
                        .offset(offset).limit(limit).all()
                    
                    return [self._format_scanned_note_full(note) for note in notes]
                else:
                    # Lightweight query without image data for list view
                    from sqlalchemy import text
                    result = session.execute(text("""
                        SELECT note_id, patient_id, nurse_id, image_mime_type, image_size, 
                               ocr_text, ocr_confidence, parsed_data, ai_summary, 
                               ai_extracted_entities, ai_confidence_score, scan_location, 
                               scan_shift, scan_timestamp, processing_status
                        FROM scanned_notes 
                        WHERE patient_id = :patient_id
                        ORDER BY scan_timestamp DESC
                        OFFSET :offset LIMIT :limit
                    """), {'patient_id': patient_id, 'offset': offset, 'limit': limit})
                    
                    return [{
                        'note_id': row[0],
                        'patient_id': row[1],
                        'nurse_id': row[2],
                        'image_mime_type': row[3],
                        'image_size': row[4],
                        'thumbnail_data': None,  # Not loaded
                        'ocr_text': row[5],
                        'ocr_confidence': row[6],
                        'parsed_data': row[7],
                        'ai_summary': row[8],
                        'ai_extracted_entities': row[9],
                        'ai_confidence_score': row[10],
                        'scan_location': row[11],
                        'scan_shift': row[12],
                        'scan_timestamp': row[13].isoformat() if row[13] else None,
                        'processing_status': row[14]
                    } for row in result]
        except Exception as e:
            logger.error(f"❌ Failed to get scanned notes for patient {patient_id}: {e}")
            return []
    
    def get_patient_scanned_notes_count(self, patient_id: str) -> int:
        """Get the total count of scanned notes for a patient"""
        try:
            with self.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text("""
                    SELECT COUNT(*) FROM scanned_notes WHERE patient_id = :patient_id
                """), {'patient_id': patient_id}).first()
                
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"❌ Failed to get scanned notes count for patient {patient_id}: {e}")
            return 0
    
    def get_multiple_scanned_notes_metadata(self, note_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get metadata for multiple scanned notes in a single query (optimized for bulk operations)"""
        if not note_ids:
            return {}
        
        try:
            with self.get_session() as session:
                from sqlalchemy import text
                placeholders = ','.join([f':note_id_{i}' for i in range(len(note_ids))])
                params = {f'note_id_{i}': note_id for i, note_id in enumerate(note_ids)}
                
                result = session.execute(text(f"""
                    SELECT note_id, patient_id, nurse_id, image_mime_type, image_size, 
                           ocr_text, ocr_confidence, parsed_data, ai_summary, 
                           ai_extracted_entities, ai_confidence_score, scan_location, 
                           scan_shift, scan_timestamp, processing_status
                    FROM scanned_notes 
                    WHERE note_id IN ({placeholders})
                """), params)
                
                return {row[0]: {
                    'note_id': row[0],
                    'patient_id': row[1],
                    'nurse_id': row[2],
                    'image_mime_type': row[3],
                    'image_size': row[4],
                    'ocr_text': row[5],
                    'ocr_confidence': row[6],
                    'parsed_data': row[7],
                    'ai_summary': row[8],
                    'ai_extracted_entities': row[9],
                    'ai_confidence_score': row[10],
                    'scan_location': row[11],
                    'scan_shift': row[12],
                    'scan_timestamp': row[13].isoformat() if row[13] else None,
                    'processing_status': row[14]
                } for row in result}
        except Exception as e:
            logger.error(f"❌ Failed to get metadata for multiple notes: {e}")
            return {}
    
    def _format_scanned_note_full(self, note) -> Dict[str, Any]:
        """Helper method to format a full scanned note"""
        return {
            'note_id': note.note_id,
            'patient_id': note.patient_id,
            'nurse_id': note.nurse_id,
            'image_mime_type': note.image_mime_type,
            'image_size': note.image_size,
            'thumbnail_data': base64.b64encode(note.thumbnail_data).decode('utf-8') if note.thumbnail_data else None,
            'ocr_text': note.ocr_text,
            'ocr_confidence': note.ocr_confidence,
            'parsed_data': note.parsed_data,
            'ai_summary': note.ai_summary,
            'ai_extracted_entities': note.ai_extracted_entities,
            'ai_confidence_score': note.ai_confidence_score,
            'scan_location': note.scan_location,
            'scan_shift': note.scan_shift,
            'scan_timestamp': note.scan_timestamp.isoformat() if note.scan_timestamp else None,
            'processing_status': note.processing_status
        }
    
    def update_processing_status(self, note_id: str, status: str, error: str = None) -> bool:
        """Update the processing status of a scanned note"""
        try:
            with self.get_session() as session:
                note = session.query(ScannedNote).filter_by(note_id=note_id).first()
                if not note:
                    return False
                
                note.processing_status = status
                if error:
                    note.processing_error = error
                note.updated_at = datetime.now()
                
                session.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to update processing status for {note_id}: {e}")
            return False
    
    def link_to_clinical_note(self, scanned_note_id: str, clinical_note_id: str) -> bool:
        """Link a scanned note to a clinical note"""
        try:
            with self.get_session() as session:
                note = session.query(ScannedNote).filter_by(note_id=scanned_note_id).first()
                if not note:
                    return False
                
                note.clinical_note_id = clinical_note_id
                note.updated_at = datetime.now()
                
                session.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to link scanned note {scanned_note_id} to clinical note {clinical_note_id}: {e}")
            return False
    
    def search_scanned_notes(self, 
                           patient_id: str = None,
                           search_text: str = None,
                           date_from: datetime = None,
                           date_to: datetime = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """Search scanned notes with various filters"""
        try:
            with self.get_session() as session:
                query = session.query(ScannedNote)
                
                if patient_id:
                    query = query.filter_by(patient_id=patient_id)
                
                if date_from:
                    query = query.filter(ScannedNote.scan_timestamp >= date_from)
                
                if date_to:
                    query = query.filter(ScannedNote.scan_timestamp <= date_to)
                
                if search_text:
                    # Search in OCR text, AI summary, and parsed data
                    search_filter = (
                        ScannedNote.ocr_text.ilike(f'%{search_text}%') |
                        ScannedNote.ai_summary.ilike(f'%{search_text}%')
                    )
                    query = query.filter(search_filter)
                
                notes = query.order_by(ScannedNote.scan_timestamp.desc()).limit(limit).all()
                
                return [{
                    'note_id': note.note_id,
                    'patient_id': note.patient_id,
                    'nurse_id': note.nurse_id,
                    'ai_summary': note.ai_summary,
                    'ocr_confidence': note.ocr_confidence,
                    'ai_confidence_score': note.ai_confidence_score,
                    'scan_timestamp': note.scan_timestamp.isoformat() if note.scan_timestamp else None,
                    'processing_status': note.processing_status
                } for note in notes]
        except Exception as e:
            logger.error(f"❌ Failed to search scanned notes: {e}")
            return []
    
    def delete_scanned_note(self, note_id: str) -> bool:
        """Delete a scanned note"""
        try:
            with self.get_session() as session:
                note = session.query(ScannedNote).filter_by(note_id=note_id).first()
                if not note:
                    return False
                
                session.delete(note)
                session.commit()
                return True
        except Exception as e:
            logger.error(f"❌ Failed to delete scanned note {note_id}: {e}")
            return False
    
    def get_scanned_note_image_only(self, note_id: str) -> Optional[bytes]:
        """Get only the image data for a scanned note (optimized for image serving)"""
        try:
            # Check cache first
            cached_image = image_cache.get(note_id)
            if cached_image is not None:
                return cached_image
            
            with self.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text("""
                    SELECT image_data FROM scanned_notes WHERE note_id = :note_id
                """), {'note_id': note_id}).first()
                
                image_data = result[0] if result else None
                
                # Cache the image data
                if image_data:
                    image_cache.set(note_id, image_data)
                
                return image_data
        except Exception as e:
            logger.error(f"❌ Failed to get image data for note {note_id}: {e}")
            return None
    
    def get_scanned_note_thumbnail_only(self, note_id: str) -> Optional[bytes]:
        """Get only the thumbnail data for a scanned note"""
        try:
            with self.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text("""
                    SELECT thumbnail_data FROM scanned_notes WHERE note_id = :note_id
                """), {'note_id': note_id}).first()
                
                return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"❌ Failed to get thumbnail data for note {note_id}: {e}")
            return None
    
    def get_scanned_note_metadata_only(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Get only metadata for a scanned note (no image data)"""
        return self.get_scanned_note(note_id, include_image_data=False)


# Global instance
enhanced_db = EnhancedDatabaseManager()
