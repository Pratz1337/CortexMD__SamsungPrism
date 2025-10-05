"""
CortexMD Advanced AR Clinical Notes System

This module provides a comprehensive Augmented Reality (AR) interface for clinical notes,
supporting both photo and video capture modes with real-time medical information extraction
and overlay capabilities.

Features:
- Live camera streaming with AR overlays
- Photo mode: Capture and analyze still images
- Video mode: Real-time video processing and analysis
- Medical information extraction using AI
- Real-time overlay rendering
- Secure database storage
- WebSocket real-time streaming
"""

import os
import io
import json
import base64
import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from PIL import Image, ImageDraw, ImageFont
from flask import Blueprint, request, jsonify, Response, stream_with_context
import threading
import queue
import time

# Try to import OpenCV, but don't fail if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Try to import Flask-SocketIO, but don't fail if not available
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    # Create dummy functions if SocketIO is not available
    def emit(*args, **kwargs):
        pass
    def join_room(*args, **kwargs):
        pass
    def leave_room(*args, **kwargs):
        pass

try:
    from .enhanced_ar_processor import enhanced_ocr_and_parse, create_thumbnail
    from ..core.database_manager import get_database
    from ..core.models import PatientInput
except ImportError:
    from api_handlers.enhanced_ar_processor import enhanced_ocr_and_parse, create_thumbnail
    from core.database_manager import get_database
    from core.models import PatientInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create AR Blueprint
ar_bp = Blueprint('ar', __name__, url_prefix='/api/ar')

# Global variables for AR processing
active_ar_sessions = {}
ar_processing_queue = queue.Queue()
ar_socketio = None

class ARSession:
    """Represents an active AR session for a patient"""
    
    def __init__(self, session_id: str, patient_id: str, mode: str = 'photo'):
        self.session_id = session_id
        self.patient_id = patient_id
        self.mode = mode  # 'photo' or 'video'
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.active = True
        self.captured_frames = []
        self.extracted_data = []
        self.current_overlay = None
        self.processing_status = 'idle'  # 'idle', 'processing', 'complete', 'error'
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def add_frame(self, frame_data: bytes, timestamp: datetime = None):
        """Add a captured frame to the session"""
        if timestamp is None:
            timestamp = datetime.now()
        
        frame_info = {
            'timestamp': timestamp,
            'data': frame_data,
            'processed': False,
            'analysis_result': None
        }
        self.captured_frames.append(frame_info)
        self.update_activity()
    
    def get_latest_overlay(self) -> Optional[Dict[str, Any]]:
        """Get the latest AR overlay data"""
        return self.current_overlay
    
    def update_overlay(self, overlay_data: Dict[str, Any]):
        """Update the current AR overlay"""
        self.current_overlay = overlay_data
        self.update_activity()

class ARProcessor:
    """Main AR processing engine"""
    
    def __init__(self):
        self.is_running = False
        self.processing_thread = None
        
    def start(self):
        """Start the AR processing thread"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            logger.info("AR Processor started")
    
    def stop(self):
        """Stop the AR processing thread"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("AR Processor stopped")
    
    def _processing_loop(self):
        """Main processing loop for AR frames"""
        while self.is_running:
            try:
                # Get next frame to process
                if not ar_processing_queue.empty():
                    task = ar_processing_queue.get_nowait()
                    self._process_ar_task(task)
                else:
                    time.sleep(0.1)  # Brief pause when no tasks
            except Exception as e:
                logger.error(f"AR processing error: {e}")
                time.sleep(1)  # Longer pause on error
    
    def _process_ar_task(self, task: Dict[str, Any]):
        """Process a single AR task"""
        try:
            session_id = task.get('session_id')
            frame_data = task.get('frame_data')
            task_type = task.get('type', 'analyze_frame')
            
            if session_id not in active_ar_sessions:
                logger.warning(f"AR session {session_id} not found")
                return
            
            session = active_ar_sessions[session_id]
            session.processing_status = 'processing'
            
            if task_type == 'analyze_frame':
                result = self._analyze_frame(frame_data, session.patient_id)
                
                # Update session with results
                if result.get('success'):
                    overlay_data = self._create_ar_overlay(result)
                    session.update_overlay(overlay_data)
                    session.processing_status = 'complete'
                    
                    # Save to database for persistent storage
                    db_result = save_ar_data_to_database(session, result)
                    if db_result.get('success'):
                        # Add database info to extracted data
                        result['clinical_note_id'] = db_result.get('clinical_note_id')
                        result['scanned_note_id'] = db_result.get('scanned_note_id')
                        session.extracted_data.append(result)
                    
                    # Emit real-time update via WebSocket
                    if ar_socketio and SOCKETIO_AVAILABLE:
                        ar_socketio.emit('ar_overlay_update', {
                            'session_id': session_id,
                            'overlay': overlay_data,
                            'analysis_result': result,
                            'database_saved': db_result.get('success', False),
                            'timestamp': datetime.now().isoformat()
                        }, room=f"ar_session_{session_id}")
                else:
                    session.processing_status = 'error'
                    
        except Exception as e:
            logger.error(f"AR task processing error: {e}")
            if session_id in active_ar_sessions:
                active_ar_sessions[session_id].processing_status = 'error'
    
    def _analyze_frame(self, frame_data: bytes, patient_id: str) -> Dict[str, Any]:
        """Analyze a single frame using enhanced AR processor"""
        try:
            logger.info(f"Analyzing frame for patient {patient_id}")
            result = enhanced_ocr_and_parse(frame_data)
            
            # Add patient context
            result['patient_id'] = patient_id
            result['analysis_timestamp'] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'patient_id': patient_id,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _create_ar_overlay(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create AR overlay data from analysis results"""
        try:
            overlay = {
                'type': 'medical_info_overlay',
                'timestamp': datetime.now().isoformat(),
                'elements': []
            }
            
            # Extract key medical information for overlay
            parsed_data = analysis_result.get('parsed_data', {})
            
            # Patient info overlay
            patient_info = parsed_data.get('patient_info', {})
            if patient_info:
                overlay['elements'].append({
                    'type': 'patient_info',
                    'position': {'x': 10, 'y': 10},
                    'data': patient_info,
                    'style': {
                        'background_color': 'rgba(59, 130, 246, 0.9)',
                        'text_color': 'white',
                        'font_size': 14,
                        'padding': 8,
                        'border_radius': 6
                    }
                })
            
            # Vitals overlay
            vitals = parsed_data.get('vitals', {})
            if vitals:
                overlay['elements'].append({
                    'type': 'vitals',
                    'position': {'x': 10, 'y': 80},
                    'data': vitals,
                    'style': {
                        'background_color': 'rgba(16, 185, 129, 0.9)',
                        'text_color': 'white',
                        'font_size': 12,
                        'padding': 6,
                        'border_radius': 4
                    }
                })
            
            # Medications overlay
            medications = parsed_data.get('clinical_data', {}).get('medications', [])
            if medications:
                overlay['elements'].append({
                    'type': 'medications',
                    'position': {'x': 10, 'y': 160},
                    'data': {'medications': medications},
                    'style': {
                        'background_color': 'rgba(245, 101, 101, 0.9)',
                        'text_color': 'white',
                        'font_size': 11,
                        'padding': 5,
                        'border_radius': 4
                    }
                })
            
            # Confidence indicators
            ocr_confidence = analysis_result.get('ocr_meta', {}).get('avg_word_confidence', 0)
            ai_confidence = parsed_data.get('confidence_score', 0) * 100
            
            overlay['elements'].append({
                'type': 'confidence_indicator',
                'position': {'x': 10, 'y': 250},
                'data': {
                    'ocr_confidence': ocr_confidence,
                    'ai_confidence': ai_confidence
                },
                'style': {
                    'background_color': 'rgba(107, 114, 128, 0.9)',
                    'text_color': 'white',
                    'font_size': 10,
                    'padding': 4,
                    'border_radius': 3
                }
            })
            
            # AI Summary overlay (bottom of screen)
            ai_summary = analysis_result.get('ai_summary', '')
            if ai_summary:
                overlay['elements'].append({
                    'type': 'ai_summary',
                    'position': {'x': 10, 'y': -80, 'anchor': 'bottom'},
                    'data': {'summary': ai_summary[:200] + '...' if len(ai_summary) > 200 else ai_summary},
                    'style': {
                        'background_color': 'rgba(139, 92, 246, 0.9)',
                        'text_color': 'white',
                        'font_size': 12,
                        'padding': 8,
                        'border_radius': 6,
                        'max_width': 300
                    }
                })
            
            return overlay
            
        except Exception as e:
            logger.error(f"Overlay creation error: {e}")
            return {
                'type': 'error_overlay',
                'timestamp': datetime.now().isoformat(),
                'elements': [{
                    'type': 'error',
                    'position': {'x': 10, 'y': 10},
                    'data': {'error': str(e)},
                    'style': {
                        'background_color': 'rgba(239, 68, 68, 0.9)',
                        'text_color': 'white',
                        'font_size': 12,
                        'padding': 6,
                        'border_radius': 4
                    }
                }]
            }

# Initialize AR Processor
ar_processor = ARProcessor()

def cleanup_expired_sessions():
    """Clean up expired AR sessions"""
    expired_sessions = []
    for session_id, session in active_ar_sessions.items():
        if session.is_expired():
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del active_ar_sessions[session_id]
        logger.info(f"Cleaned up expired AR session: {session_id}")

def save_ar_data_to_database(session: ARSession, analysis_result: Dict[str, Any]):
    """Save AR analysis data to database with clinical notes integration"""
    try:
        from database_manager import get_database
        from enhanced_database_manager import get_enhanced_database
        
        db = get_database()
        enhanced_db = get_enhanced_database()
        
        # Extract clinical data from AI analysis
        parsed_data = analysis_result.get('parsed_data', {})
        patient_info = parsed_data.get('patient_info', {})
        clinical_data = parsed_data.get('clinical_data', {})
        vitals = parsed_data.get('vitals', {})
        assessment = parsed_data.get('assessment', {})
        plan_data = parsed_data.get('plan', {})
        
        # Create comprehensive clinical note content
        clinical_note_content = f"""AR Scanned Medical Note - Enhanced AI Analysis
Patient ID: {session.patient_id}
Session ID: {session.session_id}
Processing Mode: {session.mode}
Timestamp: {datetime.now().isoformat()}

=== PATIENT INFORMATION ===
{json.dumps(patient_info, indent=2) if patient_info else 'No patient information extracted'}

=== VITAL SIGNS ===
{json.dumps(vitals, indent=2) if vitals else 'No vital signs recorded'}

=== CLINICAL DATA ===
Chief Complaint: {clinical_data.get('chief_complaint', 'Not specified')}
History of Present Illness: {clinical_data.get('history_of_present_illness', 'Not documented')}
Past Medical History: {clinical_data.get('past_medical_history', 'Not documented')}
Medications: {', '.join(clinical_data.get('medications', [])) if clinical_data.get('medications') else 'None listed'}
Allergies: {', '.join(clinical_data.get('allergies', [])) if clinical_data.get('allergies') else 'None listed'}
Social History: {clinical_data.get('social_history', 'Not documented')}

=== ASSESSMENT ===
Clinical Impression: {assessment.get('impression', 'Not provided')}
Diagnoses: {', '.join(assessment.get('diagnosis', [])) if assessment.get('diagnosis') else 'None listed'}
Differential Diagnoses: {', '.join(assessment.get('differential_diagnosis', [])) if assessment.get('differential_diagnosis') else 'None listed'}

=== PLAN ===
Treatment Plan: {plan_data.get('treatment_plan', 'Not specified')}
Prescribed Medications: {', '.join(plan_data.get('medications_prescribed', [])) if plan_data.get('medications_prescribed') else 'None prescribed'}
Follow-up Instructions: {plan_data.get('follow_up', 'Not specified')}
Discharge Instructions: {plan_data.get('discharge_instructions', 'Not applicable')}

=== AI ANALYSIS METADATA ===
AI Summary: {analysis_result.get('ai_summary', 'No summary generated')}
OCR Confidence: {analysis_result.get('ocr_meta', {}).get('avg_word_confidence', 0)}%
AI Confidence: {parsed_data.get('confidence_score', 0) * 100}%
Text Length: {analysis_result.get('ocr_meta', {}).get('total_characters', 0)} characters
Word Count: {analysis_result.get('ocr_meta', {}).get('word_count', 0)} words

=== EXTRACTED MEDICAL ENTITIES ===
{json.dumps(analysis_result.get('extracted_entities', {}), indent=2)}

=== RAW OCR TEXT ===
{analysis_result.get('ocr_text', 'No OCR text available')}
"""
        
        # Save clinical note to database
        note_id = enhanced_db.add_clinical_note(
            patient_id=session.patient_id,
            note_type='AR_SCANNED_NOTE',
            content=clinical_note_content,
            nurse_id=f'AR_SYSTEM_{session.mode.upper()}',
            location=f'AR_Scanner_{session.session_id}',
            shift='AR_Processing'
        )
        
        # Save AR session data
        ar_scan_data = {
            'session_id': session.session_id,
            'patient_id': session.patient_id,
            'mode': session.mode,
            'clinical_note_id': note_id,
            'analysis_result': analysis_result,
            'parsed_data': parsed_data,
            'ai_summary': analysis_result.get('ai_summary', ''),
            'confidence_scores': {
                'ocr_confidence': analysis_result.get('ocr_meta', {}).get('avg_word_confidence', 0),
                'ai_confidence': parsed_data.get('confidence_score', 0) * 100
            },
            'extracted_entities': analysis_result.get('extracted_entities', {}),
            'created_at': datetime.now(),
            'processing_status': 'complete'
        }
        
        # Save AR scan record (using the existing scan note functionality)
        scanned_note_id = enhanced_db.add_scanned_note(
            patient_id=session.patient_id,
            original_filename=f"AR_{session.mode}_{session.session_id}.jpg",
            storage_path=f"ar_sessions/{session.session_id}/",
            ocr_text=analysis_result.get('ocr_text', ''),
            parsed_data=parsed_data,
            ai_summary=analysis_result.get('ai_summary', ''),
            nurse_id=f'AR_SYSTEM_{session.mode.upper()}',
            confidence_score=analysis_result.get('ocr_meta', {}).get('avg_word_confidence', 0),
            processing_metadata=analysis_result.get('ocr_meta', {}),
            thumbnail_data=analysis_result.get('thumbnail_data', b'')
        )
        
        # Link the scanned note to the clinical note
        if note_id and scanned_note_id:
            enhanced_db.link_to_clinical_note(scanned_note_id, note_id)
        
        logger.info(f"AR data saved successfully - Clinical Note ID: {note_id}, Scanned Note ID: {scanned_note_id}")
        
        return {
            'success': True,
            'clinical_note_id': note_id,
            'scanned_note_id': scanned_note_id,
            'message': 'AR data saved to clinical notes database'
        }
        
    except Exception as e:
        logger.error(f"Database save error: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to save AR data to database'
        }

# AR Blueprint Routes

@ar_bp.route('/start-session', methods=['POST'])
def start_ar_session():
    """Start a new AR session"""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        mode = data.get('mode', 'photo')  # 'photo' or 'video'
        
        if not patient_id:
            return jsonify({'error': 'Patient ID is required'}), 400
        
        if mode not in ['photo', 'video']:
            return jsonify({'error': 'Mode must be "photo" or "video"'}), 400
        
        # Generate session ID
        session_id = f"ar_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Create AR session
        session = ARSession(session_id, patient_id, mode)
        active_ar_sessions[session_id] = session
        
        # Start AR processor if not running
        if not ar_processor.is_running:
            ar_processor.start()
        
        # Clean up expired sessions
        cleanup_expired_sessions()
        
        logger.info(f"Started AR session {session_id} for patient {patient_id} in {mode} mode")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'patient_id': patient_id,
            'mode': mode,
            'created_at': session.created_at.isoformat(),
            'message': f'AR session started in {mode} mode'
        })
        
    except Exception as e:
        logger.error(f"Start AR session error: {e}")
        return jsonify({'error': str(e)}), 500

@ar_bp.route('/end-session/<session_id>', methods=['POST'])
def end_ar_session(session_id: str):
    """End an AR session"""
    try:
        if session_id not in active_ar_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_ar_sessions[session_id]
        session.active = False
        
        # Save final data to database if not already saved
        saved_count = 0
        for data in session.extracted_data:
            if not data.get('clinical_note_id'):  # Only save if not already saved
                db_result = save_ar_data_to_database(session, data)
                if db_result.get('success'):
                    saved_count += 1
        
        # Remove from active sessions
        del active_ar_sessions[session_id]
        
        logger.info(f"Ended AR session {session_id}, saved {saved_count} additional records")
        
        return jsonify({
            'success': True,
            'message': 'AR session ended successfully',
            'session_id': session_id,
            'total_frames': len(session.captured_frames),
            'extracted_data_count': len(session.extracted_data),
            'final_saves': saved_count,
            'clinical_notes_created': len([d for d in session.extracted_data if d.get('clinical_note_id')])
        })
        
    except Exception as e:
        logger.error(f"End AR session error: {e}")
        return jsonify({'error': str(e)}), 500

@ar_bp.route('/capture-frame/<session_id>', methods=['POST'])
def capture_frame(session_id: str):
    """Capture and process a frame in photo mode"""
    try:
        if session_id not in active_ar_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_ar_sessions[session_id]
        
        if session.mode != 'photo':
            return jsonify({'error': 'Session not in photo mode'}), 400
        
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Add frame to session
        session.add_frame(image_data)
        
        # Queue for processing
        ar_processing_queue.put({
            'session_id': session_id,
            'frame_data': image_data,
            'type': 'analyze_frame'
        })
        
        return jsonify({
            'success': True,
            'message': 'Frame captured and queued for processing',
            'session_id': session_id,
            'frame_count': len(session.captured_frames)
        })
        
    except Exception as e:
        logger.error(f"Capture frame error: {e}")
        return jsonify({'error': str(e)}), 500

@ar_bp.route('/stream-frame/<session_id>', methods=['POST'])
def stream_frame(session_id: str):
    """Process a frame in video mode (real-time streaming)"""
    try:
        if session_id not in active_ar_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_ar_sessions[session_id]
        
        if session.mode != 'video':
            return jsonify({'error': 'Session not in video mode'}), 400
        
        # Get image from request
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        
        # For video mode, we process frames immediately
        # Only queue if not already processing
        if session.processing_status != 'processing':
            ar_processing_queue.put({
                'session_id': session_id,
                'frame_data': frame_data,
                'type': 'analyze_frame'
            })
        
        # Return current overlay
        overlay = session.get_latest_overlay()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'processing_status': session.processing_status,
            'overlay': overlay,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stream frame error: {e}")
        return jsonify({'error': str(e)}), 500

@ar_bp.route('/get-overlay/<session_id>', methods=['GET'])
def get_ar_overlay(session_id: str):
    """Get current AR overlay for a session"""
    try:
        if session_id not in active_ar_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_ar_sessions[session_id]
        overlay = session.get_latest_overlay()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'overlay': overlay,
            'processing_status': session.processing_status,
            'last_activity': session.last_activity.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Get overlay error: {e}")
        return jsonify({'error': str(e)}), 500

@ar_bp.route('/session-status/<session_id>', methods=['GET'])
def get_session_status(session_id: str):
    """Get AR session status"""
    try:
        if session_id not in active_ar_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = active_ar_sessions[session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'patient_id': session.patient_id,
            'mode': session.mode,
            'active': session.active,
            'processing_status': session.processing_status,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'frame_count': len(session.captured_frames),
            'extracted_data_count': len(session.extracted_data),
            'has_overlay': session.current_overlay is not None
        })
        
    except Exception as e:
        logger.error(f"Get session status error: {e}")
        return jsonify({'error': str(e)}), 500

@ar_bp.route('/active-sessions', methods=['GET'])
def get_active_sessions():
    """Get list of active AR sessions"""
    try:
        sessions_info = []
        
        for session_id, session in active_ar_sessions.items():
            if session.active:
                sessions_info.append({
                    'session_id': session_id,
                    'patient_id': session.patient_id,
                    'mode': session.mode,
                    'created_at': session.created_at.isoformat(),
                    'last_activity': session.last_activity.isoformat(),
                    'processing_status': session.processing_status,
                    'frame_count': len(session.captured_frames)
                })
        
        return jsonify({
            'success': True,
            'active_sessions': sessions_info,
            'total_sessions': len(sessions_info)
        })
        
    except Exception as e:
        logger.error(f"Get active sessions error: {e}")
        return jsonify({'error': str(e)}), 500

@ar_bp.route('/health', methods=['GET'])
def ar_health_check():
    """AR system health check"""
    try:
        return jsonify({
            'success': True,
            'status': 'AR system operational',
            'processor_running': ar_processor.is_running,
            'active_sessions': len(active_ar_sessions),
            'queue_size': ar_processing_queue.qsize(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"AR health check error: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket Events for Real-time AR Updates

def init_ar_socketio(socketio):
    """Initialize SocketIO for AR real-time updates"""
    global ar_socketio
    
    if not SOCKETIO_AVAILABLE:
        logger.warning("Flask-SocketIO not available, AR real-time updates disabled")
        return
    
    ar_socketio = socketio
    
    @socketio.on('join_ar_session')
    def on_join_ar_session(data):
        """Join AR session room for real-time updates"""
        session_id = data.get('session_id')
        if session_id and session_id in active_ar_sessions:
            join_room(f"ar_session_{session_id}")
            emit('ar_session_joined', {
                'session_id': session_id,
                'message': 'Joined AR session for real-time updates'
            })
        else:
            emit('error', {'message': 'Invalid AR session ID'})
    
    @socketio.on('leave_ar_session')
    def on_leave_ar_session(data):
        """Leave AR session room"""
        session_id = data.get('session_id')
        if session_id:
            leave_room(f"ar_session_{session_id}")
            emit('ar_session_left', {
                'session_id': session_id,
                'message': 'Left AR session'
            })
    
    @socketio.on('request_ar_overlay')
    def on_request_ar_overlay(data):
        """Request current AR overlay"""
        session_id = data.get('session_id')
        if session_id and session_id in active_ar_sessions:
            session = active_ar_sessions[session_id]
            overlay = session.get_latest_overlay()
            emit('ar_overlay_update', {
                'session_id': session_id,
                'overlay': overlay,
                'timestamp': datetime.now().isoformat()
            })



def initialize_ar_system():
    """Initialize the AR system"""
    try:
        logger.info("Initializing CortexMD AR System...")
        
        # Start AR processor
        ar_processor.start()
        
        logger.info("AR System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"AR System initialization error: {e}")
        return False

# Cleanup function
def cleanup_ar_system():
    """Clean up AR system resources"""
    try:
        logger.info("Cleaning up AR System...")
        
        # Stop AR processor
        ar_processor.stop()
        
        # Clear active sessions
        active_ar_sessions.clear()
        
        logger.info("AR System cleanup completed")
        
    except Exception as e:
        logger.error(f"AR System cleanup error: {e}")

# Export the blueprint and initialization functions
__all__ = ['ar_bp', 'initialize_ar_system', 'cleanup_ar_system', 'init_ar_socketio']
