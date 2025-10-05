"""
Performance Optimized Endpoints for CortexMD
Fast patient data loading with caching and optimized queries
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Any, List
import logging
import time
from datetime import datetime

# Import optimized database
try:
    from ..data_management.optimized_database import get_optimized_database
except ImportError:
    from data_management.optimized_database import get_optimized_database

logger = logging.getLogger(__name__)

# Create Blueprint for optimized endpoints
optimized_bp = Blueprint('optimized', __name__, url_prefix='/api/v2')

# ===== PERFORMANCE MONITORING =====

def monitor_performance(func):
    """Decorator to monitor endpoint performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"⚡ {func.__name__} executed in {execution_time:.3f}s")
            
            # Add performance headers
            if hasattr(result, 'headers'):
                result.headers['X-Execution-Time'] = f"{execution_time:.3f}s"
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed in {execution_time:.3f}s: {e}")
            raise
    
    wrapper.__name__ = func.__name__
    return wrapper

# ===== OPTIMIZED PATIENT ENDPOINTS =====

@optimized_bp.route('/patients', methods=['GET'])
@monitor_performance
def get_patients_optimized():
    """Get all patients with pagination and caching"""
    try:
        # Get pagination parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)  # Max 100 per page
        offset = (page - 1) * per_page
        
        # Get search parameters
        search = request.args.get('search', '').strip()
        status = request.args.get('status', '').strip()
        
        db = get_optimized_database()
        
        if search or status:
            # Use filtered query (not cached for dynamic filters)
            patients = db.get_patients_filtered(
                search=search,
                status=status,
                limit=per_page,
                offset=offset
            )
        else:
            # Use cached query for basic pagination
            patients = db.get_all_patients(limit=per_page, offset=offset)
        
        return jsonify({
            'patients': patients,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': len(patients),
                'has_more': len(patients) == per_page
            },
            'performance': {
                'cached': not (search or status),
                'query_type': 'filtered' if (search or status) else 'paginated'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_patients_optimized: {e}")
        return jsonify({'error': str(e)}), 500

@optimized_bp.route('/patients/<patient_id>', methods=['GET'])
@monitor_performance
def get_patient_details_optimized(patient_id: str):
    """Get patient details with optimized loading"""
    try:
        include_full = request.args.get('full', 'false').lower() == 'true'
        
        db = get_optimized_database()
        
        if include_full:
            # Get comprehensive patient data
            dashboard = db.get_patient_dashboard_optimized(patient_id)
        else:
            # Get basic patient info only (fastest)
            patient = db.get_patient(patient_id)
            if not patient:
                return jsonify({'error': 'Patient not found'}), 404
            
            dashboard = {
                'patient_info': patient,
                'summary_only': True
            }
        
        return jsonify(dashboard)
        
    except Exception as e:
        logger.error(f"Error in get_patient_details_optimized: {e}")
        return jsonify({'error': str(e)}), 500

@optimized_bp.route('/patients/<patient_id>/dashboard', methods=['GET'])
@monitor_performance
def get_patient_dashboard_optimized(patient_id: str):
    """Get full patient dashboard with all data (cached)"""
    try:
        db = get_optimized_database()
        dashboard = db.get_patient_dashboard_optimized(patient_id)
        
        if 'error' in dashboard:
            return jsonify(dashboard), 404
        
        return jsonify(dashboard)
        
    except Exception as e:
        logger.error(f"Error in get_patient_dashboard_optimized: {e}")
        return jsonify({'error': str(e)}), 500

@optimized_bp.route('/patients/<patient_id>/diagnoses', methods=['GET'])
@monitor_performance
def get_patient_diagnoses_optimized(patient_id: str):
    """Get patient diagnosis history (summary only for speed)"""
    try:
        limit = min(int(request.args.get('limit', 10)), 50)  # Max 50
        full_details = request.args.get('full', 'false').lower() == 'true'
        
        db = get_optimized_database()
        
        if full_details:
            # Get full diagnosis details (slower but complete)
            diagnoses = db.get_patient_diagnosis_sessions(patient_id, limit)
        else:
            # Get summary only (much faster)
            diagnoses = db.get_patient_diagnosis_sessions_summary(patient_id, limit)
        
        return jsonify({
            'diagnoses': diagnoses,
            'total': len(diagnoses),
            'summary_only': not full_details,
            'limit': limit
        })
        
    except Exception as e:
        logger.error(f"Error in get_patient_diagnoses_optimized: {e}")
        return jsonify({'error': str(e)}), 500

@optimized_bp.route('/patients/<patient_id>/diagnoses/<session_id>/full', methods=['GET'])
@monitor_performance
def get_diagnosis_full_details(patient_id: str, session_id: str):
    """Get full diagnosis details for a specific session"""
    try:
        db = get_optimized_database()
        diagnosis = db.get_patient_diagnosis_session_full(session_id)
        
        if not diagnosis:
            return jsonify({'error': 'Diagnosis session not found'}), 404
        
        if diagnosis['patient_id'] != patient_id:
            return jsonify({'error': 'Diagnosis does not belong to this patient'}), 403
        
        return jsonify(diagnosis)
        
    except Exception as e:
        logger.error(f"Error in get_diagnosis_full_details: {e}")
        return jsonify({'error': str(e)}), 500

@optimized_bp.route('/patients/<patient_id>/concern', methods=['GET'])
@monitor_performance
def get_patient_concern_optimized(patient_id: str):
    """Get patient CONCERN data (cached)"""
    try:
        db = get_optimized_database()
        concern_data = db.get_patient_concern_data_optimized(patient_id)
        
        return jsonify(concern_data)
        
    except Exception as e:
        logger.error(f"Error in get_patient_concern_optimized: {e}")
        return jsonify({'error': str(e)}), 500

# ===== BULK OPERATIONS =====

@optimized_bp.route('/patients/bulk', methods=['POST'])
@monitor_performance
def get_multiple_patients():
    """Get multiple patients by IDs (optimized bulk operation)"""
    try:
        data = request.get_json()
        patient_ids = data.get('patient_ids', [])
        
        if not patient_ids or len(patient_ids) > 50:  # Limit bulk operations
            return jsonify({'error': 'Invalid patient_ids (max 50)'}), 400
        
        db = get_optimized_database()
        patients = db.get_patients_bulk(patient_ids)
        
        return jsonify({
            'patients': patients,
            'requested': len(patient_ids),
            'found': len(patients)
        })
        
    except Exception as e:
        logger.error(f"Error in get_multiple_patients: {e}")
        return jsonify({'error': str(e)}), 500

# ===== PERFORMANCE ANALYTICS =====

@optimized_bp.route('/performance/stats', methods=['GET'])
@monitor_performance
def get_performance_stats():
    """Get database and cache performance statistics"""
    try:
        db = get_optimized_database()
        
        stats = {
            'database': {
                'connection_pool_size': db.engine.pool.size(),
                'checked_out_connections': db.engine.pool.checkedout(),
                'overflow_connections': db.engine.pool.overflow(),
                'invalid_connections': db.engine.pool.invalidated(),
            },
            'cache': {
                'enabled': db.redis_client is not None,
                'connected': False
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Test cache connection
        if db.redis_client:
            try:
                db.redis_client.ping()
                stats['cache']['connected'] = True
                
                # Get cache stats
                info = db.redis_client.info()
                stats['cache']['memory_usage'] = info.get('used_memory_human', 'Unknown')
                stats['cache']['connected_clients'] = info.get('connected_clients', 0)
                stats['cache']['total_commands'] = info.get('total_commands_processed', 0)
                
            except Exception as e:
                stats['cache']['error'] = str(e)
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in get_performance_stats: {e}")
        return jsonify({'error': str(e)}), 500

@optimized_bp.route('/cache/clear', methods=['POST'])
@monitor_performance
def clear_cache():
    """Clear application cache"""
    try:
        data = request.get_json() or {}
        pattern = data.get('pattern', '*')
        
        db = get_optimized_database()
        
        if not db.redis_client:
            return jsonify({'error': 'Cache not available'}), 400
        
        # Clear cache
        keys = db.redis_client.keys(f"*{pattern}*")
        if keys:
            deleted = db.redis_client.delete(*keys)
            return jsonify({
                'success': True,
                'keys_deleted': deleted,
                'pattern': pattern
            })
        else:
            return jsonify({
                'success': True,
                'keys_deleted': 0,
                'pattern': pattern,
                'message': 'No keys found matching pattern'
            })
        
    except Exception as e:
        logger.error(f"Error in clear_cache: {e}")
        return jsonify({'error': str(e)}), 500

# ===== HEALTH CHECK =====

@optimized_bp.route('/health', methods=['GET'])
def health_check_optimized():
    """Optimized health check endpoint"""
    try:
        db = get_optimized_database()
        
        # Quick database connectivity test
        with db.get_session() as session:
            session.execute("SELECT 1")
        
        health_status = {
            'status': 'healthy',
            'database': 'connected',
            'cache': 'connected' if db.redis_client else 'disabled',
            'timestamp': datetime.now().isoformat(),
            'version': 'v2_optimized'
        }
        
        # Test cache if available
        if db.redis_client:
            try:
                db.redis_client.ping()
            except Exception:
                health_status['cache'] = 'disconnected'
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
