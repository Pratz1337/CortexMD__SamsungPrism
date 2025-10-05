"""
Database configuration and connection management for CortexMD
Supports PostgreSQL with Redis caching
"""

import os
import asyncio
import redis
import asyncpg
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration manager"""
    
    def __init__(self):
        # PostgreSQL Configuration
        self.DATABASE_URL = os.getenv('DATABASE_URL')
        self.POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
        self.POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
        self.POSTGRES_DB = os.getenv('POSTGRES_DB', 'cortexmd_db')
        self.POSTGRES_USER = os.getenv('POSTGRES_USER', 'cortexmd_user')
        self.POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'cortexmd_password')
        
        # Redis Configuration
        self.REDIS_URL = os.getenv('REDIS_URL')
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
        self.REDIS_DB = int(os.getenv('REDIS_DB', 0))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
        
        # Session Configuration
        self.SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 3600))
        self.MAX_SESSION_AGE = int(os.getenv('MAX_SESSION_AGE', 86400))
        
    def get_postgres_dsn(self) -> str:
        """Get PostgreSQL connection string"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def get_redis_connection_params(self) -> Dict[str, Any]:
        """Get Redis connection parameters"""
        params = {
            'host': self.REDIS_HOST,
            'port': self.REDIS_PORT,
            'db': self.REDIS_DB,
            'decode_responses': True
        }
        
        if self.REDIS_PASSWORD:
            params['password'] = self.REDIS_PASSWORD
            
        return params

class CortexMDDatabase:
    """Main database manager for CortexMD"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            await self.init_postgres()
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable, continuing without database: {e}")
            
        try:
            await self.init_redis()
        except Exception as e:
            logger.warning(f"Redis unavailable, continuing without caching: {e}")
            
        if self.pool:
            await self.create_tables()
        
    async def init_postgres(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.get_postgres_dsn(),
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("✅ PostgreSQL connection pool initialized")
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL unavailable: {e}")
            self.pool = None
            
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(**self.config.get_redis_connection_params())
            # Test connection
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            logger.info("✅ Redis connection initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
            # Redis is optional for caching, continue without it
            self.redis_client = None
            
    async def create_tables(self):
        """Create database tables if they don't exist"""
        if not self.pool:
            logger.warning("⚠️ Cannot create tables - PostgreSQL not available")
            return
            
        async with self.pool.acquire() as conn:
            # Patients table (master patient record)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id VARCHAR(255) PRIMARY KEY,
                    patient_name VARCHAR(255),
                    date_of_birth DATE,
                    gender VARCHAR(20),
                    contact_info JSONB DEFAULT '{}',
                    admission_date TIMESTAMP,
                    discharge_date TIMESTAMP,
                    current_status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Diagnosis Sessions table (linked to patients)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS diagnosis_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    patient_id VARCHAR(255) NOT NULL,
                    diagnosis_type VARCHAR(50) DEFAULT 'general',
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    current_step TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    anonymize BOOLEAN DEFAULT FALSE,
                    patient_input JSONB,
                    diagnosis_result JSONB,
                    explanations JSONB,
                    enhanced_results JSONB,
                    fol_verification JSONB,
                    enhanced_verification JSONB,
                    explainability_score FLOAT DEFAULT 0.0,
                    error_message TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                )
            ''')
            
            # Textbook Content table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS textbook_content (
                    id SERIAL PRIMARY KEY,
                    textbook_name VARCHAR(255) NOT NULL,
                    edition VARCHAR(50),
                    chapter VARCHAR(255),
                    section VARCHAR(255),
                    page_number INTEGER,
                    line_number INTEGER,
                    content TEXT NOT NULL,
                    medical_conditions TEXT[],
                    keywords TEXT[],
                    content_hash VARCHAR(64) UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Chatbot Sessions table (linked to patients)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS chatbot_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    patient_id VARCHAR(255) NOT NULL,
                    conversation_history JSONB DEFAULT '[]',
                    context JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                )
            ''')
            
            # Patient Data table (for audit/history)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS patient_data (
                    id SERIAL PRIMARY KEY,
                    patient_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255),
                    data_type VARCHAR(50), -- 'text', 'image', 'fhir'
                    data_content JSONB,
                    file_path TEXT,
                    anonymized BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES diagnosis_sessions(session_id)
                )
            ''')
            
            # CONCERN Early Warning System Tables
            
            # Clinical Notes table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS clinical_notes (
                    note_id VARCHAR(255) PRIMARY KEY,
                    patient_id VARCHAR(255) NOT NULL,
                    nurse_id VARCHAR(255),
                    note_content TEXT NOT NULL,
                    note_type VARCHAR(50) DEFAULT 'nursing',
                    timestamp TIMESTAMP NOT NULL,
                    location VARCHAR(255),
                    shift VARCHAR(50),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Patient Visits table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS patient_visits (
                    visit_id VARCHAR(255) PRIMARY KEY,
                    patient_id VARCHAR(255) NOT NULL,
                    nurse_id VARCHAR(255),
                    visit_timestamp TIMESTAMP NOT NULL,
                    visit_duration INTEGER,
                    visit_type VARCHAR(50) DEFAULT 'routine',
                    location VARCHAR(255) NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # CONCERN Scores table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS concern_scores (
                    id SERIAL PRIMARY KEY,
                    patient_id VARCHAR(255) NOT NULL,
                    concern_score FLOAT NOT NULL CHECK (concern_score >= 0.0 AND concern_score <= 1.0),
                    risk_level VARCHAR(20) NOT NULL,
                    contributing_factors TEXT[],
                    metadata_patterns JSONB DEFAULT '{}',
                    timestamp TIMESTAMP NOT NULL,
                    alert_triggered BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # GradCAM Images table (for storing AI explainability visualizations)
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS gradcam_images (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    patient_id VARCHAR(255) NOT NULL,
                    original_image_path TEXT NOT NULL,
                    image_filename VARCHAR(255) NOT NULL,
                    heatmap_image BYTEA,
                    overlay_image BYTEA,
                    volume_image BYTEA,
                    analysis_data JSONB,
                    predictions JSONB,
                    activation_regions JSONB,
                    medical_interpretation JSONB,
                    processing_successful BOOLEAN DEFAULT FALSE,
                    processing_time FLOAT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES diagnosis_sessions(session_id),
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                )
            ''')
            
            # Patient Monitoring table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS patient_monitoring (
                    patient_id VARCHAR(255) PRIMARY KEY,
                    admission_date TIMESTAMP NOT NULL,
                    current_status VARCHAR(50) DEFAULT 'stable',
                    latest_concern_score FLOAT,
                    trend_data JSONB DEFAULT '[]',
                    alerts_history JSONB DEFAULT '[]',
                    nurse_visit_frequency FLOAT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_diagnosis_sessions_status ON diagnosis_sessions(status)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_diagnosis_sessions_created_at ON diagnosis_sessions(created_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_textbook_content_conditions ON textbook_content USING GIN(medical_conditions)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_textbook_content_keywords ON textbook_content USING GIN(keywords)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_chatbot_sessions_last_activity ON chatbot_sessions(last_activity)')
            
            # CONCERN EWS indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_clinical_notes_patient_id ON clinical_notes(patient_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_clinical_notes_timestamp ON clinical_notes(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_patient_visits_patient_id ON patient_visits(patient_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_patient_visits_timestamp ON patient_visits(visit_timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_concern_scores_patient_id ON concern_scores(patient_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_concern_scores_timestamp ON concern_scores(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_concern_scores_risk_level ON concern_scores(risk_level)')
            
            # Patient management indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_patients_status ON patients(current_status)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_patients_admission ON patients(admission_date)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_diagnosis_sessions_patient_id ON diagnosis_sessions(patient_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_chatbot_sessions_patient_id ON chatbot_sessions(patient_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_gradcam_images_session_id ON gradcam_images(session_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_gradcam_images_patient_id ON gradcam_images(patient_id)')
            
        logger.info("✅ Database tables created/verified")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("PostgreSQL not available")
        async with self.pool.acquire() as conn:
            yield conn

    async def save_gradcam_images(self, session_id: str, patient_id: str, gradcam_data: list):
        """Save GradCAM images to PostgreSQL database"""
        try:
            async with self.get_connection() as conn:
                for item in gradcam_data:
                    if not item.get('success', False):
                        continue
                    
                    # Convert base64 images to bytes
                    heatmap_bytes = None
                    overlay_bytes = None
                    volume_bytes = None
                    
                    if item.get('visualizations'):
                        import base64
                        vis = item['visualizations']
                        if vis.get('heatmap_image'):
                            heatmap_bytes = base64.b64decode(vis['heatmap_image'])
                        if vis.get('overlay_image'):
                            overlay_bytes = base64.b64decode(vis['overlay_image'])
                        if vis.get('volume_image'):
                            volume_bytes = base64.b64decode(vis['volume_image'])
                    
                    await conn.execute('''
                        INSERT INTO gradcam_images (
                            session_id, patient_id, original_image_path, image_filename,
                            heatmap_image, overlay_image, volume_image,
                            analysis_data, predictions, activation_regions, medical_interpretation,
                            processing_successful, processing_time, error_message
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ''', 
                    session_id, 
                    patient_id,
                    item.get('image_file', ''),
                    item.get('image_file', '').split('/')[-1] if item.get('image_file') else '',
                    heatmap_bytes,
                    overlay_bytes, 
                    volume_bytes,
                    json.dumps(item.get('analysis', {})),
                    json.dumps(item.get('predictions', [])),
                    json.dumps(item.get('activation_regions', [])),
                    json.dumps(item.get('medical_interpretation', {})),
                    item.get('success', False),
                    item.get('analysis', {}).get('processing_time', 0.0),
                    item.get('error', None)
                    )
                    
                logger.info(f"✅ Saved GradCAM images for session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving GradCAM images: {str(e)}")
            return False

    async def get_gradcam_images(self, session_id: str):
        """Retrieve GradCAM images from PostgreSQL database"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch('''
                    SELECT 
                        original_image_path, image_filename,
                        heatmap_image, overlay_image, volume_image,
                        analysis_data, predictions, activation_regions, medical_interpretation,
                        processing_successful, processing_time, error_message
                    FROM gradcam_images 
                    WHERE session_id = $1
                    ORDER BY created_at ASC
                ''', session_id)
                
                result = []
                for row in rows:
                    import base64
                    
                    # Convert bytes back to base64
                    visualizations = {}
                    if row['heatmap_image']:
                        visualizations['heatmap_image'] = base64.b64encode(row['heatmap_image']).decode('utf-8')
                    if row['overlay_image']:
                        visualizations['overlay_image'] = base64.b64encode(row['overlay_image']).decode('utf-8')
                    if row['volume_image']:
                        visualizations['volume_image'] = base64.b64encode(row['volume_image']).decode('utf-8')
                    
                    item = {
                        'success': row['processing_successful'],
                        'image_file': row['original_image_path'],
                        'visualizations': visualizations if visualizations else None,
                        'analysis': json.loads(row['analysis_data']) if row['analysis_data'] else None,
                        'predictions': json.loads(row['predictions']) if row['predictions'] else None,
                        'activation_regions': json.loads(row['activation_regions']) if row['activation_regions'] else None,
                        'medical_interpretation': json.loads(row['medical_interpretation']) if row['medical_interpretation'] else None,
                        'error': row['error_message']
                    }
                    result.append(item)
                
                logger.info(f"✅ Retrieved {len(result)} GradCAM images for session {session_id}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Error retrieving GradCAM images: {str(e)}")
            return []

    def save_gradcam_images_sync(self, session_id: str, patient_id: str, gradcam_data: list):
        """Synchronous wrapper for saving GradCAM images"""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.save_gradcam_images(session_id, patient_id, gradcam_data))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"❌ Error in sync save_gradcam_images: {str(e)}")
            return False

    def get_gradcam_images_sync(self, session_id: str):
        """Synchronous wrapper for retrieving GradCAM images"""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.get_gradcam_images(session_id))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"❌ Error in sync get_gradcam_images: {str(e)}")
            return []
            
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.close)

class SessionManager:
    """Manages diagnosis and chatbot sessions with Redis caching"""
    
    def __init__(self, db: CortexMDDatabase):
        self.db = db
        self.redis = db.redis_client
        
    async def create_diagnosis_session(self, session_id: str, patient_input: Dict, anonymize: bool = False) -> bool:
        """Create a new diagnosis session"""
        try:
            async with self.db.get_connection() as conn:
                await conn.execute('''
                    INSERT INTO diagnosis_sessions (session_id, patient_id, patient_input, anonymize, status)
                    VALUES ($1, $2, $3, $4, 'pending')
                ''', session_id, patient_input.get('patient_id'), json.dumps(patient_input), anonymize)
                
            # Cache in Redis if available
            if self.redis:
                session_data = {
                    'session_id': session_id,
                    'patient_input': patient_input,
                    'anonymize': anonymize,
                    'status': 'pending',
                    'progress': 0,
                    'created_at': datetime.now().isoformat()
                }
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.redis.setex,
                    f"session:{session_id}",
                    self.db.config.SESSION_TIMEOUT,
                    json.dumps(session_data)
                )
                
            return True
        except Exception as e:
            logger.error(f"Failed to create diagnosis session {session_id}: {e}")
            return False
            
    async def get_diagnosis_session(self, session_id: str) -> Optional[Dict]:
        """Get diagnosis session data"""
        # Try Redis cache first
        if self.redis:
            try:
                cached_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis.get, f"session:{session_id}"
                )
                if cached_data:
                    return json.loads(cached_data)#type:ignore
            except Exception as e:
                logger.warning(f"Redis cache miss for session {session_id}: {e}")
        
        # Fallback to database
        try:
            async with self.db.get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT * FROM diagnosis_sessions WHERE session_id = $1
                ''', session_id)
                
                if row:
                    session_data = dict(row)
                    # Convert JSON fields
                    for field in ['patient_input', 'diagnosis_result', 'explanations', 'enhanced_results', 'fol_verification', 'enhanced_verification']:
                        if session_data.get(field):
                            session_data[field] = json.loads(session_data[field]) if isinstance(session_data[field], str) else session_data[field]
                    
                    # Update cache if Redis is available
                    if self.redis:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.redis.setex,
                            f"session:{session_id}",
                            self.db.config.SESSION_TIMEOUT,
                            json.dumps(session_data, default=str)
                        )
                    
                    return session_data
                    
        except Exception as e:
            logger.error(f"Failed to get diagnosis session {session_id}: {e}")
            
        return None
        
    async def update_diagnosis_session(self, session_id: str, updates: Dict) -> bool:
        """Update diagnosis session"""
        try:
            # Build dynamic update query
            set_clauses = []
            values = []
            param_count = 1
            
            for key, value in updates.items():
                if key in ['patient_input', 'diagnosis_result', 'explanations', 'enhanced_results', 'fol_verification', 'enhanced_verification']:
                    set_clauses.append(f"{key} = ${param_count}")
                    values.append(json.dumps(value) if not isinstance(value, str) else value)
                else:
                    set_clauses.append(f"{key} = ${param_count}")
                    values.append(value)
                param_count += 1
            
            set_clauses.append(f"updated_at = ${param_count}")
            values.append(datetime.now())
            values.append(session_id)  # for WHERE clause
            
            query = f'''
                UPDATE diagnosis_sessions 
                SET {", ".join(set_clauses)}
                WHERE session_id = ${param_count + 1}
            '''
            
            async with self.db.get_connection() as conn:
                await conn.execute(query, *values)
            
            # Update Redis cache
            if self.redis:
                session_data = await self.get_diagnosis_session(session_id)
                if session_data:
                    session_data.update(updates)
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.redis.setex,
                        f"session:{session_id}",
                        self.db.config.SESSION_TIMEOUT,
                        json.dumps(session_data, default=str)
                    )
            
            return True
        except Exception as e:
            logger.error(f"Failed to update diagnosis session {session_id}: {e}")
            return False

    async def clear_session(self, session_id: str) -> bool:
        """Clear a specific diagnosis session from database and cache"""
        try:
            async with self.db.get_connection() as conn:
                await conn.execute('''
                    DELETE FROM diagnosis_sessions WHERE session_id = $1
                ''', session_id)
            
            # Clear from Redis cache
            if self.redis:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.redis.delete,
                    f"session:{session_id}"
                )
            
            logger.info(f"Cleared session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False

    async def clear_all_sessions(self) -> bool:
        """Clear all diagnosis sessions from database and cache"""
        try:
            async with self.db.get_connection() as conn:
                await conn.execute('DELETE FROM diagnosis_sessions')
            
            # Clear all session keys from Redis
            if self.redis:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._clear_redis_sessions
                )
            
            logger.info("Cleared all diagnosis sessions")
            return True
        except Exception as e:
            logger.error(f"Failed to clear all sessions: {e}")
            return False

    def _clear_redis_sessions(self):
        """Helper method to clear all session keys from Redis"""
        try:
            # Find all session keys
            keys = self.redis.keys("session:*")#type:ignore
            if keys:
                self.redis.delete(*keys)#type:ignore
        except Exception as e:
            logger.warning(f"Failed to clear Redis session keys: {e}")

    async def clear_expired_sessions(self) -> int:
        """Clear expired sessions from database and return count of cleared sessions"""
        try:
            # Get expiry time (default 24 hours ago)
            expiry_time = datetime.now() - timedelta(hours=24)
            
            async with self.db.get_connection() as conn:
                # Get expired session IDs first
                expired_sessions = await conn.fetch('''
                    SELECT session_id FROM diagnosis_sessions 
                    WHERE created_at < $1 OR updated_at < $1
                ''', expiry_time)
                
                # Delete expired sessions
                result = await conn.execute('''
                    DELETE FROM diagnosis_sessions 
                    WHERE created_at < $1 OR updated_at < $1
                ''', expiry_time)
                
                # Clear expired sessions from Redis cache
                if self.redis and expired_sessions:
                    for session in expired_sessions:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.redis.delete,
                            f"session:{session['session_id']}"
                        )
                
                cleared_count = len(expired_sessions)
                logger.info(f"Cleared {cleared_count} expired sessions")
                return cleared_count
                
        except Exception as e:
            logger.error(f"Failed to clear expired sessions: {e}")
            return 0

# Global database instance
db_instance: Optional[CortexMDDatabase] = None
session_manager: Optional[SessionManager] = None

async def initialize_database():
    """Initialize global database instance"""
    global db_instance, session_manager
    
    db_instance = CortexMDDatabase()
    await db_instance.initialize()
    session_manager = SessionManager(db_instance)
    
    return db_instance, session_manager

async def get_database() -> CortexMDDatabase:
    """Get global database instance"""
    global db_instance
    if not db_instance:
        await initialize_database()
    return db_instance#type:ignore

async def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global session_manager
    if not session_manager:
        await initialize_database()
    return session_manager#type:ignore
