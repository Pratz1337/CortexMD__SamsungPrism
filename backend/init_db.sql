-- CortexMD Database Initialization Script
-- Creates database schema and user if they don't exist

-- Create the cortexmd_user if it doesn't exist
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cortexmd_user') THEN
      CREATE USER cortexmd_user WITH PASSWORD 'cortexmd_password';
   END IF;
END
$$;

-- Grant privileges to cortexmd_user
GRANT ALL PRIVILEGES ON DATABASE cortexmd_db TO cortexmd_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cortexmd_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cortexmd_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO cortexmd_user;

-- Alter default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO cortexmd_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO cortexmd_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO cortexmd_user;

-- Create basic tables for CortexMD application
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_query TEXT,
    diagnosis_result JSONB,
    confidence_scores JSONB,
    fol_verification JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS medical_images (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES analysis_sessions(session_id),
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    mime_type VARCHAR(100),
    analysis_result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES analysis_sessions(session_id),
    feedback_type VARCHAR(50) NOT NULL,
    feedback_text TEXT,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_session_id ON analysis_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_medical_images_session_id ON medical_images(session_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_session_id ON user_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_created_at ON analysis_sessions(created_at);

-- Insert some sample data for testing
INSERT INTO analysis_sessions (session_id, user_query, diagnosis_result, confidence_scores) 
VALUES ('test_session_001', 'Sample medical query for testing', '{"diagnosis": "test"}', '{"confidence": 0.95}')
ON CONFLICT (session_id) DO NOTHING;

-- Grant usage on sequences
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cortexmd_user;

COMMENT ON DATABASE cortexmd_db IS 'CortexMD Medical Analysis Platform Database';
COMMENT ON TABLE analysis_sessions IS 'Stores medical analysis sessions and results';
COMMENT ON TABLE medical_images IS 'Stores uploaded medical images and their analysis results';
COMMENT ON TABLE user_feedback IS 'Stores user feedback and ratings for the system';
