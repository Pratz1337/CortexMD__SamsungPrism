-- Migration: Add persistent CONCERN severity tracking table
-- This table stores cumulative severity scores per patient that persist across backend restarts

-- Create concern_severity_tracking table
CREATE TABLE IF NOT EXISTS concern_severity_tracking (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    
    -- Cumulative severity tracking
    cumulative_severity FLOAT DEFAULT 0.0,
    total_diagnoses INTEGER DEFAULT 0,
    average_severity FLOAT DEFAULT 0.0,
    
    -- Last diagnosis severity components
    last_diagnosis_confidence FLOAT,
    last_fol_verification FLOAT,
    last_enhanced_verification FLOAT,
    last_explainability_score FLOAT,
    last_imaging_present BOOLEAN DEFAULT FALSE,
    last_computed_severity FLOAT,
    
    -- Current risk assessment
    current_risk_level VARCHAR(20) DEFAULT 'low', -- low, medium, high, critical
    current_risk_score FLOAT DEFAULT 0.0,
    
    -- Historical tracking
    max_severity_reached FLOAT DEFAULT 0.0,
    severity_history JSONB DEFAULT '[]', -- Array of {timestamp, severity, diagnosis_id}
    
    -- Metadata
    last_diagnosis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    first_diagnosis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(patient_id)
);

-- Create indexes for better query performance
CREATE INDEX idx_concern_severity_patient_id ON concern_severity_tracking(patient_id);
CREATE INDEX idx_concern_severity_risk_level ON concern_severity_tracking(current_risk_level);
CREATE INDEX idx_concern_severity_updated_at ON concern_severity_tracking(updated_at DESC);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_concern_severity_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-update updated_at
CREATE TRIGGER update_concern_severity_tracking_updated_at 
BEFORE UPDATE ON concern_severity_tracking 
FOR EACH ROW 
EXECUTE FUNCTION update_concern_severity_updated_at();

-- Create a view for easy access to concern severity with patient info
CREATE OR REPLACE VIEW v_concern_severity_summary AS
SELECT 
    cst.patient_id,
    p.patient_name,
    cst.cumulative_severity,
    cst.total_diagnoses,
    cst.average_severity,
    cst.current_risk_level,
    cst.current_risk_score,
    cst.max_severity_reached,
    cst.last_diagnosis_timestamp,
    cst.first_diagnosis_timestamp,
    cst.severity_history
FROM concern_severity_tracking cst
LEFT JOIN patients p ON cst.patient_id = p.patient_id
ORDER BY cst.current_risk_score DESC, cst.updated_at DESC;

-- Function to calculate risk level from severity score
CREATE OR REPLACE FUNCTION calculate_risk_level(severity_score FLOAT)
RETURNS VARCHAR(20) AS $$
BEGIN
    IF severity_score >= 0.8 THEN
        RETURN 'critical';
    ELSIF severity_score >= 0.6 THEN
        RETURN 'high';
    ELSIF severity_score >= 0.3 THEN
        RETURN 'medium';
    ELSE
        RETURN 'low';
    END IF;
END;
$$ language 'plpgsql';

COMMENT ON TABLE concern_severity_tracking IS 'Persistent storage for cumulative CONCERN severity scores that survive backend restarts';
COMMENT ON COLUMN concern_severity_tracking.cumulative_severity IS 'Sum of all severity scores for this patient';
COMMENT ON COLUMN concern_severity_tracking.total_diagnoses IS 'Total number of diagnoses performed for this patient';
COMMENT ON COLUMN concern_severity_tracking.average_severity IS 'Average severity score across all diagnoses';
COMMENT ON COLUMN concern_severity_tracking.severity_history IS 'JSON array of historical severity entries with timestamps';
