-- Migration: Add scanned_notes table for AR scanner feature
-- This table stores scanned medical note images with OCR text and AI summaries

-- Create scanned_notes table
CREATE TABLE IF NOT EXISTS scanned_notes (
    id SERIAL PRIMARY KEY,
    note_id VARCHAR(255) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    patient_id VARCHAR(255) NOT NULL,
    nurse_id VARCHAR(255) DEFAULT 'AR_SCANNER',
    
    -- Image storage
    image_data BYTEA NOT NULL,  -- Store the actual image in PostgreSQL
    image_mime_type VARCHAR(50) DEFAULT 'image/png',
    image_size INTEGER,
    thumbnail_data BYTEA,  -- Small preview image
    
    -- OCR and text extraction
    ocr_text TEXT,  -- Raw OCR output
    ocr_confidence FLOAT,
    ocr_metadata JSONB DEFAULT '{}',
    
    -- Parsed structured data
    parsed_data JSONB DEFAULT '{}',  -- Structured fields extracted from text
    
    -- AI-generated content
    ai_summary TEXT,  -- AI-generated summary of the note
    ai_extracted_entities JSONB DEFAULT '{}',  -- Medical entities extracted by AI
    ai_confidence_score FLOAT,
    ai_model_used VARCHAR(100),
    
    -- Clinical note link
    clinical_note_id VARCHAR(255),  -- Links to clinical_notes table
    
    -- Metadata
    scan_location VARCHAR(100),
    scan_shift VARCHAR(20),
    scan_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    processing_error TEXT,
    
    -- Audit fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (clinical_note_id) REFERENCES clinical_notes(note_id) ON DELETE SET NULL
);

-- Create indexes for better query performance
CREATE INDEX idx_scanned_notes_patient_id ON scanned_notes(patient_id);
CREATE INDEX idx_scanned_notes_nurse_id ON scanned_notes(nurse_id);
CREATE INDEX idx_scanned_notes_scan_timestamp ON scanned_notes(scan_timestamp DESC);
CREATE INDEX idx_scanned_notes_processing_status ON scanned_notes(processing_status);

-- Add image storage columns to existing clinical_notes table (optional)
ALTER TABLE clinical_notes 
ADD COLUMN IF NOT EXISTS has_scanned_image BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS scanned_note_id VARCHAR(255) REFERENCES scanned_notes(note_id) ON DELETE SET NULL;

-- Create a view for easy access to scanned notes with patient info
CREATE OR REPLACE VIEW v_scanned_notes_summary AS
SELECT 
    sn.note_id,
    sn.patient_id,
    p.patient_name,
    sn.nurse_id,
    sn.ai_summary,
    sn.ocr_confidence,
    sn.processing_status,
    sn.scan_timestamp,
    sn.scan_location,
    sn.parsed_data,
    sn.ai_extracted_entities
FROM scanned_notes sn
LEFT JOIN patients p ON sn.patient_id = p.patient_id
ORDER BY sn.scan_timestamp DESC;

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-update updated_at
CREATE TRIGGER update_scanned_notes_updated_at 
BEFORE UPDATE ON scanned_notes 
FOR EACH ROW 
EXECUTE FUNCTION update_updated_at_column();
