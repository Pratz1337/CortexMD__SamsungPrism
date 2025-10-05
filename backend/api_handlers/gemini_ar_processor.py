import io
import os
import re
import json
import base64
import logging
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime
import uuid

from PIL import Image, ImageOps, ImageFilter
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    pytesseract = None
    TESS_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

# Optional: Allow configuring tesseract path via env var on Windows
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESS_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logger = logging.getLogger(__name__)

# Initialize Gemini client via ai_key_manager (supports multiple keys)
gemini_model = None
if GEMINI_AVAILABLE:
    try:
        from ai_key_manager import get_gemini_model
        gemini_model = get_gemini_model('gemini-1.5-flash')
        if gemini_model:
            logger.info("Gemini client initialized successfully via ai_key_manager")
        else:
            logger.warning("Gemini model not available via ai_key_manager (no keys)")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client via ai_key_manager: {e}")
        gemini_model = None
else:
    gemini_model = None


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Basic preprocessing to improve OCR quality."""
    # Convert to grayscale, increase contrast, slight sharpen
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    
    # Enhanced binarization with adaptive thresholding
    g = g.point(lambda x: 255 if x > 160 else 0, mode='1')
    
    # Additional noise reduction
    g = g.filter(ImageFilter.MedianFilter(size=3))
    
    return g


def run_fast_ocr(image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Run fast OCR on the provided image bytes.
    """
    try:
        # Try to open the image with PIL
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        processed = _preprocess_for_ocr(img)
    except Exception as e:
        logger.error(f"Failed to open image with PIL: {e}")
        return "Image processing failed", {
            'engine': 'error',
            'avg_word_confidence': 0.0,
            'word_count': 0,
            'line_count': 0,
            'total_characters': 0,
            'processing_timestamp': datetime.now().isoformat()
        }

    if not TESS_AVAILABLE:
        # Return basic text if Tesseract is not available
        return "Medical note image processed - OCR not available", {
            'engine': 'fallback',
            'avg_word_confidence': 0.0,
            'word_count': 0,
            'line_count': 0,
            'total_characters': 0,
            'processing_timestamp': datetime.now().isoformat()
        }

    try:
        # Get basic OCR text
        text = pytesseract.image_to_string(processed)
        
        # Get confidence data
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data.get('conf', []) if str(c).isdigit()]
        avg_conf = sum(confs)/len(confs) if confs else 0
        
        # Count words and lines
        word_count = len([w for w in data.get('text', []) if w.strip()])
        line_count = len([l for l in text.split('\n') if l.strip()])

        meta = {
            'engine': 'tesseract_fast',
            'avg_word_confidence': round(avg_conf, 2),
            'word_count': word_count,
            'line_count': line_count,
            'total_characters': len(text),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return text.strip(), meta
        
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return "OCR processing failed", {
            'engine': 'error',
            'avg_word_confidence': 0.0,
            'word_count': 0,
            'line_count': 0,
            'total_characters': 0,
            'processing_timestamp': datetime.now().isoformat()
        }


def gemini_parse_medical_text(ocr_text: str) -> Dict[str, Any]:
    """
    Use Gemini AI to parse and structure medical text from OCR output.
    """
    if not gemini_model or not GEMINI_AVAILABLE:
        logger.warning("Gemini not available, using fallback parsing")
        return _fallback_parse_medical_text(ocr_text)
    
    try:
        prompt = f"""
        You are an expert medical AI assistant. Parse the following OCR text from a medical document and extract ALL medical information comprehensively.
        
        OCR Text:
        {ocr_text}
        
        Please extract and return a JSON object that captures ALL medical data found in the text. Be thorough and include:
        
        1. ALL patient information and demographics
        2. ALL vital signs, measurements, and physiological data
        3. ALL medications (current, prescribed, discontinued, dosages, frequencies)
        4. ALL diagnoses, conditions, medical problems (primary, secondary, differential)
        5. ALL procedures, tests, examinations, and interventions
        6. ALL clinical findings, observations, and examination results
        7. ALL allergies, adverse reactions, and contraindications
        8. ALL laboratory results, test values, and imaging findings
        9. ALL treatment plans, recommendations, and follow-up instructions
        10. ALL medical history, family history, and social history
        11. ANY other medical information present in the text
        
        Structure your response as a comprehensive JSON object:
        {{
            "comprehensive_medical_data": {{
                "patient_information": {{
                    "name": "extract if found",
                    "patient_id": "extract if found",
                    "mrn": "extract if found",
                    "age": "extract if found",
                    "date_of_birth": "extract if found",
                    "gender": "extract if found",
                    "contact_info": "extract if found"
                }},
                "document_details": {{
                    "document_type": "determine type (progress note, discharge summary, etc.)",
                    "date": "extract document date if found",
                    "time": "extract time if found",
                    "provider": "extract provider name if found",
                    "department": "extract department if found",
                    "facility": "extract facility name if found"
                }},
                "vital_signs_and_measurements": {{
                    "blood_pressure": "extract with units",
                    "heart_rate": "extract with units",
                    "temperature": "extract with units",
                    "respiratory_rate": "extract with units",
                    "oxygen_saturation": "extract with units",
                    "weight": "extract with units",
                    "height": "extract with units",
                    "bmi": "extract if calculated",
                    "other_vitals": ["list any other vital signs or measurements"]
                }},
                "medications_comprehensive": {{
                    "current_medications": ["list all current medications with dosages and frequencies"],
                    "newly_prescribed": ["list all newly prescribed medications with details"],
                    "discontinued_medications": ["list any stopped medications"],
                    "medication_changes": ["list any dose changes or modifications"],
                    "allergic_medications": ["list medications patient is allergic to"]
                }},
                "diagnoses_and_conditions": {{
                    "primary_diagnosis": "main diagnosis with ICD codes if present",
                    "secondary_diagnoses": ["list all secondary diagnoses"],
                    "differential_diagnoses": ["list possible differential diagnoses"],
                    "chronic_conditions": ["list ongoing chronic conditions"],
                    "acute_conditions": ["list acute conditions"],
                    "ruled_out_conditions": ["list conditions that were ruled out"]
                }},
                "clinical_assessment": {{
                    "chief_complaint": "extract patient's main complaint",
                    "history_of_present_illness": "extract HPI details",
                    "review_of_systems": ["extract ROS findings"],
                    "physical_examination": {{
                        "general": "general appearance notes",
                        "cardiovascular": "CV exam findings",
                        "respiratory": "respiratory exam findings",
                        "gastrointestinal": "GI exam findings",
                        "neurological": "neuro exam findings",
                        "musculoskeletal": "MSK exam findings",
                        "other_systems": ["any other system examinations"]
                    }},
                    "clinical_impression": "provider's clinical impression"
                }},
                "laboratory_and_diagnostics": {{
                    "blood_tests": ["list all blood test results with values and reference ranges"],
                    "chemistry_panel": ["list chemistry results"],
                    "hematology": ["list CBC and hematology results"],
                    "microbiology": ["list culture and sensitivity results"],
                    "pathology": ["list pathology results"],
                    "imaging_studies": ["list all imaging with findings"],
                    "other_diagnostics": ["list any other diagnostic tests"]
                }},
                "procedures_and_interventions": {{
                    "procedures_performed": ["list all procedures done"],
                    "planned_procedures": ["list scheduled procedures"],
                    "surgical_history": ["list past surgeries"],
                    "interventions": ["list any medical interventions"]
                }},
                "allergies_and_adverse_reactions": {{
                    "drug_allergies": ["list all drug allergies with reactions"],
                    "food_allergies": ["list food allergies"],
                    "environmental_allergies": ["list environmental allergies"],
                    "adverse_drug_reactions": ["list any ADRs"],
                    "contraindications": ["list any contraindications"]
                }},
                "treatment_and_management": {{
                    "immediate_treatment": "current treatment plan",
                    "ongoing_management": "long-term management plan",
                    "discharge_instructions": "discharge or follow-up instructions",
                    "lifestyle_modifications": ["list lifestyle recommendations"],
                    "patient_education": ["list educational points provided"],
                    "follow_up_appointments": ["list scheduled follow-ups"]
                }},
                "medical_history": {{
                    "past_medical_history": ["list all past medical conditions"],
                    "surgical_history": ["list all past surgeries"],
                    "family_history": ["list relevant family history"],
                    "social_history": ["list social history including habits, occupation"],
                    "psychiatric_history": ["list any psychiatric conditions"]
                }},
                "additional_clinical_data": {{
                    "progress_notes": ["any progress or nursing notes"],
                    "physician_orders": ["any new orders or instructions"],
                    "consultation_notes": ["any specialist consultation notes"],
                    "care_coordination": ["any care coordination notes"],
                    "other_medical_information": ["any other relevant medical data found"]
                }}
            }},
            "metadata": {{
                "confidence_score": 0.90,
                "processing_timestamp": "{datetime.now().isoformat()}",
                "ai_model": "gemini-1.5-flash",
                "extraction_completeness": "comprehensive",
                "notes": "Any processing notes or observations"
            }}
        }}
        
        CRITICAL INSTRUCTIONS:
        - Extract EVERY piece of medical information found in the text
        - Preserve exact medical terminology and values
        - Include units of measurement whenever present
        - Use null or empty arrays only for truly missing information
        - If medical information doesn't fit existing categories, add new fields
        - Maintain clinical accuracy and context
        - Include dosages, frequencies, and routes for medications
        - Preserve timing and temporal relationships in clinical data
        
        Return only valid JSON without any markdown formatting.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response text
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Parse the JSON response
        try:
            parsed_data = json.loads(response_text)
            return parsed_data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Gemini response as JSON: {e}")
            logger.warning(f"Raw response: {response_text[:500]}...")
            return _fallback_parse_medical_text(ocr_text)
            
    except Exception as e:
        logger.error(f"Gemini parsing failed: {e}")
        return _fallback_parse_medical_text(ocr_text)


def _fallback_parse_medical_text(text: str) -> Dict[str, Any]:
    """
    Comprehensive fallback parsing using regex patterns when AI parsing fails.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lower = text.lower()

    def grab(pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    def grab_all(pattern: str, flags: int = re.IGNORECASE) -> List[str]:
        matches = re.findall(pattern, text, flags)
        return [m.strip() if isinstance(m, str) else m[0].strip() for m in matches if m]

    # Patient information extraction
    patient_name = grab(r"(?:patient\s*(?:name)?|name)\s*[:\-]\s*([^\n]+)")
    patient_id = grab(r"(?:mrn|patient\s*id|id|medical\s*record)\s*[:\-]\s*([^\n]+)")
    age = grab(r"(?:age)\s*[:\-]\s*(\d{1,3})")
    gender = grab(r"(?:gender|sex)\s*[:\-]\s*([^\n]+)")
    dob = grab(r"(?:dob|date\s*of\s*birth)\s*[:\-]\s*([^\n]+)")

    # Vitals extraction with enhanced patterns
    vitals = {}
    bp = re.search(r"\b(?:bp|blood\s*pressure)\s*[:\-]?\s*(\d{2,3}\/?\d{2,3})\s*(?:mmhg)?", lower)
    hr = re.search(r"\b(?:hr|heart\s*rate|pulse)\s*[:\-]?\s*(\d{2,3})\s*(?:bpm)?", lower)
    temp = re.search(r"\b(?:temp|temperature)\s*[:\-]?\s*([\d\.]+\s*(?:°?[fc])?)", lower)
    rr = re.search(r"\b(?:rr|resp(?:iratory)?\s*rate)\s*[:\-]?\s*(\d{2,3})", lower)
    spo2 = re.search(r"\b(?:spo2|o2\s*(?:sat)?|oxygen\s*saturation)\s*[:\-]?\s*(\d{2,3})%?", lower)
    weight = re.search(r"\b(?:weight|wt)\s*[:\-]?\s*([\d\.]+\s*(?:kg|lbs?))", lower)
    height = re.search(r"\b(?:height|ht)\s*[:\-]?\s*([\d\.]+\s*(?:cm|ft|in))", lower)

    if bp: vitals['blood_pressure'] = bp.group(1)
    if hr: vitals['heart_rate'] = hr.group(1)
    if temp: vitals['temperature'] = temp.group(1)
    if rr: vitals['respiratory_rate'] = rr.group(1)
    if spo2: vitals['oxygen_saturation'] = spo2.group(1)
    if weight: vitals['weight'] = weight.group(1)
    if height: vitals['height'] = height.group(1)

    # Medication extraction
    medications = grab_all(r"(?:medication|drug|rx)\s*[:\-]?\s*([^\n]+)")
    
    # Diagnosis extraction
    diagnoses = grab_all(r"(?:diagnosis|dx|impression)\s*[:\-]?\s*([^\n]+)")
    
    # Laboratory results extraction
    lab_results = grab_all(r"(?:lab|laboratory|test)\s*[:\-]?\s*([^\n]+)")
    
    # Allergies extraction
    allergies = grab_all(r"(?:allerg|nkda|adverse)\s*[:\-]?\s*([^\n]+)")

    return {
        "comprehensive_medical_data": {
            "patient_information": {
                "name": patient_name,
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "date_of_birth": dob
            },
            "vital_signs_and_measurements": vitals if vitals else {},
            "medications_comprehensive": {
                "current_medications": medications,
                "newly_prescribed": [],
                "discontinued_medications": []
            },
            "diagnoses_and_conditions": {
                "primary_diagnosis": diagnoses[0] if diagnoses else None,
                "secondary_diagnoses": diagnoses[1:] if len(diagnoses) > 1 else []
            },
            "clinical_assessment": {
                "chief_complaint": grab(r"(?:chief\s*complaint|cc)\s*[:\-]\s*([^\n]+)"),
                "history_of_present_illness": grab(r"(?:hpi|history\s*of\s*present\s*illness)\s*[:\-]\s*([^\n]+)")
            },
            "laboratory_and_diagnostics": {
                "blood_tests": lab_results,
                "other_diagnostics": []
            },
            "allergies_and_adverse_reactions": {
                "drug_allergies": allergies,
                "food_allergies": [],
                "environmental_allergies": []
            },
            "treatment_and_management": {
                "immediate_treatment": grab(r"(?:plan|treatment)\s*[:\-]\s*([^\n]+)"),
                "ongoing_management": None,
                "follow_up_appointments": []
            },
            "additional_clinical_data": {
                "other_medical_information": [line for line in lines if len(line) > 10]
            }
        },
        "metadata": {
            "confidence_score": 0.5,  # Lower confidence for fallback
            "processing_timestamp": datetime.now().isoformat(),
            "ai_model": "fallback_regex",
            "extraction_completeness": "basic"
        }
    }


def generate_gemini_summary(parsed_data: Dict[str, Any], ocr_text: str) -> str:
    """
    Generate a Gemini-powered comprehensive summary of the medical note.
    """
    if not gemini_model or not GEMINI_AVAILABLE:
        return _generate_fallback_summary(parsed_data, ocr_text)
    
    try:
        prompt = f"""
        You are an expert medical AI assistant. Based on the following comprehensive medical data and original OCR text, 
        generate a detailed, professional clinical summary suitable for healthcare providers.
        
        Comprehensive Medical Data:
        {json.dumps(parsed_data, indent=2)}
        
        Original OCR Text:
        {ocr_text}
        
        Please provide a comprehensive clinical summary that includes:
        1. Patient identification and key demographics
        2. Primary and secondary diagnoses with clinical significance
        3. Key vital signs and laboratory findings with clinical context
        4. Current medications and any changes made
        5. Assessment of clinical status and severity
        6. Treatment plan and follow-up requirements
        7. Any critical findings or concerns that require attention
        
        Format the summary as a professional clinical note that would be suitable for:
        - Provider handoffs
        - Continuity of care
        - Medical record documentation
        
        Keep it comprehensive yet concise, medically accurate, and clinically relevant.
        Include specific values and findings where available.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Gemini summary generation failed: {e}")
        return _generate_fallback_summary(parsed_data, ocr_text)


def _generate_fallback_summary(parsed_data: Dict[str, Any], ocr_text: str) -> str:
    """
    Generate a comprehensive summary when AI generation fails.
    """
    try:
        medical_data = parsed_data.get('comprehensive_medical_data', {})
        patient_info = medical_data.get('patient_information', {})
        vitals = medical_data.get('vital_signs_and_measurements', {})
        diagnoses = medical_data.get('diagnoses_and_conditions', {})
        medications = medical_data.get('medications_comprehensive', {})
        
        summary_parts = []
        
        # Patient info
        if patient_info.get('name'):
            summary_parts.append(f"Patient: {patient_info['name']}")
        if patient_info.get('age'):
            summary_parts.append(f"Age: {patient_info['age']}")
        if patient_info.get('gender'):
            summary_parts.append(f"Gender: {patient_info['gender']}")
        
        # Diagnoses
        if diagnoses.get('primary_diagnosis'):
            summary_parts.append(f"Primary Diagnosis: {diagnoses['primary_diagnosis']}")
        
        # Key vitals
        if vitals:
            vital_parts = []
            for key, value in vitals.items():
                if value:
                    vital_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            if vital_parts:
                summary_parts.append(f"Vitals: {', '.join(vital_parts)}")
        
        # Medications
        current_meds = medications.get('current_medications', [])
        if current_meds:
            summary_parts.append(f"Current Medications: {', '.join(current_meds[:3])}{'...' if len(current_meds) > 3 else ''}")
        
        if summary_parts:
            return ". ".join(summary_parts) + f". Comprehensive medical data extracted from {len(ocr_text)} characters of clinical text."
        else:
            return f"Clinical note processed and analyzed. Extracted comprehensive medical data from {len(ocr_text)} characters of text."
    except Exception as e:
        logger.error(f"Fallback summary generation failed: {e}")
        return f"Clinical note scanned and processed. OCR extracted {len(ocr_text)} characters of text."


def extract_comprehensive_medical_entities(parsed_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract comprehensive medical entities from parsed data for enhanced searchability.
    """
    entities = {
        "medications": [],
        "diagnoses": [],
        "procedures": [],
        "symptoms": [],
        "allergies": [],
        "laboratory_tests": [],
        "vital_signs": [],
        "providers": [],
        "facilities": []
    }
    
    try:
        medical_data = parsed_data.get('comprehensive_medical_data', {})
        
        # Extract medications from all sources
        medications = medical_data.get('medications_comprehensive', {})
        for med_type in ['current_medications', 'newly_prescribed', 'discontinued_medications']:
            if medications.get(med_type):
                entities["medications"].extend(medications[med_type])
        
        # Extract diagnoses
        diagnoses = medical_data.get('diagnoses_and_conditions', {})
        if diagnoses.get('primary_diagnosis'):
            entities["diagnoses"].append(diagnoses['primary_diagnosis'])
        if diagnoses.get('secondary_diagnoses'):
            entities["diagnoses"].extend(diagnoses['secondary_diagnoses'])
        if diagnoses.get('chronic_conditions'):
            entities["diagnoses"].extend(diagnoses['chronic_conditions'])
        
        # Extract procedures
        procedures = medical_data.get('procedures_and_interventions', {})
        if procedures.get('procedures_performed'):
            entities["procedures"].extend(procedures['procedures_performed'])
        if procedures.get('planned_procedures'):
            entities["procedures"].extend(procedures['planned_procedures'])
        
        # Extract allergies
        allergies = medical_data.get('allergies_and_adverse_reactions', {})
        for allergy_type in ['drug_allergies', 'food_allergies', 'environmental_allergies']:
            if allergies.get(allergy_type):
                entities["allergies"].extend(allergies[allergy_type])
        
        # Extract lab tests
        labs = medical_data.get('laboratory_and_diagnostics', {})
        for lab_type in ['blood_tests', 'chemistry_panel', 'hematology', 'imaging_studies']:
            if labs.get(lab_type):
                entities["laboratory_tests"].extend(labs[lab_type])
        
        # Extract vital signs
        vitals = medical_data.get('vital_signs_and_measurements', {})
        for vital, value in vitals.items():
            if value:
                entities["vital_signs"].append(f"{vital}: {value}")
        
        # Extract providers and facilities
        doc_details = medical_data.get('document_details', {})
        if doc_details.get('provider'):
            entities["providers"].append(doc_details['provider'])
        if doc_details.get('facility'):
            entities["facilities"].append(doc_details['facility'])
        
        # Remove duplicates and empty items
        for key in entities:
            entities[key] = list(set([item for item in entities[key] if item and str(item).strip()]))
    
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
    
    return entities


def create_thumbnail(image_bytes: bytes, max_size: Tuple[int, int] = (200, 200)) -> bytes:
    """
    Create a thumbnail of the scanned image for preview purposes.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Failed to create thumbnail: {e}")
        return b''


def fast_ocr_and_parse_gemini(image_bytes: bytes) -> Dict[str, Any]:
    """
    Fast pipeline using Gemini: OCR, AI parsing, summary generation, and entity extraction.
    """
    try:
        logger.info(f"Starting Gemini OCR processing for {len(image_bytes)} bytes")
        
        # Step 1: Fast OCR
        ocr_text, ocr_meta = run_fast_ocr(image_bytes)
        logger.info(f"OCR completed: {len(ocr_text)} characters extracted")
        
        # Step 2: Gemini-powered parsing
        parsed_data = gemini_parse_medical_text(ocr_text)
        logger.info("Gemini parsing completed")
        
        # Step 3: Generate comprehensive summary
        ai_summary = generate_gemini_summary(parsed_data, ocr_text)
        logger.info("Gemini summary generated")
        
        # Step 4: Extract comprehensive medical entities
        entities = extract_comprehensive_medical_entities(parsed_data)
        logger.info("Comprehensive medical entities extracted")
        
        # Step 5: Create thumbnail
        thumbnail_data = create_thumbnail(image_bytes)
        logger.info("Thumbnail created")
        
        return {
            'success': True,
            'ocr_text': ocr_text,
            'ocr_meta': ocr_meta,
            'parsed_data': parsed_data,
            'ai_summary': ai_summary,
            'extracted_entities': entities,
            'thumbnail_data': thumbnail_data,
            'processing_timestamp': datetime.now().isoformat(),
            'ai_processor': 'gemini-1.5-flash'
        }
        
    except Exception as e:
        logger.error(f"Gemini OCR and parsing failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a basic success response even if processing fails
        return {
            'success': True,
            'error': str(e),
            'ocr_text': 'Image processing failed - using fallback text extraction',
            'ocr_meta': {'engine': 'fallback', 'avg_word_confidence': 0.0},
            'parsed_data': {
                'comprehensive_medical_data': {
                    'patient_information': {'name': 'Processing Failed'},
                    'metadata': {'confidence_score': 0.0, 'ai_model': 'fallback'}
                }
            },
            'ai_summary': f'Gemini processing encountered an issue: {str(e)}. Please try with a different image format.',
            'extracted_entities': {},
            'thumbnail_data': b'',
            'processing_timestamp': datetime.now().isoformat(),
            'ai_processor': 'gemini-fallback'
        }
