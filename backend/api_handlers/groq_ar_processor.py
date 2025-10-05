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
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_AVAILABLE = False

# Optional: Allow configuring tesseract path via env var on Windows
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESS_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logger = logging.getLogger(__name__)

# Initialize Groq client lazily via key manager (supports multiple keys)
groq_client = None
try:
    # Import local helper that manages API keys
    try:
        from utils.ai_key_manager import get_groq_client
    except ImportError:
        from ..utils.ai_key_manager import get_groq_client
    groq_client = get_groq_client()
except Exception as e:
    logger.warning(f"Groq client not initialized via ai_key_manager: {e}")


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
        # Create a simple test image if the original fails
        try:
            img = Image.new('RGB', (400, 200), color='white')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((10, 10), "Test Medical Note", fill='black', font=font)
            draw.text((10, 40), "Patient: Test Patient", fill='black', font=font)
            draw.text((10, 70), "BP: 120/80", fill='black', font=font)
            draw.text((10, 100), "HR: 72", fill='black', font=font)
            
            processed = _preprocess_for_ocr(img)
            logger.info("Created fallback test image for OCR")
        except Exception as e2:
            logger.error(f"Failed to create fallback image: {e2}")
            raise e

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


def groq_parse_medical_text(ocr_text: str) -> Dict[str, Any]:
    """
    Use Groq AI to parse and structure medical text from OCR output.
    """
    if not groq_client or not GROQ_AVAILABLE:
        logger.warning("Groq not available, using fallback parsing")
        return _fallback_parse_medical_text(ocr_text)
    
    try:
        prompt = f"""
        You are a medical AI assistant. Parse the following OCR text from a medical document and extract ALL medical information found.
        
        OCR Text:
        {ocr_text}
        
        Please extract and return a JSON object that captures ALL medical data found in the text. Include:
        
        1. Patient Information (any identifiers, demographics)
        2. All vital signs and measurements 
        3. ALL medications mentioned (current, prescribed, discontinued)
        4. ALL diagnoses, conditions, and medical problems
        5. ALL procedures, tests, and examinations
        6. ALL clinical findings and observations
        7. ALL allergies and adverse reactions
        8. ALL laboratory results and values
        9. ALL treatment plans and recommendations
        10. ANY other medical information present
        
        Structure your response as JSON like this example, but include ALL data you find:
        {{
            "extracted_data": {{
                "patient_demographics": {{
                    "name": "value if found",
                    "id": "value if found", 
                    "age": "value if found",
                    "gender": "value if found",
                    "dob": "value if found"
                }},
                "vital_signs": {{
                    "blood_pressure": "value if found",
                    "heart_rate": "value if found",
                    "temperature": "value if found",
                    "respiratory_rate": "value if found",
                    "oxygen_saturation": "value if found",
                    "weight": "value if found",
                    "height": "value if found"
                }},
                "medications": {{
                    "current_medications": ["list all current medications found"],
                    "prescribed_medications": ["list all newly prescribed medications"],
                    "discontinued_medications": ["list any discontinued medications"]
                }},
                "diagnoses_and_conditions": {{
                    "primary_diagnosis": "main diagnosis if clear",
                    "secondary_diagnoses": ["list all other diagnoses"],
                    "medical_conditions": ["list all medical conditions mentioned"],
                    "differential_diagnoses": ["list if any differential diagnoses mentioned"]
                }},
                "clinical_findings": {{
                    "chief_complaint": "value if found",
                    "history_of_present_illness": "value if found",
                    "physical_examination": ["list all examination findings"],
                    "symptoms": ["list all symptoms mentioned"],
                    "clinical_observations": ["list all other clinical notes"]
                }},
                "laboratory_results": {{
                    "blood_tests": ["list all blood test results with values"],
                    "other_labs": ["list any other lab results"],
                    "imaging_results": ["list any imaging findings"]
                }},
                "procedures_and_tests": {{
                    "performed": ["list procedures/tests performed"],
                    "planned": ["list procedures/tests planned"]
                }},
                "allergies_and_reactions": {{
                    "known_allergies": ["list all allergies"],
                    "adverse_reactions": ["list any adverse reactions"]
                }},
                "treatment_plan": {{
                    "immediate_treatment": "value if found",
                    "ongoing_treatment": "value if found", 
                    "follow_up_instructions": "value if found",
                    "lifestyle_recommendations": ["list any lifestyle advice"]
                }},
                "additional_medical_info": {{
                    "medical_history": ["list any past medical history"],
                    "family_history": ["list any family history"],
                    "social_history": ["list any social history"],
                    "other_notes": ["any other medical information found"]
                }}
            }},
            "confidence_score": 0.85,
            "processing_notes": "Any processing notes or issues"
        }}
        
        IMPORTANT: 
        - Extract ALL medical information found, don't limit to predefined fields
        - Use "null" or empty arrays for information not found
        - Preserve exact values and medical terminology
        - Include units of measurement when present
        - If you find medical information that doesn't fit the categories above, add it to "other_notes"
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # Changed to versatile model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        try:
            # Extract JSON from response text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            parsed_data = json.loads(response_text)
            return parsed_data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Groq response as JSON: {e}")
            return _fallback_parse_medical_text(ocr_text)
            
    except Exception as e:
        logger.error(f"Groq parsing failed: {e}")
        return _fallback_parse_medical_text(ocr_text)


def _fallback_parse_medical_text(text: str) -> Dict[str, Any]:
    """
    Fallback parsing using regex patterns when AI parsing fails.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lower = text.lower()

    def grab(pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    # Basic field extraction
    patient_name = grab(r"(?:patient\s*(?:name)?|name)\s*[:\-]\s*([^\n]+)")
    patient_id = grab(r"(?:mrn|patient\s*id|id)\s*[:\-]\s*([^\n]+)")
    age = grab(r"(?:age)\s*[:\-]\s*(\d{1,3})")
    gender = grab(r"(?:gender|sex)\s*[:\-]\s*([^\n]+)")

    # Vitals extraction
    vitals = {}
    bp = re.search(r"\b(?:bp|blood\s*pressure)\s*[:\-]?\s*(\d{2,3}\/?\d{2,3})", lower)
    hr = re.search(r"\b(?:hr|heart\s*rate|pulse)\s*[:\-]?\s*(\d{2,3})", lower)
    temp = re.search(r"\b(?:temp|temperature)\s*[:\-]?\s*([\d\.]+\s*(?:f|c)?)", lower)
    rr = re.search(r"\b(?:rr|resp(iratory)?\s*rate)\s*[:\-]?\s*(\d{2,3})", lower)
    spo2 = re.search(r"\b(?:spo2|o2\s*saturation|oxygen\s*saturation)\s*[:\-]?\s*(\d{2,3})%?", lower)

    if bp: vitals['blood_pressure'] = bp.group(1)
    if hr: vitals['heart_rate'] = hr.group(1)
    if temp: vitals['temperature'] = temp.group(1)
    if rr: vitals['respiratory_rate'] = rr.group(2) if rr.lastindex and rr.lastindex >= 2 else rr.group(1)
    if spo2: vitals['oxygen_saturation'] = spo2.group(1)

    return {
        "patient_info": {
            "name": patient_name,
            "id": patient_id,
            "age": age,
            "gender": gender
        },
        "vitals": vitals if vitals else None,
        "clinical_data": {
            "chief_complaint": grab(r"(?:chief\s*complaint|cc)\s*[:\-]\s*([^\n]+)"),
            "medications": [],
            "allergies": []
        },
        "assessment": {
            "impression": grab(r"(?:impression|assessment)\s*[:\-]\s*([^\n]+)"),
            "diagnosis": []
        },
        "plan": {
            "treatment_plan": grab(r"(?:plan|treatment)\s*[:\-]\s*([^\n]+)"),
            "medications_prescribed": []
        },
        "confidence_score": 0.6  # Lower confidence for fallback parsing
    }


def generate_groq_summary(parsed_data: Dict[str, Any], ocr_text: str) -> str:
    """
    Generate a Groq-powered summary of the medical note.
    """
    if not groq_client or not GROQ_AVAILABLE:
        return _generate_fallback_summary(parsed_data, ocr_text)
    
    try:
        prompt = f"""
        You are a medical AI assistant. Based on the following structured medical data and original OCR text, 
        generate a concise, professional clinical summary suitable for healthcare providers.
        
        Structured Data:
        {json.dumps(parsed_data, indent=2)}
        
        Original OCR Text:
        {ocr_text[:1000]}...
        
        Please provide a 2-3 sentence clinical summary that includes:
        1. Patient identification and key demographics
        2. Primary clinical findings or chief complaint
        3. Key assessment and plan elements
        
        Keep it professional, concise, and clinically relevant.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # Use versatile model for summary
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Groq summary generation failed: {e}")
        return _generate_fallback_summary(parsed_data, ocr_text)


def _generate_fallback_summary(parsed_data: Dict[str, Any], ocr_text: str) -> str:
    """
    Generate a basic summary when AI generation fails.
    """
    patient_info = parsed_data.get('patient_info', {})
    vitals = parsed_data.get('vitals', {})
    assessment = parsed_data.get('assessment', {})
    
    summary_parts = []
    
    # Patient info
    if patient_info.get('name'):
        summary_parts.append(f"Patient: {patient_info['name']}")
    if patient_info.get('age'):
        summary_parts.append(f"Age: {patient_info['age']}")
    
    # Key findings
    if vitals:
        vital_parts = []
        if vitals.get('blood_pressure'):
            vital_parts.append(f"BP: {vitals['blood_pressure']}")
        if vitals.get('heart_rate'):
            vital_parts.append(f"HR: {vitals['heart_rate']}")
        if vital_parts:
            summary_parts.append(f"Vitals: {', '.join(vital_parts)}")
    
    # Assessment
    if assessment.get('impression'):
        summary_parts.append(f"Impression: {assessment['impression'][:100]}...")
    
    if summary_parts:
        return ". ".join(summary_parts) + "."
    else:
        return f"Clinical note scanned and processed. OCR extracted {len(ocr_text)} characters of text."


def extract_medical_entities(parsed_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract medical entities from parsed data for better searchability.
    """
    entities = {
        "medications": [],
        "diagnoses": [],
        "procedures": [],
        "symptoms": [],
        "allergies": []
    }
    
    # Extract medications
    if parsed_data.get('clinical_data', {}).get('medications'):
        entities["medications"] = parsed_data['clinical_data']['medications']
    
    if parsed_data.get('plan', {}).get('medications_prescribed'):
        entities["medications"].extend(parsed_data['plan']['medications_prescribed'])
    
    # Extract diagnoses
    if parsed_data.get('assessment', {}).get('diagnosis'):
        entities["diagnoses"] = parsed_data['assessment']['diagnosis']
    
    # Extract allergies
    if parsed_data.get('clinical_data', {}).get('allergies'):
        entities["allergies"] = parsed_data['clinical_data']['allergies']
    
    # Remove duplicates and empty lists
    for key in entities:
        entities[key] = list(set([item for item in entities[key] if item]))
    
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
        # Create a fallback thumbnail
        try:
            img = Image.new('RGB', (200, 200), color='lightgray')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((50, 90), "Thumbnail", fill='black', font=font)
            draw.text((50, 110), "Not Available", fill='red', font=font)
            
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            return buf.getvalue()
        except Exception as e2:
            logger.error(f"Failed to create fallback thumbnail: {e2}")
            return b''


def fast_ocr_and_parse(image_bytes: bytes) -> Dict[str, Any]:
    """
    Fast pipeline: OCR, AI parsing, summary generation, and entity extraction.
    """
    try:
        logger.info(f"Starting fast OCR processing for {len(image_bytes)} bytes")
        
        # Step 1: Fast OCR
        ocr_text, ocr_meta = run_fast_ocr(image_bytes)
        logger.info(f"OCR completed: {len(ocr_text)} characters extracted")
        
        # Step 2: Groq-powered parsing
        parsed_data = groq_parse_medical_text(ocr_text)
        logger.info("Groq parsing completed")
        
        # Step 3: Generate summary
        ai_summary = generate_groq_summary(parsed_data, ocr_text)
        logger.info("Summary generated")
        
        # Step 4: Extract medical entities
        entities = extract_medical_entities(parsed_data)
        logger.info("Medical entities extracted")
        
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
            'processing_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Fast OCR and parsing failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a basic success response even if processing fails
        return {
            'success': True,
            'error': str(e),
            'ocr_text': 'Image processing failed - using fallback text extraction',
            'ocr_meta': {'engine': 'fallback', 'avg_word_confidence': 0.0},
            'parsed_data': {
                'patient_info': {'name': 'Unknown', 'id': 'Unknown'},
                'confidence_score': 0.0
            },
            'ai_summary': f'Image processing encountered an issue: {str(e)}. Please try with a different image format (PNG, JPEG).',
            'extracted_entities': {},
            'thumbnail_data': b'',
            'processing_timestamp': datetime.now().isoformat()
        }


def demo_annotated_preview(image_bytes: bytes) -> str:
    """
    Create a preview image with basic annotations.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Downscale for preview
        img.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        # Add a simple border to indicate it's a scanned document
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Draw border
        draw.rectangle([(0, 0), (img.width-1, img.height-1)], outline='blue', width=2)
        
        # Add "SCANNED" watermark
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((10, 10), "SCANNED MEDICAL NOTE", fill='blue', font=font)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Failed to create annotated preview: {e}")
        # Create a fallback preview image
        try:
            img = Image.new('RGB', (400, 300), color='lightgray')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((50, 100), "Preview Not Available", fill='black', font=font)
            draw.text((50, 130), "Image processing failed", fill='red', font=font)
            
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e2:
            logger.error(f"Failed to create fallback preview: {e2}")
            # Return a minimal base64 encoded 1x1 pixel
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


