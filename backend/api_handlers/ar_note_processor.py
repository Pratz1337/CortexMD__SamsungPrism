import io
import os
import re
import json
import base64
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageOps, ImageFilter

try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    pytesseract = None  # type: ignore
    TESS_AVAILABLE = False

# Optional: Allow configuring tesseract path via env var on Windows
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESS_AVAILABLE and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD  # type: ignore


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Basic preprocessing to improve OCR quality."""
    # Convert to grayscale, increase contrast, slight sharpen, adaptive thresholding-like effect
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.SHARPEN)
    # Simple binarization
    g = g.point(lambda x: 255 if x > 160 else 0, mode='1')
    return g


def run_ocr(image_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Run OCR on the provided image bytes.

    Returns: (full_text, meta)
    meta may include engine info and confidence hints when available.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    processed = _preprocess_for_ocr(img)

    if not TESS_AVAILABLE:
        raise RuntimeError(
            "Tesseract OCR not available. Please install Tesseract and set TESSERACT_CMD if needed."
        )

    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)  # type: ignore
    text = pytesseract.image_to_string(processed)  # type: ignore

    # Collect simple confidence statistics
    confs = [int(c) for c in data.get('conf', []) if str(c).isdigit()]
    avg_conf = sum(confs)/len(confs) if confs else None

    meta = {
        'engine': 'tesseract',
        'lang': 'eng',
        'avg_word_confidence': avg_conf
    }
    return text, meta


def parse_structured_fields(text: str) -> Dict[str, Any]:
    """
    Best-effort parsing of common fields from a clinical note.
    This is a simple heuristic extractor; can be replaced by an LLM downstream if needed.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lower = text.lower()

    def grab(pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    # Common fields
    patient = grab(r"(?:patient\s*(?:name)?|name)\s*[:\-]\s*([^\n]+)")
    mrn = grab(r"(?:mrn|patient\s*id|id)\s*[:\-]\s*([^\n]+)")
    age = grab(r"(?:age)\s*[:\-]\s*(\d{1,3})")
    gender = grab(r"(?:gender|sex)\s*[:\-]\s*([^\n]+)")
    date = grab(r"(?:date|dos|exam\s*date)\s*[:\-]\s*([^\n]+)")

    # Vitals (simple patterns)
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

    # Sections: Medications, Allergies, Impression/Assessment, Plan
    def extract_section(section_name: str) -> Optional[str]:
        pattern = rf"{section_name}[:\-]\s*(.*?)(?:\n\s*\n|\n\s*[A-Z][A-Za-z ]+[:\-]|$)"
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    meds = extract_section("medications|current\s*meds|rx")
    allergies = extract_section("allergies")
    impression = extract_section("impression|assessment")
    plan = extract_section("plan|recommendations")
    chief_complaint = extract_section("chief\s*complaint|cc")

    # Fallback: short summary from first lines
    summary = None
    if not impression and lines:
        summary = lines[0][:200]

    parsed = {
        'patient_name': patient,
        'patient_id_text': mrn,
        'age': age,
        'gender': gender,
        'date': date,
        'vitals': vitals or None,
        'chief_complaint': chief_complaint,
        'medications': meds,
        'allergies': allergies,
        'impression': impression,
        'plan': plan,
        'summary': impression or summary,
    }
    return {k: v for k, v in parsed.items() if v}


def ocr_and_parse(image_bytes: bytes) -> Dict[str, Any]:
    """Full pipeline: OCR then structured parsing."""
    full_text, meta = run_ocr(image_bytes)
    parsed = parse_structured_fields(full_text)
    return {
        'success': True,
        'text': full_text,
        'parsed': parsed,
        'meta': meta,
    }


def demo_annotated_preview(image_bytes: bytes) -> str:
    """
    Create a small preview image (base64 PNG) to show in UI. No real AR boxes yet.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Downscale for preview
    img.thumbnail((800, 800))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

