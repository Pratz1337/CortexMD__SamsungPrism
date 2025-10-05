"""
Comprehensive Diagnosis Report Generator
Generates detailed PDF reports from CortexMD diagnosis results
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from reportlab.platypus import Paragraph
logger = logging.getLogger(__name__)

class DiagnosisReportGenerator:
    """Generates comprehensive PDF reports from diagnosis results"""

    def __init__(self):
        self.template_dir = Path(__file__).parent.parent / "templates"
        self.template_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(self, session_id: str, diagnosis_data: Dict[str, Any]) -> bytes:
        """
        Generate a comprehensive PDF report from diagnosis data

        Args:
            session_id: The diagnosis session ID
            diagnosis_data: Complete diagnosis results dictionary

        Returns:
            PDF report as bytes
        """
        try:
            # Extract all relevant data sections
            report_data = self._collect_all_diagnosis_data(diagnosis_data, session_id)

            # Generate PDF report
            pdf_content = self._generate_pdf_report(report_data)

            return pdf_content

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            # Return a fallback JSON report
            return self._generate_fallback_json_report(session_id, diagnosis_data)

    def _collect_all_diagnosis_data(self, diagnosis_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Collect and organize all diagnosis data into report sections"""

        # Basic session information
        report_data = {
            'session_id': session_id,
            'generated_at': datetime.now().isoformat(),
            'report_version': '2.1',
            'sections': [],
            'original_data': diagnosis_data  # Store original data for enhanced PDF generator
        }

        # 1. Patient Information Section
        patient_info = self._extract_patient_information(diagnosis_data)
        if patient_info:
            report_data['sections'].append({
                'title': 'Patient Information',
                'type': 'patient_info',
                'data': patient_info,
                'priority': 1
            })

        # 2. Primary Diagnosis Section
        primary_diagnosis = self._extract_primary_diagnosis(diagnosis_data)
        if primary_diagnosis:
            report_data['sections'].append({
                'title': 'Primary Diagnosis',
                'type': 'primary_diagnosis',
                'data': primary_diagnosis,
                'priority': 2
            })

        # 3. Clinical Assessment Section
        clinical_assessment = self._extract_clinical_assessment(diagnosis_data)
        if clinical_assessment:
            report_data['sections'].append({
                'title': 'Clinical Assessment',
                'type': 'clinical_assessment',
                'data': clinical_assessment,
                'priority': 3
            })

        # 4. AI Analysis Results Section
        ai_analysis = self._extract_ai_analysis(diagnosis_data)
        if ai_analysis:
            report_data['sections'].append({
                'title': 'AI Analysis Results',
                'type': 'ai_analysis',
                'data': ai_analysis,
                'priority': 4
            })

        # 5. FOL Logic Verification Section
        fol_verification = self._extract_fol_verification(diagnosis_data)
        if fol_verification:
            report_data['sections'].append({
                'title': 'First-Order Logic Verification',
                'type': 'fol_verification',
                'data': fol_verification,
                'priority': 5
            })

        # 6. Medical Imaging Analysis Section
        imaging_analysis = self._extract_imaging_analysis(diagnosis_data)
        if imaging_analysis:
            report_data['sections'].append({
                'title': 'Medical Imaging Analysis',
                'type': 'imaging_analysis',
                'data': imaging_analysis,
                'priority': 6
            })

        # 7. Differential Diagnoses Section
        differential_diagnoses = self._extract_differential_diagnoses(diagnosis_data)
        if differential_diagnoses:
            report_data['sections'].append({
                'title': 'Differential Diagnoses',
                'type': 'differential_diagnoses',
                'data': differential_diagnoses,
                'priority': 7
            })

        # 8. Treatment Recommendations Section
        treatment_recommendations = self._extract_treatment_recommendations(diagnosis_data)
        if treatment_recommendations:
            report_data['sections'].append({
                'title': 'Treatment Recommendations',
                'type': 'treatment_recommendations',
                'data': treatment_recommendations,
                'priority': 8
            })

        # 9. Advanced Verification Section
        advanced_verification = self._extract_advanced_verification(diagnosis_data)
        if advanced_verification:
            report_data['sections'].append({
                'title': 'Advanced Medical Verification',
                'type': 'advanced_verification',
                'data': advanced_verification,
                'priority': 9
            })

        # 10. Ontology Analysis Section
        ontology_analysis = self._extract_ontology_analysis(diagnosis_data)
        if ontology_analysis:
            report_data['sections'].append({
                'title': 'Medical Ontology Analysis',
                'type': 'ontology_analysis',
                'data': ontology_analysis,
                'priority': 10
            })

        # 11. Performance Metrics Section
        performance_metrics = self._extract_performance_metrics(diagnosis_data)
        if performance_metrics:
            report_data['sections'].append({
                'title': 'AI Performance Metrics',
                'type': 'performance_metrics',
                'data': performance_metrics,
                'priority': 11
            })

        # 12. Sources and References Section
        sources_references = self._extract_sources_references(diagnosis_data)
        if sources_references:
            report_data['sections'].append({
                'title': 'Sources and References',
                'type': 'sources_references',
                'data': sources_references,
                'priority': 12
            })

        # Sort sections by priority
        report_data['sections'].sort(key=lambda x: x['priority'])

        return report_data

    def _extract_patient_information(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract patient information from diagnosis data"""
        patient_info = {}

        # Try to get patient ID from various sources
        patient_id = (
            data.get('patient_id') or
            getattr(data.get('patient_input'), 'patient_id', None) or
            data.get('session_id', 'Unknown')
        )
        patient_info['patient_id'] = patient_id

        # Extract patient input data
        patient_input = data.get('patient_input')
        if patient_input:
            if isinstance(patient_input, dict):
                patient_info['clinical_text'] = patient_input.get('text_data', '')
                patient_info['image_count'] = len(patient_input.get('image_paths', []))
                patient_info['has_fhir_data'] = bool(patient_input.get('fhir_data'))
            else:
                # Handle object format
                patient_info['clinical_text'] = getattr(patient_input, 'text_data', '')
                patient_info['image_count'] = len(getattr(patient_input, 'image_paths', []))
                patient_info['has_fhir_data'] = bool(getattr(patient_input, 'fhir_data', None))

        # Session metadata
        patient_info['session_timestamp'] = data.get('timestamp', datetime.now().isoformat())
        patient_info['processing_time'] = data.get('processing_time', 0)

        return patient_info if patient_info else None

    def _extract_primary_diagnosis(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract primary diagnosis information"""
        primary_diagnosis = data.get('primary_diagnosis', {})

        if not primary_diagnosis:
            # Try to extract from diagnosis_result
            diagnosis_result = data.get('diagnosis_result', {})
            if diagnosis_result and isinstance(diagnosis_result, dict):
                primary_diagnosis = {
                    'condition': diagnosis_result.get('primary_diagnosis', 'Unknown'),
                    'confidence': diagnosis_result.get('confidence_score', 0),
                    'description': diagnosis_result.get('clinical_impression', ''),
                    'icd_code': diagnosis_result.get('icd_code', '')
                }

        if primary_diagnosis:
            return {
                'condition': primary_diagnosis.get('condition', 'Unknown'),
                'confidence_score': primary_diagnosis.get('confidence', 0),
                'confidence_percentage': round(primary_diagnosis.get('confidence', 0) * 100, 1),
                'description': primary_diagnosis.get('description', ''),
                'icd_code': primary_diagnosis.get('icd_code', ''),
                'clinical_impression': primary_diagnosis.get('clinical_impression', ''),
                'reasoning_paths': primary_diagnosis.get('reasoning_paths', []),
                'recommendations': primary_diagnosis.get('clinical_recommendations', [])
            }

        return None

    def _extract_clinical_assessment(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract clinical assessment information"""
        assessment = {
            'urgency_level': data.get('urgency_level', 'medium'),
            'confidence_metrics': data.get('confidence_metrics', {}),
            'explanations': data.get('ui_data', {}).get('explanations', []),
            'clinical_recommendations': data.get('clinical_recommendations', [])
        }

        return assessment if any(assessment.values()) else None

    def _extract_ai_analysis(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract AI analysis results"""
        ai_analysis = {}

        # Extract from various AI result sources
        if data.get('diagnosis_result'):
            diag_result = data['diagnosis_result']
            ai_analysis['ai_model_used'] = diag_result.get('ai_model_used', 'CortexMD AI')
            ai_analysis['processing_method'] = diag_result.get('processing_method', 'Multi-modal AI analysis')
            ai_analysis['data_quality_score'] = diag_result.get('data_quality_assessment', {}).get('score', 0.8)

        # Enhanced explanations
        if data.get('enhanced_results'):
            ai_analysis['enhanced_explanations'] = data['enhanced_results'].get('explanations', [])

        return ai_analysis if ai_analysis else None

    def _extract_fol_verification(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract FOL verification results"""
        fol_data = data.get('fol_verification', {})

        if fol_data:
            return {
                'status': fol_data.get('status', 'Unknown'),
                'overall_confidence': fol_data.get('overall_confidence', 0),
                'verified_predicates': fol_data.get('verified_predicates', 0),
                'total_predicates': fol_data.get('total_predicates', 0),
                'success_rate': fol_data.get('success_rate', 0),
                'verification_time': fol_data.get('verification_time', 0),
                'medical_reasoning_summary': fol_data.get('medical_reasoning_summary', ''),
                'verification_summary': fol_data.get('verification_summary', ''),
                'detailed_verification': fol_data.get('detailed_verification', [])
            }

        return None

    def _extract_imaging_analysis(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract medical imaging analysis results"""
        imaging_data = {}

        # GradCAM visualization data
        heatmap_data = data.get('heatmap_data', [])
        heatmap_visualization = data.get('heatmap_visualization', {})

        if heatmap_data:
            imaging_data['heatmap_analysis'] = {
                'total_images': len(heatmap_data),
                'successful_heatmaps': sum(1 for h in heatmap_data if h.get('success')),
                'model_type': heatmap_visualization.get('model_type', 'AI GradCAM'),
                'heatmap_data': heatmap_data[:5]  # Limit to first 5 for report
            }

        # Image paths
        image_paths = data.get('image_paths', [])
        if image_paths:
            imaging_data['image_paths'] = image_paths
            imaging_data['total_images'] = len(image_paths)

        return imaging_data if imaging_data else None

    def _extract_differential_diagnoses(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract differential diagnoses"""
        differential = data.get('differential_diagnoses', [])

        if differential:
            return {
                'diagnoses': differential,
                'total_count': len(differential),
                'top_diagnoses': differential[:3]  # Focus on top 3
            }

        return None

    def _extract_treatment_recommendations(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract treatment recommendations"""
        treatment_data = {}

        # Recommended tests
        recommended_tests = data.get('recommended_tests', [])
        if recommended_tests:
            treatment_data['recommended_tests'] = recommended_tests

        # Treatment options
        treatment_options = data.get('treatment_options', [])
        if treatment_options:
            treatment_data['treatment_options'] = treatment_options

        # Clinical recommendations
        clinical_recommendations = data.get('clinical_recommendations', [])
        if clinical_recommendations:
            treatment_data['clinical_recommendations'] = clinical_recommendations

        return treatment_data if treatment_data else None

    def _extract_advanced_verification(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract advanced verification results"""
        advanced_data = {}

        # Enhanced verification
        enhanced_verification = data.get('enhanced_verification', {})
        if enhanced_verification:
            advanced_data['enhanced_verification'] = {
                'overall_status': enhanced_verification.get('overall_status'),
                'overall_confidence': enhanced_verification.get('overall_confidence'),
                'evidence_strength': enhanced_verification.get('evidence_strength'),
                'sources_count': enhanced_verification.get('sources_count'),
                'textbook_references': enhanced_verification.get('textbook_references', [])[:3]
            }

        # Online verification
        online_verification = data.get('online_verification', {})
        if online_verification:
            advanced_data['online_verification'] = {
                'verification_status': online_verification.get('verification_status'),
                'confidence_score': online_verification.get('confidence_score'),
                'sources_count': len(online_verification.get('sources', [])),
                'clinical_notes': online_verification.get('clinical_notes')
            }

        return advanced_data if advanced_data else None

    def _extract_ontology_analysis(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract ontology analysis results"""
        ontology_data = data.get('ontology_analysis', {})

        if ontology_data:
            return {
                'diagnosis_term': ontology_data.get('diagnosis_term'),
                'normalized_diagnosis': ontology_data.get('normalized_diagnosis'),
                'term_count': ontology_data.get('term_count', 0),
                'synonym_count': ontology_data.get('synonym_count', 0),
                'confidence': ontology_data.get('confidence', 0),
                'ontology_source': ontology_data.get('ontology_source')
            }

        return None

    def _extract_performance_metrics(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract AI performance metrics"""
        metrics = {}

        # Processing time
        processing_time = data.get('processing_time')
        if processing_time:
            metrics['processing_time_seconds'] = processing_time

        # Confidence metrics
        confidence_metrics = data.get('confidence_metrics', {})
        if confidence_metrics:
            metrics['confidence_metrics'] = confidence_metrics

        # Explainability score
        explainability_score = data.get('explainability_score')
        if explainability_score is not None:
            metrics['explainability_score'] = explainability_score

        return metrics if metrics else None

    def _extract_sources_references(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract sources and references"""
        sources_data = {}

        # Sources from various sections
        sources = data.get('sources', [])
        if sources:
            sources_data['medical_sources'] = sources

        # Online verification sources
        online_sources = data.get('online_verification', {}).get('sources', [])
        if online_sources:
            sources_data['online_sources'] = online_sources[:5]  # Limit for report

        return sources_data if sources_data else None

    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> bytes:
        """Generate PDF report from collected data using enhanced PDF generator"""
        try:
            # Use the enhanced PDF generator for better reliability
            from .enhanced_pdf_generator import EnhancedPDFGenerator
            
            pdf_generator = EnhancedPDFGenerator()
            
            # Extract the original diagnosis data from report_data
            diagnosis_data = report_data.get('original_data', {})
            session_id = report_data.get('session_id', 'unknown')
            
            # Generate PDF using the enhanced generator
            pdf_bytes = pdf_generator.generate_diagnosis_report(session_id, diagnosis_data)
            
            logger.info(f"Successfully generated PDF report using enhanced generator: {len(pdf_bytes)} bytes")
            return pdf_bytes

        except Exception as e:
            logger.error(f"Enhanced PDF generation failed: {e}")
            
            # Fallback to original method if enhanced generator fails
            try:
                return self._generate_legacy_pdf_report(report_data)
            except Exception as fallback_error:
                logger.error(f"Legacy PDF generation also failed: {fallback_error}")
                # Return JSON fallback
                return self._generate_fallback_json_report(report_data['session_id'], report_data)

    def _generate_legacy_pdf_report(self, report_data: Dict[str, Any]) -> bytes:
        """Legacy PDF generation method as fallback"""
        try:
            # Try to use reportlab for PDF generation
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            import io

            # Create PDF buffer
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )

            section_title_style = ParagraphStyle(
                'SectionTitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=15,
                textColor=colors.blue
            )

            content_style = styles['Normal']
            content_style.fontSize = 10
            content_style.leading = 14

            # Build PDF content
            story = []

            # Title page
            story.append(Paragraph("CortexMD Diagnosis Report", title_style))
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph(f"Session ID: {report_data['session_id']}", styles['Heading3']))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", content_style))
            story.append(Paragraph("Report Version: 2.0", content_style))
            story.append(Spacer(1, 1*inch))

            # Generate content for each section
            for section in report_data.get('sections', []):
                story.append(Paragraph(section['title'], section_title_style))
                story.append(Spacer(1, 0.2*inch))

                # Generate section content based on type
                section_content = self._generate_section_content(section, styles)
                story.extend(section_content)
                story.append(Spacer(1, 0.5*inch))

            # Footer
            story.append(Spacer(1, 1*inch))
            story.append(Paragraph("This report was generated by CortexMD AI Diagnostic System", content_style))
            story.append(Paragraph("For medical professional use only", content_style))

            # Build PDF
            doc.build(story)
            buffer.seek(0)
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            # Validate PDF
            if len(pdf_bytes) < 1000 or not pdf_bytes.startswith(b'%PDF'):
                raise Exception("Generated PDF appears to be corrupted")
                
            return pdf_bytes

        except ImportError:
            logger.warning("ReportLab not available, falling back to JSON report")
            return self._generate_fallback_json_report(report_data['session_id'], report_data)

    def _generate_section_content(self, section: Dict[str, Any], styles) -> List:
        """Generate PDF content for a specific section"""
        content = []
        section_type = section['type']
        data = section['data']

        try:
            if section_type == 'patient_info':
                content.extend(self._generate_patient_info_content(data, styles))
            elif section_type == 'primary_diagnosis':
                content.extend(self._generate_primary_diagnosis_content(data, styles))
            elif section_type == 'clinical_assessment':
                content.extend(self._generate_clinical_assessment_content(data, styles))
            elif section_type == 'ai_analysis':
                content.extend(self._generate_ai_analysis_content(data, styles))
            elif section_type == 'fol_verification':
                content.extend(self._generate_fol_verification_content(data, styles))
            elif section_type == 'imaging_analysis':
                content.extend(self._generate_imaging_analysis_content(data, styles))
            elif section_type == 'differential_diagnoses':
                content.extend(self._generate_differential_diagnoses_content(data, styles))
            elif section_type == 'treatment_recommendations':
                content.extend(self._generate_treatment_recommendations_content(data, styles))
            elif section_type == 'advanced_verification':
                content.extend(self._generate_advanced_verification_content(data, styles))
            elif section_type == 'performance_metrics':
                content.extend(self._generate_performance_metrics_content(data, styles))
            else:
                # Generic content generator
                content.append(Paragraph(f"Data: {json.dumps(data, indent=2, default=str)}", styles['Normal']))

        except Exception as e:
            logger.error(f"Error generating content for section {section_type}: {e}")
            content.append(Paragraph(f"Error generating section content: {str(e)}", styles['Normal']))

        return content

    def _generate_patient_info_content(self, data: Dict[str, Any], styles) -> List:
        """Generate patient information section content"""
        content = []
        content.append(Paragraph(f"<b>Patient ID:</b> {data.get('patient_id', 'Unknown')}", styles['Normal']))
        content.append(Paragraph(f"<b>Session Timestamp:</b> {data.get('session_timestamp', 'Unknown')}", styles['Normal']))
        content.append(Paragraph(f"<b>Processing Time:</b> {data.get('processing_time', 0):.2f} seconds", styles['Normal']))

        clinical_text = data.get('clinical_text', '')
        if clinical_text:
            content.append(Paragraph("<b>Clinical Information:</b>", styles['Normal']))
            content.append(Paragraph(clinical_text[:500] + "..." if len(clinical_text) > 500 else clinical_text, styles['Normal']))

        if data.get('image_count', 0) > 0:
            content.append(Paragraph(f"<b>Medical Images:</b> {data['image_count']} image(s) analyzed", styles['Normal']))

        return content

    def _generate_primary_diagnosis_content(self, data: Dict[str, Any], styles) -> List:
        """Generate primary diagnosis section content"""
        content = []
        content.append(Paragraph(f"<b>Diagnosis:</b> {data.get('condition', 'Unknown')}", styles['Normal']))
        content.append(Paragraph(f"<b>Confidence:</b> {data.get('confidence_percentage', 0)}%", styles['Normal']))

        if data.get('icd_code'):
            content.append(Paragraph(f"<b>ICD Code:</b> {data['icd_code']}", styles['Normal']))

        description = data.get('description', '')
        if description:
            content.append(Paragraph("<b>Description:</b>", styles['Normal']))
            content.append(Paragraph(description, styles['Normal']))

        return content

    def _generate_clinical_assessment_content(self, data: Dict[str, Any], styles) -> List:
        """Generate clinical assessment section content"""
        content = []
        content.append(Paragraph(f"<b>Urgency Level:</b> {data.get('urgency_level', 'Medium').upper()}", styles['Normal']))

        confidence_metrics = data.get('confidence_metrics', {})
        if confidence_metrics:
            content.append(Paragraph("<b>Confidence Metrics:</b>", styles['Normal']))
            for key, value in confidence_metrics.items():
                content.append(Paragraph(f"  • {key.replace('_', ' ').title()}: {value}", styles['Normal']))

        explanations = data.get('explanations', [])
        if explanations:
            content.append(Paragraph("<b>AI Explanations:</b>", styles['Normal']))
            for i, exp in enumerate(explanations[:3], 1):  # Limit to 3 explanations
                content.append(Paragraph(f"{i}. {exp[:200]}{'...' if len(exp) > 200 else ''}", styles['Normal']))

        return content

    def _generate_ai_analysis_content(self, data: Dict[str, Any], styles) -> List:
        """Generate AI analysis section content"""
        content = []
        content.append(Paragraph(f"<b>AI Model:</b> {data.get('ai_model_used', 'CortexMD AI')}", styles['Normal']))
        content.append(Paragraph(f"<b>Processing Method:</b> {data.get('processing_method', 'Multi-modal analysis')}", styles['Normal']))

        if data.get('data_quality_score'):
            content.append(Paragraph(f"<b>Data Quality Score:</b> {data['data_quality_score']:.2f}", styles['Normal']))

        return content

    def _generate_fol_verification_content(self, data: Dict[str, Any], styles) -> List:
        """Generate FOL verification section content"""
        content = []
        content.append(Paragraph(f"<b>Verification Status:</b> {data.get('status', 'Unknown')}", styles['Normal']))
        content.append(Paragraph(f"<b>Overall Confidence:</b> {data.get('overall_confidence', 0):.2f}", styles['Normal']))
        content.append(Paragraph(f"<b>Verified Predicates:</b> {data.get('verified_predicates', 0)}/{data.get('total_predicates', 0)}", styles['Normal']))
        content.append(Paragraph(f"<b>Success Rate:</b> {data.get('success_rate', 0):.1%}", styles['Normal']))

        if data.get('verification_summary'):
            content.append(Paragraph("<b>Verification Summary:</b>", styles['Normal']))
            content.append(Paragraph(data['verification_summary'], styles['Normal']))

        return content

    def _generate_imaging_analysis_content(self, data: Dict[str, Any], styles) -> List:
        """Generate imaging analysis section content"""
        content = []

        heatmap_analysis = data.get('heatmap_analysis', {})
        if heatmap_analysis:
            content.append(Paragraph(f"<b>GradCAM Analysis:</b>", styles['Normal']))
            content.append(Paragraph(f"  • Total Images: {heatmap_analysis.get('total_images', 0)}", styles['Normal']))
            content.append(Paragraph(f"  • Successful Heatmaps: {heatmap_analysis.get('successful_heatmaps', 0)}", styles['Normal']))
            content.append(Paragraph(f"  • AI Model: {heatmap_analysis.get('model_type', 'Unknown')}", styles['Normal']))

        total_images = data.get('total_images', 0)
        if total_images > 0:
            content.append(Paragraph(f"<b>Medical Images Analyzed:</b> {total_images}", styles['Normal']))

        return content

    def _generate_differential_diagnoses_content(self, data: Dict[str, Any], styles) -> List:
        """Generate differential diagnoses section content"""
        content = []
        content.append(Paragraph(f"<b>Total Differential Diagnoses:</b> {data.get('total_count', 0)}", styles['Normal']))

        top_diagnoses = data.get('top_diagnoses', [])
        for i, diagnosis in enumerate(top_diagnoses, 1):
            content.append(Paragraph(f"{i}. {diagnosis.get('condition', 'Unknown')} (Confidence: {diagnosis.get('confidence', 0):.1%})", styles['Normal']))

        return content

    def _generate_treatment_recommendations_content(self, data: Dict[str, Any], styles) -> List:
        """Generate treatment recommendations section content"""
        content = []

        recommended_tests = data.get('recommended_tests', [])
        if recommended_tests:
            content.append(Paragraph("<b>Recommended Tests:</b>", styles['Normal']))
            for test in recommended_tests:
                content.append(Paragraph(f"  • {test}", styles['Normal']))

        treatment_options = data.get('treatment_options', [])
        if treatment_options:
            content.append(Paragraph("<b>Treatment Options:</b>", styles['Normal']))
            for treatment in treatment_options:
                content.append(Paragraph(f"  • {treatment}", styles['Normal']))

        return content

    def _generate_advanced_verification_content(self, data: Dict[str, Any], styles) -> List:
        """Generate advanced verification section content"""
        content = []

        enhanced_verification = data.get('enhanced_verification', {})
        if enhanced_verification:
            content.append(Paragraph("<b>Enhanced Medical Verification:</b>", styles['Normal']))
            content.append(Paragraph(f"  • Status: {enhanced_verification.get('overall_status', 'Unknown')}", styles['Normal']))
            content.append(Paragraph(f"  • Confidence: {enhanced_verification.get('overall_confidence', 0):.2f}", styles['Normal']))
            content.append(Paragraph(f"  • Sources: {enhanced_verification.get('sources_count', 0)}", styles['Normal']))

        online_verification = data.get('online_verification', {})
        if online_verification:
            content.append(Paragraph("<b>Online Medical Verification:</b>", styles['Normal']))
            content.append(Paragraph(f"  • Status: {online_verification.get('verification_status', 'Unknown')}", styles['Normal']))
            content.append(Paragraph(f"  • Confidence: {online_verification.get('confidence_score', 0):.2f}", styles['Normal']))
            content.append(Paragraph(f"  • Sources: {online_verification.get('sources_count', 0)}", styles['Normal']))

        return content

    def _generate_performance_metrics_content(self, data: Dict[str, Any], styles) -> List:
        """Generate performance metrics section content"""
        content = []

        processing_time = data.get('processing_time_seconds')
        if processing_time:
            content.append(Paragraph(f"<b>Processing Time:</b> {processing_time:.2f} seconds", styles['Normal']))

        confidence_metrics = data.get('confidence_metrics', {})
        if confidence_metrics:
            content.append(Paragraph("<b>AI Confidence Metrics:</b>", styles['Normal']))
            for key, value in confidence_metrics.items():
                content.append(Paragraph(f"  • {key.replace('_', ' ').title()}: {value}", styles['Normal']))

        explainability_score = data.get('explainability_score')
        if explainability_score is not None:
            content.append(Paragraph(f"<b>Explainability Score:</b> {explainability_score:.2f}", styles['Normal']))

        return content

    def _generate_fallback_json_report(self, session_id: str, diagnosis_data: Dict[str, Any]) -> bytes:
        """Generate a fallback JSON report when PDF generation fails"""
        try:
            report_data = {
                'session_id': session_id,
                'generated_at': datetime.now().isoformat(),
                'report_type': 'CortexMD Diagnosis Report (JSON Fallback)',
                'version': '2.0',
                'diagnosis_data': diagnosis_data
            }

            # Convert to JSON bytes
            json_content = json.dumps(report_data, indent=2, default=str)
            return json_content.encode('utf-8')

        except Exception as e:
            logger.error(f"Error generating fallback JSON report: {e}")
            # Return minimal error report
            error_report = {
                'error': 'Report generation failed',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'message': 'Unable to generate diagnosis report due to technical issues'
            }
            return json.dumps(error_report, indent=2).encode('utf-8')

# Global instance for reuse
report_generator = DiagnosisReportGenerator()
