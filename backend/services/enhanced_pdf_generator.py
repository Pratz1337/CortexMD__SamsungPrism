"""
Enhanced PDF Report Generator for CortexMD
Fixes corruption issues and provides robust PDF generation
"""

import os
import json
import io
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

class EnhancedPDFGenerator:
    """Enhanced PDF generator with corruption fixes"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom PDF styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.HexColor('#2563eb')
        )

        # Section header style
        self.section_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#1e40af'),
            borderWidth=1,
            borderColor=colors.HexColor('#e5e7eb'),
            borderPadding=5,
            backColor=colors.HexColor('#f8fafc')
        )

        # Subsection style
        self.subsection_style = ParagraphStyle(
            'SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#374151')
        )

        # Normal text style
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=6
        )

        # Code style for technical data
        self.code_style = ParagraphStyle(
            'CodeStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            fontName='Courier',
            backColor=colors.HexColor('#f3f4f6'),
            borderWidth=1,
            borderColor=colors.HexColor('#d1d5db'),
            borderPadding=8,
            leftIndent=10,
            rightIndent=10
        )

    def generate_diagnosis_report(self, session_id: str, diagnosis_data: Dict[str, Any]) -> bytes:
        """
        Generate a comprehensive PDF diagnosis report
        
        Args:
            session_id: Diagnosis session identifier
            diagnosis_data: Complete diagnosis data
            
        Returns:
            PDF bytes or raises exception
        """
        try:
            # Create PDF in memory
            buffer = io.BytesIO()
            
            # Create document with proper margins
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72,
                title=f"CortexMD Diagnosis Report - {session_id}",
                author="CortexMD AI System"
            )

            # Build the story (content)
            story = self._build_report_story(session_id, diagnosis_data)
            
            # Generate PDF
            doc.build(story)
            
            # Get PDF bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            # Validate PDF was created properly
            if len(pdf_bytes) < 1000:  # PDF should be at least 1KB
                raise Exception("Generated PDF is too small, likely corrupted")
            
            # Check PDF header
            if not pdf_bytes.startswith(b'%PDF'):
                raise Exception("Generated file does not have proper PDF header")
            
            logger.info(f"Successfully generated PDF report for session {session_id}, size: {len(pdf_bytes)} bytes")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")
            raise Exception(f"PDF generation failed: {str(e)}")

    def _build_report_story(self, session_id: str, diagnosis_data: Dict[str, Any]) -> List:
        """Build the complete PDF story/content"""
        story = []
        
        # Header
        story.extend(self._create_header(session_id))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(diagnosis_data))
        
        # Patient Information
        story.extend(self._create_patient_section(diagnosis_data))
        
        # Diagnosis Results
        story.extend(self._create_diagnosis_section(diagnosis_data))
        
        # AI Analysis
        story.extend(self._create_ai_analysis_section(diagnosis_data))
        
        # Clinical Assessment
        story.extend(self._create_clinical_section(diagnosis_data))
        
        # Medical Entities
        story.extend(self._create_entities_section(diagnosis_data))
        
        # Technical Details
        story.extend(self._create_technical_section(diagnosis_data))
        
        # Footer
        story.extend(self._create_footer())
        
        return story

    def _create_header(self, session_id: str) -> List:
        """Create report header"""
        content = []
        
        # Title
        content.append(Paragraph("CortexMD AI Diagnosis Report", self.title_style))
        content.append(Spacer(1, 0.3*inch))
        
        # Report metadata
        metadata_data = [
            ['Session ID:', session_id],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Report Version:', '2.1'],
            ['System:', 'CortexMD AI Diagnostic Platform']
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ]))
        
        content.append(metadata_table)
        content.append(Spacer(1, 0.5*inch))
        
        return content

    def _create_executive_summary(self, diagnosis_data: Dict[str, Any]) -> List:
        """Create executive summary section"""
        content = []
        
        content.append(Paragraph("Executive Summary", self.section_style))
        
        # Extract key information for summary
        primary_diagnosis = diagnosis_data.get('primary_diagnosis', {})
        confidence = diagnosis_data.get('confidence_score', 0)
        
        if primary_diagnosis:
            condition = primary_diagnosis.get('condition', 'Not specified')
            confidence_pct = primary_diagnosis.get('confidence_percentage', confidence * 100)
            
            summary_text = f"""
            <b>Primary Diagnosis:</b> {condition}<br/>
            <b>Confidence Level:</b> {confidence_pct:.1f}%<br/>
            <b>Urgency:</b> {diagnosis_data.get('urgency_level', 'Medium')}<br/>
            <b>Processing Time:</b> {diagnosis_data.get('processing_time', 0):.2f} seconds
            """
            content.append(Paragraph(summary_text, self.normal_style))
        
        # AI Summary if available
        ai_summary = diagnosis_data.get('ai_summary', '')
        if ai_summary:
            content.append(Paragraph("<b>AI Summary:</b>", self.subsection_style))
            # Clean and truncate summary
            clean_summary = ai_summary.replace('\n', '<br/>').replace('\r', '')
            if len(clean_summary) > 500:
                clean_summary = clean_summary[:500] + "..."
            content.append(Paragraph(clean_summary, self.normal_style))
        
        content.append(Spacer(1, 0.3*inch))
        return content

    def _create_patient_section(self, diagnosis_data: Dict[str, Any]) -> List:
        """Create patient information section"""
        content = []
        
        content.append(Paragraph("Patient Information", self.section_style))
        
        patient_data = [
            ['Patient ID:', diagnosis_data.get('patient_id', 'Unknown')],
            ['Session Type:', diagnosis_data.get('session_type', 'Clinical Note Scan')],
            ['Processing Timestamp:', diagnosis_data.get('processing_timestamp', 'Not available')]
        ]
        
        # Add clinical text if available
        clinical_text = diagnosis_data.get('clinical_text', '') or diagnosis_data.get('parsed_data', {}).get('extracted_text', '')
        if clinical_text:
            # Truncate long clinical text
            if len(clinical_text) > 300:
                clinical_text = clinical_text[:300] + "..."
            patient_data.append(['Clinical Text:', clinical_text])
        
        # Add scan information
        ocr_confidence = diagnosis_data.get('ocr_confidence', 0)
        if ocr_confidence > 0:
            patient_data.append(['OCR Confidence:', f"{ocr_confidence:.1f}%"])
        
        word_count = diagnosis_data.get('word_count', 0)
        if word_count > 0:
            patient_data.append(['Word Count:', str(word_count)])
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f9fafb')),
        ]))
        
        content.append(patient_table)
        content.append(Spacer(1, 0.3*inch))
        
        return content

    def _create_diagnosis_section(self, diagnosis_data: Dict[str, Any]) -> List:
        """Create diagnosis results section"""
        content = []
        
        content.append(Paragraph("Diagnosis Results", self.section_style))
        
        # Primary diagnosis
        primary_diagnosis = diagnosis_data.get('primary_diagnosis', {})
        if primary_diagnosis:
            content.append(Paragraph("Primary Diagnosis", self.subsection_style))
            
            diag_data = [
                ['Condition:', primary_diagnosis.get('condition', 'Not specified')],
                ['Confidence:', f"{primary_diagnosis.get('confidence_percentage', 0):.1f}%"],
                ['ICD Code:', primary_diagnosis.get('icd_code', 'Not available')],
            ]
            
            description = primary_diagnosis.get('description', '')
            if description:
                if len(description) > 200:
                    description = description[:200] + "..."
                diag_data.append(['Description:', description])
            
            diag_table = Table(diag_data, colWidths=[1.5*inch, 4.5*inch])
            diag_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecfdf5')),
            ]))
            
            content.append(diag_table)
        
        # Differential diagnoses if available
        differential = diagnosis_data.get('differential_diagnoses', [])
        if differential and len(differential) > 0:
            content.append(Paragraph("Differential Diagnoses", self.subsection_style))
            
            for i, diff_diag in enumerate(differential[:3], 1):  # Limit to top 3
                if isinstance(diff_diag, dict):
                    condition = diff_diag.get('condition', f'Condition {i}')
                    probability = diff_diag.get('probability', 0)
                    content.append(Paragraph(f"{i}. {condition} (Probability: {probability:.1f}%)", self.normal_style))
                else:
                    content.append(Paragraph(f"{i}. {str(diff_diag)}", self.normal_style))
        
        content.append(Spacer(1, 0.3*inch))
        return content

    def _create_ai_analysis_section(self, diagnosis_data: Dict[str, Any]) -> List:
        """Create AI analysis section"""
        content = []
        
        content.append(Paragraph("AI Analysis", self.section_style))
        
        ai_data = [
            ['AI Model:', diagnosis_data.get('ai_model_used', 'CortexMD Multi-Modal AI')],
            ['Processing Method:', diagnosis_data.get('processing_method', 'OCR + NLP + Medical Entity Recognition')],
            ['AI Confidence:', f"{diagnosis_data.get('ai_confidence', 0) * 100:.1f}%"],
        ]
        
        processing_time = diagnosis_data.get('processing_time', 0)
        if processing_time > 0:
            ai_data.append(['Processing Time:', f"{processing_time:.2f} seconds"])
        
        data_quality = diagnosis_data.get('data_quality_score', 0)
        if data_quality > 0:
            ai_data.append(['Data Quality Score:', f"{data_quality:.2f}/5.0"])
        
        ai_table = Table(ai_data, colWidths=[2*inch, 4*inch])
        ai_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fef3c7')),
        ]))
        
        content.append(ai_table)
        content.append(Spacer(1, 0.3*inch))
        
        return content

    def _create_clinical_section(self, diagnosis_data: Dict[str, Any]) -> List:
        """Create clinical assessment section"""
        content = []
        
        content.append(Paragraph("Clinical Assessment", self.section_style))
        
        urgency = diagnosis_data.get('urgency_level', 'Medium')
        urgency_color = {
            'High': colors.red,
            'Medium': colors.orange,
            'Low': colors.green
        }.get(urgency, colors.black)
        
        content.append(Paragraph(f"<b>Urgency Level:</b> <font color='{urgency_color}'>{urgency.upper()}</font>", self.normal_style))
        
        # Confidence metrics
        confidence_metrics = diagnosis_data.get('confidence_metrics', {})
        if confidence_metrics:
            content.append(Paragraph("Confidence Breakdown:", self.subsection_style))
            
            for key, value in confidence_metrics.items():
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    content.append(Paragraph(f"• {formatted_key}: {value:.2f}", self.normal_style))
                else:
                    content.append(Paragraph(f"• {formatted_key}: {str(value)}", self.normal_style))
        
        content.append(Spacer(1, 0.3*inch))
        return content

    def _create_entities_section(self, diagnosis_data: Dict[str, Any]) -> List:
        """Create extracted medical entities section"""
        content = []
        
        entities = diagnosis_data.get('extracted_entities', {})
        if not entities:
            return content
        
        content.append(Paragraph("Extracted Medical Entities", self.section_style))
        
        entity_data = []
        for key, value in entities.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, list):
                formatted_value = ', '.join(str(v) for v in value) if value else 'None found'
            else:
                formatted_value = str(value) if value else 'Not found'
            
            # Truncate long values
            if len(formatted_value) > 100:
                formatted_value = formatted_value[:100] + "..."
                
            entity_data.append([formatted_key + ':', formatted_value])
        
        if entity_data:
            entity_table = Table(entity_data, colWidths=[2*inch, 4*inch])
            entity_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f9ff')),
            ]))
            
            content.append(entity_table)
        
        content.append(Spacer(1, 0.3*inch))
        return content

    def _create_technical_section(self, diagnosis_data: Dict[str, Any]) -> List:
        """Create technical details section"""
        content = []
        
        content.append(Paragraph("Technical Details", self.section_style))
        
        # Show limited parsed data
        parsed_data = diagnosis_data.get('parsed_data', {})
        if parsed_data:
            content.append(Paragraph("System Output (Sample):", self.subsection_style))
            
            # Create a clean, limited representation of parsed data
            clean_data = {}
            for key, value in parsed_data.items():
                if key in ['extracted_text', 'confidence_score', 'processing_timestamp']:
                    if isinstance(value, str) and len(value) > 100:
                        clean_data[key] = value[:100] + "..."
                    else:
                        clean_data[key] = value
            
            try:
                json_str = json.dumps(clean_data, indent=2, default=str)
                content.append(Paragraph(f"<pre>{json_str}</pre>", self.code_style))
            except:
                content.append(Paragraph("Technical data available in system logs", self.normal_style))
        
        content.append(Spacer(1, 0.3*inch))
        return content

    def _create_footer(self) -> List:
        """Create report footer"""
        content = []
        
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph("Report Generation Information", self.section_style))
        
        footer_text = """
        <b>Important Notice:</b><br/>
        This report was generated by the CortexMD AI Diagnostic System. 
        The information contained in this report is intended for medical professional use only 
        and should not be used as a substitute for professional medical judgment.<br/><br/>
        
        <b>Disclaimer:</b><br/>
        AI-generated diagnoses should always be reviewed and validated by qualified healthcare professionals. 
        This system is designed to assist, not replace, clinical decision-making.<br/><br/>
        
        <b>Data Privacy:</b><br/>
        All patient data is processed in accordance with healthcare privacy regulations and 
        organizational data protection policies.
        """
        
        content.append(Paragraph(footer_text, self.normal_style))
        
        return content

    def generate_fallback_json_report(self, session_id: str, diagnosis_data: Dict[str, Any]) -> bytes:
        """Generate a fallback JSON report if PDF fails"""
        try:
            fallback_data = {
                'session_id': session_id,
                'report_type': 'JSON_FALLBACK',
                'generated_at': datetime.now().isoformat(),
                'diagnosis_data': diagnosis_data,
                'note': 'This is a fallback JSON report. PDF generation failed.'
            }
            
            json_str = json.dumps(fallback_data, indent=2, default=str)
            return json_str.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to generate fallback JSON report: {e}")
            error_data = {
                'error': 'Failed to generate report',
                'session_id': session_id,
                'message': str(e)
            }
            return json.dumps(error_data).encode('utf-8')
