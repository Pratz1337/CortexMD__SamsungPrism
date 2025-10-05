// Professional Medical Report Generator
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { saveAs } from 'file-saver';
import { Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel, Table, TableRow, TableCell, WidthType } from 'docx';
import { DiagnosisResult } from '@/types';

export interface ReportOptions {
  format: 'pdf' | 'html' | 'docx' | 'print';
  template: 'standard' | 'detailed' | 'summary';
  includeCharts: boolean;
  includeDiagnosticImages: boolean;
  hospitalInfo?: {
    name: string;
    address: string;
    phone: string;
    logo?: string;
  };
  physicianInfo?: {
    name: string;
    title: string;
    license: string;
    signature?: string;
  };
}

export class MedicalReportGenerator {
  private data: DiagnosisResult;
  private options: ReportOptions;

  constructor(data: DiagnosisResult, options: ReportOptions) {
    this.data = data;
    this.options = options;
  }

  // Main generation method
  async generateReport(): Promise<void> {
    switch (this.options.format) {
      case 'pdf':
        return this.generatePDFReport();
      case 'html':
        return this.generateHTMLReport();
      case 'docx':
        return this.generateWordReport();
      case 'print':
        return this.printReport();
      default:
        throw new Error('Unsupported format');
    }
  }

  // PDF Report Generation
  private async generatePDFReport(): Promise<void> {
    const doc = new jsPDF('portrait', 'mm', 'a4');
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    let currentY = 20;

    // Add header
    currentY = this.addPDFHeader(doc, currentY, pageWidth);
    
    // Add patient info section
    currentY = this.addPDFPatientInfo(doc, currentY, pageWidth);
    
    // Add diagnosis section
    currentY = this.addPDFDiagnosis(doc, currentY, pageWidth, pageHeight);
    
    // Add recommendations
    currentY = this.addPDFRecommendations(doc, currentY, pageWidth, pageHeight);
    
    // Add verification details if detailed template
    if (this.options.template === 'detailed') {
      currentY = this.addPDFVerificationDetails(doc, currentY, pageWidth, pageHeight);
    }
    
    // Add footer
    this.addPDFFooter(doc, pageWidth, pageHeight);

    // Save the PDF
    const filename = `medical_report_${this.data.session_id}.pdf`;
    doc.save(filename);
  }

  private addPDFHeader(doc: jsPDF, startY: number, pageWidth: number): number {
    let currentY = startY;

    // Hospital info
    const hospitalName = this.options.hospitalInfo?.name || 'CortexMD Medical AI System';
    const hospitalAddress = this.options.hospitalInfo?.address || 'AI-Powered Medical Diagnosis Platform';
    
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text(hospitalName, pageWidth / 2, currentY, { align: 'center' });
    currentY += 10;

    doc.setFontSize(12);
    doc.setFont('helvetica', 'normal');
    doc.text(hospitalAddress, pageWidth / 2, currentY, { align: 'center' });
    currentY += 15;

    // Title
    doc.setFontSize(18);
    doc.setFont('helvetica', 'bold');
    doc.text('AI MEDICAL DIAGNOSIS REPORT', pageWidth / 2, currentY, { align: 'center' });
    currentY += 15;

    // Divider line
    doc.setLineWidth(0.5);
    doc.line(20, currentY, pageWidth - 20, currentY);
    currentY += 10;

    return currentY;
  }

  private addPDFPatientInfo(doc: jsPDF, startY: number, pageWidth: number): number {
    let currentY = startY;

    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('PATIENT INFORMATION', 20, currentY);
    currentY += 8;

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    
    const info = [
      ['Report ID:', this.data.session_id],
      ['Date Generated:', new Date(this.data.timestamp).toLocaleString()],
      ['Analysis Time:', `${this.data.processing_time?.toFixed(2) || 'N/A'} seconds`],
      ['AI Confidence:', `${((this.data.confidence_metrics?.overall_confidence || 0) * 100).toFixed(1)}%`],
      ['Urgency Level:', (this.data.urgency_level || 'MEDIUM').toUpperCase()],
    ];

    info.forEach(([label, value]) => {
      doc.text(label, 25, currentY);
      doc.text(value, 80, currentY);
      currentY += 6;
    });

    currentY += 5;
    return currentY;
  }

  private addPDFDiagnosis(doc: jsPDF, startY: number, pageWidth: number, pageHeight: number): number {
    let currentY = startY;

    // Check if we need a new page
    if (currentY > pageHeight - 60) {
      doc.addPage();
      currentY = 20;
    }

    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('PRIMARY DIAGNOSIS', 20, currentY);
    currentY += 8;

    // Primary diagnosis box
    doc.setFillColor(240, 248, 255);
    doc.rect(20, currentY, pageWidth - 40, 25, 'F');
    doc.setDrawColor(59, 130, 246);
    doc.setLineWidth(0.5);
    doc.rect(20, currentY, pageWidth - 40, 25);

    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.text(this.data.primary_diagnosis.condition, 25, currentY + 8);
    
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Confidence: ${(this.data.primary_diagnosis.confidence * 100).toFixed(1)}%`, 25, currentY + 16);
    
    if (this.data.primary_diagnosis.icd_code) {
      doc.text(`ICD Code: ${this.data.primary_diagnosis.icd_code}`, 25, currentY + 22);
    }

    currentY += 35;

    // Clinical impression
    if (this.data.primary_diagnosis.description) {
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text('CLINICAL IMPRESSION:', 20, currentY);
      currentY += 8;

      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      const description = this.data.primary_diagnosis.description;
      const lines = doc.splitTextToSize(description, pageWidth - 40);
      doc.text(lines, 25, currentY);
      currentY += lines.length * 5 + 10;
    }

    // Differential diagnoses
    if (this.data.differential_diagnoses?.length > 0) {
      currentY = this.addPDFDifferentialDiagnoses(doc, currentY, pageWidth, pageHeight);
    }

    return currentY;
  }

  private addPDFDifferentialDiagnoses(doc: jsPDF, startY: number, pageWidth: number, pageHeight: number): number {
    let currentY = startY;

    // Check if we need a new page
    if (currentY > pageHeight - 80) {
      doc.addPage();
      currentY = 20;
    }

    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.text('DIFFERENTIAL DIAGNOSES', 20, currentY);
    currentY += 10;

    this.data.differential_diagnoses.slice(0, 5).forEach((diagnosis, index) => {
      if (currentY > pageHeight - 30) {
        doc.addPage();
        currentY = 20;
      }

      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text(`${index + 1}. ${diagnosis.condition}`, 25, currentY);
      currentY += 5;

      doc.setFont('helvetica', 'normal');
      doc.text(`Confidence: ${(diagnosis.confidence * 100).toFixed(1)}%`, 30, currentY);
      currentY += 5;

      if (diagnosis.reasoning) {
        const reasoningLines = doc.splitTextToSize(diagnosis.reasoning, pageWidth - 60);
        doc.text(reasoningLines, 30, currentY);
        currentY += reasoningLines.length * 4 + 5;
      }

      currentY += 3;
    });

    return currentY;
  }

  private addPDFRecommendations(doc: jsPDF, startY: number, pageWidth: number, pageHeight: number): number {
    let currentY = startY;

    // Check if we need a new page
    if (currentY > pageHeight - 60) {
      doc.addPage();
      currentY = 20;
    }

    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.text('RECOMMENDED TESTS & TREATMENT', 20, currentY);
    currentY += 10;

    // Recommended tests
    if (this.data.recommended_tests?.length > 0) {
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('Recommended Tests:', 25, currentY);
      currentY += 6;

      doc.setFont('helvetica', 'normal');
      this.data.recommended_tests.forEach((test, index) => {
        if (currentY > pageHeight - 20) {
          doc.addPage();
          currentY = 20;
        }
        doc.text(`• ${test}`, 30, currentY);
        currentY += 5;
      });
      currentY += 5;
    }

    // Treatment recommendations
    if (this.data.treatment_recommendations && 
        this.data.treatment_recommendations.recommended_tests && 
        this.data.treatment_recommendations.recommended_tests.length > 0) {
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('Treatment Recommendations:', 25, currentY);
      currentY += 6;

      doc.setFont('helvetica', 'normal');
      this.data.treatment_recommendations.recommended_tests.forEach((treatment, index) => {
        if (currentY > pageHeight - 20) {
          doc.addPage();
          currentY = 20;
        }
        doc.text(`• ${treatment}`, 30, currentY);
        currentY += 5;
      });
    }

    return currentY;
  }

  private addPDFVerificationDetails(doc: jsPDF, startY: number, pageWidth: number, pageHeight: number): number {
    let currentY = startY;

    // Check if we need a new page
    if (currentY > pageHeight - 80) {
      doc.addPage();
      currentY = 20;
    }

    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.text('AI VERIFICATION DETAILS', 20, currentY);
    currentY += 10;

    // FOL Verification
    if (this.data.fol_verification) {
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('First-Order Logic Verification:', 25, currentY);
      currentY += 6;

      doc.setFont('helvetica', 'normal');
      doc.text(`Status: ${this.data.fol_verification.status || 'N/A'}`, 30, currentY);
      currentY += 5;
      doc.text(`Confidence: ${((this.data.fol_verification.overall_confidence || 0) * 100).toFixed(1)}%`, 30, currentY);
      currentY += 5;

      if (this.data.fol_verification.verification_summary) {
        const summaryLines = doc.splitTextToSize(this.data.fol_verification.verification_summary, pageWidth - 60);
        doc.text(summaryLines, 30, currentY);
        currentY += summaryLines.length * 4 + 10;
      }
    }

    // Enhanced Verification
    if (this.data.enhanced_verification) {
      if (currentY > pageHeight - 40) {
        doc.addPage();
        currentY = 20;
      }

      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text('Enhanced AI Verification:', 25, currentY);
      currentY += 6;

      doc.setFont('helvetica', 'normal');
      doc.text(`Status: ${this.data.enhanced_verification.overall_status || 'N/A'}`, 30, currentY);
      currentY += 5;
      doc.text(`Evidence Strength: ${this.data.enhanced_verification.evidence_strength || 'N/A'}`, 30, currentY);
      currentY += 5;
      doc.text(`Sources: ${this.data.enhanced_verification.sources_count || 0}`, 30, currentY);
      currentY += 10;
    }

    return currentY;
  }

  private addPDFFooter(doc: jsPDF, pageWidth: number, pageHeight: number): void {
    const totalPages = doc.internal.pages.length - 1;
    
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      
      // Footer line
      doc.setLineWidth(0.3);
      doc.line(20, pageHeight - 20, pageWidth - 20, pageHeight - 20);
      
      // Footer text
      doc.setFontSize(8);
      doc.setFont('helvetica', 'normal');
      doc.text('Generated by CortexMD AI Medical Diagnosis System', 20, pageHeight - 15);
      doc.text(`Page ${i} of ${totalPages}`, pageWidth - 20, pageHeight - 15, { align: 'right' });
      doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, pageHeight - 15, { align: 'center' });
      
      // Disclaimer
      doc.setFontSize(7);
      doc.text('This AI-generated report should be reviewed by a qualified medical professional before clinical use.', 
               pageWidth / 2, pageHeight - 10, { align: 'center' });
    }
  }

  // HTML Report Generation
  private async generateHTMLReport(): Promise<void> {
    const htmlContent = this.generateHTMLContent();
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const filename = `medical_report_${this.data.session_id}.html`;
    saveAs(blob, filename);
  }

  private generateHTMLContent(): string {
    const hospitalName = this.options.hospitalInfo?.name || 'CortexMD Medical AI System';
    const physicianName = this.options.physicianInfo?.name || 'AI Medical Assistant';
    
    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Report - ${this.data.session_id}</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .report-container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 3px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
        .hospital-name { font-size: 28px; font-weight: bold; color: #007bff; margin-bottom: 10px; }
        .report-title { font-size: 24px; font-weight: bold; color: #333; margin-top: 20px; }
        .section { margin: 30px 0; }
        .section-title { font-size: 18px; font-weight: bold; color: #007bff; border-bottom: 2px solid #007bff; padding-bottom: 5px; margin-bottom: 15px; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .info-item { margin: 10px 0; }
        .info-label { font-weight: bold; color: #555; }
        .primary-diagnosis { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .diagnosis-title { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
        .confidence-bar { background: rgba(255,255,255,0.3); height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .confidence-fill { background: #4CAF50; height: 100%; border-radius: 10px; }
        .differential-list { list-style: none; padding: 0; }
        .differential-item { background: #f8f9fa; margin: 10px 0; padding: 15px; border-left: 4px solid #007bff; }
        .recommendation-box { background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 15px 0; }
        .verification-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .verification-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745; }
        .footer { border-top: 2px solid #dee2e6; padding-top: 20px; margin-top: 40px; text-align: center; color: #666; }
        .disclaimer { background: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; margin-top: 20px; border-left: 4px solid #ffc107; }
        @media print { body { background: white; } .report-container { box-shadow: none; } }
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <div class="hospital-name">${hospitalName}</div>
            <div class="report-title">AI Medical Diagnosis Report</div>
            <div style="margin-top: 15px; color: #666;">Generated on ${new Date(this.data.timestamp).toLocaleString()}</div>
        </div>

        <div class="section">
            <div class="section-title">Patient Information</div>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Report ID:</span> ${this.data.session_id}
                </div>
                <div class="info-item">
                    <span class="info-label">Date Generated:</span> ${new Date(this.data.timestamp).toLocaleDateString()}
                </div>
                <div class="info-item">
                    <span class="info-label">Analysis Time:</span> ${this.data.processing_time?.toFixed(2) || 'N/A'} seconds
                </div>
                <div class="info-item">
                    <span class="info-label">AI Confidence:</span> ${((this.data.confidence_metrics?.overall_confidence || 0) * 100).toFixed(1)}%
                </div>
                <div class="info-item">
                    <span class="info-label">Urgency Level:</span> <span style="background: ${this.getUrgencyColor(this.data.urgency_level)}; padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold;">${(this.data.urgency_level || 'MEDIUM').toUpperCase()}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Attending Physician:</span> ${physicianName}
                </div>
            </div>
        </div>

        <div class="section">
            <div class="primary-diagnosis">
                <div class="diagnosis-title">${this.data.primary_diagnosis.condition}</div>
                <div style="margin: 10px 0;">
                    <div>Confidence Score: ${(this.data.primary_diagnosis.confidence * 100).toFixed(1)}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${(this.data.primary_diagnosis.confidence * 100)}%"></div>
                    </div>
                </div>
                ${this.data.primary_diagnosis.icd_code ? `<div>ICD Code: ${this.data.primary_diagnosis.icd_code}</div>` : ''}
                ${this.data.primary_diagnosis.description ? `<div style="margin-top: 15px; font-style: italic;">${this.data.primary_diagnosis.description}</div>` : ''}
            </div>
        </div>

        ${this.data.differential_diagnoses?.length > 0 ? `
        <div class="section">
            <div class="section-title">Differential Diagnoses</div>
            <ul class="differential-list">
                ${this.data.differential_diagnoses.slice(0, 5).map((diagnosis, index) => `
                <li class="differential-item">
                    <strong>${index + 1}. ${diagnosis.condition}</strong>
                    <div style="margin: 8px 0;">Confidence: ${(diagnosis.confidence * 100).toFixed(1)}%</div>
                    ${diagnosis.reasoning ? `<div style="color: #666; font-style: italic;">${diagnosis.reasoning}</div>` : ''}
                </li>
                `).join('')}
            </ul>
        </div>
        ` : ''}

        ${this.data.recommended_tests?.length > 0 || this.data.treatment_recommendations ? `
        <div class="section">
            <div class="section-title">Recommendations</div>
            ${this.data.recommended_tests?.length > 0 ? `
            <div class="recommendation-box">
                <h4 style="margin-top: 0; color: #2d5a27;">Recommended Tests:</h4>
                <ul>
                    ${this.data.recommended_tests.map(test => `<li>${test}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            ${this.data.treatment_recommendations && 
              this.data.treatment_recommendations.recommended_tests && 
              this.data.treatment_recommendations.recommended_tests.length > 0 ? `
            <div class="recommendation-box">
                <h4 style="margin-top: 0; color: #2d5a27;">Treatment Recommendations:</h4>
                <ul>
                    ${this.data.treatment_recommendations.recommended_tests.map(treatment => `<li>${treatment}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
        </div>
        ` : ''}

        ${this.options.template === 'detailed' ? this.generateHTMLVerificationSection() : ''}

        <div class="footer">
            <div>Generated by CortexMD AI Medical Diagnosis System</div>
            <div style="margin-top: 10px; font-size: 12px;">Session ID: ${this.data.session_id} | Generated: ${new Date().toLocaleString()}</div>
        </div>

        <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> This AI-generated report is for informational purposes only and should be reviewed by a qualified medical professional before clinical use. It is not intended to replace professional medical advice, diagnosis, or treatment.
        </div>
    </div>
</body>
</html>
    `;
  }

  private generateHTMLVerificationSection(): string {
    return `
        <div class="section">
            <div class="section-title">AI Verification Details</div>
            <div class="verification-grid">
                ${this.data.fol_verification ? `
                <div class="verification-card">
                    <h4 style="margin-top: 0; color: #28a745;">First-Order Logic Verification</h4>
                    <div><strong>Status:</strong> ${this.data.fol_verification.status || 'N/A'}</div>
                    <div><strong>Confidence:</strong> ${((this.data.fol_verification.overall_confidence || 0) * 100).toFixed(1)}%</div>
                    ${this.data.fol_verification.verification_summary ? `<div style="margin-top: 10px; font-style: italic;">${this.data.fol_verification.verification_summary}</div>` : ''}
                </div>
                ` : ''}
                ${this.data.enhanced_verification ? `
                <div class="verification-card">
                    <h4 style="margin-top: 0; color: #28a745;">Enhanced AI Verification</h4>
                    <div><strong>Status:</strong> ${this.data.enhanced_verification.overall_status || 'N/A'}</div>
                    <div><strong>Evidence Strength:</strong> ${this.data.enhanced_verification.evidence_strength || 'N/A'}</div>
                    <div><strong>Sources:</strong> ${this.data.enhanced_verification.sources_count || 0}</div>
                </div>
                ` : ''}
                ${this.data.online_verification ? `
                <div class="verification-card">
                    <h4 style="margin-top: 0; color: #28a745;">Online Verification</h4>
                    <div><strong>Status:</strong> ${this.data.online_verification.verification_status || 'N/A'}</div>
                    <div><strong>Confidence:</strong> ${((this.data.online_verification.confidence_score || 0) * 100).toFixed(1)}%</div>
                    <div><strong>Sources:</strong> ${this.data.online_verification.sources?.length || 0}</div>
                </div>
                ` : ''}
            </div>
        </div>
    `;
  }

  private getUrgencyColor(urgency?: string): string {
    switch (urgency?.toLowerCase()) {
      case 'critical': return '#dc3545';
      case 'high': return '#fd7e14';
      case 'medium': return '#ffc107';
      case 'low': return '#28a745';
      default: return '#6c757d';
    }
  }

  // Word Document Generation
  private async generateWordReport(): Promise<void> {
    const doc = new Document({
      sections: [
        {
          properties: {},
          children: [
            // Title
            new Paragraph({
              text: "AI Medical Diagnosis Report",
              heading: HeadingLevel.TITLE,
              alignment: AlignmentType.CENTER,
            }),
            
            // Hospital info
            new Paragraph({
              text: this.options.hospitalInfo?.name || 'CortexMD Medical AI System',
              alignment: AlignmentType.CENTER,
            }),
            
            new Paragraph({
              text: `Generated: ${new Date(this.data.timestamp).toLocaleString()}`,
              alignment: AlignmentType.CENTER,
            }),

            // Patient Information Section
            new Paragraph({
              text: "Patient Information",
              heading: HeadingLevel.HEADING_1,
            }),

            new Paragraph({
              children: [
                new TextRun({ text: "Report ID: ", bold: true }),
                new TextRun(this.data.session_id),
              ],
            }),

            new Paragraph({
              children: [
                new TextRun({ text: "Analysis Time: ", bold: true }),
                new TextRun(`${this.data.processing_time?.toFixed(2) || 'N/A'} seconds`),
              ],
            }),

            new Paragraph({
              children: [
                new TextRun({ text: "AI Confidence: ", bold: true }),
                new TextRun(`${((this.data.confidence_metrics?.overall_confidence || 0) * 100).toFixed(1)}%`),
              ],
            }),

            // Primary Diagnosis Section
            new Paragraph({
              text: "Primary Diagnosis",
              heading: HeadingLevel.HEADING_1,
            }),

            new Paragraph({
              text: this.data.primary_diagnosis.condition,
              heading: HeadingLevel.HEADING_2,
            }),

            new Paragraph({
              children: [
                new TextRun({ text: "Confidence: ", bold: true }),
                new TextRun(`${(this.data.primary_diagnosis.confidence * 100).toFixed(1)}%`),
              ],
            }),

            ...(this.data.primary_diagnosis.icd_code ? [
              new Paragraph({
                children: [
                  new TextRun({ text: "ICD Code: ", bold: true }),
                  new TextRun(this.data.primary_diagnosis.icd_code),
                ],
              })
            ] : []),

            ...(this.data.primary_diagnosis.description ? [
              new Paragraph({
                children: [
                  new TextRun({ text: "Clinical Impression: ", bold: true }),
                ],
              }),
              new Paragraph({
                text: this.data.primary_diagnosis.description,
              })
            ] : []),

            // Add more sections as needed...
          ],
        },
      ],
    });

    const blob = await Packer.toBlob(doc);
    const filename = `medical_report_${this.data.session_id}.docx`;
    saveAs(blob, filename);
  }

  // Print functionality
  private async printReport(): Promise<void> {
    const htmlContent = this.generateHTMLContent();
    const printWindow = window.open('', '_blank');
    if (printWindow) {
      printWindow.document.write(htmlContent);
      printWindow.document.close();
      printWindow.focus();
      setTimeout(() => {
        printWindow.print();
      }, 1000);
    }
  }
}

// Factory function for easy usage
export const generateMedicalReport = async (data: DiagnosisResult, options: ReportOptions) => {
  const generator = new MedicalReportGenerator(data, options);
  return generator.generateReport();
};

// Quick generation functions
export const generatePDFReport = (data: DiagnosisResult, options?: Partial<ReportOptions>) => {
  return generateMedicalReport(data, {
    format: 'pdf',
    template: 'standard',
    includeCharts: true,
    includeDiagnosticImages: true,
    ...options,
  });
};

export const generateHTMLReport = (data: DiagnosisResult, options?: Partial<ReportOptions>) => {
  return generateMedicalReport(data, {
    format: 'html',
    template: 'detailed',
    includeCharts: true,
    includeDiagnosticImages: true,
    ...options,
  });
};

export const generateWordReport = (data: DiagnosisResult, options?: Partial<ReportOptions>) => {
  return generateMedicalReport(data, {
    format: 'docx',
    template: 'standard',
    includeCharts: false,
    includeDiagnosticImages: false,
    ...options,
  });
};
