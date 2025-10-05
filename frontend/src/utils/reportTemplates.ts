// Specialized Medical Report Templates
import { DiagnosisResult } from '@/types'

export interface ReportTemplate {
  id: string
  name: string
  description: string
  specialty: string
  sections: string[]
  customFields: { [key: string]: string }
  styling?: {
    primaryColor: string
    headerStyle: 'formal' | 'modern' | 'clinical'
    layout: 'single-column' | 'two-column'
  }
}

export const reportTemplates: ReportTemplate[] = [
  {
    id: 'general-medicine',
    name: 'General Medicine Report',
    description: 'Standard medical report for general practice',
    specialty: 'General Medicine',
    sections: [
      'patient-info',
      'chief-complaint', 
      'diagnosis',
      'differential-diagnosis',
      'recommendations',
      'follow-up',
      'verification'
    ],
    customFields: {
      'chief-complaint': 'Primary presenting symptoms and duration',
      'follow-up': 'Recommended follow-up schedule'
    },
    styling: {
      primaryColor: '#3b82f6',
      headerStyle: 'formal',
      layout: 'single-column'
    }
  },
  {
    id: 'cardiology',
    name: 'Cardiology Report',
    description: 'Specialized cardiac assessment report',
    specialty: 'Cardiology',
    sections: [
      'patient-info',
      'cardiac-history',
      'diagnosis', 
      'risk-stratification',
      'ecg-findings',
      'recommendations',
      'medication-review',
      'verification'
    ],
    customFields: {
      'cardiac-history': 'Previous cardiac events, family history',
      'risk-stratification': 'Cardiovascular risk assessment',
      'ecg-findings': 'Electrocardiogram interpretation',
      'medication-review': 'Current cardiac medications and adjustments'
    },
    styling: {
      primaryColor: '#dc2626',
      headerStyle: 'clinical',
      layout: 'two-column'
    }
  },
  {
    id: 'radiology',
    name: 'Radiology Report',
    description: 'Diagnostic imaging interpretation report',
    specialty: 'Radiology',
    sections: [
      'patient-info',
      'study-info',
      'technique',
      'findings',
      'impression',
      'recommendations',
      'ai-analysis',
      'verification'
    ],
    customFields: {
      'study-info': 'Examination type, date, indication',
      'technique': 'Imaging protocol and parameters',
      'findings': 'Detailed radiological findings',
      'impression': 'Radiological impression and significance',
      'ai-analysis': 'AI-assisted image analysis results'
    },
    styling: {
      primaryColor: '#059669',
      headerStyle: 'modern',
      layout: 'single-column'
    }
  },
  {
    id: 'emergency',
    name: 'Emergency Medicine Report',
    description: 'Urgent care and emergency department report',
    specialty: 'Emergency Medicine',
    sections: [
      'patient-info',
      'presentation',
      'triage-assessment',
      'vital-signs',
      'diagnosis',
      'interventions',
      'disposition',
      'verification'
    ],
    customFields: {
      'presentation': 'Mode of arrival, chief complaint, onset',
      'triage-assessment': 'Initial triage category and priority',
      'interventions': 'Emergency interventions performed',
      'disposition': 'Patient disposition and discharge planning'
    },
    styling: {
      primaryColor: '#ea580c',
      headerStyle: 'clinical',
      layout: 'two-column'
    }
  },
  {
    id: 'pathology',
    name: 'Pathology Report',
    description: 'Histopathological examination report',
    specialty: 'Pathology',
    sections: [
      'patient-info',
      'specimen-info',
      'gross-description',
      'microscopic-findings',
      'diagnosis',
      'staging',
      'molecular-markers',
      'verification'
    ],
    customFields: {
      'specimen-info': 'Specimen type, site, collection method',
      'gross-description': 'Macroscopic specimen description',
      'microscopic-findings': 'Histological examination results',
      'staging': 'Tumor staging (if applicable)',
      'molecular-markers': 'Immunohistochemistry and molecular studies'
    },
    styling: {
      primaryColor: '#7c3aed',
      headerStyle: 'formal',
      layout: 'single-column'
    }
  },
  {
    id: 'psychiatry',
    name: 'Psychiatric Evaluation Report',
    description: 'Mental health assessment report',
    specialty: 'Psychiatry',
    sections: [
      'patient-info',
      'mental-status',
      'psychiatric-history',
      'diagnosis',
      'risk-assessment',
      'treatment-plan',
      'medication-review',
      'verification'
    ],
    customFields: {
      'mental-status': 'Mental status examination findings',
      'psychiatric-history': 'Previous psychiatric treatment and hospitalizations',
      'risk-assessment': 'Suicide and violence risk assessment',
      'treatment-plan': 'Therapeutic interventions and goals'
    },
    styling: {
      primaryColor: '#0891b2',
      headerStyle: 'modern',
      layout: 'single-column'
    }
  },
  {
    id: 'pediatrics',
    name: 'Pediatric Report',
    description: 'Child and adolescent medical report',
    specialty: 'Pediatrics',
    sections: [
      'patient-info',
      'growth-development',
      'immunization-status',
      'diagnosis',
      'developmental-assessment',
      'recommendations',
      'parent-education',
      'verification'
    ],
    customFields: {
      'growth-development': 'Growth charts, developmental milestones',
      'immunization-status': 'Vaccination history and due vaccines',
      'developmental-assessment': 'Age-appropriate development evaluation',
      'parent-education': 'Guidance for parents/caregivers'
    },
    styling: {
      primaryColor: '#ec4899',
      headerStyle: 'modern',
      layout: 'two-column'
    }
  }
]

// Template selector based on diagnosis
export const selectOptimalTemplate = (diagnosisData: DiagnosisResult): ReportTemplate => {
  const diagnosis = diagnosisData.primary_diagnosis.condition.toLowerCase()
  
  // Simple keyword matching for template selection
  if (diagnosis.includes('cardiac') || diagnosis.includes('heart') || diagnosis.includes('coronary')) {
    return reportTemplates.find(t => t.id === 'cardiology') || reportTemplates[0]
  }
  
  if (diagnosis.includes('psychiatric') || diagnosis.includes('depression') || diagnosis.includes('anxiety')) {
    return reportTemplates.find(t => t.id === 'psychiatry') || reportTemplates[0]
  }
  
  if (diagnosis.includes('fracture') || diagnosis.includes('imaging') || diagnosis.includes('scan')) {
    return reportTemplates.find(t => t.id === 'radiology') || reportTemplates[0]
  }
  
  if (diagnosis.includes('emergency') || diagnosis.includes('acute') || diagnosis.includes('urgent')) {
    return reportTemplates.find(t => t.id === 'emergency') || reportTemplates[0]
  }
  
  if (diagnosis.includes('cancer') || diagnosis.includes('tumor') || diagnosis.includes('malignant')) {
    return reportTemplates.find(t => t.id === 'pathology') || reportTemplates[0]
  }
  
  if (diagnosis.includes('child') || diagnosis.includes('pediatric') || diagnosis.includes('infant')) {
    return reportTemplates.find(t => t.id === 'pediatrics') || reportTemplates[0]
  }
  
  // Default to general medicine
  return reportTemplates[0]
}

// Generate template-specific content
export const generateTemplateContent = (
  template: ReportTemplate, 
  diagnosisData: DiagnosisResult
): { [key: string]: string } => {
  const content: { [key: string]: string } = {}
  
  template.sections.forEach(section => {
    switch (section) {
      case 'patient-info':
        content[section] = generatePatientInfo(diagnosisData)
        break
      case 'diagnosis':
        content[section] = generateDiagnosisSection(diagnosisData)
        break
      case 'recommendations':
        content[section] = generateRecommendations(diagnosisData)
        break
      case 'verification':
        content[section] = generateVerificationSection(diagnosisData)
        break
      default:
        content[section] = generateGenericSection(section, diagnosisData, template)
    }
  })
  
  return content
}

const generatePatientInfo = (data: DiagnosisResult): string => {
  return `
Report ID: ${data.session_id}
Date: ${new Date(data.timestamp).toLocaleDateString()}
Analysis Time: ${data.processing_time?.toFixed(2) || 'N/A'} seconds
Urgency Level: ${(data.urgency_level || 'MEDIUM').toUpperCase()}
AI Confidence: ${((data.confidence_metrics?.overall_confidence || 0) * 100).toFixed(1)}%
  `.trim()
}

const generateDiagnosisSection = (data: DiagnosisResult): string => {
  let content = `Primary Diagnosis: ${data.primary_diagnosis.condition}\n`
  content += `Confidence: ${(data.primary_diagnosis.confidence * 100).toFixed(1)}%\n`
  
  if (data.primary_diagnosis.icd_code) {
    content += `ICD Code: ${data.primary_diagnosis.icd_code}\n`
  }
  
  if (data.primary_diagnosis.description) {
    content += `\nClinical Impression:\n${data.primary_diagnosis.description}\n`
  }
  
  return content
}

const generateRecommendations = (data: DiagnosisResult): string => {
  let content = ''
  
  if (data.recommended_tests?.length > 0) {
    content += 'Recommended Tests:\n'
    data.recommended_tests.forEach((test, index) => {
      content += `${index + 1}. ${test}\n`
    })
    content += '\n'
  }
  
  if (data.treatment_recommendations?.recommended_tests && data.treatment_recommendations.recommended_tests.length > 0) {
    content += 'Treatment Recommendations:\n'
    data.treatment_recommendations.recommended_tests.forEach((treatment, index) => {
      content += `${index + 1}. ${treatment}\n`
    })
  }
  
  return content
}

const generateVerificationSection = (data: DiagnosisResult): string => {
  let content = 'AI Verification Results:\n\n'
  
  if (data.fol_verification) {
    content += `FOL Verification: ${data.fol_verification.status || 'N/A'}\n`
    content += `FOL Confidence: ${((data.fol_verification.overall_confidence || 0) * 100).toFixed(1)}%\n\n`
  }
  
  if (data.enhanced_verification) {
    content += `Enhanced Verification: ${data.enhanced_verification.overall_status || 'N/A'}\n`
    content += `Evidence Strength: ${data.enhanced_verification.evidence_strength || 'N/A'}\n`
    content += `Sources: ${data.enhanced_verification.sources_count || 0}\n\n`
  }
  
  if (data.online_verification) {
    content += `Online Verification: ${data.online_verification.verification_status || 'N/A'}\n`
    content += `Online Confidence: ${((data.online_verification.confidence_score || 0) * 100).toFixed(1)}%\n`
  }
  
  return content
}

const generateGenericSection = (
  section: string, 
  data: DiagnosisResult, 
  template: ReportTemplate
): string => {
  const customField = template.customFields[section]
  if (customField) {
    return `${customField}\n\n[This section would be populated with specific ${section} data in a full implementation]`
  }
  
  return `[${section.replace('-', ' ').toUpperCase()} section - to be implemented based on specific requirements]`
}

export default {
  reportTemplates,
  selectOptimalTemplate,
  generateTemplateContent
}
