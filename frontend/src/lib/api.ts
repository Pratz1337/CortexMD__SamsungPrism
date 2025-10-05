import axios from 'axios';
import {
  PatientInput,
  DiagnosisResult,
  DiagnosisResultUI,
  ProcessingStatus,
  UMLSSearchResult,
  SystemHealth,
  ChatResponse,
} from '@/types';

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_URL || 'http://localhost:5000'
).replace(/\/+$/, ''); // Remove trailing slashes to prevent double slashes

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export class DiagnosisAPI {
  static async submitDiagnosis(patientInput: PatientInput, files?: File[], claraOptions?: any, patientId?: string): Promise<{ session_id: string }> {
    console.log('🔧 DiagnosisAPI.submitDiagnosis called with:')
    console.log('  - patientInput:', patientInput)
    console.log('  - files count:', files?.length || 0)
    console.log('  - claraOptions:', claraOptions)
    console.log('  - patientId:', patientId)
    
    // Test backend connectivity first
    try {
      console.log('🏥 Testing backend connectivity...')
      const healthCheck = await api.get('/sessions/status')
      console.log('✅ Backend is reachable:', healthCheck.status)
    } catch (healthError) {
      console.error('❌ Backend connectivity test failed:', healthError)
      if (axios.isAxiosError(healthError)) {
        console.error('❌ Health check error details:')
        console.error('  - Status:', healthError.response?.status || 'No response')
        console.error('  - Message:', healthError.message)
        console.error('  - Base URL:', API_BASE_URL)
      }
    }
    
    const formData = new FormData();
    
    // If patientId is provided, fetch patient details first
    let enhancedClinicalText = '';
    if (patientId) {
      try {
        // Fetch patient details from database
        const patientResponse = await api.get(`/api/patients/${patientId}`);
        const patientData = patientResponse.data;
        
        // Create enhanced clinical text with patient context
        enhancedClinicalText = `
PATIENT MEDICAL RECORD CONTEXT:
Patient ID: ${patientId}
Name: ${patientData.patient_info?.patient_name || 'Not specified'}
Gender: ${patientData.patient_info?.gender || 'Not specified'}
Date of Birth: ${patientData.patient_info?.date_of_birth || 'Not specified'}
Admission Date: ${patientData.patient_info?.admission_date || 'Not specified'}
Current Status: ${patientData.patient_info?.current_status || 'active'}

CURRENT PRESENTING SYMPTOMS & CLINICAL DATA:
${patientInput.symptoms}

Additional Medical History: ${patientInput.medical_history || 'None provided'}
Current Medications: ${patientInput.current_medications || 'None provided'}
Allergies: ${patientInput.allergies || 'None provided'}

Vital Signs:
${patientInput.vital_signs?.temperature ? `Temperature: ${patientInput.vital_signs.temperature}°F` : ''}
${patientInput.vital_signs?.blood_pressure ? `Blood Pressure: ${patientInput.vital_signs.blood_pressure}` : ''}
${patientInput.vital_signs?.heart_rate ? `Heart Rate: ${patientInput.vital_signs.heart_rate} bpm` : ''}
${patientInput.vital_signs?.respiratory_rate ? `Respiratory Rate: ${patientInput.vital_signs.respiratory_rate}` : ''}
${patientInput.vital_signs?.oxygen_saturation ? `Oxygen Saturation: ${patientInput.vital_signs.oxygen_saturation}%` : ''}

HISTORICAL MEDICAL CONTEXT:
${patientData.concern_data ? `Recent CONCERN Risk Level: ${patientData.concern_data.current_risk_level} (Score: ${patientData.concern_data.current_concern_score})` : ''}
${patientData.concern_data?.risk_factors ? `Risk Factors: ${patientData.concern_data.risk_factors.join(', ')}` : ''}
${patientData.diagnosis_history?.length ? `Previous Diagnoses: ${patientData.diagnosis_history.slice(0, 3).map((d: any) => d.primary_diagnosis).join(', ')}` : 'No previous diagnoses on record'}
`.trim();

        // Add patient context as headers
        formData.append('patient_context', JSON.stringify({
          patient_id: patientId,
          patient_info: patientData.patient_info,
          concern_data: patientData.concern_data,
          recent_diagnosis_count: patientData.diagnosis_history?.length || 0
        }));
        
      } catch (error) {
        console.warn('Could not fetch patient details, proceeding with basic info:', error);
        // Fallback to basic clinical text
        enhancedClinicalText = `
Patient ID: ${patientId}
Symptoms: ${patientInput.symptoms}
Medical History: ${patientInput.medical_history || 'None provided'}
Current Medications: ${patientInput.current_medications || 'None provided'}
Allergies: ${patientInput.allergies || 'None provided'}`.trim();
      }
    } else {
      // Create clinical text from patient input (original behavior)
      enhancedClinicalText = `
Patient Information:
Age: ${patientInput.age}
Gender: ${patientInput.gender}

Symptoms: ${patientInput.symptoms}

Medical History: ${patientInput.medical_history || 'None provided'}

Current Medications: ${patientInput.current_medications || 'None provided'}

Allergies: ${patientInput.allergies || 'None provided'}

Vital Signs:
${patientInput.vital_signs?.temperature ? `Temperature: ${patientInput.vital_signs.temperature}°F` : ''}
${patientInput.vital_signs?.blood_pressure ? `Blood Pressure: ${patientInput.vital_signs.blood_pressure}` : ''}
${patientInput.vital_signs?.heart_rate ? `Heart Rate: ${patientInput.vital_signs.heart_rate} bpm` : ''}
${patientInput.vital_signs?.respiratory_rate ? `Respiratory Rate: ${patientInput.vital_signs.respiratory_rate}` : ''}
${patientInput.vital_signs?.oxygen_saturation ? `Oxygen Saturation: ${patientInput.vital_signs.oxygen_saturation}%` : ''}`.trim();
    }

// Add the clinical text to form data
formData.append('clinical_text', enhancedClinicalText);
    // Add FHIR data if provided
    if (patientInput.fhir_data) {
      formData.append('fhir_data', patientInput.fhir_data);
    }

    // Add media files with correct field names
    if (files && files.length > 0) {
      files.forEach((file) => {
        // Determine if it's a video or image based on MIME type
        if (file.type.startsWith('video/')) {
          formData.append('videos', file);
        } else {
          formData.append('images', file);
        }
      });
    }

    // Use patient-specific endpoint if patientId is provided
    const endpoint = patientId ? `/api/patients/${patientId}/diagnose` : '/diagnose';
    
    console.log('🌐 Making API request to:', API_BASE_URL + endpoint)
    console.log('📋 FormData contents:')
    const formDataEntries = Array.from(formData.entries());
    formDataEntries.forEach(([key, value]) => {
      if (value instanceof File) {
        console.log(`  - ${key}: File(${value.name}, ${value.size} bytes, ${value.type})`)
      } else {
        console.log(`  - ${key}:`, typeof value === 'string' ? value.substring(0, 100) + '...' : value)
      }
    })
    
    try {
      const response = await api.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('✅ API response received:', response.status, response.statusText)
      console.log('📄 Response data:', response.data)
      
      return response.data;
    } catch (apiError) {
      console.error('❌ API request failed:', apiError)
      console.error('❌ API error details:')
      if (axios.isAxiosError(apiError)) {
        console.error('  - Status:', apiError.response?.status)
        console.error('  - Status text:', apiError.response?.statusText)
        console.error('  - Response data:', apiError.response?.data)
        console.error('  - Request URL:', apiError.config?.url)
        console.error('  - Request method:', apiError.config?.method)
      }
      throw apiError;
    }
  }

  static async getProcessingStatus(sessionId: string): Promise<ProcessingStatus> {
    const response = await api.get(`/status/${sessionId}`);
    return response.data;
  }

  static async getDiagnosisResults(sessionId: string): Promise<DiagnosisResult> {
    const response = await api.get(`/results/${sessionId}`);
    const backendData = response.data;

    // Extract explanations from backend response (dynamic, not hardcoded)
    const extractExplanations = (data: any): string[] => {
      const explanations: string[] = [];
      
      // Helper function to convert explanation to string (handles both string and object formats)
      const explanationToString = (exp: any): string => {
        if (typeof exp === 'string') {
          return exp;
        } else if (exp && typeof exp === 'object' && exp.text) {
          // Handle structured explanation object: {text, confidence, verified, source}
          const confidence = exp.confidence ? `${(exp.confidence * 100).toFixed(1)}% confidence` : '0.0% confidence';
          const verifiedText = exp.verified ? '✓ FOL Verified' : '✗ Unverified';
          // Format: text [XX.X% confidence] [✓/✗ Status]
          return `${exp.text} [${confidence}] [${verifiedText}]`;
        }
        return String(exp || '');
      };
      
      // 1. First priority: Use explanations directly from backend if available
      if (data.ui_data?.explanations && Array.isArray(data.ui_data.explanations)) {
        console.log('📝 Using ui_data explanations from backend:', data.ui_data.explanations.length, 'explanations');
        explanations.push(...data.ui_data.explanations.map(explanationToString));
      }
      
      // 2. Second priority: Use clinical impression and reasoning paths
      if (data.diagnosis?.clinical_impression) {
        console.log('📝 Adding clinical impression from backend');
        explanations.push(`**Clinical Impression**: ${data.diagnosis.clinical_impression}`);
      }
      
      if (data.diagnosis?.reasoning_paths && Array.isArray(data.diagnosis.reasoning_paths)) {
        console.log('📝 Adding reasoning paths from backend:', data.diagnosis.reasoning_paths.length, 'paths');
        data.diagnosis.reasoning_paths.forEach((path: string, index: number) => {
          explanations.push(`**Clinical Reasoning ${index + 1}**: ${path}`);
        });
      }
      
      // 3. Third priority: Use verification summaries
      if (data.fol_verification?.verification_summary) {
        console.log('📝 Adding FOL verification summary');
        explanations.push(`**FOL Verification**: ${data.fol_verification.verification_summary}`);
      }
      
      if (data.enhanced_verification?.evidence_summary) {
        console.log('📝 Adding enhanced verification evidence summary');
        explanations.push(`**Evidence Summary**: ${data.enhanced_verification.evidence_summary}`);
      }
      
      if (data.online_verification?.verification_summary) {
        console.log('📝 Adding online verification summary');
        explanations.push(`**Medical Verification**: ${data.online_verification.verification_summary}`);
      }
      
      // 4. Fourth priority: Use detailed verification results
      if (data.fol_verification?.detailed_verification && Array.isArray(data.fol_verification.detailed_verification)) {
        console.log('📝 Adding detailed verification results:', data.fol_verification.detailed_verification.length, 'results');
        data.fol_verification.detailed_verification.slice(0, 2).forEach((verification: any, index: number) => {
          if (verification.explanation || verification.reasoning) {
            explanations.push(`**Detailed Analysis ${index + 1}**: ${verification.explanation || verification.reasoning}`);
          }
        });
      }
      
      // 5. Fifth priority: Use medical reasoning summary
      if (data.fol_verification?.medical_reasoning_summary) {
        console.log('📝 Adding medical reasoning summary');
        explanations.push(`**Medical Reasoning**: ${data.fol_verification.medical_reasoning_summary}`);
      }
      
      // 6. Last resort: Generate a basic explanation only if no backend explanations exist
      if (explanations.length === 0 && data.diagnosis?.primary_diagnosis) {
        console.log('📝 No backend explanations found, creating basic fallback for:', data.diagnosis.primary_diagnosis);
        const basicExplanation = `Based on the clinical presentation and analysis, the patient's symptoms are consistent with **${data.diagnosis.primary_diagnosis}**. ${
          data.diagnosis.confidence_score 
            ? `This diagnosis has a confidence score of ${(data.diagnosis.confidence_score * 100).toFixed(1)}%.`
            : ''
        }`;
        explanations.push(basicExplanation);
      }
      
      console.log('📝 Final explanations count:', explanations.length);
      
      // Return top 4 explanations, formatted and trimmed
      // Handle both string and object formats safely
      return explanations
        .filter(exp => {
          if (typeof exp === 'string') {
            return exp && exp.trim().length > 0;
          } else if (exp && typeof exp === 'object') {
            // If it's an object, it was already converted to string above
            return String(exp).trim().length > 0;
          }
          return false;
        })
        .slice(0, 4)
        .map(exp => typeof exp === 'string' ? exp.trim() : String(exp).trim());
    };
    
    // Extract confidence scores from various sources
    const extractConfidenceScores = (data: any) => {
      const scores: Record<string, number> = {};
      
      if (data.diagnosis?.confidence_score) {
        scores.primary_diagnosis = data.diagnosis.confidence_score;
      }
      if (data.fol_verification?.overall_confidence) {
        scores.fol_verification = data.fol_verification.overall_confidence;
      }
      if (data.enhanced_verification?.overall_confidence) {
        scores.enhanced_verification = data.enhanced_verification.overall_confidence;
      }
      if (data.enhanced_verification?.textbook_confidence) {
        scores.textbook_verification = data.enhanced_verification.textbook_confidence;
      }
      if (data.enhanced_verification?.online_confidence) {
        scores.online_verification = data.enhanced_verification.online_confidence;
      }
      if (data.online_verification?.confidence_score) {
        scores.online_search = data.online_verification.confidence_score;
      }
      
      return scores;
    };

    // Create DiagnosisResultsUI for rendering
    const createDiagnosisResultsUI = (data: any): DiagnosisResultUI => {
      return {
        explanations: extractExplanations(data),
        confidenceScores: extractConfidenceScores(data),
        verificationStatus: {
          fol_verified: data.fol_verification?.verified || false,
          enhanced_verified: data.enhanced_verification?.overall_status === 'verified',
          online_verified: data.online_verification?.verification_status === 'verified'
        },
        sources: {
          textbook_references: data.enhanced_verification?.textbook_references || [],
          online_sources: data.online_verification?.sources || [],
          total_sources: (data.enhanced_verification?.sources_count || 0) + 
                        (data.online_verification?.sources?.length || 0)
        }
      };
    };

    // Transform backend response to match frontend DiagnosisResult interface
    const transformedData: DiagnosisResult = {
      session_id: backendData.session_id || sessionId,
      primary_diagnosis: {
        condition: backendData.diagnosis?.primary_diagnosis || 'Unknown condition',
        confidence: backendData.diagnosis?.confidence_score || 0,
        icd_code: backendData.diagnosis?.icd_code || '',
        description: backendData.diagnosis?.clinical_impression || 'No description available',
        clinical_impression: backendData.diagnosis?.clinical_impression,
        top_diagnoses: backendData.diagnosis?.top_diagnoses || [],
        reasoning_paths: backendData.diagnosis?.reasoning_paths || [],
        clinical_recommendations: backendData.diagnosis?.clinical_recommendations || [],
        data_quality_assessment: backendData.diagnosis?.data_quality_assessment,
        data_utilization: backendData.diagnosis?.data_utilization || []
      },
      differential_diagnoses: (backendData.diagnosis?.top_diagnoses || []).map((diag: any) => ({
        condition: diag.diagnosis || diag.condition || 'Unknown',
        confidence: diag.confidence || 0,
        reasoning: diag.reasoning || 'No reasoning provided',
        icd_code: diag.icd_code || ''
      })),
      recommended_tests: backendData.diagnosis?.clinical_recommendations || [],
      treatment_recommendations: {
        recommended_tests: backendData.diagnosis?.recommended_tests || backendData.diagnosis?.clinical_recommendations || [],
        treatment_options: backendData.diagnosis?.treatment_options || backendData.diagnosis?.clinical_recommendations || [],
      },
      urgency_level: this.determineUrgencyLevel(backendData.diagnosis?.confidence_score || 0),
      fol_verification: {
        // New structure
        status: backendData.fol_verification?.status || 'UNVERIFIED',
        overall_confidence: backendData.fol_verification?.overall_confidence || 0,
        verification_summary: backendData.fol_verification?.verification_summary || 'No FOL verification available',
        verified_explanations: backendData.fol_verification?.verified_explanations || 0,
        total_explanations: backendData.fol_verification?.total_explanations || 0,
        success_rate: backendData.fol_verification?.success_rate || 0,
        detailed_verification: backendData.fol_verification?.detailed_verification || [],
        error: backendData.fol_verification?.error,
        
        // Legacy fields for backward compatibility
        verified: backendData.fol_verification?.verified || (backendData.fol_verification?.status === 'VERIFIED' || backendData.fol_verification?.status === 'FULLY_VERIFIED'),
        predicates: backendData.fol_verification?.predicates || [],
        explanation: backendData.fol_verification?.explanation || backendData.fol_verification?.verification_summary || 'No FOL verification available',
        total_predicates: backendData.fol_verification?.total_predicates,
        verified_predicates: backendData.fol_verification?.verified_predicates,
        verification_time: backendData.fol_verification?.verification_time,
        detailed_results: backendData.fol_verification?.detailed_results,
        medical_reasoning_summary: backendData.fol_verification?.medical_reasoning_summary,
        disease_probabilities: backendData.fol_verification?.disease_probabilities,
        clinical_recommendations: backendData.fol_verification?.clinical_recommendations
      },
      advanced_fol_extraction: backendData.advanced_fol_extraction ? {
        extracted_predicates: backendData.advanced_fol_extraction.extracted_predicates || [],
        nlp_entities: backendData.advanced_fol_extraction.nlp_entities || [],
        logic_rules: backendData.advanced_fol_extraction.logic_rules || [],
        confidence_scores: backendData.advanced_fol_extraction.confidence_scores || {},
        extraction_method: backendData.advanced_fol_extraction.extraction_method || 'NLP',
        predicate_count: backendData.advanced_fol_extraction.predicate_count || 0,
        entity_count: backendData.advanced_fol_extraction.entity_count || 0
      } : undefined,
      ontology_analysis: backendData.ontology_analysis ? {
        diagnosis_term: backendData.ontology_analysis.diagnosis_term,
        normalized_diagnosis: backendData.ontology_analysis.normalized_diagnosis,
        diagnosis_cui: backendData.ontology_analysis.diagnosis_cui,
        diagnosis_definition: backendData.ontology_analysis.diagnosis_definition,
        extracted_terms: backendData.ontology_analysis.extracted_terms || [],
        normalized_terms: backendData.ontology_analysis.normalized_terms || [],
        synonyms: backendData.ontology_analysis.synonyms || [],
        synonym_count: backendData.ontology_analysis.synonym_count || 0,
        ontology_source: backendData.ontology_analysis.ontology_source,
        confidence: backendData.ontology_analysis.confidence || 0,
        term_count: backendData.ontology_analysis.term_count || 0
      } : undefined,
      enhanced_verification: backendData.enhanced_verification ? {
        overall_status: backendData.enhanced_verification.overall_status,
        overall_confidence: backendData.enhanced_verification.overall_confidence,
        evidence_strength: backendData.enhanced_verification.evidence_strength,
        consensus_analysis: backendData.enhanced_verification.consensus_analysis,
        clinical_recommendations: backendData.enhanced_verification.clinical_recommendations || [],
        evidence_summary: backendData.enhanced_verification.evidence_summary,
        sources_count: backendData.enhanced_verification.sources_count,
        textbook_confidence: backendData.enhanced_verification.textbook_confidence,
        textbook_references: backendData.enhanced_verification.textbook_references || [],
        online_confidence: backendData.enhanced_verification.online_confidence,
        online_sources: backendData.enhanced_verification.online_sources || [],
        verification_timestamp: backendData.enhanced_verification.verification_timestamp,
        contradictions: backendData.enhanced_verification.contradictions || []
      } : undefined,
      online_verification: backendData.online_verification ? {
        search_strategies_used: backendData.online_verification.search_strategies_used || true,
        bibliography: backendData.online_verification.bibliography || true,
        verification_status: backendData.online_verification.verification_status,
        confidence_score: backendData.online_verification.confidence_score,
        sources: backendData.online_verification.sources || [],
        supporting_evidence: backendData.online_verification.supporting_evidence || [],
        contradicting_evidence: backendData.online_verification.contradicting_evidence || [],
        clinical_notes: backendData.online_verification.clinical_notes,
        verification_summary: backendData.online_verification.verification_summary,
        timestamp: backendData.online_verification.timestamp
      } : undefined,
      clara_results: backendData.clara_results ? {
        imaging: backendData.clara_results.imaging,
        genomics: backendData.clara_results.genomics
      } : undefined,
      confidence_metrics: {
        overall_confidence: backendData.diagnosis?.confidence_score || 0,
        data_quality: backendData.diagnosis?.data_quality_assessment?.score || 0.8,
        source_reliability: 0.85, // Default value
        model_agreement: backendData.explainability_score || 0.75
      },
      processing_time: backendData.metadata?.processing_time || 0,
      explainability_score: backendData.explainability_score,
      metadata: backendData.metadata,
      sources: this.extractSources(backendData),
      timestamp: backendData.metadata?.created_at || new Date().toISOString(),
      enhanced: backendData.enhanced,
      ui_data: createDiagnosisResultsUI(backendData),
      // Add heatmap data directly from backend (bypassing database)
      heatmap_data: backendData.heatmap_data || [],
      heatmap_visualization: backendData.heatmap_visualization || { available: false },
      image_paths: backendData.image_paths || []
    };

    return transformedData;
  }

  private static determineUrgencyLevel(confidence: number): 'low' | 'medium' | 'high' | 'critical' {
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.7) return 'medium';
    if (confidence >= 0.5) return 'low';
    return 'medium'; // Default fallback
  }

  private static extractSources(backendData: any): Array<{ title: string; url: string; relevance: number; credibility: 'excellent' | 'good' | 'fair' | 'poor' }> {
    const sources: Array<{ title: string; url: string; relevance: number; credibility: 'excellent' | 'good' | 'fair' | 'poor' }> = [];
    
    // Extract from explanations
    if (backendData.explanations) {
      backendData.explanations.forEach((exp: any, index: number) => {
        sources.push({
          title: `Medical Explanation ${index + 1}`,
          url: '#',
          relevance: exp.confidence || 0.8,
          credibility: this.mapConfidenceToCredibility(exp.confidence || 0.8)
        });
      });
    }
    
    // Extract from online verification sources
    if (backendData.online_verification?.sources) {
      backendData.online_verification.sources.forEach((source: any) => {
        sources.push({
          title: source.title || 'Medical Source',
          url: source.url || '#',
          relevance: source.confidence || 0.8,
          credibility: this.mapConfidenceToCredibility(source.confidence || 0.8)
        });
      });
    }
    
    return sources;
  }

  private static mapConfidenceToCredibility(confidence: number): 'excellent' | 'good' | 'fair' | 'poor' {
    if (confidence >= 0.9) return 'excellent';
    if (confidence >= 0.7) return 'good';
    if (confidence >= 0.5) return 'fair';
    return 'poor';
  }

  static async downloadResults(sessionId: string): Promise<{ blob: Blob; contentType: string; suggestedFilename: string }> {
    const response = await api.get(`/download/${sessionId}`, {
      responseType: 'blob',
    });
    const contentType = response.headers['content-type'];
    const isPdf = contentType === 'application/pdf';
    const extension = isPdf ? '.pdf' : '.json';
    const suggestedFilename = `diagnosis_results_${sessionId}${extension}`;
    return {
      blob: response.data,
      contentType,
      suggestedFilename
    };
  }

  static async clearSession(sessionId: string): Promise<void> {
    await api.delete(`/clear-session/${sessionId}`);
  }

  static async clearAllSessions(): Promise<void> {
    await api.delete('/clear-all-sessions');
  }

  static async getSystemHealth(): Promise<SystemHealth> {
    const response = await api.get('/api/health');
    return response.data;
  }

  // Clinical Notes API
  static async addClinicalNote(noteData: {
    patient_id: string;
    nurse_id: string;
    content: string;
    location?: string;
    shift?: string;
  }) {
    const response = await api.post('/api/concern/add-note', noteData);
    return response.data;
  }

  static async getPatientClinicalNotes(patientId: string, limit: number = 50) {
    const response = await api.get(`/api/concern/patient/${patientId}/notes?limit=${limit}`);
    return response.data;
  }

  // AR-lite Scan Note (OCR)
  static async submitClinicalNoteScan(patientId: string, file: File, options?: { nurseId?: string; location?: string; shift?: string; }) {
    console.log('🔍 API: Starting clinical note scan...');
    console.log('🔍 API: Patient ID:', patientId);
    console.log('🔍 API: File:', file.name, 'Size:', file.size);
    console.log('🔍 API: Options:', options);
    console.log('🔍 API: Backend URL:', API_BASE_URL);
    
    const formData = new FormData();
    formData.append('patient_id', patientId);
    formData.append('nurse_id', options?.nurseId || 'AR_SCANNER');
    if (options?.location) formData.append('location', options.location);
    if (options?.shift) formData.append('shift', options.shift);
    formData.append('image', file);

    console.log('🔍 API: FormData created, sending request to:', `${API_BASE_URL}/api/concern/scan-note`);

    try {
      const response = await api.post('/api/concern/scan-note', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      });
      console.log('🔍 API: Response received:', response.status, response.data);
      return response.data;
    } catch (error: any) {
      console.error('🔍 API: Request failed:', error);
      console.error('🔍 API: Error response:', error?.response?.data);
      console.error('🔍 API: Error status:', error?.response?.status);
      throw error;
    }
  }

  // Scanned Notes API
  static async getScannedNotes(patientId: string, limit: number = 50) {
    const response = await api.get(`/api/concern/scanned-notes/${patientId}?limit=${limit}`);
    return response.data;
  }

  static async getScannedNote(noteId: string) {
    const response = await api.get(`/api/concern/scanned-note/${noteId}`);
    return response.data;
  }

  static async searchScannedNotes(params: {
    patient_id?: string;
    search_text?: string;
    date_from?: string;
    date_to?: string;
    limit?: number;
  }) {
    const queryParams = new URLSearchParams();
    if (params.patient_id) queryParams.append('patient_id', params.patient_id);
    if (params.search_text) queryParams.append('search_text', params.search_text);
    if (params.date_from) queryParams.append('date_from', params.date_from);
    if (params.date_to) queryParams.append('date_to', params.date_to);
    if (params.limit) queryParams.append('limit', params.limit.toString());
    
    const response = await api.get(`/api/concern/search-scanned-notes?${queryParams.toString()}`);
    return response.data;
  }

  // Patient Visits API
  static async addPatientVisit(visitData: {
    patient_id: string;
    nurse_id: string;
    location: string;
    visit_type?: string;
    duration_minutes?: number;
    notes?: string;
  }) {
    const response = await api.post('/api/concern/add-visit', visitData);
    return response.data;
  }

  static async getPatientVisits(patientId: string, limit: number = 50) {
    const response = await api.get(`/api/concern/patient/${patientId}/visits?limit=${limit}`);
    return response.data;
  }

  // CONCERN Early Warning System
  static async calculateRiskScore(patientId: string, data?: any) {
    const response = await api.post(`/api/concern/patient/${patientId}/calculate`, data || {});
    return response.data;
  }
  
  static async updateRiskThresholds(thresholds: {
    critical?: number;
    high?: number;
    medium?: number;
    low?: number;
  }) {
    const response = await api.post('/api/concern/thresholds', thresholds);
    return response.data;
  }
  
  static async getCriticalPatients() {
    const response = await api.get('/api/concern/critical');
    return response.data;
  }
  
  static async triggerAlert(patientId: string, alertData: {
    type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    message: string;
    metadata?: any;
  }) {
    const response = await api.post(`/api/concern/patient/${patientId}/alert`, alertData);
    return response.data;
  }
  
  static async getAlerts(patientId?: string) {
    const endpoint = patientId ? `/api/concern/alerts/${patientId}` : '/api/concern/alerts';
    const response = await api.get(endpoint);
    return response.data;
  }
  
  static async acknowledgeAlert(alertId: string) {
    const response = await api.put(`/api/concern/alerts/${alertId}/acknowledge`);
    return response.data;
  }
  
  static async getRiskTrends(patientId: string, timeframe: string = '7d') {
    const response = await api.get(`/api/concern/patient/${patientId}/trends?timeframe=${timeframe}`);
    return response.data;
  }
  
  static async getBulkRiskAssessment() {
    const response = await api.get('/api/concern/bulk-assessment');
    return response.data;
  }

  static async getRealtimeMetrics(patientId: string) {
    const response = await api.get(`/api/concern/patient/${patientId}/metrics/realtime`);
    return response.data;
  }
}

export class ChatAPI {
  static async sendMessage(message: string, sessionId?: string, diagnosisSessionId?: string, images?: File[], enableExplainability: boolean = true): Promise<ChatResponse> {
    if (images && images.length > 0) {
      // If images are included, use form data
      const formData = new FormData();
      formData.append('message', message);
      
      if (sessionId) {
        formData.append('session_id', sessionId);
      }
      
      if (diagnosisSessionId) {
        formData.append('diagnosis_session_id', diagnosisSessionId);
      }

      images.forEach((image, index) => {
        formData.append(`image_${index}`, image);
      });
      
      // Add explainability parameters
      formData.append('enable_explainability', enableExplainability.toString());
      formData.append('enable_fol_verification', 'true');
      formData.append('confidence_threshold', '0.6');
      formData.append('include_reasoning_steps', 'true');
      formData.append('medical_context', 'true');

      const response = await api.post('/chat', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } else {
      // Use JSON for text-only messages
      const response = await api.post('/chat', {
        message,
        session_id: sessionId,
        diagnosis_session_id: diagnosisSessionId,
        enable_explainability: enableExplainability,
        enable_fol_verification: true,
        confidence_threshold: 0.6,
        include_reasoning_steps: true,
        medical_context: true
      });
      return response.data;
    }
  }

  static async sendVoiceMessage(audioBlob: Blob, sessionId?: string): Promise<ChatResponse> {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice_message.wav');
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const response = await api.post('/api/voice-chat', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async analyzeImage(image: File, sessionId?: string): Promise<any> {
    const formData = new FormData();
    formData.append('image', image);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const response = await api.post('/api/analyze-image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async getChatHistory(sessionId?: string): Promise<any> {
    const url = sessionId ? `/api/chat/history/${sessionId}` : '/api/chat/history';
    const response = await api.get(url);
    return response.data;
  }

  static async clearChatHistory(sessionId?: string): Promise<any> {
    const url = sessionId ? `/api/chat/clear/${sessionId}` : '/api/chat/clear';
    const response = await api.delete(url);
    return response.data;
  }
}

export class UMLSApi {
  static async searchConcepts(query: string, searchType: string = 'exact', maxResults: number = 20): Promise<UMLSSearchResult> {
    const response = await api.post('/api/umls/search', {
      query,
      search_type: searchType,
      max_results: maxResults,
    });
    return response.data;
  }

  static async getConceptDetails(cui: string): Promise<any> {
    const response = await api.get(`/api/umls/concept-details/${cui}`);
    return response.data;
  }

  static async lookupCode(code: string): Promise<any> {
    const response = await api.post('/api/umls/lookup-code', {
      code,
    });
    return response.data;
  }

  static async lookupCodesFromFile(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/api/umls/lookup-codes-file', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async normalizeTerm(term: string): Promise<string> {
    const response = await api.post('/ontology/normalize', {
      terms: [term],
    });
    return response.data.normalized_terms?.[0] || term;
  }
}

export class SystemApi {
  static async getHealthStatus(): Promise<any> {
    const response = await api.get('/api/health');
    return response.data;
  }

  static async getSystemStatus(): Promise<any> {
    const response = await api.get('/api/system-status');
    return response.data;
  }

  static async getDatabaseStats(): Promise<any> {
    const response = await api.get('/api/database-stats');
    return response.data;
  }

  static async getPerformanceMetrics(): Promise<any> {
    const response = await api.get('/api/performance');
    return response.data;
  }

  static async clearCache(): Promise<any> {
    const response = await api.post('/api/clear-cache');
    return response.data;
  }

  static async exportData(format: string = 'json'): Promise<any> {
    const response = await api.get(`/api/export-data?format=${format}`, {
      responseType: 'blob',
    });
    return response.data;
  }
}

export class AudioAPI {
  static async transcribeAudio(audioBlob: Blob): Promise<{ text: string }> {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    const response = await api.post('/audio/transcribe', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }
}

// Utility function to handle API errors
export const handleApiError = (error: any): string => {
  if (error.response) {
    // Server responded with error status
    return error.response.data?.message || error.response.data?.error || 'Server error occurred';
  } else if (error.request) {
    // Request was made but no response received
    return 'Network error - please check your connection';
  } else {
    // Something else happened
    return error.message || 'An unexpected error occurred';
  }
};

export { api };
