import { ReactNode } from 'react';

export interface PatientInput {
  symptoms: string;
  medical_history: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  current_medications: string;
  allergies: string;
  vital_signs: {
    temperature?: number;
    blood_pressure?: string;
    heart_rate?: number;
    respiratory_rate?: number;
    oxygen_saturation?: number;
  };
  images?: File[];
  fhir_data?: string;
  clinical_text?: string; // Add this for backend compatibility
  patient_id?: string;
  anonymize?: boolean;
  clara_options?: {
    dicom_processing: boolean;
    '3d_reconstruction': boolean;
    image_segmentation: boolean;
    genomic_analysis: boolean;
    variant_calling: boolean;
    multi_omics: boolean;
  };
}

export interface DiagnosisResultUI {
  explanations: string[];
  confidenceScores: Record<string, number>;
  verificationStatus: {
    fol_verified: boolean;
    enhanced_verified: boolean;
    online_verified: boolean;
  };
  sources: {
    textbook_references: Array<{
      title?: string;
      page_number?: number;
      chapter?: string;
      section?: string;
      relevant_quote?: string;
      relevance_score?: number;
      confidence_score?: number;
      source_citation?: string;
    }>;
    online_sources: Array<{
      title: string;
      url: string;
      domain: string;
      content_snippet: string;
      relevance_score: number;
      credibility_score: number;
      citation_format: string;
      source_type: string;
      date_accessed: string;
    }>;
    total_sources: number;
  };
}

export interface DiagnosisResult {
  session_id: string;
  primary_diagnosis: {
    condition: string;
    confidence: number;
    icd_code?: string;
    description: string;
    clinical_impression?: string;
    top_diagnoses?: Array<{
      diagnosis: string;
      confidence: number;
      reasoning?: string;
    }>;
    reasoning_paths?: string[];
    clinical_recommendations?: string[];
    data_quality_assessment?: {
      score: number;
      quality_score?: number;
    };
    data_utilization?: string[];
  };
  differential_diagnoses: Array<{
    condition: string;
    confidence: number;
    reasoning: string;
    icd_code?: string;
  }>;
  recommended_tests: string[];
  treatment_recommendations: {
    recommended_tests?: string[];
    treatment_options?: string[];
  };
  urgency_level: 'low' | 'medium' | 'high' | 'critical';
  fol_verification: {
    status: 'VERIFIED' | 'UNVERIFIED' | 'FAILED' | 'FULLY_VERIFIED';
    overall_confidence: number;
    verification_summary: string;
    verified_explanations: number;
    total_explanations: number;
    success_rate: number;
    detailed_verification?: Array<{
      explanation_index: number;
      explanation_id: string;
      fol_report?: {
        total_predicates: number;
        verified_predicates: number;
        failed_predicates: number;
        overall_confidence: number;
        verification_time: number;
        detailed_results: Array<{
          predicate_index: number;
          predicate_string: string;
          verification_status: 'VERIFIED' | 'FAILED';
          confidence_level: string;
          reasoning: string;
          evaluation_method: string;
          clinical_significance?: string;
        }>;
        medical_reasoning_summary: string;
        disease_probabilities: Record<string, number>;
        clinical_recommendations: string[];
      };
      verified: boolean;
      confidence: number;
      error?: string;
    }>;
    error?: string;
    // Legacy fields for backward compatibility
    verified?: boolean;
    predicates?: string[];
    explanation?: string;
    total_predicates?: number;
    verified_predicates?: number;
    verification_time?: number;
    detailed_results?: any;
    medical_reasoning_summary?: string;
    disease_probabilities?: any;
    clinical_recommendations?: string[];
  };
  advanced_fol_extraction?: {
    extracted_predicates?: string[];
    nlp_entities?: any[];
    logic_rules?: any[];
    confidence_scores?: Record<string, number>;
    extraction_method?: string;
    predicate_count?: number;
    entity_count?: number;
  };
  ontology_analysis?: {
    diagnosis_term?: string;
    normalized_diagnosis?: string;
    diagnosis_cui?: string;
    diagnosis_definition?: string;
    extracted_terms?: string[];
    normalized_terms?: string[];
    synonyms?: string[];
    synonym_count?: number;
    ontology_source?: string;
    confidence?: number;
    term_count?: number;
    // Enhanced UMLS, SNOMED, ICD-10 mappings
    umls_mapping?: {
      cui: string;
      name: string;
      source: string;
      confidence: number;
      details?: {
        definition?: string;
        semantic_types?: Array<{
          name: string;
          uri?: string;
        }>;
        atoms?: any[];
        relations?: any[];
      };
      search_term?: string;
      normalized_term?: string;
      retrieved_at?: string;
    };
    snomed_mapping?: {
      concept_id: string;
      preferred_term?: string;
      name?: string;
      confidence: number;
      source?: string;
      retrieved_at?: string;
    };
    icd10_mapping?: {
      code: string;
      description?: string;
      name?: string;
      confidence: number;
      source?: string;
      retrieved_at?: string;
    };
    best_match?: {
      source: string;
      confidence: number;
      primary_code: string;
      primary_name: string;
      data?: any;
    };
    mapping_completeness?: {
      sources_found: number;
      total_sources: number;
      average_confidence: number;
      completeness_percentage: number;
      coverage_analysis?: string;
    };
  };
  enhanced_verification?: {
    overall_status?: string;
    overall_confidence?: number;
    evidence_strength?: string;
    consensus_analysis?: string;
    clinical_recommendations?: string[];
    evidence_summary?: string;
    sources_count?: number;
    textbook_confidence?: number;
    textbook_references?: Array<{
      title?: string;
      page_number?: number;
      chapter?: string;
      section?: string;
      relevant_quote?: string;
      relevance_score?: number;
      confidence_score?: number;
      source_citation?: string;
    }>;
    online_confidence?: number;
    online_sources?: Array<{
      title?: string;
      url?: string;
      source_type?: string;
      reliability_score?: number;
      relevant_excerpt?: string;
      publication_date?: string;
      authors?: string;
    }>;
    verification_timestamp?: string;
    contradictions?: any[];
  };
  online_verification?: {
    search_strategies_used?: string[];
    bibliography?: string[];
    verification_status?: string;
    confidence_score?: number;
    sources?: Array<{
      [x: string]: ReactNode;
      title: string;
      url: string;
      domain: string;
      content_snippet: string;
      relevance_score: number;
      credibility_score: number;
      citation_format: string;
      source_type: string;
      date_accessed: string;
    }>;
    supporting_evidence?: string[];
    contradicting_evidence?: string[];
    clinical_notes?: string;
    verification_summary?: string;
    timestamp?: string;
  };
  clara_results?: {
    imaging?: {
      dicom_processed?: boolean;
      enhancement_applied?: boolean;
      volume_data?: any;
      segmentation_data?: any;
    };
    genomics?: {
      analysis_completed?: boolean;
      gpu_accelerated?: boolean;
      quality_metrics?: any;
      variants?: any;
      integration?: any;
    };
  };
  confidence_metrics: {
    overall_confidence: number;
    data_quality: number;
    source_reliability: number;
    model_agreement: number;
  };
  processing_time: number;
  explainability_score?: number;
  metadata?: {
    created_at?: string;
    completed_at?: string;
    anonymized?: boolean;
    clara_features_used?: string[];
    clara_available?: boolean;
    enhanced_processing?: boolean;
  };
  sources: Array<{
    title: string;
    url: string;
    relevance: number;
    credibility: 'excellent' | 'good' | 'fair' | 'poor';
  }>;
  timestamp: string;
  enhanced?: boolean;
  ui_data?: DiagnosisResultUI;
  image_paths?: string[];
  heatmap_visualization?: {
    available: boolean;
    total_images?: number;
    successful_heatmaps?: number;
    model_type?: string;
    error?: string;
  };
  heatmap_data?: Array<{
    success: boolean;
    image_file?: string;
    error?: string;
    analysis?: {
      predicted_class?: string;
      confidence_score?: number;
      processing_time?: number;
      activation_regions_count?: number;
    };
    visualizations?: {
      heatmap_image?: string;
      overlay_image?: string;
      volume_image?: string;
    };
    predictions?: Array<{
      class: string;
      confidence: number;
      probability: number;
    }>;
    activation_regions?: Array<{
      id: number;
      slice?: number;
      bbox_2d?: number[];
      area?: number;
      max_activation?: number;
      mean_activation?: number;
      centroid?: number[];
    }>;
    medical_interpretation?: {
      primary_finding?: string;
      confidence_level?: string;
      attention_areas?: number;
      clinical_notes?: string[];
    };
  }>;
}

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: string;
  metrics?: {
    confidence: number;
    fol_verified: boolean;
    explainability: number;
    response_time: number;
    medical_accuracy?: number;
    source_reliability?: number;
  };
  images?: string[];
  detailed_explanations?: DetailedExplanation[];
  fol_verification?: FOLVerificationDetails;
  medical_reasoning?: MedicalReasoning;
  reasoning_steps?: Array<{
    id: number;
    step_type: 'premise' | 'inference' | 'conclusion' | 'verification';
    content: string;
    confidence: number;
    verified: boolean;
    reasoning: string;
    dependencies?: number[];
    evidence_sources?: string[];
    medical_context?: string;
  }>;
  confidence_breakdown?: {
    overall: number;
    reasoning: number;
    evidence: number;
    medical_accuracy: number;
  };
  explainability_data?: {
    reasoning_clarity: number;
    step_by_step_available: boolean;
    confidence_intervals: number[];
    verification_methods: string[];
  };
}

export interface DetailedExplanation {
  type: string;
  content: string;
  confidence: number;
  icon: string;
}

export interface FOLVerificationDetails {
  total_predicates: number;
  verified_count: number;
  verification_score: number;
  status: 'VERIFIED' | 'UNVERIFIED' | 'PENDING';
  reasoning: string;
  predicate_breakdown?: PredicateDetail[];
  verification_evidence?: VerificationEvidence;
  medical_reasoning_summary?: string;
}

export interface PredicateDetail {
  id: number;
  predicate: string;
  verified: boolean;
  confidence: number;
  reasoning: string;
  medical_context: string;
  status: 'VERIFIED' | 'UNVERIFIED' | 'PENDING';
}

export interface VerificationEvidence {
  supporting_evidence: string[];
  verification_methods: string[];
  confidence_factors: string[];
  medical_summary?: string;
}

export interface MedicalReasoning {
  primary_reasoning: string;
  context_analysis: string;
  clinical_relevance: number;
  evidence_strength: number;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  metrics?: {
    confidence: number;
    fol_verified: boolean;
    explainability: number;
    response_time: number;
    medical_accuracy?: number;
    source_reliability?: number;
  };
  detailed_explanations?: DetailedExplanation[];
  fol_verification?: FOLVerificationDetails;
  medical_reasoning?: MedicalReasoning;
  reasoning_steps?: Array<{
    id: number;
    step_type: 'premise' | 'inference' | 'conclusion' | 'verification';
    content: string;
    confidence: number;
    verified: boolean;
    reasoning: string;
    dependencies?: number[];
    evidence_sources?: string[];
    medical_context?: string;
  }>;
  confidence_breakdown?: {
    overall: number;
    reasoning: number;
    evidence: number;
    medical_accuracy: number;
  };
  explainability_data?: {
    reasoning_clarity: number;
    step_by_step_available: boolean;
    confidence_intervals: number[];
    verification_methods: string[];
  };
}

export interface ProcessingStatus {
  session_id: string;
  status: 'processing' | 'completed' | 'error';
  progress: number;
  stage: string;
  message: string;
  estimated_time_remaining?: number;
}

export interface UMLSConcept {
  cui: string;
  preferred_name: string;
  synonyms: string[];
  semantic_types: string[];
  definitions: string[];
  sources: string[];
}

export interface UMLSSearchResult {
  concepts: UMLSConcept[];
  total_results: number;
  search_term: string;
  normalized_term?: string;
}

export interface SystemHealth {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  api_configured: boolean;
  features: {
    dynamic_ai_diagnosis: boolean;
    fol_verification: boolean;
    enhanced_explanations: boolean;
    multimodal_input: boolean;
    real_time_processing: boolean;
    chatbot_interface: boolean;
    ontology_mapping: boolean;
    parallel_processing: boolean;
    predicate_api: boolean;
  };
}

export interface AudioRecording {
  blob: Blob;
  duration: number;
  url: string;
}

export interface FileUploadProgress {
  file: File;
  progress: number;
  status: 'uploading' | 'completed' | 'error';
  error?: string;
}
