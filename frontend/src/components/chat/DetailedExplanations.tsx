'use client';

import { useState } from 'react';
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import type { DetailedExplanation, FOLVerificationDetails, MedicalReasoning } from '@/types';
import { MedicalMarkdownText } from '@/utils/markdown';
import { ConfidenceVisualization } from './ConfidenceVisualization';
import { ReasoningBreakdown } from './ReasoningBreakdown';

interface DetailedExplanationsProps {
  explanations: DetailedExplanation[];
  folVerification?: FOLVerificationDetails;
  medicalReasoning?: MedicalReasoning;
  confidenceMetrics?: {
    confidence: number;
    fol_verified: boolean;
    explainability: number;
    response_time: number;
    medical_accuracy?: number;
    source_reliability?: number;
  };
  reasoningSteps?: Array<{
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
}

export function DetailedExplanations({ 
  explanations, 
  folVerification, 
  medicalReasoning,
  confidenceMetrics,
  reasoningSteps
}: DetailedExplanationsProps) {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    explanations: true,
    folVerification: false,
    predicates: false,
    evidence: false,
    reasoning: false,
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'VERIFIED': return 'text-green-600 bg-green-100';
      case 'UNVERIFIED': return 'text-red-600 bg-red-100';
      case 'PENDING': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="mt-4 space-y-4">
      {/* Enhanced Confidence Metrics */}
      {confidenceMetrics && (
        <ConfidenceVisualization 
          metrics={confidenceMetrics}
          compact={false}
          showDetails={true}
        />
      )}

      {/* Step-by-step Reasoning Breakdown */}
      {reasoningSteps && reasoningSteps.length > 0 && (
        <ReasoningBreakdown
          steps={reasoningSteps}
          overallConfidence={confidenceMetrics?.confidence || 0.5}
          verificationStatus={confidenceMetrics?.fol_verified ? 'verified' : 'unverified'}
          processingTime={confidenceMetrics?.response_time || 0}
        />
      )}

      {/* Detailed Explanations Section */}
      {explanations && explanations.length > 0 && (
        <div className="border rounded-lg overflow-hidden">
          <button
            onClick={() => toggleSection('explanations')}
            className="w-full px-4 py-3 bg-blue-50 hover:bg-blue-100 flex items-center justify-between transition-colors"
          >
            <div className="flex items-center space-x-2">
              <span className="text-blue-600 font-medium">🧠 Detailed Medical Explanations</span>
              <span className="text-sm text-blue-500">({explanations.length} perspectives)</span>
            </div>
            {expandedSections.explanations ? (
              <ChevronDownIcon className="w-5 h-5 text-blue-600" />
            ) : (
              <ChevronRightIcon className="w-5 h-5 text-blue-600" />
            )}
          </button>

          {expandedSections.explanations && (
            <div className="p-4 bg-white border-t">
              <div className="grid gap-4">
                {explanations.map((explanation, index) => (
                  <div
                    key={index}
                    className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
                  >
                    <div className="flex items-start space-x-3">
                      <span className="text-2xl flex-shrink-0">{explanation.icon}</span>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold text-gray-800">{explanation.type}</h4>
                          <span className={`text-sm font-medium ${getConfidenceColor(explanation.confidence)}`}>
                            {(explanation.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <MedicalMarkdownText 
                          showConfidence={false}
                          className="text-gray-700"
                        >
                          {explanation.content}
                        </MedicalMarkdownText>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* FOL Verification Section */}
      {folVerification && (
        <div className="border rounded-lg overflow-hidden">
          <button
            onClick={() => toggleSection('folVerification')}
            className="w-full px-4 py-3 bg-purple-50 hover:bg-purple-100 flex items-center justify-between transition-colors"
          >
            <div className="flex items-center space-x-2">
              <span className="text-purple-600 font-medium">🔍 FOL Logic Verification</span>
              <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(folVerification.status)}`}>
                {folVerification.status}
              </span>
            </div>
            {expandedSections.folVerification ? (
              <ChevronDownIcon className="w-5 h-5 text-purple-600" />
            ) : (
              <ChevronRightIcon className="w-5 h-5 text-purple-600" />
            )}
          </button>

          {expandedSections.folVerification && (
            <div className="p-4 bg-white border-t space-y-4">
              {/* Verification Summary */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-3 bg-gray-50 rounded-lg">
                <div className="text-center">
                  <div className="text-lg font-bold text-purple-600">{folVerification.verified_count}</div>
                  <div className="text-xs text-gray-600">Verified</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-600">{folVerification.total_predicates}</div>
                  <div className="text-xs text-gray-600">Total</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-green-600">{folVerification.verification_score}%</div>
                  <div className="text-xs text-gray-600">Score</div>
                </div>
                <div className="text-center">
                  <div className={`text-lg font-bold ${getStatusColor(folVerification.status).split(' ')[0]}`}>
                    {folVerification.status}
                  </div>
                  <div className="text-xs text-gray-600">Status</div>
                </div>
              </div>

              {/* Reasoning */}
              <div className="p-3 bg-blue-50 rounded-lg">
                <MedicalMarkdownText className="text-sm text-blue-800">
                  {folVerification.reasoning}
                </MedicalMarkdownText>
              </div>

              {/* Predicate Breakdown */}
              {folVerification.predicate_breakdown && folVerification.predicate_breakdown.length > 0 && (
                <div>
                  <button
                    onClick={() => toggleSection('predicates')}
                    className="flex items-center space-x-2 text-purple-600 hover:text-purple-800 font-medium mb-3"
                  >
                    {expandedSections.predicates ? (
                      <ChevronDownIcon className="w-4 h-4" />
                    ) : (
                      <ChevronRightIcon className="w-4 h-4" />
                    )}
                    <span>View Predicate Details ({folVerification.predicate_breakdown.length})</span>
                  </button>

                  {expandedSections.predicates && (
                    <div className="space-y-2">
                      {folVerification.predicate_breakdown.map((predicate) => (
                        <div
                          key={predicate.id}
                          className={`p-3 rounded-lg border-l-4 ${
                            predicate.verified 
                              ? 'border-green-500 bg-green-50' 
                              : 'border-red-500 bg-red-50'
                          }`}
                        >
                          <div className="flex items-start justify-between mb-2">
                            <span className="font-medium text-sm">#{predicate.id}</span>
                            <div className="flex items-center space-x-2">
                              <span className={`text-xs font-medium ${getConfidenceColor(predicate.confidence)}`}>
                                {(predicate.confidence * 100).toFixed(1)}%
                              </span>
                              <span className={`px-2 py-1 text-xs rounded ${getStatusColor(predicate.status)}`}>
                                {predicate.verified ? '✓' : '✗'}
                              </span>
                            </div>
                          </div>
                          <MedicalMarkdownText className="text-sm text-gray-700 mb-2">
                            {predicate.predicate}
                          </MedicalMarkdownText>
                          <MedicalMarkdownText className="text-xs text-gray-600">
                            {predicate.reasoning}
                          </MedicalMarkdownText>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Verification Evidence */}
              {folVerification.verification_evidence && (
                <div>
                  <button
                    onClick={() => toggleSection('evidence')}
                    className="flex items-center space-x-2 text-purple-600 hover:text-purple-800 font-medium mb-3"
                  >
                    {expandedSections.evidence ? (
                      <ChevronDownIcon className="w-4 h-4" />
                    ) : (
                      <ChevronRightIcon className="w-4 h-4" />
                    )}
                    <span>Supporting Evidence</span>
                  </button>

                  {expandedSections.evidence && (
                    <div className="space-y-3">
                      <div>
                        <h5 className="text-sm font-semibold text-gray-700 mb-2">Supporting Evidence:</h5>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {folVerification.verification_evidence.supporting_evidence.map((evidence, idx) => (
                            <li key={idx} className="flex items-start space-x-2">
                              <span className="text-green-500 flex-shrink-0">✓</span>
                              <span>{evidence}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h5 className="text-sm font-semibold text-gray-700 mb-2">Verification Methods:</h5>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {folVerification.verification_evidence.verification_methods.map((method, idx) => (
                            <li key={idx} className="flex items-start space-x-2">
                              <span className="text-blue-500 flex-shrink-0">🔬</span>
                              <span>{method}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h5 className="text-sm font-semibold text-gray-700 mb-2">Confidence Factors:</h5>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {folVerification.verification_evidence.confidence_factors.map((factor, idx) => (
                            <li key={idx} className="flex items-start space-x-2">
                              <span className="text-yellow-500 flex-shrink-0">📊</span>
                              <span>{factor}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Medical Reasoning Section */}
      {medicalReasoning && (
        <div className="border rounded-lg overflow-hidden">
          <button
            onClick={() => toggleSection('reasoning')}
            className="w-full px-4 py-3 bg-green-50 hover:bg-green-100 flex items-center justify-between transition-colors"
          >
            <div className="flex items-center space-x-2">
              <span className="text-green-600 font-medium">🧪 Medical Reasoning Analysis</span>
            </div>
            {expandedSections.reasoning ? (
              <ChevronDownIcon className="w-5 h-5 text-green-600" />
            ) : (
              <ChevronRightIcon className="w-5 h-5 text-green-600" />
            )}
          </button>

          {expandedSections.reasoning && (
            <div className="p-4 bg-white border-t space-y-3">
              <div className="p-3 bg-green-50 rounded-lg">
                <h5 className="font-semibold text-green-800 mb-2">Primary Reasoning:</h5>
                <MedicalMarkdownText className="text-green-700 text-sm">
                  {medicalReasoning.primary_reasoning}
                </MedicalMarkdownText>
              </div>

              <div className="p-3 bg-blue-50 rounded-lg">
                <h5 className="font-semibold text-blue-800 mb-2">Context Analysis:</h5>
                <MedicalMarkdownText className="text-blue-700 text-sm">
                  {medicalReasoning.context_analysis}
                </MedicalMarkdownText>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-yellow-50 rounded-lg text-center">
                  <div className="text-lg font-bold text-yellow-600">
                    {(medicalReasoning.clinical_relevance * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-yellow-700">Clinical Relevance</div>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg text-center">
                  <div className="text-lg font-bold text-purple-600">
                    {(medicalReasoning.evidence_strength * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-purple-700">Evidence Strength</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
