"use client"

import React, { useState } from 'react'
import { 
  ChevronDownIcon, 
  ChevronRightIcon,
  BeakerIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  LightBulbIcon,
  DocumentTextIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline'
import { MedicalMarkdownText } from '@/utils/markdown'

interface ReasoningStep {
  id: number
  step_type: 'premise' | 'inference' | 'conclusion' | 'verification'
  content: string
  confidence: number
  verified: boolean
  reasoning: string
  dependencies?: number[]
  evidence_sources?: string[]
  medical_context?: string
}

interface ReasoningBreakdownProps {
  steps: ReasoningStep[]
  overallConfidence: number
  verificationStatus: 'verified' | 'partial' | 'unverified'
  processingTime: number
  title?: string
}

export function ReasoningBreakdown({
  steps,
  overallConfidence,
  verificationStatus,
  processingTime,
  title = "AI Reasoning Process"
}: ReasoningBreakdownProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())

  const toggleStep = (stepId: number) => {
    const newExpanded = new Set(expandedSteps)
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId)
    } else {
      newExpanded.add(stepId)
    }
    setExpandedSteps(newExpanded)
  }

  const getStepIcon = (stepType: string, verified: boolean) => {
    const iconClass = verified ? "text-green-600" : "text-red-600"
    
    switch (stepType) {
      case 'premise':
        return <DocumentTextIcon className={`w-4 h-4 ${iconClass}`} />
      case 'inference':
        return <CpuChipIcon className={`w-4 h-4 ${iconClass}`} />
      case 'conclusion':
        return <LightBulbIcon className={`w-4 h-4 ${iconClass}`} />
      case 'verification':
        return <BeakerIcon className={`w-4 h-4 ${iconClass}`} />
      default:
        return <CheckCircleIcon className={`w-4 h-4 ${iconClass}`} />
    }
  }

  const getVerificationColor = (status: string) => {
    switch (status) {
      case 'verified':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'partial':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'unverified':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const verifiedSteps = steps.filter(step => step.verified).length
  const totalSteps = steps.length

  return (
    <div className="mt-4 border rounded-lg overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 bg-indigo-50 hover:bg-indigo-100 flex items-center justify-between transition-colors"
      >
        <div className="flex items-center space-x-3">
          <BeakerIcon className="w-5 h-5 text-indigo-600" />
          <div className="text-left">
            <div className="font-medium text-indigo-800">{title}</div>
            <div className="text-xs text-indigo-600">
              {verifiedSteps}/{totalSteps} steps verified • {(overallConfidence * 100).toFixed(1)}% confidence
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`px-2 py-1 rounded-full text-xs font-medium border ${getVerificationColor(verificationStatus)}`}>
            {verificationStatus.toUpperCase()}
          </div>
          {isExpanded ? (
            <ChevronDownIcon className="w-4 h-4 text-indigo-600" />
          ) : (
            <ChevronRightIcon className="w-4 h-4 text-indigo-600" />
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="bg-white border-t">
          {/* Summary Stats */}
          <div className="p-4 bg-gray-50 border-b">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-lg font-bold text-indigo-600">{totalSteps}</div>
                <div className="text-xs text-gray-600">Total Steps</div>
              </div>
              <div>
                <div className="text-lg font-bold text-green-600">{verifiedSteps}</div>
                <div className="text-xs text-gray-600">Verified</div>
              </div>
              <div>
                <div className="text-lg font-bold text-purple-600">
                  {(overallConfidence * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600">Confidence</div>
              </div>
              <div>
                <div className="text-lg font-bold text-blue-600">{processingTime.toFixed(2)}s</div>
                <div className="text-xs text-gray-600">Process Time</div>
              </div>
            </div>
          </div>

          {/* Reasoning Steps */}
          <div className="p-4 space-y-3">
            {steps.map((step, index) => (
              <div key={step.id} className="border rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleStep(step.id)}
                  className={`w-full p-3 flex items-start space-x-3 hover:bg-gray-50 transition-colors ${
                    step.verified ? 'bg-green-50' : 'bg-red-50'
                  }`}
                >
                  {/* Step Number & Icon */}
                  <div className="flex items-center space-x-2 flex-shrink-0">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      step.verified ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {index + 1}
                    </div>
                    {getStepIcon(step.step_type, step.verified)}
                  </div>

                  {/* Step Content Preview */}
                  <div className="flex-1 text-left">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium capitalize text-gray-800">
                        {step.step_type.replace('_', ' ')}
                      </span>
                      <div className="flex items-center space-x-2">
                        <span className={`text-xs font-medium ${
                          step.confidence >= 0.8 ? 'text-green-600' :
                          step.confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {(step.confidence * 100).toFixed(0)}%
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          step.verified ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>
                          {step.verified ? '✓' : '✗'}
                        </span>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 line-clamp-2">
                      {step.content}
                    </div>
                  </div>

                  <ChevronDownIcon className={`w-4 h-4 text-gray-400 transition-transform ${
                    expandedSteps.has(step.id) ? 'rotate-180' : ''
                  }`} />
                </button>

                {/* Expanded Step Details */}
                {expandedSteps.has(step.id) && (
                  <div className="border-t bg-white p-4 space-y-3">
                    {/* Main Content */}
                    <div>
                      <h5 className="text-sm font-semibold text-gray-800 mb-2">Step Content:</h5>
                      <div className="p-3 bg-gray-50 rounded-lg">
                        <MedicalMarkdownText className="text-sm text-gray-700">
                          {step.content}
                        </MedicalMarkdownText>
                      </div>
                    </div>

                    {/* Reasoning */}
                    {step.reasoning && (
                      <div>
                        <h5 className="text-sm font-semibold text-gray-800 mb-2">Reasoning:</h5>
                        <div className="p-3 bg-blue-50 rounded-lg">
                          <MedicalMarkdownText className="text-sm text-blue-800">
                            {step.reasoning}
                          </MedicalMarkdownText>
                        </div>
                      </div>
                    )}

                    {/* Medical Context */}
                    {step.medical_context && (
                      <div>
                        <h5 className="text-sm font-semibold text-gray-800 mb-2">Medical Context:</h5>
                        <div className="p-3 bg-green-50 rounded-lg">
                          <MedicalMarkdownText className="text-sm text-green-800">
                            {step.medical_context}
                          </MedicalMarkdownText>
                        </div>
                      </div>
                    )}

                    {/* Dependencies */}
                    {step.dependencies && step.dependencies.length > 0 && (
                      <div>
                        <h5 className="text-sm font-semibold text-gray-800 mb-2">Depends on steps:</h5>
                        <div className="flex flex-wrap gap-1">
                          {step.dependencies.map(dep => (
                            <span key={dep} className="px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded">
                              Step {dep}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Evidence Sources */}
                    {step.evidence_sources && step.evidence_sources.length > 0 && (
                      <div>
                        <h5 className="text-sm font-semibold text-gray-800 mb-2">Evidence Sources:</h5>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {step.evidence_sources.map((source, idx) => (
                            <li key={idx} className="flex items-start space-x-2">
                              <DocumentTextIcon className="w-3 h-3 text-gray-400 mt-0.5 flex-shrink-0" />
                              <span>{source}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Confidence Breakdown */}
                    <div className="pt-2 border-t">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-600">Step Confidence:</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full transition-all duration-300 ${
                                step.confidence >= 0.8 ? 'bg-green-500' :
                                step.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}
                              style={{ width: `${step.confidence * 100}%` }}
                            ></div>
                          </div>
                          <span className={`font-medium ${
                            step.confidence >= 0.8 ? 'text-green-600' :
                            step.confidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                          }`}>
                            {(step.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Overall Summary */}
          <div className="p-4 bg-indigo-50 border-t">
            <div className="flex items-center space-x-2 text-indigo-800">
              <LightBulbIcon className="w-4 h-4" />
              <span className="text-sm font-medium">
                Reasoning completed with {(overallConfidence * 100).toFixed(1)}% overall confidence
              </span>
              <ClockIcon className="w-4 h-4" />
              <span className="text-sm">
                {processingTime.toFixed(2)}s processing time
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
