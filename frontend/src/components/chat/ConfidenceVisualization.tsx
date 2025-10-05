"use client"

import React from 'react'
import { 
  ShieldCheckIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon,
  XCircleIcon,
  InformationCircleIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'

interface ConfidenceMetrics {
  confidence: number
  fol_verified: boolean
  explainability: number
  response_time: number
  medical_accuracy?: number
  source_reliability?: number
}

interface ConfidenceVisualizationProps {
  metrics: ConfidenceMetrics
  compact?: boolean
  showDetails?: boolean
}

export function ConfidenceVisualization({ 
  metrics, 
  compact = false, 
  showDetails = true 
}: ConfidenceVisualizationProps) {
  const getConfidenceLevel = (score: number): { 
    level: string, 
    color: string, 
    bgColor: string,
    icon: React.ElementType 
  } => {
    if (score >= 0.9) return { 
      level: 'Excellent', 
      color: 'text-green-600', 
      bgColor: 'bg-green-50 border-green-200',
      icon: CheckCircleIcon
    }
    if (score >= 0.8) return { 
      level: 'High', 
      color: 'text-blue-600', 
      bgColor: 'bg-blue-50 border-blue-200',
      icon: ShieldCheckIcon
    }
    if (score >= 0.7) return { 
      level: 'Good', 
      color: 'text-indigo-600', 
      bgColor: 'bg-indigo-50 border-indigo-200',
      icon: InformationCircleIcon
    }
    if (score >= 0.6) return { 
      level: 'Moderate', 
      color: 'text-yellow-600', 
      bgColor: 'bg-yellow-50 border-yellow-200',
      icon: ExclamationTriangleIcon
    }
    return { 
      level: 'Low', 
      color: 'text-red-600', 
      bgColor: 'bg-red-50 border-red-200',
      icon: XCircleIcon
    }
  }

  const getFolVerificationStatus = (verified: boolean) => {
    return verified ? {
      status: 'Verified',
      color: 'text-green-600',
      bgColor: 'bg-green-100',
      icon: '✓'
    } : {
      status: 'Unverified',
      color: 'text-red-600',
      bgColor: 'bg-red-100',
      icon: '✗'
    }
  }

  const confidenceInfo = getConfidenceLevel(metrics.confidence)
  const folStatus = getFolVerificationStatus(metrics.fol_verified)

  if (compact) {
    return (
      <div className="flex items-center space-x-2">
        {/* Main Confidence Badge */}
        <div className={`inline-flex items-center px-2 py-1 rounded-full border text-xs font-medium ${confidenceInfo.bgColor} ${confidenceInfo.color}`}>
          <confidenceInfo.icon className="w-3 h-3 mr-1" />
          {(metrics.confidence * 100).toFixed(0)}%
        </div>

        {/* FOL Verification Badge */}
        <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${folStatus.bgColor} ${folStatus.color}`}>
          <span className="mr-1">{folStatus.icon}</span>
          FOL
        </div>

        {/* Explainability Score */}
        {metrics.explainability && (
          <div className="inline-flex items-center px-2 py-1 rounded-full bg-purple-100 text-purple-600 text-xs font-medium">
            <BeakerIcon className="w-3 h-3 mr-1" />
            {(metrics.explainability * 100).toFixed(0)}%
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="mt-3 p-3 bg-gray-50 rounded-lg border">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-gray-700 flex items-center">
          <BeakerIcon className="w-4 h-4 mr-1" />
          AI Confidence Analysis
        </h4>
        <div className="text-xs text-gray-500">
          Response time: {metrics.response_time.toFixed(2)}s
        </div>
      </div>

      {/* Main Confidence Score */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div className={`p-3 rounded-lg border ${confidenceInfo.bgColor}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-600 mb-1">Overall Confidence</div>
              <div className={`text-lg font-bold ${confidenceInfo.color}`}>
                {(metrics.confidence * 100).toFixed(1)}%
              </div>
              <div className={`text-xs ${confidenceInfo.color}`}>
                {confidenceInfo.level}
              </div>
            </div>
            <confidenceInfo.icon className={`w-6 h-6 ${confidenceInfo.color}`} />
          </div>
          
          {/* Progress bar */}
          <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${
                metrics.confidence >= 0.8 ? 'bg-green-500' :
                metrics.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${metrics.confidence * 100}%` }}
            ></div>
          </div>
        </div>

        {/* FOL Verification */}
        <div className={`p-3 rounded-lg border ${folStatus.bgColor}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-600 mb-1">FOL Verification</div>
              <div className={`text-sm font-bold ${folStatus.color}`}>
                {folStatus.status}
              </div>
              <div className="text-xs text-gray-600">
                Logic-based verification
              </div>
            </div>
            <div className={`w-8 h-8 rounded-full ${folStatus.bgColor} flex items-center justify-center ${folStatus.color} font-bold`}>
              {folStatus.icon}
            </div>
          </div>
        </div>

        {/* Explainability Score */}
        <div className="p-3 rounded-lg border bg-purple-50 border-purple-200">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-600 mb-1">Explainability</div>
              <div className="text-lg font-bold text-purple-600">
                {metrics.explainability ? (metrics.explainability * 100).toFixed(1) : 'N/A'}%
              </div>
              <div className="text-xs text-purple-600">
                Reasoning clarity
              </div>
            </div>
            <BeakerIcon className="w-6 h-6 text-purple-600" />
          </div>
          
          {metrics.explainability && (
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div 
                className="h-2 rounded-full bg-purple-500 transition-all duration-300"
                style={{ width: `${metrics.explainability * 100}%` }}
              ></div>
            </div>
          )}
        </div>
      </div>

      {/* Additional Metrics */}
      {showDetails && (metrics.medical_accuracy || metrics.source_reliability) && (
        <div className="grid grid-cols-2 gap-3 pt-3 border-t">
          {metrics.medical_accuracy && (
            <div className="text-center">
              <div className="text-xs text-gray-600 mb-1">Medical Accuracy</div>
              <div className="text-sm font-bold text-blue-600">
                {(metrics.medical_accuracy * 100).toFixed(1)}%
              </div>
            </div>
          )}
          {metrics.source_reliability && (
            <div className="text-center">
              <div className="text-xs text-gray-600 mb-1">Source Reliability</div>
              <div className="text-sm font-bold text-indigo-600">
                {(metrics.source_reliability * 100).toFixed(1)}%
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Confidence indicator for message headers
export function ConfidenceIndicator({ confidence }: { confidence: number }) {
  const getIndicatorColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-500'
    if (score >= 0.6) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="flex items-center space-x-1">
      <div className="flex space-x-1">
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            className={`w-2 h-2 rounded-full ${
              i < Math.round(confidence * 5) 
                ? getIndicatorColor(confidence)
                : 'bg-gray-300'
            }`}
          />
        ))}
      </div>
      <span className="text-xs text-gray-500 ml-1">
        {(confidence * 100).toFixed(0)}%
      </span>
    </div>
  )
}

// Mini confidence badge for inline use
export function ConfidenceBadge({ 
  confidence, 
  verified, 
  size = 'sm' 
}: { 
  confidence: number
  verified: boolean
  size?: 'xs' | 'sm' | 'md' 
}) {
  const sizeClasses = {
    xs: 'text-xs px-1.5 py-0.5',
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-3 py-1.5'
  }

  const confidenceColor = confidence >= 0.8 ? 'green' : confidence >= 0.6 ? 'yellow' : 'red'

  return (
    <div className="flex items-center space-x-1">
      <span className={`inline-flex items-center rounded-full font-medium ${sizeClasses[size]} bg-${confidenceColor}-100 text-${confidenceColor}-700`}>
        {(confidence * 100).toFixed(0)}%
      </span>
      <span className={`inline-flex items-center rounded-full font-medium ${sizeClasses[size]} ${
        verified ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
      }`}>
        {verified ? '✓' : '✗'}
      </span>
    </div>
  )
}
