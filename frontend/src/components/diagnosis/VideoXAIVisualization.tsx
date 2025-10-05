"use client"

import { useState } from 'react'
import { VideoAnalysisResult } from '@/services/videoAnalysisService'
import { 
  ChartBarIcon, 
  EyeIcon, 
  ClockIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/solid'

interface VideoXAIVisualizationProps {
  analysisResult: VideoAnalysisResult
  patientId: string
}

export function VideoXAIVisualization({ analysisResult, patientId }: VideoXAIVisualizationProps) {
  const [selectedFrame, setSelectedFrame] = useState<number>(0)
  const [showAttentionMap, setShowAttentionMap] = useState(true)
  const [activeTab, setActiveTab] = useState<'temporal' | 'findings' | 'xai' | 'frames'>('temporal')

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-600 bg-red-100'
      case 'medium': return 'text-yellow-600 bg-yellow-100'
      case 'low': return 'text-green-600 bg-green-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-6">
        <h3 className="text-2xl font-bold mb-2">Video Analysis Results with XAI</h3>
        <div className="grid grid-cols-4 gap-4 text-sm">
          <div>
            <span className="opacity-75">Duration:</span>
            <span className="block font-semibold">{analysisResult.duration.toFixed(1)}s</span>
          </div>
          <div>
            <span className="opacity-75">FPS:</span>
            <span className="block font-semibold">{analysisResult.fps}</span>
          </div>
          <div>
            <span className="opacity-75">Total Frames:</span>
            <span className="block font-semibold">{analysisResult.total_frames}</span>
          </div>
          <div>
            <span className="opacity-75">Key Frames:</span>
            <span className="block font-semibold">{analysisResult.key_frames.length}</span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex">
          {[
            { id: 'temporal', label: 'Temporal Analysis', icon: ClockIcon },
            { id: 'findings', label: 'Medical Findings', icon: ExclamationTriangleIcon },
            { id: 'xai', label: 'XAI Explanations', icon: EyeIcon },
            { id: 'frames', label: 'Frame Analysis', icon: ChartBarIcon }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 px-6 py-3 border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-purple-600 text-purple-600'
                  : 'border-transparent text-gray-600 hover:text-gray-800'
              }`}
            >
              <tab.icon className="w-5 h-5" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="p-6">
        {/* Temporal Analysis Tab */}
        {activeTab === 'temporal' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold mb-4 flex items-center">
                <ArrowPathIcon className="w-5 h-5 mr-2 text-purple-600" />
                Motion Patterns Detected
              </h4>
              <div className="flex flex-wrap gap-2">
                {analysisResult.temporal_analysis.motion_patterns.map((pattern, index) => (
                  <span key={index} className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                    {pattern}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-4">Timeline of Changes</h4>
              <div className="space-y-3">
                {analysisResult.temporal_analysis.changes_detected.map((change, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className="w-20 text-sm text-gray-500">
                      {change.timestamp.toFixed(1)}s
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(change.severity)}`}>
                      {change.severity}
                    </div>
                    <div className="flex-1 text-gray-700">{change.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Medical Findings Tab */}
        {activeTab === 'findings' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold mb-4 flex items-center">
                <ExclamationTriangleIcon className="w-5 h-5 mr-2 text-red-600" />
                Abnormalities Detected
              </h4>
              <div className="space-y-3">
                {analysisResult.medical_findings.abnormalities.map((abnormality, index) => (
                  <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h5 className="font-semibold text-red-800">{abnormality.type}</h5>
                        <p className="text-gray-700 mt-1">{abnormality.description}</p>
                        <div className="mt-2 text-sm text-gray-600">
                          Frames: {abnormality.frame_range[0]} - {abnormality.frame_range[1]}
                        </div>
                      </div>
                      <div className={`text-lg font-bold ${getConfidenceColor(abnormality.confidence)}`}>
                        {(abnormality.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-4 flex items-center">
                <CheckCircleIcon className="w-5 h-5 mr-2 text-green-600" />
                Normal Findings
              </h4>
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <ul className="space-y-2">
                  {analysisResult.medical_findings.normal_findings.map((finding, index) => (
                    <li key={index} className="flex items-start">
                      <CheckCircleIcon className="w-4 h-4 text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* XAI Explanations Tab */}
        {activeTab === 'xai' && (
          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-semibold mb-4 flex items-center">
                <InformationCircleIcon className="w-5 h-5 mr-2 text-blue-600" />
                AI Decision Path
              </h4>
              <div className="bg-blue-50 rounded-lg p-4">
                <div className="space-y-2">
                  {analysisResult.xai_explanation.decision_path.map((step, index) => (
                    <div key={index} className="flex items-center">
                      <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs mr-3">
                        {index + 1}
                      </div>
                      <div className="flex-1 text-gray-700">{step}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-4">Feature Importance</h4>
              <div className="space-y-3">
                {Object.entries(analysisResult.xai_explanation.feature_importance)
                  .sort(([, a], [, b]) => b - a)
                  .map(([feature, importance]) => (
                    <div key={feature}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium">{feature}</span>
                        <span className="text-gray-600">{(importance * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-purple-500 to-indigo-500 h-2 rounded-full"
                          style={{ width: `${importance * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        )}

        {/* Frame Analysis Tab */}
        {activeTab === 'frames' && (
          <div className="space-y-6">
            {/* Frame Selector */}
            <div>
              <h4 className="text-lg font-semibold mb-4">Select Frame for Detailed Analysis</h4>
              <div className="grid grid-cols-5 gap-2">
                {analysisResult.xai_explanation.attention_maps.map((map, index) => (
                  <button
                    key={index}
                    onClick={() => setSelectedFrame(index)}
                    className={`p-2 rounded border-2 transition-all ${
                      selectedFrame === index
                        ? 'border-purple-600 bg-purple-50'
                        : 'border-gray-300 hover:border-gray-400'
                    }`}
                  >
                    <div className="text-sm font-medium">Frame {map.frame}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Selected Frame Analysis */}
            {selectedFrame !== null && analysisResult.xai_explanation.attention_maps[selectedFrame] && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Attention Map */}
                <div>
                  <h5 className="font-semibold mb-3">Attention Heatmap</h5>
                  <div className="relative bg-gray-100 rounded-lg overflow-hidden">
                    {showAttentionMap && (
                      <img
                        src={analysisResult.xai_explanation.attention_maps[selectedFrame].heatmap_url}
                        alt="Attention heatmap"
                        className="w-full h-auto"
                      />
                    )}
                    <button
                      onClick={() => setShowAttentionMap(!showAttentionMap)}
                      className="absolute top-2 right-2 bg-white px-3 py-1 rounded shadow-md text-sm"
                    >
                      {showAttentionMap ? 'Hide' : 'Show'} Heatmap
                    </button>
                  </div>
                </div>

                {/* Regions of Interest */}
                <div>
                  <h5 className="font-semibold mb-3">Regions of Interest</h5>
                  <div className="space-y-2">
                    {analysisResult.xai_explanation.attention_maps[selectedFrame].regions_of_interest.map((roi, index) => (
                      <div key={index} className="bg-gray-50 rounded-lg p-3">
                        <div className="flex justify-between items-center">
                          <span className="font-medium">{roi.label}</span>
                          <span className={`text-sm font-bold ${getConfidenceColor(roi.importance)}`}>
                            {(roi.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          Position: ({roi.coordinates.x}, {roi.coordinates.y}) - 
                          Size: {roi.coordinates.width}x{roi.coordinates.height}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Export Options */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-600">
              Analysis ID: {analysisResult.video_id}
            </div>
            <div className="space-x-3">
              <button className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition">
                Download Report
              </button>
              <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition">
                Add to Patient Record
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
