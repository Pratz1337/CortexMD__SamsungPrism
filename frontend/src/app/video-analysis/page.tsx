"use client"

import { useState } from 'react'
import { VideoInput } from '@/components/diagnosis/VideoInput'
import { VideoXAIVisualization } from '@/components/diagnosis/VideoXAIVisualization'
import { VideoAnalysisResult } from '@/services/videoAnalysisService'

export default function VideoAnalysisPage() {
  const [analysisResult, setAnalysisResult] = useState<VideoAnalysisResult | null>(null)
  const [extractedFrames, setExtractedFrames] = useState<Array<{ url: string; timestamp: number }>>([])
  const [showDemo, setShowDemo] = useState(false)

  // Demo data for showcasing XAI capabilities
  const demoResult: VideoAnalysisResult = {
    video_id: "demo-12345",
    duration: 45.5,
    fps: 30,
    total_frames: 1365,
    key_frames: [
      { timestamp: 5.2, frameNumber: 156, analysis: { detected_features: ['cardiac_motion', 'valve_movement'], confidence: 0.92, roi_coordinates: [{ x: 120, y: 80, width: 200, height: 180 }] } },
      { timestamp: 12.8, frameNumber: 384, analysis: { detected_features: ['abnormal_flow_pattern'], confidence: 0.87, roi_coordinates: [{ x: 150, y: 100, width: 180, height: 160 }] } },
      { timestamp: 23.4, frameNumber: 702, analysis: { detected_features: ['tissue_anomaly'], confidence: 0.94, roi_coordinates: [{ x: 200, y: 120, width: 150, height: 140 }] } },
      { timestamp: 35.1, frameNumber: 1053, analysis: { detected_features: ['normal_structure'], confidence: 0.89 } },
      { timestamp: 42.7, frameNumber: 1281, analysis: { detected_features: ['measurement_landmarks'], confidence: 0.91 } }
    ],
    temporal_analysis: {
      motion_patterns: ['Periodic cardiac motion', 'Irregular flow in left ventricle', 'Stable tissue boundaries'],
      changes_detected: [
        { timestamp: 12.8, description: 'Turbulent flow pattern detected in left ventricle', severity: 'high' },
        { timestamp: 23.4, description: 'Hypoechoic mass observed near septum', severity: 'medium' },
        { timestamp: 28.5, description: 'Slight valve regurgitation noted', severity: 'low' }
      ]
    },
    medical_findings: {
      abnormalities: [
        {
          type: 'Mitral Valve Regurgitation',
          confidence: 0.88,
          frame_range: [350, 450],
          description: 'Moderate mitral regurgitation observed with turbulent flow pattern. Color Doppler shows retrograde flow into left atrium during systole.'
        },
        {
          type: 'Left Ventricular Hypertrophy',
          confidence: 0.92,
          frame_range: [600, 800],
          description: 'Increased left ventricular wall thickness measuring approximately 14mm. Pattern consistent with concentric hypertrophy.'
        }
      ],
      normal_findings: [
        'Normal right ventricular size and function',
        'Aortic valve appears structurally normal with appropriate opening',
        'No pericardial effusion detected',
        'Normal inferior vena cava diameter with appropriate respiratory variation'
      ]
    },
    xai_explanation: {
      attention_maps: [
        {
          frame: 156,
          heatmap_url: '/api/placeholder/heatmap1.png',
          regions_of_interest: [
            { label: 'Mitral Valve', importance: 0.95, coordinates: { x: 150, y: 100, width: 100, height: 80 } },
            { label: 'Left Atrium', importance: 0.82, coordinates: { x: 120, y: 60, width: 120, height: 100 } }
          ]
        },
        {
          frame: 384,
          heatmap_url: '/api/placeholder/heatmap2.png',
          regions_of_interest: [
            { label: 'Turbulent Flow', importance: 0.91, coordinates: { x: 180, y: 120, width: 80, height: 60 } },
            { label: 'Valve Leaflets', importance: 0.87, coordinates: { x: 160, y: 110, width: 90, height: 70 } }
          ]
        },
        {
          frame: 702,
          heatmap_url: '/api/placeholder/heatmap3.png',
          regions_of_interest: [
            { label: 'LV Wall', importance: 0.93, coordinates: { x: 200, y: 140, width: 60, height: 120 } },
            { label: 'Septum', importance: 0.88, coordinates: { x: 180, y: 130, width: 40, height: 100 } }
          ]
        }
      ],
      decision_path: [
        'Detected periodic motion pattern consistent with cardiac ultrasound',
        'Identified valve structures and chamber boundaries using trained CNN model',
        'Analyzed flow patterns using optical flow algorithms',
        'Detected retrograde flow in frames 350-450 indicating regurgitation',
        'Measured wall thickness using automated segmentation',
        'Cross-referenced findings with medical knowledge base',
        'Generated confidence scores based on feature clarity and model certainty'
      ],
      feature_importance: {
        'Flow Pattern Analysis': 0.92,
        'Wall Thickness Measurement': 0.88,
        'Valve Motion Tracking': 0.85,
        'Chamber Size Assessment': 0.79,
        'Tissue Echogenicity': 0.74,
        'Temporal Consistency': 0.71
      }
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Medical Video Analysis with Explainable AI
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            Upload medical videos for AI-powered analysis with transparent, explainable insights.
            Our XAI system provides frame-by-frame analysis, attention mapping, and detailed explanations
            of all findings.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Video Input</h3>
              <p className="text-gray-600 text-sm">
                Support for ultrasound, endoscopy, fluoroscopy, and other medical video formats
              </p>
            </div>
            
            <div className="bg-purple-50 rounded-lg p-6 border border-purple-200">
              <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">AI Analysis</h3>
              <p className="text-gray-600 text-sm">
                Temporal analysis, motion detection, and abnormality identification
              </p>
            </div>
            
            <div className="bg-green-50 rounded-lg p-6 border border-green-200">
              <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mb-4">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">XAI Insights</h3>
              <p className="text-gray-600 text-sm">
                Attention maps, decision paths, and feature importance visualization
              </p>
            </div>
          </div>

          <button
            onClick={() => setShowDemo(true)}
            className="mt-6 px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-lg hover:from-purple-700 hover:to-indigo-700 transition"
          >
            View Demo Analysis
          </button>
        </div>

        {/* Video Input Section */}
        {!analysisResult && !showDemo && (
          <VideoInput
            patientId="demo-patient"
            onAnalysisComplete={setAnalysisResult}
            onFramesExtracted={setExtractedFrames}
          />
        )}

        {/* Results Section */}
        {(analysisResult || showDemo) && (
          <VideoXAIVisualization
            analysisResult={showDemo ? demoResult : analysisResult!}
            patientId="demo-patient"
          />
        )}

        {/* Demo Mode Notice */}
        {showDemo && (
          <div className="mt-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-yellow-800 text-sm">
              <strong>Demo Mode:</strong> This is a demonstration of XAI video analysis capabilities.
              Upload your own medical video to see real analysis results.
            </p>
            <button
              onClick={() => {
                setShowDemo(false)
                setAnalysisResult(null)
              }}
              className="mt-2 text-yellow-600 hover:text-yellow-700 underline text-sm"
            >
              Exit Demo Mode
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
