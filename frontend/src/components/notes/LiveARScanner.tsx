"use client"

import React, { useRef, useState, useEffect } from 'react'
import { DiagnosisAPI } from '@/lib/api'

interface Props {
  patientId: string
  onAdded?: () => void
  onCapture?: (imageBlob: Blob) => void
}

interface ScanResult {
  success: boolean
  patient_id: string
  scanned_note_id: string
  clinical_note_id: string
  parsed_data: any
  ai_summary: string
  extracted_entities: any
  ocr_confidence: number
  ai_confidence: number
  text_length: number
  word_count: number
  preview_image: string
  patient_dashboard: any
  processing_timestamp: string
}

interface CameraConstraints {
  video: {
    width: { ideal: number; min: number; max: number }
    height: { ideal: number; min: number; max: number }
    facingMode: string
    aspectRatio: { ideal: number }
    frameRate: { ideal: number; min: number }
  }
  audio: false
}

const LiveARScanner: React.FC<Props> = ({ patientId, onAdded, onCapture }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  const [isStreaming, setIsStreaming] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [cameraPermission, setCameraPermission] = useState<'pending' | 'granted' | 'denied'>('pending')

  // Camera constraints for better document scanning
  const getCameraConstraints = (): CameraConstraints => {
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    
    return {
      video: {
        width: { ideal: isMobile ? 1920 : 1280, min: 640, max: 1920 },
        height: { ideal: isMobile ? 1080 : 720, min: 480, max: 1080 },
        facingMode: isMobile ? 'environment' : 'user', // Use back camera on mobile
        aspectRatio: { ideal: 16/9 },
        frameRate: { ideal: 30, min: 15 }
      },
      audio: false
    }
  }

  const requestPermissionFirst = async () => {
    try {
      console.log('🎥 Requesting camera permission first...')
      
      // Check for getUserMedia support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not supported in this browser')
      }

      // Test with minimal constraints first to get permission
      const testStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240 },
        audio: false
      })
      
      console.log('🎥 Permission granted, stopping test stream')
      testStream.getTracks().forEach(track => track.stop())
      
      return true
    } catch (err: any) {
      console.error('🎥 Permission request failed:', err)
      setCameraPermission('denied')
      setError(
        err.name === 'NotAllowedError' 
          ? 'Camera access denied. Please click "Allow" when prompted for camera access.' 
          : err.name === 'NotFoundError'
          ? 'No camera device found. Please connect a camera and try again.'
          : err.name === 'NotReadableError'
          ? 'Camera is already in use by another application.'
          : `Camera error: ${err.message || 'Unknown error'}`
      )
      return false
    }
  }

  // Browser detection for specific instructions
  const getBrowserInstructions = () => {
    const userAgent = navigator.userAgent
    if (userAgent.includes('Chrome') && !userAgent.includes('Edge')) {
      return {
        browser: 'Chrome',
        instructions: 'Click the camera icon in the address bar and select "Allow"'
      }
    } else if (userAgent.includes('Firefox')) {
      return {
        browser: 'Firefox', 
        instructions: 'Click on the shield icon and allow camera access'
      }
    } else if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
      return {
        browser: 'Safari',
        instructions: 'Go to Safari > Settings for This Website > Camera: Allow'
      }
    } else if (userAgent.includes('Edge')) {
      return {
        browser: 'Edge',
        instructions: 'Click the camera icon in the address bar and select "Allow"'
      }
    } else {
      return {
        browser: 'Your browser',
        instructions: 'Allow camera access when prompted'
      }
    }
  }

  const startCamera = async () => {
    try {
      console.log('🎥 Starting camera...', { userAgent: navigator.userAgent })
      setError(null)
      setCameraPermission('pending')
      
      // First request permission with minimal constraints
      const hasPermission = await requestPermissionFirst()
      if (!hasPermission) {
        return
      }

      const constraints = getCameraConstraints()
      console.log('🎥 Camera constraints:', constraints)
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      console.log('🎥 Camera stream obtained:', { id: stream.id, tracks: stream.getTracks().length })
      
      if (videoRef.current) {
        // Set up video element for optimal display
        const video = videoRef.current
        video.srcObject = stream
        streamRef.current = stream
        
        // Add attributes for better mobile support
        video.setAttribute('playsinline', 'true')
        video.setAttribute('muted', 'true')
        video.setAttribute('autoplay', 'true')
        
        // Ensure video plays properly
        video.onloadedmetadata = async () => {
          console.log('🎥 Video metadata loaded:', { 
            videoWidth: video.videoWidth, 
            videoHeight: video.videoHeight,
            readyState: video.readyState 
          })
          
          try {
            await video.play()
            console.log('🎥 Video playing successfully')
            setIsStreaming(true)
            setCameraPermission('granted')
          } catch (playError: any) {
            console.error('🎥 Video play failed:', playError)
            setError(`Failed to play video: ${playError.message}`)
          }
        }
        
        video.onerror = (e) => {
          console.error('🎥 Video element error:', e)
          setError('Video display error occurred')
        }
        
        // Fallback: try to play immediately if metadata is already loaded
        if (video.readyState >= 1) {
          try {
            await video.play()
            console.log('🎥 Video playing immediately')
            setIsStreaming(true)
            setCameraPermission('granted')
          } catch (playError: any) {
            console.log('🎥 Immediate play failed, waiting for loadedmetadata')
          }
        }
      }
    } catch (err: any) {
      console.error('🎥 Camera access failed:', err)
      setCameraPermission('denied')
      setError(
        err.name === 'NotAllowedError' 
          ? 'Camera access denied. Please refresh the page and allow camera access.' 
          : err.name === 'NotFoundError'
          ? 'No camera device found. Please connect a camera and try again.'
          : err.name === 'NotReadableError'
          ? 'Camera is already in use by another application.'
          : `Camera error: ${err.message || 'Unknown error'}`
      )
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    setCapturedImage(null)
  }

  const capturePhoto = async () => {
    if (!videoRef.current || !canvasRef.current) {
      setError('Camera not ready')
      return
    }

    const canvas = canvasRef.current
    const video = videoRef.current
    const context = canvas.getContext('2d')

    if (!context) {
      setError('Cannot get canvas context')
      return
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw current frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (blob) {
        try {
          setIsProcessing(true)
          setError(null)
          
          // Create preview
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8)
          setCapturedImage(dataUrl)
          
          // Create File object from blob
          const file = new File([blob], 'captured_note.jpg', { type: 'image/jpeg' })
          
          // Call the onCapture callback if provided
          if (onCapture) {
            onCapture(blob)
          }

          console.log('Starting AR scan for captured image...')
          console.log('Patient ID:', patientId)
          console.log('Image size:', blob.size, 'bytes')

          // Submit to AR scanner API
          const data = await DiagnosisAPI.submitClinicalNoteScan(patientId, file, {
            nurseId: 'AR_LIVE_SCANNER',
            location: 'AR_Capture',
            shift: 'Live'
          })

          console.log('AR scan response:', data)
          setResult(data)
          
          if (onAdded) {
            onAdded()
          }

        } catch (error: any) {
          console.error('AR processing error:', error)
          setError(error?.response?.data?.error || error?.message || 'Failed to process captured image')
        } finally {
          setIsProcessing(false)
        }
      }
    }, 'image/jpeg', 0.8)
  }

  const retakePhoto = () => {
    setCapturedImage(null)
    setResult(null)
    setError(null)
    setShowDetails(false)
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-gray-900">📷 Live AR Scanner</h3>
          <div className="flex gap-2">
            {!isStreaming ? (
              <button
                onClick={startCamera}
                disabled={cameraPermission === 'denied'}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                🔴 Start Camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                ⏹️ Stop Camera
              </button>
            )}
          </div>
        </div>

        {/* Camera Permission Status */}
        {cameraPermission === 'pending' && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-yellow-800 text-sm">📋 Requesting camera permission...</p>
          </div>
        )}

        {cameraPermission === 'denied' && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <h4 className="text-red-800 font-medium mb-2">Camera Access Required</h4>
            <p className="text-red-700 text-sm mb-3">
              To use the live AR scanner, please enable camera permissions and try again.
            </p>
            <div className="bg-red-100 p-3 rounded text-xs text-red-800 mb-3">
              <strong>{getBrowserInstructions().browser} Instructions:</strong>
              <br />
              {getBrowserInstructions().instructions}
            </div>
            <p className="text-red-600 text-xs mb-3">
              📱 On mobile: Make sure to allow camera access when prompted
              <br />
              💻 On desktop: Look for camera permissions in your browser
            </p>
            <div className="flex gap-2">
              <button
                onClick={startCamera}
                className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700 transition-colors"
              >
                🔄 Try Again
              </button>
              <a
                href="/test-camera"
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors"
              >
                📷 Test Camera
              </a>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 text-sm">
              <strong>Error:</strong> {error}
            </p>
          </div>
        )}

        {/* Main Camera/Capture Area */}
        <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ minHeight: '400px' }}>
          {isStreaming && !capturedImage && (
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                controls={false}
                webkit-playsinline="true"
                x-webkit-airplay="deny"
                className="w-full h-auto max-h-96 object-cover bg-gray-900"
                style={{ 
                  transform: 'scaleX(-1)', // Mirror for better UX
                  minHeight: '300px',
                  maxHeight: '400px'
                }}
                onLoadedData={() => console.log('🎥 Video data loaded')}
                onCanPlay={() => console.log('🎥 Video can play')}
                onPlaying={() => console.log('🎥 Video is playing')}
                onError={(e) => {
                  console.error('🎥 Video element error:', e)
                  setError('Video display failed')
                }}
              />
              
              {/* Overlay guide for document alignment */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-4 border-2 border-white border-dashed rounded-lg opacity-50"></div>
                <div className="absolute top-6 left-6 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
                  📄 Align medical note within frame
                </div>
              </div>
              
              {/* Capture Button */}
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
                <button
                  onClick={capturePhoto}
                  disabled={isProcessing}
                  className="w-16 h-16 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white rounded-full flex items-center justify-center transition-all duration-200 shadow-lg"
                  style={{ boxShadow: '0 4px 20px rgba(239, 68, 68, 0.4)' }}
                >
                  {isProcessing ? (
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                  ) : (
                    '📷'
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Captured Image Preview */}
          {capturedImage && !result && (
            <div className="relative">
              <img
                src={capturedImage}
                alt="Captured medical note"
                className="w-full h-auto max-h-96 object-contain"
              />
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2">
                <button
                  onClick={retakePhoto}
                  disabled={isProcessing}
                  className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 transition-colors"
                >
                  🔄 Retake
                </button>
              </div>
              
              {isProcessing && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                  <div className="bg-white rounded-lg p-4 text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                    <p className="text-gray-700 font-medium">Processing with AI...</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* No camera state */}
          {!isStreaming && !capturedImage && (
            <div className="flex flex-col items-center justify-center h-96 text-gray-500">
              <div className="text-6xl mb-4">📹</div>
              <p className="text-lg font-medium mb-2">Live AR Camera Scanner</p>
              <p className="text-sm text-center max-w-md">
                Click "Start Camera" to begin live scanning of clinical notes.
                The camera will open and you can capture notes in real-time.
              </p>
            </div>
          )}
        </div>

        {/* Processing Results */}
        {result && (
          <div className="mt-6 space-y-4">
            {/* Success Message */}
            <div className="alert alert-success bg-green-50 border border-green-200 rounded-lg p-4">
              <strong>✅ Success!</strong> Note captured and analyzed with AI.
              <div className="text-sm mt-1">
                Scanned Note ID: {result.scanned_note_id} | Clinical Note ID: {result.clinical_note_id}
              </div>
            </div>

            {/* AI Summary */}
            {result.ai_summary && (
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-2">🤖 AI Summary</h4>
                <p className="text-blue-700">{result.ai_summary}</p>
              </div>
            )}

            {/* Preview and Stats */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium mb-2">📷 Captured Image</h4>
                {result.preview_image && (
                  <img
                    src={result.preview_image}
                    alt="Processed Note Preview"
                    className="rounded border max-w-full shadow-sm"
                  />
                )}
              </div>
              
              <div>
                <h4 className="font-medium mb-2">📊 Processing Statistics</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>OCR Confidence:</span>
                    <span className={`font-medium ${
                      result.ocr_confidence > 80 ? 'text-green-600' : 
                      result.ocr_confidence > 60 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {result.ocr_confidence?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>AI Confidence:</span>
                    <span className={`font-medium ${
                      result.ai_confidence > 0.8 ? 'text-green-600' : 
                      result.ai_confidence > 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {(result.ai_confidence * 100)?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Text Length:</span>
                    <span>{result.text_length} characters</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Word Count:</span>
                    <span>{result.word_count} words</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Processed:</span>
                    <span>{new Date(result.processing_timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Extracted Medical Entities */}
            {result.extracted_entities && Object.keys(result.extracted_entities).length > 0 && (
              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                <h4 className="font-semibold text-green-800 mb-2">🏥 Extracted Medical Entities</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  {Object.entries(result.extracted_entities).map(([key, value]: [string, any]) => (
                    <div key={key}>
                      <span className="font-medium text-green-700 capitalize">
                        {key.replace('_', ' ')}:
                      </span>
                      <div className="text-green-600">
                        {Array.isArray(value) ? (
                          value.length > 0 ? value.join(', ') : 'None found'
                        ) : (
                          value || 'Not found'
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-2 pt-4">
              <button 
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                onClick={retakePhoto}
              >
                📷 Capture Another
              </button>
              <button 
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                onClick={() => setShowDetails(!showDetails)}
              >
                {showDetails ? 'Hide' : 'Show'} Details
              </button>
            </div>

            {/* Detailed Parsed Data */}
            {showDetails && (
              <div className="bg-gray-50 p-4 rounded-lg border">
                <h4 className="font-medium mb-2">🔍 Detailed Parsed Data</h4>
                <pre className="p-3 bg-white rounded border text-xs overflow-auto max-h-96">
                  {JSON.stringify(result.parsed_data || {}, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {/* Hidden canvas for image capture */}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Usage Instructions */}
      <div className="bg-blue-50 px-6 py-4 border-t">
        <h4 className="font-medium text-blue-900 mb-2">📖 How to Use Live AR Scanner</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Click "Start Camera" to activate your device camera</li>
          <li>• Position the medical note within the frame guidelines</li>
          <li>• Tap the red capture button to take a photo</li>
          <li>• The AI will automatically process and extract medical information</li>
          <li>• All data is securely stored in the patient database</li>
        </ul>
      </div>
    </div>
  )
}

export default LiveARScanner
