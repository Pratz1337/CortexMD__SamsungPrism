"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react'
import { DiagnosisAPI, API_BASE_URL } from '@/lib/api'

interface Props {
  patientId: string
  onAdded?: () => void
}

interface ARSession {
  session_id: string
  patient_id: string
  mode: 'photo' | 'video'
  created_at: string
  active: boolean
}

interface AROverlay {
  type: string
  timestamp: string
  elements: Array<{
    type: string
    position: { x: number; y: number; anchor?: string }
    data: any
    style: {
      background_color: string
      text_color: string
      font_size: number
      padding: number
      border_radius: number
      max_width?: number
    }
  }>
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
  processing_timestamp: string
}

const EnhancedARScanner: React.FC<Props> = ({ patientId, onAdded }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  
  const [mode, setMode] = useState<'photo' | 'video'>('photo')
  const [isStreaming, setIsStreaming] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [arSession, setArSession] = useState<ARSession | null>(null)
  const [currentOverlay, setCurrentOverlay] = useState<AROverlay | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [cameraInfo, setCameraInfo] = useState<any>(null)

  // Camera constraints optimized for document scanning
  const getCameraConstraints = useCallback(() => {
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    
    return {
      video: {
        width: { ideal: isMobile ? 1280 : 1280, min: 480, max: 1920 },
        height: { ideal: isMobile ? 720 : 720, min: 360, max: 1080 },
        facingMode: isMobile ? 'environment' : 'user',
        aspectRatio: { ideal: 16/9 },
        frameRate: { ideal: 30, min: 15, max: 30 }
      },
      audio: false
    }
  }, [])

  // Start AR session
  const startARSession = async () => {
    try {
      console.log('Starting AR session...', { patientId, mode })
      
      const response = await fetch(`${API_BASE_URL}/api/ar/start-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: patientId,
          mode: mode
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.success) {
        setArSession(data)
        console.log('AR session started:', data.session_id)
        return data.session_id
      } else {
        throw new Error(data.error || 'Failed to start AR session')
      }
    } catch (error: any) {
      console.error('Failed to start AR session:', error)
      setError(`Failed to start AR session: ${error.message}`)
      return null
    }
  }

  // End AR session
  const endARSession = async () => {
    if (!arSession) return

    try {
      const response = await fetch(`${API_BASE_URL}/api/ar/end-session/${arSession.session_id}`, {
        method: 'POST'
      })

      if (response.ok) {
        console.log('AR session ended')
      }
    } catch (error) {
      console.error('Failed to end AR session:', error)
    }

    setArSession(null)
    setCurrentOverlay(null)
  }

  // Start camera
  const startCamera = async () => {
    try {
      setError(null)
      console.log('Enhanced AR: Starting camera...')

      // Use exact same constraints as Camera Test
      const constraints = {
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      }

      console.log('Enhanced AR: Requesting camera with constraints:', constraints)
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      
      console.log('Enhanced AR: Camera stream obtained:', stream)
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream

        videoRef.current.onloadedmetadata = async () => {
          try {
            await videoRef.current!.play()
            setIsStreaming(true)
            setCameraInfo({
              videoWidth: videoRef.current!.videoWidth,
              videoHeight: videoRef.current!.videoHeight
            })
            console.log('Enhanced AR: Video is playing successfully!')
            
            // Start AR session for video mode
            if (mode === 'video') {
              const sessionId = await startARSession()
              if (sessionId) {
                startVideoProcessing()
              }
            }
          } catch (playError) {
            console.error('Enhanced AR: Video play failed:', playError)
            setError(`Failed to play video: ${playError}`)
          }
        }

        videoRef.current.onerror = (e) => {
          console.error('Enhanced AR: Video error:', e)
          setError('Video element error')
        }
      }
    } catch (err: any) {
      console.error('Enhanced AR: Camera access failed:', err)
      setError(`Camera error: ${err.message}`)
    }
  }

  // Stop camera
  const stopCamera = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setIsStreaming(false)
    setCapturedImage(null)
    endARSession()
  }

  // Start video processing for real-time AR
  const startVideoProcessing = () => {
    if (!videoRef.current || !canvasRef.current || mode !== 'video') return

    const processFrame = async () => {
      if (!videoRef.current || !canvasRef.current || !arSession) {
        return
      }

      const canvas = canvasRef.current
      const video = videoRef.current
      const context = canvas.getContext('2d')

      if (!context) return

      // Set canvas size to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Draw current frame
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convert to blob and send for processing (throttled)
      if (!isProcessing) {
        canvas.toBlob(async (blob) => {
          if (blob && arSession) {
            try {
              const formData = new FormData()
              formData.append('frame', blob, 'frame.jpg')

              const response = await fetch(`${API_BASE_URL}/api/ar/stream-frame/${arSession.session_id}`, {
                method: 'POST',
                body: formData
              })

              if (response.ok) {
                const data = await response.json()
                if (data.overlay) {
                  setCurrentOverlay(data.overlay)
                }
              }
            } catch (error) {
              console.error('Frame processing error:', error)
            }
          }
        }, 'image/jpeg', 0.8)
      }

      // Continue processing
      if (isStreaming && mode === 'video') {
        animationFrameRef.current = requestAnimationFrame(processFrame)
      }
    }

    // Start processing loop
    animationFrameRef.current = requestAnimationFrame(processFrame)
  }

  // Capture photo
  const capturePhoto = async () => {
    console.log('Enhanced AR: Starting photo capture...')
    
    if (!videoRef.current || !canvasRef.current) {
      console.error('Enhanced AR: Camera or canvas not ready')
      setError('Camera not ready')
      return
    }

    const canvas = canvasRef.current
    const video = videoRef.current
    const context = canvas.getContext('2d')

    if (!context) {
      console.error('Enhanced AR: Cannot get canvas context')
      setError('Cannot get canvas context')
      return
    }

    console.log('Enhanced AR: Capturing frame from video...', {
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight
    })

    // Set canvas size and draw frame
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert to blob
    canvas.toBlob(async (blob) => {
      if (blob) {
        console.log('Enhanced AR: Image captured, blob size:', blob.size)
        try {
          setIsProcessing(true)
          setError(null)
          
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8)
          setCapturedImage(dataUrl)
          console.log('Enhanced AR: Image set for preview')

          // Process with AI regardless of AR session
          try {
            const formData = new FormData()
            formData.append('image', blob, 'captured_note.jpg')

            const backendUrl = API_BASE_URL
            
            // Try AR endpoint first if session exists
            if (arSession) {
              const response = await fetch(`${backendUrl}/api/ar/capture-frame/${arSession.session_id}`, {
                method: 'POST',
                body: formData
              })
              
              if (response.ok) {
                const arData = await response.json()
                console.log('AR processing result:', arData)
              }
            }
            
            // Always send to diagnosis API for database storage and AI processing
            const file = new File([blob], 'captured_note.jpg', { type: 'image/jpeg' })
            const data = await DiagnosisAPI.submitClinicalNoteScan(patientId, file, {
              nurseId: 'AR_ENHANCED_SCANNER',
              location: 'AR_Photo_Capture',
              shift: 'Live'
            })

            setResult(data)
            console.log('Enhanced AR: Processing completed successfully')
            
          } catch (apiError) {
            console.error('Enhanced AR: API processing error:', apiError)
            setError('Failed to process image with AI')
          }
        } catch (error: any) {
          console.error('Enhanced AR: Photo capture error:', error)
          setError(error?.response?.data?.error || error?.message || 'Failed to process captured image')
        } finally {
          setIsProcessing(false)
          console.log('Enhanced AR: Photo capture process completed')
        }
      } else {
        console.error('Enhanced AR: Failed to create blob from canvas')
        setError('Failed to capture image')
        setIsProcessing(false)
      }
    }, 'image/jpeg', 0.8)
  }

  // Draw AR overlay on canvas
  const drawAROverlay = useCallback(() => {
    if (!overlayCanvasRef.current || !currentOverlay) return

    const canvas = overlayCanvasRef.current
    const context = canvas.getContext('2d')
    if (!context) return

    // Clear previous overlay
    context.clearRect(0, 0, canvas.width, canvas.height)

    // Draw overlay elements
    currentOverlay.elements.forEach(element => {
      const { position, data, style } = element
      
      // Set styles
      context.fillStyle = style.background_color
      context.font = `${style.font_size}px Arial`
      
      // Calculate position
      let x = position.x
      let y = position.y
      
      if (position.anchor === 'bottom') {
        y = canvas.height + position.y
      }

      // Draw background
      const text = Object.entries(data).map(([key, value]) => 
        `${key}: ${Array.isArray(value) ? value.join(', ') : value}`
      ).join('\n')
      
      const lines = text.split('\n')
      const lineHeight = style.font_size + 4
      const maxWidth = style.max_width || 250
      const padding = style.padding

      // Background rectangle
      const bgHeight = lines.length * lineHeight + padding * 2
      const bgWidth = Math.min(maxWidth, Math.max(...lines.map(line => 
        context.measureText(line).width
      ))) + padding * 2

      context.fillStyle = style.background_color
      context.fillRect(x, y, bgWidth, bgHeight)

      // Text
      context.fillStyle = style.text_color
      lines.forEach((line, index) => {
        context.fillText(
          line,
          x + padding,
          y + padding + (index + 1) * lineHeight,
          maxWidth
        )
      })
    })
  }, [currentOverlay])

  // Update overlay canvas when overlay changes
  useEffect(() => {
    if (mode === 'video' && isStreaming) {
      drawAROverlay()
    }
  }, [currentOverlay, mode, isStreaming, drawAROverlay])

  // Set up overlay canvas size
  useEffect(() => {
    if (overlayCanvasRef.current && videoRef.current && isStreaming) {
      const video = videoRef.current
      const canvas = overlayCanvasRef.current
      
      const updateCanvasSize = () => {
        canvas.width = video.clientWidth
        canvas.height = video.clientHeight
      }

      updateCanvasSize()
      video.addEventListener('loadedmetadata', updateCanvasSize)
      
      return () => {
        video.removeEventListener('loadedmetadata', updateCanvasSize)
      }
    }
  }, [isStreaming])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  // Mode change handler
  const handleModeChange = (newMode: 'photo' | 'video') => {
    if (isStreaming) {
      stopCamera()
    }
    setMode(newMode)
    setResult(null)
    setCapturedImage(null)
    setCurrentOverlay(null)
    setError(null)
  }

  const retakePhoto = () => {
    setCapturedImage(null)
    setResult(null)
    setError(null)
    setShowDetails(false)
  }

  return (
    <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
      {/* Header with Mode Selection */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-2xl font-bold text-gray-900 flex items-center">
            <span className="text-3xl mr-3">📱</span>
            Enhanced AR Clinical Scanner
          </h3>
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-600">Mode:</span>
            <div className="flex bg-white rounded-lg p-1 shadow-sm">
              <button
                onClick={() => handleModeChange('photo')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  mode === 'photo'
                    ? 'bg-blue-500 text-white shadow-sm'
                    : 'text-gray-600 hover:text-blue-500'
                }`}
              >
                📷 Photo
              </button>
              <button
                onClick={() => handleModeChange('video')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  mode === 'video'
                    ? 'bg-purple-500 text-white shadow-sm'
                    : 'text-gray-600 hover:text-purple-500'
                }`}
              >
                🎥 Video
              </button>
            </div>
          </div>
        </div>

        {/* Mode Description */}
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <div className="flex items-center mb-2">
            <span className="text-2xl mr-2">
              {mode === 'photo' ? '📷' : '🎥'}
            </span>
            <h4 className="font-semibold text-gray-900">
              {mode === 'photo' ? 'Photo Mode' : 'Video Mode'}
            </h4>
          </div>
          <p className="text-sm text-gray-600">
            {mode === 'photo' 
              ? 'Capture high-quality photos of medical notes for detailed analysis and database storage.'
              : 'Real-time video processing with live AR overlays showing extracted medical information as you move the camera.'
            }
          </p>
        </div>
      </div>

      {/* Camera Controls */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {!isStreaming ? (
              <button
                onClick={startCamera}
                disabled={isStreaming}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
              >
                <span className="text-xl mr-2">🔴</span>
                Start {mode === 'photo' ? 'Camera' : 'Live AR'}
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl hover:from-red-600 hover:to-pink-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
              >
                <span className="text-xl mr-2">⏹️</span>
                Stop {mode === 'photo' ? 'Camera' : 'AR Session'}
              </button>
            )}
          </div>

          {/* AR Session Status */}
          {arSession && (
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-gray-600">AR Session Active</span>
              <span className="text-gray-400">({arSession.session_id.slice(-8)})</span>
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mx-6 mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">⚠️</span>
            <div>
              <p className="font-semibold text-red-800">Error</p>
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Camera/AR View */}
      <div className="relative bg-gray-900 overflow-hidden" style={{ minHeight: '500px' }}>
        {/* Video Display */}
        <div className="relative bg-black rounded-lg overflow-hidden" style={{ minHeight: '400px' }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-auto"
            style={{ 
              minHeight: '400px',
              backgroundColor: '#000',
              display: 'block'
            }}
          />
          
          {isStreaming && (
            <div className="absolute top-4 left-4 bg-green-600 text-white px-3 py-1 rounded text-sm">
              ✅ Camera Active
            </div>
          )}
          
          {!isStreaming && !error && (
            <div className="absolute inset-0 flex items-center justify-center text-white">
              <div className="text-center">
                <div className="text-6xl mb-4">📷</div>
                <p className="mb-4">Enhanced AR Scanner</p>
                <button
                  onClick={startCamera}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  Start Camera
                </button>
              </div>
            </div>
          )}
        </div>

        {isStreaming && !capturedImage && (
          <div className="relative">
            {/* AR Overlay Canvas (for video mode) */}
            {mode === 'video' && (
              <canvas
                ref={overlayCanvasRef}
                className="absolute inset-0 pointer-events-none"
                style={{ 
                  mixBlendMode: 'multiply'
                }}
              />
            )}

            {/* Document Alignment Guide */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute inset-6 border-2 border-white border-dashed rounded-lg opacity-60"></div>
              <div className="absolute top-8 left-8 bg-black bg-opacity-60 text-white px-3 py-2 rounded-lg text-sm font-medium">
                📄 Align medical note within frame
                {mode === 'video' && (
                  <div className="text-xs text-gray-300 mt-1">
                    Real-time AR processing active
                  </div>
                )}
              </div>
            </div>

            {/* Capture Button (photo mode only) */}
            {mode === 'photo' && (
              <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2">
                <button
                  onClick={capturePhoto}
                  disabled={isProcessing}
                  className="w-20 h-20 bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white rounded-full flex items-center justify-center transition-all duration-200 shadow-2xl hover:shadow-red-500/25 transform hover:scale-105"
                >
                  {isProcessing ? (
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                  ) : (
                    <span className="text-3xl">📷</span>
                  )}
                </button>
              </div>
            )}

            {/* Video Mode Info */}
            {mode === 'video' && currentOverlay && (
              <div className="absolute bottom-6 left-6 bg-black bg-opacity-70 text-white px-4 py-2 rounded-lg text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span>Live AR Processing</span>
                </div>
                <div className="text-xs text-gray-300 mt-1">
                  {currentOverlay.elements.length} overlay elements detected
                </div>
              </div>
            )}
          </div>
        )}

        {/* Captured Image Preview (photo mode) */}
        {capturedImage && mode === 'photo' && (
          <div className="relative">
            <img
              src={capturedImage}
              alt="Captured medical note"
              className="w-full h-auto max-h-96 object-contain"
            />
            <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 flex gap-3">
              <button
                onClick={retakePhoto}
                disabled={isProcessing}
                className="px-6 py-3 bg-gray-600 text-white rounded-xl hover:bg-gray-700 disabled:opacity-50 transition-colors font-medium"
              >
                🔄 Retake
              </button>
            </div>
            
            {isProcessing && (
              <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center">
                <div className="bg-white rounded-2xl p-6 text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-700 font-medium text-lg">AI Processing...</p>
                  <p className="text-gray-500 text-sm mt-1">Analyzing medical content</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* No camera state */}
        {!isStreaming && !capturedImage && (
          <div className="flex flex-col items-center justify-center h-96 text-gray-400">
            <div className="text-8xl mb-6">
              {mode === 'photo' ? '📷' : '🎥'}
            </div>
            <h3 className="text-2xl font-bold mb-4 text-gray-600">
              {mode === 'photo' ? 'Photo Mode AR Scanner' : 'Live Video AR Scanner'}
            </h3>
            <p className="text-center max-w-md text-gray-500 leading-relaxed">
              {mode === 'photo' 
                ? 'Capture high-quality photos of medical notes for detailed AI analysis and secure database storage.'
                : 'Experience real-time AR overlays with live medical information extraction as you move your camera over documents.'
              }
            </p>
          </div>
        )}

        {/* Hidden canvas for image processing */}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {/* Results Section (photo mode) */}
      {result && mode === 'photo' && (
        <div className="p-6 border-t border-gray-100">
          <div className="space-y-6">
            {/* Success Message */}
            <div className="bg-green-50 border border-green-200 rounded-xl p-4">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">✅</span>
                <div>
                  <p className="font-semibold text-green-800">Success! Medical note processed with enhanced AR system</p>
                  <p className="text-xs text-green-700 mt-1">
                    Scanned Note ID: {result.scanned_note_id} | Clinical Note ID: {result.clinical_note_id}
                  </p>
                </div>
              </div>
            </div>

            {/* AI Summary */}
            {result.ai_summary && (
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
                <div className="flex items-center mb-4">
                  <span className="text-2xl mr-3">🤖</span>
                  <h4 className="text-lg font-bold text-blue-800">AI-Generated Summary</h4>
                </div>
                <p className="text-blue-700 leading-relaxed">{result.ai_summary}</p>
              </div>
            )}

            {/* Processing Statistics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <span className="text-xl mr-2">📊</span>
                  Processing Statistics
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600">OCR Confidence:</span>
                    <span className={`font-bold text-lg ${
                      result.ocr_confidence > 80 ? 'text-green-600' :
                      result.ocr_confidence > 60 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {result.ocr_confidence?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600">AI Confidence:</span>
                    <span className={`font-bold text-lg ${
                      result.ai_confidence > 0.8 ? 'text-green-600' :
                      result.ai_confidence > 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {(result.ai_confidence * 100)?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600">Text Length:</span>
                    <span className="font-medium text-gray-900">{result.text_length} characters</span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-gray-600">Word Count:</span>
                    <span className="font-medium text-gray-900">{result.word_count} words</span>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <span className="text-xl mr-2">📷</span>
                  Processed Image
                </h4>
                {result.preview_image && (
                  <img
                    src={result.preview_image}
                    alt="Processed Note Preview"
                    className="rounded-lg border border-gray-200 max-w-full shadow-md"
                  />
                )}
              </div>
            </div>

            {/* Extracted Medical Entities */}
            {result.extracted_entities && Object.keys(result.extracted_entities).length > 0 && (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
                <div className="flex items-center mb-4">
                  <span className="text-2xl mr-3">🏥</span>
                  <h4 className="text-lg font-bold text-green-800">Extracted Medical Entities</h4>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(result.extracted_entities).map(([key, value]: [string, any]) => (
                    <div key={key} className="bg-white rounded-lg p-3 border border-green-200">
                      <span className="font-semibold text-green-700 capitalize text-sm block mb-1">
                        {key.replace('_', ' ')}:
                      </span>
                      <div className="text-green-600 text-sm">
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
            <div className="flex gap-3 pt-4">
              <button 
                className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors font-medium"
                onClick={retakePhoto}
              >
                📷 Capture Another
              </button>
              <button 
                className="px-6 py-3 border border-gray-300 rounded-xl hover:bg-gray-50 transition-colors font-medium"
                onClick={() => setShowDetails(!showDetails)}
              >
                {showDetails ? 'Hide' : 'Show'} Detailed Data
              </button>
            </div>

            {/* Detailed Parsed Data */}
            {showDetails && (
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
                <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <span className="text-xl mr-2">🔍</span>
                  Detailed Parsed Data
                </h4>
                <pre className="bg-white p-4 rounded-lg border border-gray-200 text-xs overflow-auto max-h-96 text-gray-800">
                  {JSON.stringify(result.parsed_data || {}, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Usage Instructions */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 px-6 py-4 border-t">
        <h4 className="font-bold text-indigo-900 mb-3 flex items-center">
          <span className="text-xl mr-2">📖</span>
          Enhanced AR Scanner Instructions
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h5 className="font-semibold text-indigo-800 mb-2">📷 Photo Mode:</h5>
            <ul className="text-indigo-700 space-y-1">
              <li>• Click "Start Camera" to activate photo capture mode</li>
              <li>• Position medical note within the frame guidelines</li>
              <li>• Tap the red capture button to take a high-quality photo</li>
              <li>• AI will process and extract medical information</li>
              <li>• Results are saved to the patient database</li>
            </ul>
          </div>
          <div>
            <h5 className="font-semibold text-purple-800 mb-2">🎥 Video Mode:</h5>
            <ul className="text-purple-700 space-y-1">
              <li>• Click "Start Live AR" to begin real-time processing</li>
              <li>• Move camera over medical documents</li>
              <li>• Watch live AR overlays appear with extracted data</li>
              <li>• Information updates in real-time as you scan</li>
              <li>• Perfect for quick reference and verification</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EnhancedARScanner
