"use client"

import React, { useRef, useState, useEffect } from 'react'

interface CameraTestResult {
  supported: boolean
  devices: MediaDeviceInfo[]
  error?: string
  permissionGranted?: boolean
  streamActive?: boolean
}

const CameraTestPage: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [testResult, setTestResult] = useState<CameraTestResult | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentStream, setCurrentStream] = useState<MediaStream | null>(null)

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev.slice(-19), `[${timestamp}] ${message}`])
    console.log(`🔍 Camera Test: ${message}`)
  }

  const runCameraTest = async () => {
    addLog('Starting camera diagnostics...')
    setTestResult(null)
    
    const result: CameraTestResult = {
      supported: false,
      devices: []
    }

    try {
      // Test 1: Check API support
      addLog('Testing browser API support...')
      if (!navigator.mediaDevices) {
        throw new Error('MediaDevices API not supported')
      }
      if (!navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not supported')
      }
      addLog('✅ Browser APIs supported')

      // Test 2: Check device enumeration
      addLog('Enumerating camera devices...')
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        const cameras = devices.filter(device => device.kind === 'videoinput')
        result.devices = cameras
        addLog(`Found ${cameras.length} camera device(s)`)
        
        cameras.forEach((device, index) => {
          addLog(`Camera ${index + 1}: ${device.label || 'Unknown'} (${device.deviceId.substring(0, 10)}...)`)
        })

        if (cameras.length === 0) {
          throw new Error('No camera devices found')
        }
      } catch (enumError: any) {
        addLog(`❌ Device enumeration failed: ${enumError.message}`)
        result.error = enumError.message
      }

      // Test 3: Request permissions
      addLog('Testing camera permission...')
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false
        })
        
        addLog('✅ Camera permission granted')
        result.permissionGranted = true
        result.supported = true
        
        // Stop the test stream immediately
        stream.getTracks().forEach(track => track.stop())
        addLog('Test stream stopped')

      } catch (permError: any) {
        addLog(`❌ Permission failed: ${permError.name} - ${permError.message}`)
        result.error = permError.message
        result.permissionGranted = false
      }

    } catch (error: any) {
      addLog(`❌ Test failed: ${error.message}`)
      result.error = error.message
    }

    setTestResult(result)
    addLog('Camera test completed')
  }

  const startLiveCamera = async () => {
    if (isStreaming) {
      stopLiveCamera()
      return
    }

    try {
      addLog('Starting live camera...')
      
      const constraints = {
        video: {
          width: { ideal: 1280, min: 640, max: 1920 },
          height: { ideal: 720, min: 480, max: 1080 },
          facingMode: 'user',
          aspectRatio: { ideal: 16/9 },
          frameRate: { ideal: 30, min: 15 }
        },
        audio: false
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      addLog('Camera stream obtained')

      if (videoRef.current) {
        const video = videoRef.current
        video.srcObject = stream
        setCurrentStream(stream)

        video.onloadedmetadata = async () => {
          addLog(`Video metadata loaded: ${video.videoWidth}x${video.videoHeight}`)
          try {
            await video.play()
            addLog('✅ Video playing successfully')
            setIsStreaming(true)
          } catch (playError: any) {
            addLog(`❌ Video play failed: ${playError.message}`)
          }
        }
      }

    } catch (error: any) {
      addLog(`❌ Live camera failed: ${error.name} - ${error.message}`)
    }
  }

  const stopLiveCamera = () => {
    if (currentStream) {
      addLog('Stopping camera...')
      currentStream.getTracks().forEach(track => track.stop())
      setCurrentStream(null)
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    addLog('Camera stopped')
  }

  const clearLogs = () => {
    setLogs([])
  }

  // Initial test on load
  useEffect(() => {
    runCameraTest()
    return () => {
      stopLiveCamera()
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              📷 Camera Test & Diagnostics
            </h1>
            <p className="text-gray-600">
              Test your device's camera functionality for AR scanning
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Camera Test Results */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">🔍 Camera Test Results</h2>
                <button
                  onClick={runCameraTest}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  🔄 Re-test
                </button>
              </div>

              {testResult ? (
                <div className="space-y-4">
                  <div className={`p-3 rounded-lg border ${
                    testResult.supported 
                      ? 'bg-green-50 border-green-200 text-green-800' 
                      : 'bg-red-50 border-red-200 text-red-800'
                  }`}>
                    <div className="font-medium mb-1">
                      {testResult.supported ? '✅ Camera Supported' : '❌ Camera Issues'}
                    </div>
                    {testResult.error && (
                      <div className="text-sm">Error: {testResult.error}</div>
                    )}
                  </div>

                  <div className="space-y-2">
                    <div className="text-sm">
                      <strong>Devices Found:</strong> {testResult.devices.length}
                    </div>
                    {testResult.devices.map((device, index) => (
                      <div key={device.deviceId} className="text-xs bg-gray-50 p-2 rounded">
                        Camera {index + 1}: {device.label || 'Unknown Device'}
                      </div>
                    ))}
                  </div>

                  <div className="text-sm space-y-1">
                    <div>
                      <strong>Permission:</strong> {
                        testResult.permissionGranted === true ? '✅ Granted' :
                        testResult.permissionGranted === false ? '❌ Denied' : '⏳ Pending'
                      }
                    </div>
                    <div>
                      <strong>Browser:</strong> {navigator.userAgent.split(' ')[0]}
                    </div>
                    <div>
                      <strong>HTTPS:</strong> {window.location.protocol === 'https:' ? '✅' : '❌'} {window.location.protocol}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              )}
            </div>

            {/* Live Camera Test */}
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">📹 Live Camera Test</h2>
                <button
                  onClick={startLiveCamera}
                  className={`px-4 py-2 rounded-lg transition-colors ${
                    isStreaming 
                      ? 'bg-red-600 text-white hover:bg-red-700' 
                      : 'bg-green-600 text-white hover:bg-green-700'
                  }`}
                >
                  {isStreaming ? '⏹️ Stop' : '▶️ Start'} Camera
                </button>
              </div>

              <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ minHeight: '240px' }}>
                {isStreaming ? (
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-auto object-cover"
                    style={{ minHeight: '240px', transform: 'scaleX(-1)' }}
                  />
                ) : (
                  <div className="flex items-center justify-center h-60 text-gray-500">
                    <div className="text-center">
                      <div className="text-4xl mb-2">📷</div>
                      <p>Click "Start Camera" to test live video</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Debug Logs */}
          <div className="mt-6 bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">📋 Debug Logs</h2>
              <button
                onClick={clearLogs}
                className="px-3 py-1 text-sm bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
              >
                Clear
              </button>
            </div>
            
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-64 overflow-y-auto">
              {logs.length > 0 ? (
                logs.map((log, index) => (
                  <div key={index} className="mb-1">{log}</div>
                ))
              ) : (
                <div className="text-gray-500">No logs yet...</div>
              )}
            </div>
          </div>

          {/* Instructions */}
          <div className="mt-6 bg-blue-50 rounded-lg p-6 border border-blue-200">
            <h3 className="font-semibold text-blue-900 mb-3">🛠️ Troubleshooting Guide</h3>
            <div className="space-y-2 text-sm text-blue-800">
              <div><strong>Permission Denied:</strong> Click the camera icon in your browser's address bar and allow camera access</div>
              <div><strong>No Camera Found:</strong> Make sure your camera is connected and not being used by another app</div>
              <div><strong>HTTPS Required:</strong> Some browsers require HTTPS for camera access</div>
              <div><strong>Mobile Issues:</strong> Try refreshing the page and tapping "Allow" when prompted</div>
              <div><strong>Still not working?</strong> Try a different browser (Chrome, Firefox, Safari)</div>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

export default CameraTestPage
