"use client"

import React, { useRef, useState, useEffect } from 'react'

const ARCameraDebug: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [debugInfo, setDebugInfo] = useState<any>({})
  const [logs, setLogs] = useState<string[]>([])

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev.slice(-9), `[${timestamp}] ${message}`])
    console.log(`🐛 AR Debug: ${message}`)
  }

  const checkBrowserSupport = () => {
    const info = {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      vendor: navigator.vendor,
      protocol: window.location.protocol,
      hostname: window.location.hostname,
      mediaDevicesSupported: !!navigator.mediaDevices,
      getUserMediaSupported: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
      enumerateDevicesSupported: !!(navigator.mediaDevices && navigator.mediaDevices.enumerateDevices),
      isSecureContext: window.isSecureContext,
      isMobile: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    }
    setDebugInfo(info)
    addLog('Browser support check completed')
    return info
  }

  const testDeviceEnumeration = async () => {
    try {
      addLog('Testing device enumeration...')
      
      if (!navigator.mediaDevices?.enumerateDevices) {
        throw new Error('enumerateDevices not supported')
      }

      const deviceList = await navigator.mediaDevices.enumerateDevices()
      const cameras = deviceList.filter(device => device.kind === 'videoinput')
      
      setDevices(cameras)
      addLog(`Found ${cameras.length} camera device(s)`)
      
      cameras.forEach((device, index) => {
        addLog(`Camera ${index + 1}: ${device.label || 'Unknown'} (${device.deviceId.substring(0, 10)}...)`)
      })

      if (cameras.length === 0) {
        addLog('⚠️ No camera devices found!')
      }

      return cameras
    } catch (err: any) {
      addLog(`❌ Device enumeration failed: ${err.message}`)
      throw err
    }
  }

  const testCameraPermissions = async () => {
    try {
      addLog('Testing camera permissions...')
      
      // Test with minimal constraints first
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240 },
        audio: false
      })
      
      addLog('✅ Camera permission granted')
      
      // Immediately stop the test stream
      stream.getTracks().forEach(track => track.stop())
      
      return true
    } catch (err: any) {
      addLog(`❌ Camera permission failed: ${err.name} - ${err.message}`)
      
      if (err.name === 'NotAllowedError') {
        setError('Camera access denied. Please allow camera permissions and try again.')
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please connect a camera device.')
      } else if (err.name === 'NotReadableError') {
        setError('Camera is being used by another application.')
      } else {
        setError(`Camera error: ${err.message}`)
      }
      
      throw err
    }
  }

  const startCameraStream = async () => {
    try {
      addLog('Starting camera stream...')
      setError(null)
      
      // Check browser support first
      const browserInfo = checkBrowserSupport()
      
      if (!browserInfo.getUserMediaSupported) {
        throw new Error('getUserMedia not supported')
      }

      // Test device enumeration
      await testDeviceEnumeration()
      
      // Test permissions
      await testCameraPermissions()
      
      // Start actual stream
      const constraints = {
        video: {
          width: { ideal: 1280, min: 320, max: 1920 },
          height: { ideal: 720, min: 240, max: 1080 },
          facingMode: browserInfo.isMobile ? 'environment' : 'user',
          frameRate: { ideal: 30, min: 15 }
        },
        audio: false
      }
      
      addLog('Requesting camera stream with constraints...')
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      
      if (!videoRef.current) {
        throw new Error('Video element not available')
      }

      const video = videoRef.current
      video.srcObject = stream
      
      // Set up event listeners
      video.onloadedmetadata = () => {
        addLog(`Video metadata loaded: ${video.videoWidth}x${video.videoHeight}`)
      }
      
      video.oncanplay = () => {
        addLog('Video can play')
      }
      
      video.onplaying = () => {
        addLog('Video is playing')
        setIsStreaming(true)
      }
      
      video.onerror = (e) => {
        addLog(`Video error: ${e}`)
        setError('Video playback error')
      }
      
      // Try to play
      await video.play()
      addLog('✅ Camera stream started successfully')
      
    } catch (err: any) {
      addLog(`❌ Camera stream failed: ${err.name} - ${err.message}`)
      setIsStreaming(false)
      
      if (!error) { // Don't override specific error messages
        setError(`Failed to start camera: ${err.message}`)
      }
    }
  }

  const stopCameraStream = () => {
    addLog('Stopping camera stream...')
    
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach(track => {
        track.stop()
        addLog(`Stopped track: ${track.kind}`)
      })
      videoRef.current.srcObject = null
    }
    
    setIsStreaming(false)
    addLog('Camera stream stopped')
  }

  useEffect(() => {
    // Initial browser check
    checkBrowserSupport()
    
    return () => {
      stopCameraStream()
    }
  }, [])

  return (
    <div className="p-6 bg-white rounded-lg border shadow-sm max-w-4xl">
      <h3 className="text-xl font-bold mb-4 text-gray-900">🔧 AR Camera Debug Tool</h3>
      
      {/* Controls */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={startCameraStream}
          disabled={isStreaming}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
        >
          {isStreaming ? '✅ Camera Active' : '🎥 Start Camera'}
        </button>
        
        {isStreaming && (
          <button
            onClick={stopCameraStream}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            ⏹️ Stop Camera
          </button>
        )}
        
        <button
          onClick={checkBrowserSupport}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          🔍 Check Browser
        </button>
        
        <button
          onClick={testDeviceEnumeration}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          📱 List Cameras
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="font-semibold text-red-800 mb-2">❌ Error</h4>
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Preview */}
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-800">📹 Camera Preview</h4>
          <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ minHeight: '300px' }}>
            {isStreaming ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-auto object-cover"
                style={{ minHeight: '300px', maxHeight: '400px' }}
              />
            ) : (
              <div className="flex items-center justify-center h-80 text-gray-500">
                <div className="text-center">
                  <div className="text-4xl mb-2">📷</div>
                  <p>Camera preview will appear here</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Debug Information */}
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-800">🔍 Debug Information</h4>
          
          {/* Browser Info */}
          <div className="bg-gray-50 p-3 rounded border">
            <h5 className="font-medium text-gray-700 mb-2">Browser Support</h5>
            <div className="text-xs space-y-1">
              <div>Platform: {debugInfo.platform}</div>
              <div>Protocol: {debugInfo.protocol}</div>
              <div>Secure Context: {debugInfo.isSecureContext ? '✅' : '❌'}</div>
              <div>MediaDevices: {debugInfo.mediaDevicesSupported ? '✅' : '❌'}</div>
              <div>getUserMedia: {debugInfo.getUserMediaSupported ? '✅' : '❌'}</div>
              <div>Mobile: {debugInfo.isMobile ? '✅' : '❌'}</div>
            </div>
          </div>

          {/* Available Cameras */}
          <div className="bg-gray-50 p-3 rounded border">
            <h5 className="font-medium text-gray-700 mb-2">Available Cameras ({devices.length})</h5>
            {devices.length > 0 ? (
              <div className="space-y-1 text-xs">
                {devices.map((device, index) => (
                  <div key={device.deviceId} className="truncate">
                    {index + 1}. {device.label || `Camera ${index + 1}`}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-500">No cameras detected</p>
            )}
          </div>

          {/* Debug Logs */}
          <div className="bg-gray-50 p-3 rounded border">
            <h5 className="font-medium text-gray-700 mb-2">Debug Logs</h5>
            <div className="max-h-48 overflow-y-auto">
              {logs.length > 0 ? (
                <div className="space-y-1 text-xs font-mono">
                  {logs.map((log, index) => (
                    <div key={index} className="text-gray-600">{log}</div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-gray-500">No logs yet</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="font-semibold text-blue-800 mb-2">📖 Troubleshooting Steps</h4>
        <ol className="text-sm text-blue-700 space-y-1 list-decimal list-inside">
          <li>Click "Check Browser" to verify compatibility</li>
          <li>Click "List Cameras" to see available devices</li>
          <li>Click "Start Camera" and allow permissions when prompted</li>
          <li>Check the debug logs for specific error messages</li>
          <li>If camera doesn't work, try a different browser or device</li>
        </ol>
      </div>
    </div>
  )
}

export default ARCameraDebug
