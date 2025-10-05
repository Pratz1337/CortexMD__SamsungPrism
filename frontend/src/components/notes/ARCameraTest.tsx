"use client"

import React, { useRef, useState, useEffect } from 'react'

const ARCameraTest: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])

  const testCameraAccess = async () => {
    try {
      setError(null)
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Camera API not supported in this browser')
        return
      }

      // Get available devices
      const deviceList = await navigator.mediaDevices.enumerateDevices()
      const cameras = deviceList.filter(device => device.kind === 'videoinput')
      setDevices(cameras)
      
      console.log('📷 Available cameras:', cameras)

      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment' // Prefer back camera
        },
        audio: false
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsStreaming(true)
        await videoRef.current.play()
        console.log('✅ Camera stream started successfully')
      }

    } catch (err: any) {
      console.error('❌ Camera test failed:', err)
      setError(`Camera error: ${err.message || 'Unknown error'}`)
      setIsStreaming(false)
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
  }

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="p-6 bg-white rounded-lg border shadow-sm">
      <h3 className="text-lg font-semibold mb-4">📷 AR Camera Test</h3>
      
      {/* Controls */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={testCameraAccess}
          disabled={isStreaming}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {isStreaming ? '✅ Camera Active' : '🔴 Start Camera Test'}
        </button>
        
        {isStreaming && (
          <button
            onClick={stopCamera}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            ⏹️ Stop Camera
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded">
          <p className="text-red-700 text-sm">
            <strong>Error:</strong> {error}
          </p>
        </div>
      )}

      {/* Camera Info */}
      {devices.length > 0 && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded">
          <p className="text-blue-800 text-sm font-medium mb-2">
            📱 Found {devices.length} camera(s):
          </p>
          <ul className="text-xs text-blue-700">
            {devices.map((device, index) => (
              <li key={device.deviceId}>
                {index + 1}. {device.label || `Camera ${index + 1}`} 
                ({device.deviceId.substring(0, 8)}...)
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Video Preview */}
      <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ minHeight: '300px' }}>
        {isStreaming ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-auto max-h-80 object-cover"
          />
        ) : (
          <div className="flex items-center justify-center h-60 text-gray-500">
            <div className="text-center">
              <div className="text-4xl mb-2">📷</div>
              <p>Click "Start Camera Test" to test camera functionality</p>
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="mt-4 text-sm text-gray-600">
        <p><strong>Instructions:</strong></p>
        <ul className="list-disc pl-5 space-y-1 mt-2">
          <li>Click "Start Camera Test" to request camera permissions</li>
          <li>Allow camera access when prompted by your browser</li>
          <li>You should see a live video feed if everything works correctly</li>
          <li>On mobile devices, this will try to use the back camera</li>
        </ul>
      </div>
    </div>
  )
}

export default ARCameraTest
