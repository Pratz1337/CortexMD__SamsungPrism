"use client"

import React, { useRef, useState, useEffect } from 'react'

const CameraTest: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cameraInfo, setCameraInfo] = useState<any>(null)
  const [streamInfo, setStreamInfo] = useState<any>(null)

  const startCamera = async () => {
    try {
      setError(null)
      console.log('Starting camera test...')

      // Simple constraints first
      const constraints = {
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      }

      console.log('Requesting camera with constraints:', constraints)
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      
      console.log('Camera stream obtained:', stream)
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setStreamInfo({
          id: stream.id,
          active: stream.active,
          tracks: stream.getVideoTracks().map(track => ({
            label: track.label,
            kind: track.kind,
            enabled: track.enabled,
            muted: track.muted,
            readyState: track.readyState,
            settings: track.getSettings(),
            constraints: track.getConstraints()
          }))
        })

        videoRef.current.onloadedmetadata = async () => {
          try {
            await videoRef.current!.play()
            setIsStreaming(true)
            setCameraInfo({
              videoWidth: videoRef.current!.videoWidth,
              videoHeight: videoRef.current!.videoHeight,
              duration: videoRef.current!.duration
            })
            console.log('Video is playing successfully!')
          } catch (playError) {
            console.error('Video play failed:', playError)
            setError(`Failed to play video: ${playError}`)
          }
        }

        videoRef.current.onerror = (e) => {
          console.error('Video error:', e)
          setError('Video element error')
        }
      }
    } catch (err: any) {
      console.error('Camera access failed:', err)
      setError(`Camera error: ${err.message}`)
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setIsStreaming(false)
    setCameraInfo(null)
    setStreamInfo(null)
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6 text-center">📱 AR Camera Test</h2>
      
      <div className="space-y-6">
        {/* Controls */}
        <div className="flex justify-center gap-4">
          <button
            onClick={startCamera}
            disabled={isStreaming}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            Start Camera Test
          </button>
          <button
            onClick={stopCamera}
            disabled={!isStreaming}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
          >
            Stop Camera
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="font-bold text-red-800">Error:</h3>
            <p className="text-red-600">{error}</p>
          </div>
        )}

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
                <p>Click "Start Camera Test" to begin</p>
              </div>
            </div>
          )}
        </div>

        {/* Debug Info */}
        {streamInfo && (
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-bold mb-2">Stream Information:</h3>
            <pre className="text-sm bg-white p-3 rounded border overflow-auto">
              {JSON.stringify(streamInfo, null, 2)}
            </pre>
          </div>
        )}

        {cameraInfo && (
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-bold mb-2">Video Information:</h3>
            <pre className="text-sm bg-white p-3 rounded border">
              {JSON.stringify(cameraInfo, null, 2)}
            </pre>
          </div>
        )}

        {/* Instructions */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h3 className="font-bold text-yellow-800 mb-2">📋 Test Instructions:</h3>
          <ul className="text-yellow-700 space-y-1 text-sm">
            <li>• Make sure you're accessing via HTTPS (https://192.168.1.6:3000)</li>
            <li>• Allow camera permissions when prompted</li>
            <li>• Check if the video feed appears in the black box above</li>
            <li>• Review the debug information below for troubleshooting</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default CameraTest
