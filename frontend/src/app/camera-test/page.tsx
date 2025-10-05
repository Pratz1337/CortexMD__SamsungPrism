"use client"

import React, { useState, useEffect } from 'react'
import { testCameraAccess, detectBrowserIssues, getCameraDevices } from '@/utils/cameraUtils'

export const dynamic = 'force-dynamic'

export default function CameraTestPage() {
  const [testResults, setTestResults] = useState<{
    supported: boolean
    devices: MediaDeviceInfo[]
    error?: string
  } | null>(null)
  const [browserIssues, setBrowserIssues] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [browserInfo, setBrowserInfo] = useState<{
    userAgent: string
    platform: string
    protocol: string
    hostname: string
    mediaDevicesSupported: boolean
    getUserMediaSupported: boolean
  } | null>(null)

  useEffect(() => {
    setBrowserIssues(detectBrowserIssues())
    if (typeof window !== 'undefined') {
      setBrowserInfo({
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        protocol: window.location.protocol,
        hostname: window.location.hostname,
        mediaDevicesSupported: !!navigator.mediaDevices,
        getUserMediaSupported: !!navigator.mediaDevices?.getUserMedia
      })
    }
  }, [])

  const runCameraTest = async () => {
    setIsLoading(true)
    try {
      const results = await testCameraAccess()
      setTestResults(results)
    } catch (error: any) {
      setTestResults({
        supported: false,
        devices: [],
        error: `Test failed: ${error.message}`
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">📷 Camera Diagnostics</h1>
          <p className="text-gray-600">Test camera functionality and diagnose issues</p>
        </div>

        {/* Test Camera Button */}
        <div className="text-center mb-8">
          <button
            onClick={runCameraTest}
            disabled={isLoading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Testing...</span>
              </div>
            ) : (
              '🔍 Test Camera Access'
            )}
          </button>
        </div>

        {/* Browser Issues */}
        {browserIssues.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <h3 className="text-lg font-semibold text-yellow-800 mb-2">⚠️ Potential Browser Issues</h3>
            <ul className="list-disc list-inside space-y-1">
              {browserIssues.map((issue, index) => (
                <li key={index} className="text-yellow-700 text-sm">{issue}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Test Results */}
        {testResults && (
          <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">📊 Test Results</h3>
            
            <div className={`p-4 rounded-lg mb-4 ${
              testResults.supported 
                ? 'bg-green-50 border border-green-200' 
                : 'bg-red-50 border border-red-200'
            }`}>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-lg">{testResults.supported ? '✅' : '❌'}</span>
                <span className={`font-semibold ${
                  testResults.supported ? 'text-green-800' : 'text-red-800'
                }`}>
                  Camera {testResults.supported ? 'Supported' : 'Not Supported'}
                </span>
              </div>
              
              {testResults.error && (
                <p className="text-red-700 text-sm">{testResults.error}</p>
              )}
            </div>

            {/* Device List */}
            <div className="mb-4">
              <h4 className="font-semibold mb-2">📱 Available Cameras ({testResults.devices.length})</h4>
              {testResults.devices.length > 0 ? (
                <div className="space-y-2">
                  {testResults.devices.map((device, index) => (
                    <div key={device.deviceId} className="bg-gray-50 p-3 rounded border">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="font-medium">{device.label || `Camera ${index + 1}`}</p>
                          <p className="text-sm text-gray-600">Device ID: {device.deviceId.substring(0, 20)}...</p>
                        </div>
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                          {device.kind}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">No camera devices found</p>
              )}
            </div>
          </div>
        )}

        {/* Browser Info */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
          <h3 className="text-lg font-semibold mb-4">🌐 Browser Information</h3>
          <div className="space-y-2 text-sm">
            {browserInfo ? (
              <>
                <div className="flex justify-between">
                  <span className="font-medium">User Agent:</span>
                  <span className="text-gray-600 text-right max-w-md truncate">{browserInfo.userAgent}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Platform:</span>
                  <span className="text-gray-600">{browserInfo.platform}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Protocol:</span>
                  <span className="text-gray-600">{browserInfo.protocol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Hostname:</span>
                  <span className="text-gray-600">{browserInfo.hostname}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">MediaDevices API:</span>
                  <span className={`${browserInfo.mediaDevicesSupported ? 'text-green-600' : 'text-red-600'}`}>
                    {browserInfo.mediaDevicesSupported ? 'Supported' : 'Not Supported'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">getUserMedia:</span>
                  <span className={`${browserInfo.getUserMediaSupported ? 'text-green-600' : 'text-red-600'}`}>
                    {browserInfo.getUserMediaSupported ? 'Supported' : 'Not Supported'}
                  </span>
                </div>
              </>
            ) : (
              <div className="text-gray-500">Loading browser information...</div>
            )}
          </div>
        </div>

        {/* Instructions */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-800 mb-4">📖 Troubleshooting Tips</h3>
          <div className="space-y-3 text-sm text-blue-700">
            <div>
              <strong>Camera not working?</strong>
              <ul className="list-disc list-inside mt-1 space-y-1">
                <li>Make sure you allow camera permissions when prompted</li>
                <li>Check if another application is using the camera</li>
                <li>Try refreshing the page</li>
                <li>Ensure you're using HTTPS (required for most browsers)</li>
              </ul>
            </div>
            <div>
              <strong>Video not showing?</strong>
              <ul className="list-disc list-inside mt-1 space-y-1">
                <li>Check browser console for errors</li>
                <li>Try a different browser</li>
                <li>Disable browser extensions temporarily</li>
                <li>Clear browser cache and cookies</li>
              </ul>
            </div>
            <div>
              <strong>Still having issues?</strong>
              <ul className="list-disc list-inside mt-1 space-y-1">
                <li>Open browser developer tools (F12) and check the Console tab</li>
                <li>Look for camera-related error messages</li>
                <li>Try the AR scanner with these diagnostic results</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
