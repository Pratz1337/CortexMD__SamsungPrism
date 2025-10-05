"use client"

import { useState, useEffect, useRef } from "react"
import { DiagnosisAPI, API_BASE_URL } from "@/lib/api"
import { useDiagnosisStore } from "@/store/diagnosisStore"

interface DebugLogEntry {
  timestamp: string
  level: string
  message: string
  stage: string
}

interface ProcessingDetails {
  fol_verification?: {
    status: string
    overall_confidence: number
    verified_explanations: number
    total_explanations: number
    verification_summary: string
  }
  enhanced_verification?: {
    overall_status: string
    overall_confidence: number
    evidence_strength: string
    sources_count: number
  }
  online_verification?: {
    verification_status: string
    confidence_score: number
    sources_count: number
  }
}

interface DebugData {
  timestamp: string
  session_id: string
  status: string
  progress: number
  current_step: string
  processing_details: ProcessingDetails
  logs: DebugLogEntry[]
}

export function DebugVisualization() {
  const { currentSessionId } = useDiagnosisStore()
  const [debugData, setDebugData] = useState<DebugData | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [logs, setLogs] = useState<DebugLogEntry[]>([])
  const [expandedSections, setExpandedSections] = useState({
    logs: true,
    processing: true,
    verification: true
  })
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!currentSessionId) return

    let eventSource: EventSource | null = null

    const connectToDebugStream = () => {
      if (eventSource) {
        eventSource.close()
      }

      eventSource = new EventSource(`${API_BASE_URL}/stream/debug/${currentSessionId}`)

      eventSource.onopen = () => {
        setIsConnected(true)
        console.log('Debug stream connected')
      }

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.event === 'debug' && data.data) {
            const debugData: DebugData = data.data
            setDebugData(debugData)

            // Update logs
            if (debugData.logs && debugData.logs.length > 0) {
              setLogs(prevLogs => {
                const newLogs = [...prevLogs, ...debugData.logs]
                // Keep only last 50 logs to prevent memory issues
                return newLogs.slice(-50)
              })
            }
          }
        } catch (error) {
          console.error('Error parsing debug stream data:', error)
        }
      }

      eventSource.onerror = () => {
        setIsConnected(false)
        console.log('Debug stream connection lost, attempting to reconnect...')

        // Try to reconnect after 5 seconds
        setTimeout(() => {
          connectToDebugStream()
        }, 5000)
      }
    }

    connectToDebugStream()

    return () => {
      if (eventSource) {
        eventSource.close()
      }
    }
  }, [currentSessionId])

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  const getLogLevelColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
      case 'CRITICAL':
        return 'text-red-600 bg-red-50'
      case 'WARNING':
        return 'text-orange-600 bg-orange-50'
      case 'SUCCESS':
        return 'text-green-600 bg-green-50'
      default:
        return 'text-blue-600 bg-blue-50'
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  if (!debugData) {
    return (
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          <div className="text-gray-500 text-sm">🔌 Debug Stream</div>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <div className="text-xs text-gray-500">
            {isConnected ? 'Connected' : 'Connecting...'}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-gray-700 to-gray-900 text-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-xl">🔍</div>
            <div>
              <h3 className="text-lg font-bold">Real-time Debug Output</h3>
              <div className="text-xs opacity-80">
                Session: {debugData.session_id}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <div className="text-xs">
              {isConnected ? 'Live' : 'Disconnected'}
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
        {/* Current Status */}
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-semibold text-blue-800">
                {debugData.current_step}
              </div>
              <div className="text-sm text-blue-600">
                Progress: {debugData.progress}%
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-blue-600">
                {formatTimestamp(debugData.timestamp)}
              </div>
            </div>
          </div>
        </div>

        {/* Processing Details */}
        {debugData.processing_details && Object.keys(debugData.processing_details).length > 0 && (
          <div>
            <button
              onClick={() => toggleSection('processing')}
              className="flex items-center justify-between w-full p-2 bg-gray-100 rounded-lg hover:bg-gray-200"
            >
              <span className="font-semibold text-gray-800">Processing Details</span>
              <span className="text-gray-600">
                {expandedSections.processing ? '▼' : '▶'}
              </span>
            </button>

            {expandedSections.processing && (
              <div className="mt-2 space-y-2">
                {debugData.processing_details.fol_verification && (
                  <div className="bg-purple-50 rounded-lg p-3">
                    <div className="font-semibold text-purple-800 mb-2">🧠 FOL Verification</div>
                    <div className="text-sm space-y-1">
                      <div>Status: {debugData.processing_details.fol_verification.status}</div>
                      <div>Confidence: {debugData.processing_details.fol_verification.overall_confidence.toFixed(2)}</div>
                      <div>Verified: {debugData.processing_details.fol_verification.verified_explanations}/{debugData.processing_details.fol_verification.total_explanations}</div>
                      {debugData.processing_details.fol_verification.verification_summary && (
                        <div className="text-xs text-purple-700 mt-2 p-2 bg-purple-100 rounded">
                          {debugData.processing_details.fol_verification.verification_summary}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {debugData.processing_details.enhanced_verification && (
                  <div className="bg-green-50 rounded-lg p-3">
                    <div className="font-semibold text-green-800 mb-2">🧪 Enhanced Verification</div>
                    <div className="text-sm space-y-1">
                      <div>Status: {debugData.processing_details.enhanced_verification.overall_status}</div>
                      <div>Confidence: {debugData.processing_details.enhanced_verification.overall_confidence.toFixed(2)}</div>
                      <div>Sources: {debugData.processing_details.enhanced_verification.sources_count}</div>
                      <div>Evidence: {debugData.processing_details.enhanced_verification.evidence_strength}</div>
                    </div>
                  </div>
                )}

                {debugData.processing_details.online_verification && (
                  <div className="bg-orange-50 rounded-lg p-3">
                    <div className="font-semibold text-orange-800 mb-2">🌐 Online Verification</div>
                    <div className="text-sm space-y-1">
                      <div>Status: {debugData.processing_details.online_verification.verification_status}</div>
                      <div>Confidence: {debugData.processing_details.online_verification.confidence_score.toFixed(2)}</div>
                      <div>Sources: {debugData.processing_details.online_verification.sources_count}</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Live Logs */}
        <div>
          <button
            onClick={() => toggleSection('logs')}
            className="flex items-center justify-between w-full p-2 bg-gray-100 rounded-lg hover:bg-gray-200"
          >
            <span className="font-semibold text-gray-800">Live Logs ({logs.length})</span>
            <span className="text-gray-600">
              {expandedSections.logs ? '▼' : '▶'}
            </span>
          </button>

          {expandedSections.logs && (
            <div className="mt-2 bg-black rounded-lg p-3 max-h-48 overflow-y-auto font-mono text-xs">
              <div className="space-y-1">
                {logs.map((log, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <div className="text-gray-500 min-w-[60px]">
                      {formatTimestamp(log.timestamp)}
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-medium ${getLogLevelColor(log.level)}`}>
                      {log.level}
                    </div>
                    <div className="text-gray-300 flex-1">
                      {log.message}
                    </div>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}
        </div>

        {/* Verification Status */}
        <div>
          <button
            onClick={() => toggleSection('verification')}
            className="flex items-center justify-between w-full p-2 bg-gray-100 rounded-lg hover:bg-gray-200"
          >
            <span className="font-semibold text-gray-800">Verification Status</span>
            <span className="text-gray-600">
              {expandedSections.verification ? '▼' : '▶'}
            </span>
          </button>

          {expandedSections.verification && (
            <div className="mt-2 space-y-2">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <div className="bg-blue-50 rounded-lg p-3 text-center">
                  <div className="text-2xl mb-1">🧠</div>
                  <div className="font-semibold text-blue-800">FOL Logic</div>
                  <div className="text-xs text-blue-600">
                    {debugData.processing_details.fol_verification?.status || 'Pending'}
                  </div>
                  <div className="text-xs text-blue-500">
                    {debugData.processing_details.fol_verification?.overall_confidence ?
                      `${(debugData.processing_details.fol_verification.overall_confidence * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                </div>

                <div className="bg-green-50 rounded-lg p-3 text-center">
                  <div className="text-2xl mb-1">🧪</div>
                  <div className="font-semibold text-green-800">Enhanced</div>
                  <div className="text-xs text-green-600">
                    {debugData.processing_details.enhanced_verification?.overall_status || 'Pending'}
                  </div>
                  <div className="text-xs text-green-500">
                    {debugData.processing_details.enhanced_verification?.overall_confidence ?
                      `${(debugData.processing_details.enhanced_verification.overall_confidence * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                </div>

                <div className="bg-orange-50 rounded-lg p-3 text-center">
                  <div className="text-2xl mb-1">🌐</div>
                  <div className="font-semibold text-orange-800">Online</div>
                  <div className="text-xs text-orange-600">
                    {debugData.processing_details.online_verification?.verification_status || 'Pending'}
                  </div>
                  <div className="text-xs text-orange-500">
                    {debugData.processing_details.online_verification?.confidence_score ?
                      `${(debugData.processing_details.online_verification.confidence_score * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

