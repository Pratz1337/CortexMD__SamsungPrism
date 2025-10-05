"use client"

import { useState, useEffect, useCallback } from "react"
import { toast } from "react-hot-toast"
import { DiagnosisForm } from "@/components/diagnosis/DiagnosisForm"
import { DiagnosisResults } from "@/components/diagnosis/DiagnosisResults_new"
import { ProcessingStatus } from "@/components/diagnosis/ProcessingStatus"
import { ChatInterface } from "@/components/chat/ChatInterface"
import { ClinicalNotesView } from "@/components/patient/ClinicalNotesView"
import { PatientVisitsView } from "@/components/patient/PatientVisitsView"
import { useDiagnosisStore } from "@/store/diagnosisStore"
import { api, API_BASE_URL, DiagnosisAPI } from "@/lib/api"
import { MedicalMarkdownText } from "@/utils/markdown"
import { ConcernTrendChart } from "./ConcernTrendChart"
import { HealthcareLoadingScreen } from "@/components/ui/HealthcareLoadingScreen"
import { debounce, throttle, getCachedResponse, setCachedResponse } from "@/utils/debounce"

interface PatientDashboardProps {
  patientId: string
  onBack: () => void
}

interface PatientData {
  patient_info: {
    patient_id: string
    patient_name: string
    current_status: string
    admission_date: string
  }
  diagnosis_history: Array<{
    session_id: string
    created_at: string
    updated_at?: string
    status: string
    primary_diagnosis: string
    confidence_score: number
    processing_time?: number
    ai_model_used?: string
    verification_status?: string
    symptoms_summary?: string
    diagnosis_result?: any // Full diagnosis details
    patient_input?: any // Full patient input
  }>
  concern_data: {
    current_concern_score: number
    current_risk_level: string
    risk_factors: string[]
    visits_24h: number
    notes_24h: number
    last_assessment?: string
    trend_direction?: string
    alert_triggered?: boolean
    depth_metrics?: any
    score_trend: Array<{
      score: number
      level: string
      timestamp: string
    }>
  }
  chat_history: any[]
  total_diagnoses: number
  loading_complete?: boolean
  performance?: {
    cached?: boolean
    response_time_ms?: number
    basic_load_time_ms?: number
    total_load_time_ms?: number
  }
}

export function PatientDashboard({ patientId, onBack }: PatientDashboardProps) {
  const [patientData, setPatientData] = useState<PatientData | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<"overview" | "diagnosis" | "notes" | "visits" | "chat">("overview")
  const [startingNewDiagnosis, setStartingNewDiagnosis] = useState(false)
  const [expandedDiagnoses, setExpandedDiagnoses] = useState<Record<string, boolean>>({})
  const [realtimeMetrics, setRealtimeMetrics] = useState<any>(null)
  const [metricsLoading, setMetricsLoading] = useState(false)

  const {
    diagnosisResults,
    processingStatus,
    currentSessionId,
    isLoading,
    setCurrentSessionId,
    setDiagnosisResults,
    setProcessingStatus,
    setIsLoading,
  } = useDiagnosisStore()

  const fetchPatientData = useCallback(async () => {
    const startTime = performance.now()
    
    try {
      // PHASE 1: Get basic patient info instantly with ultra-fast caching
      const patientResponse = await api.get(`/api/patients/${patientId}?fast=true`)
      const basicPatientData = patientResponse.data

      // Show basic data immediately
      const basicCombinedData = {
        ...basicPatientData,
        diagnosis_history: [],
        total_diagnoses: 0,
        loading_complete: false
      }
      
      setPatientData(basicCombinedData)
      setLoading(false) // Stop loading spinner immediately
      
      const basicLoadTime = performance.now() - startTime
      console.log(`⚡ Basic patient data loaded in ${basicLoadTime.toFixed(1)}ms`)
      
      // PHASE 2: Load diagnosis history in parallel (background)
      setTimeout(async () => {
        try {
          const diagnosisResponse = await api.get(`/api/patients/${patientId}/diagnosis/history`)
          const diagnosisData = diagnosisResponse.data
          const diagnosisHistory = diagnosisData.diagnosis_history || []
          
          // Update with complete data
          const completeCombinedData = {
            ...basicPatientData,
            diagnosis_history: diagnosisHistory,
            total_diagnoses: diagnosisHistory.length,
            loading_complete: true,
            performance: {
              basic_load_time_ms: basicLoadTime,
              total_load_time_ms: performance.now() - startTime
            }
          }
          
          setPatientData(completeCombinedData)
          
          const totalLoadTime = performance.now() - startTime
          console.log(`🚀 Complete patient data loaded in ${totalLoadTime.toFixed(1)}ms`)
          
        } catch (e) {
          console.warn("Failed to fetch diagnosis history:", e)
          // Keep basic data even if diagnosis history fails
          setPatientData(prev => prev ? { ...prev, loading_complete: true } : null)
        }
      }, 10) // Load diagnosis history after 10ms delay
      
    } catch (error) {
      console.error("Error fetching patient data:", error)
      toast.error("Failed to load patient data")
      setLoading(false)
    }
  }, [patientId])

  const startNewDiagnosis = () => {
    // Clear previous diagnosis session
    setCurrentSessionId(null)
    setDiagnosisResults(null)
    setProcessingStatus(null)
    setIsLoading(false)
    setStartingNewDiagnosis(true)
    setActiveTab("diagnosis")
  }

  const toggleDiagnosisExpansion = (sessionId: string) => {
    setExpandedDiagnoses((prev) => ({
      ...prev,
      [sessionId]: !prev[sessionId],
    }))
  }

  const fetchRealtimeMetrics = useCallback(async () => {
    if (metricsLoading || loading) return // Prevent concurrent calls
    
    try {
      // Check cache first
      const cacheKey = `realtime_metrics_${patientId}`
      const cached = getCachedResponse(cacheKey)
      if (cached) {
        setRealtimeMetrics(cached)
        return
      }
      
      setMetricsLoading(true)
      const data = await DiagnosisAPI.getRealtimeMetrics(patientId)
      setRealtimeMetrics(data)
      
      // Cache the response
      setCachedResponse(cacheKey, data)
    } catch (error) {
      console.error("Error fetching realtime metrics:", error)
    } finally {
      setMetricsLoading(false)
    }
  }, [patientId, metricsLoading, loading])

  const refreshConcernData = useCallback(debounce(async () => {
    if (loading) return // Prevent concurrent refreshes
    
    try {
      setLoading(true)

      // Force recalculate CONCERN score
      await DiagnosisAPI.calculateRiskScore(patientId)

      // Refresh patient data and metrics
      await fetchPatientData()
      await fetchRealtimeMetrics()

      toast.success("CONCERN risk assessment updated!")
    } catch (error) {
      console.error("Error refreshing CONCERN data:", error)
      toast.error("Failed to refresh CONCERN data")
    } finally {
      setLoading(false)
    }
  }, 3000), [patientId, loading, fetchPatientData, fetchRealtimeMetrics]) // 3 second debounce

  // Throttled version to prevent excessive calls
  const throttledFetchRealtimeMetrics = useCallback(throttle(fetchRealtimeMetrics, 5000), [fetchRealtimeMetrics])

  useEffect(() => {
    fetchPatientData()
    // Initial fetch after a short delay to prevent conflicts
    setTimeout(() => {
      throttledFetchRealtimeMetrics()
    }, 1000)
  }, [patientId, fetchPatientData, throttledFetchRealtimeMetrics])

  // Auto-refresh metrics every 5 minutes (reduced frequency)
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    
    // Only set interval if not loading
    if (!loading && !metricsLoading) {
      interval = setInterval(() => {
        if (!loading && !metricsLoading) {
          throttledFetchRealtimeMetrics()
        }
      }, 300000) // 5 minutes instead of 1 minute
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [patientId, loading, metricsLoading])

  // Debounced Real-time CONCERN updates via SSE (only when not loading)
  useEffect(() => {
    if (!patientId || loading) return
    if (typeof window === "undefined" || !("EventSource" in window)) return

    let es: EventSource | null = null
    let reconnectTimeout: NodeJS.Timeout | null = null
    let connected = false

    const connectSSE = () => {
      if (es && connected) {
        es.close()
      }

      try {
        es = new EventSource(`${API_BASE_URL}/stream/concern/${patientId}`)
        
        es.onopen = () => {
          connected = true
          console.log(`SSE connected for patient ${patientId}`)
        }
        
        es.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data)
            if (payload.event === "concern" && payload.data && !loading) {
              setPatientData((prev) =>
                prev
                  ? {
                      ...prev,
                      concern_data: payload.data,
                    }
                  : prev,
              )
            }
          } catch (e) {
            console.error("CONCERN SSE parse error:", e)
          }
        }

        es.onerror = () => {
          connected = false
          if (es) {
            es.close()
          }
          // Only reconnect if patient hasn't changed and we're not loading
          if (!loading) {
            reconnectTimeout = setTimeout(connectSSE, 15000) // 15 seconds delay
          }
        }
      } catch (error) {
        console.error("Failed to create SSE connection:", error)
      }
    }

    // Initial connection with delay to prevent overlap with API calls
    const initialTimeout = setTimeout(connectSSE, 2000)

    return () => {
      connected = false
      if (initialTimeout) clearTimeout(initialTimeout)
      if (reconnectTimeout) clearTimeout(reconnectTimeout)
      if (es) {
        es.close()
      }
    }
  }, [patientId, loading])

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "critical":
        return "text-red-700 bg-red-100 border-red-300"
      case "high":
        return "text-orange-700 bg-orange-100 border-orange-300"
      case "medium":
        return "text-yellow-700 bg-yellow-100 border-yellow-300"
      case "low":
        return "text-green-700 bg-green-100 border-green-300"
      default:
        return "text-gray-700 bg-gray-100 border-gray-300"
    }
  }

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case "critical":
        return "🚨"
      case "high":
        return "⚠️"
      case "medium":
        return "⚡"
      case "low":
        return "✅"
      default:
        return "❓"
    }
  }

  // Show loading only if we have no patient data at all
  if (loading && !patientData) {
    return (
      <div className="flex items-center justify-center py-12">
        <HealthcareLoadingScreen 
          variant="heartbeat" 
          message="Loading patient data..." 
          className="min-h-0"
        />
      </div>
    )
  }

  if (!patientData) {
    return (
      <div className="text-center py-12">
        <div className="text-4xl mb-4">❌</div>
        <h3 className="text-xl font-semibold mb-2">Patient Not Found</h3>
        <p className="text-gray-600 mb-4">Could not load data for patient {patientId}</p>
        <button onClick={onBack} className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
          Back to Patients
        </button>
      </div>
    )
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case "overview":
        return (
          <div className="space-y-6">
            {/* Patient Info Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Patient Information</h3>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    patientData.patient_info.current_status === "active"
                      ? "bg-green-100 text-green-800"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  {patientData.patient_info.current_status.toUpperCase()}
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-gray-600">Patient ID</div>
                  <div className="font-medium">{patientData.patient_info.patient_id}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Patient Name</div>
                  <div className="font-medium">{patientData.patient_info.patient_name}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Admission Date</div>
                  <div className="font-medium">
                    {new Date(patientData.patient_info.admission_date).toLocaleDateString()}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Total Diagnoses</div>
                  <div className="font-medium">
                    {patientData.total_diagnoses}
                    {!patientData.loading_complete && (
                      <span className="ml-2 text-xs text-blue-500 animate-pulse">Loading...</span>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Performance Indicator */}
              {patientData.performance && (
                <div className="mt-4 p-2 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <span>
                      {patientData.performance.cached ? '⚡ Cached' : '🐢 Database'} - 
                      {patientData.performance.response_time_ms || patientData.performance.basic_load_time_ms}ms
                    </span>
                    {patientData.loading_complete === false && (
                      <span className="text-blue-500 animate-pulse">Loading diagnosis history...</span>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Real-Time Metrics */}
            {realtimeMetrics && (
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg shadow p-6 border border-indigo-200">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                  <span className="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></span>
                  Real-Time Patient Metrics
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white p-4 rounded-lg">
                    <div className="text-sm text-gray-600">Heart Rate</div>
                    <div className="text-2xl font-bold text-red-600">
                      {realtimeMetrics.vitals?.heart_rate || "--"} <span className="text-sm">bpm</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {realtimeMetrics.vitals?.heart_rate_trend || "stable"}
                    </div>
                  </div>
                  <div className="bg-white p-4 rounded-lg">
                    <div className="text-sm text-gray-600">Blood Pressure</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {realtimeMetrics.vitals?.blood_pressure || "--"}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">{realtimeMetrics.vitals?.bp_trend || "normal"}</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg">
                    <div className="text-sm text-gray-600">O₂ Saturation</div>
                    <div className="text-2xl font-bold text-green-600">
                      {realtimeMetrics.vitals?.oxygen_saturation || "--"}
                      <span className="text-sm">%</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">{realtimeMetrics.vitals?.o2_trend || "normal"}</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg">
                    <div className="text-sm text-gray-600">Temperature</div>
                    <div className="text-2xl font-bold text-amber-600">
                      {realtimeMetrics.vitals?.temperature || "--"}
                      <span className="text-sm">°F</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">{realtimeMetrics.vitals?.temp_trend || "normal"}</div>
                  </div>
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <div className="text-xs text-gray-500">
                    Last updated: {new Date(realtimeMetrics.last_updated || Date.now()).toLocaleTimeString()}
                  </div>
                  <div className="flex items-center space-x-2">
                    {metricsLoading && (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-indigo-600"></div>
                    )}
                    <span className="text-xs text-indigo-600">Live monitoring active</span>
                  </div>
                </div>
              </div>
            )}

            {/* Enhanced Interactive CONCERN Trend Chart */}
            {patientData.concern_data?.score_trend && patientData.concern_data.score_trend.length > 0 && (
              <ConcernTrendChart
                trendData={patientData.concern_data.score_trend}
                currentScore={patientData.concern_data.current_concern_score || 0}
                currentLevel={patientData.concern_data.current_risk_level || "low"}
                patientName={patientData.patient_info.patient_name}
                patientId={patientId}
                showAdvancedMetrics={true}
                depthMetrics={patientData.concern_data.depth_metrics}
              />
            )}

            {/* CONCERN Risk Assessment */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold">CONCERN Risk Assessment</h3>
                <button
                  onClick={refreshConcernData}
                  className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm flex items-center gap-1"
                  disabled={loading}
                >
                  <span>🔄</span>
                  {loading ? "Refreshing..." : "Refresh"}
                </button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div
                  className={`p-4 rounded-lg border-2 ${getRiskColor(patientData.concern_data?.current_risk_level || "low")}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">Risk Level</span>
                    <span className="text-2xl">
                      {getRiskIcon(patientData.concern_data?.current_risk_level || "low")}
                    </span>
                  </div>
                  <div className="text-2xl font-bold">
                    {(patientData.concern_data?.current_risk_level || "low").toUpperCase()}
                  </div>
                  <div className="text-sm opacity-75">
                    Score: {((patientData.concern_data?.current_concern_score || 0) * 100).toFixed(1)}%
                  </div>
                  {patientData.concern_data?.last_assessment && (
                    <div className="text-xs opacity-60 mt-1">
                      Updated: {new Date(patientData.concern_data.last_assessment).toLocaleTimeString()}
                    </div>
                  )}
                </div>

                <div className="p-4 bg-blue-50 rounded-lg border-2 border-blue-200">
                  <div className="font-medium text-blue-900 mb-2">24h Activity</div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>Visits:</span>
                      <span className="font-bold">{patientData.concern_data?.visits_24h || 0}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Notes:</span>
                      <span className="font-bold">{patientData.concern_data?.notes_24h || 0}</span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-purple-50 rounded-lg border-2 border-purple-200">
                  <div className="flex justify-between items-center mb-2">
                    <div className="font-medium text-purple-900">Risk Factors</div>
                    {patientData.concern_data?.trend_direction && (
                      <div className="text-sm flex items-center gap-1">
                        <span>
                          {patientData.concern_data.trend_direction === "increasing"
                            ? "📈"
                            : patientData.concern_data.trend_direction === "decreasing"
                              ? "📉"
                              : "➡️"}
                        </span>
                        <span className="text-xs text-purple-600 capitalize">
                          {patientData.concern_data.trend_direction}
                        </span>
                      </div>
                    )}
                  </div>
                  <div className="text-sm">
                    {(patientData.concern_data?.risk_factors || []).length > 0 ? (
                      <div className="space-y-1">
                        {(patientData.concern_data?.risk_factors || []).slice(0, 2).map((factor, index) => (
                          <div key={index} className="text-xs">
                            • {factor}
                          </div>
                        ))}
                        {(patientData.concern_data?.risk_factors || []).length > 2 && (
                          <div className="text-xs text-purple-600">
                            +{(patientData.concern_data?.risk_factors || []).length - 2} more
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-gray-500">No risk factors detected</div>
                    )}
                  </div>
                  {patientData.concern_data?.alert_triggered && (
                    <div className="mt-2 px-2 py-1 bg-red-100 text-red-700 text-xs rounded font-medium">
                      🚨 Alert Active
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Recent Diagnosis History */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Recent Diagnoses</h3>
                <button
                  onClick={startNewDiagnosis}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                >
                  🩺 New Diagnosis
                </button>
              </div>

              {patientData.diagnosis_history.length > 0 ? (
                <div className="space-y-4">
                  {patientData.diagnosis_history.slice(0, 5).map((diagnosis, index) => {
                    const isExpanded = expandedDiagnoses[diagnosis.session_id]
                    const diagnosisResult = diagnosis.diagnosis_result || {}

                    return (
                      <div
                        key={diagnosis.session_id}
                        className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden"
                      >
                        {/* Main Diagnosis Header - Always Visible */}
                        <div
                          className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                          onClick={() => toggleDiagnosisExpansion(diagnosis.session_id)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-3">
                                <div className="flex-shrink-0">
                                  <div
                                    className={`w-3 h-3 rounded-full ${
                                      diagnosis.confidence_score >= 0.8
                                        ? "bg-green-500"
                                        : diagnosis.confidence_score >= 0.6
                                          ? "bg-yellow-500"
                                          : "bg-red-500"
                                    }`}
                                  ></div>
                                </div>
                                <div>
                                  <div className="font-semibold text-gray-900 text-lg">
                                    {diagnosis.primary_diagnosis || "Medical Analysis Completed"}
                                  </div>
                                  <div className="text-sm text-gray-600 mt-1">
                                    {new Date(diagnosis.created_at).toLocaleDateString("en-US", {
                                      year: "numeric",
                                      month: "long",
                                      day: "numeric",
                                      hour: "2-digit",
                                      minute: "2-digit",
                                    })}{" "}
                                    • Status: {diagnosis.status}
                                  </div>
                                </div>
                              </div>
                            </div>

                            <div className="flex items-center space-x-4">
                              {diagnosis.confidence_score > 0 && (
                                <div className="text-right">
                                  <div
                                    className={`text-lg font-bold ${
                                      diagnosis.confidence_score >= 0.8
                                        ? "text-green-600"
                                        : diagnosis.confidence_score >= 0.6
                                          ? "text-yellow-600"
                                          : "text-red-600"
                                    }`}
                                  >
                                    {(diagnosis.confidence_score * 100).toFixed(1)}%
                                  </div>
                                  <div className="text-xs text-gray-500">Confidence</div>
                                </div>
                              )}

                              <div className="text-gray-400">{isExpanded ? "▼" : "▶"}</div>
                            </div>
                          </div>
                        </div>

                        {/* Expandable Details Section */}
                        {isExpanded && (
                          <div className="border-t border-gray-200 bg-gray-50">
                            <div className="p-6 space-y-6">
                              {/* Clinical Summary */}
                              <div>
                                <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                                  <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                                  Clinical Summary
                                </h4>
                                {diagnosis.symptoms_summary && (
                                  <p className="text-gray-700 text-sm leading-relaxed bg-white p-3 rounded border">
                                    {diagnosis.symptoms_summary}
                                  </p>
                                )}
                              </div>

                              {/* Diagnosis Details */}
                              {diagnosisResult.primary_diagnosis && (
                                <div>
                                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                                    <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                    Diagnosis Details
                                  </h4>
                                  <div className="bg-white p-4 rounded border space-y-3">
                                    <div>
                                      <span className="font-medium text-gray-600">Primary Diagnosis:</span>
                                      <p className="text-gray-900 mt-1">{diagnosisResult.primary_diagnosis}</p>
                                    </div>

                                    {diagnosisResult.clinical_impression && (
                                      <div>
                                        <span className="font-medium text-gray-600">Clinical Impression:</span>
                                        <p className="text-gray-900 mt-1">{diagnosisResult.clinical_impression}</p>
                                      </div>
                                    )}

                                    {diagnosisResult.reasoning_paths && diagnosisResult.reasoning_paths.length > 0 && (
                                      <div>
                                        <span className="font-medium text-gray-600">Medical Reasoning:</span>
                                        <div className="mt-1 space-y-2">
                                          {diagnosisResult.reasoning_paths.map((reasoning: string, idx: number) => (
                                            <MedicalMarkdownText
                                              key={idx}
                                              className="text-gray-900 text-sm leading-relaxed"
                                            >
                                              {reasoning}
                                            </MedicalMarkdownText>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              )}

                              {/* Clinical Recommendations */}
                              {diagnosisResult.clinical_recommendations &&
                                diagnosisResult.clinical_recommendations.length > 0 && (
                                  <div>
                                    <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                                      <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                                      Clinical Recommendations
                                    </h4>
                                    <div className="bg-white p-4 rounded border">
                                      <ul className="space-y-2">
                                        {diagnosisResult.clinical_recommendations.map((rec: string, idx: number) => (
                                          <li key={idx} className="text-gray-900 text-sm flex items-start">
                                            <span className="text-purple-500 mr-2 mt-1">•</span>
                                            {rec}
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  </div>
                                )}

                              {/* Processing Information */}
                              <div className="border-t pt-4">
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                  <div>
                                    <span className="text-gray-500">Processing Time:</span>
                                    <p className="font-medium">{diagnosis.processing_time?.toFixed(2) || "0.00"}s</p>
                                  </div>
                                  <div>
                                    <span className="text-gray-500">AI Model:</span>
                                    <p className="font-medium">{diagnosis.ai_model_used || "CortexMD"}</p>
                                  </div>
                                  <div>
                                    <span className="text-gray-500">Verification:</span>
                                    <p className="font-medium">{diagnosis.verification_status || "Complete"}</p>
                                  </div>
                                  <div>
                                    <span className="text-gray-500">Session ID:</span>
                                    <p className="font-mono text-xs">{diagnosis.session_id.slice(0, 8)}...</p>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  <div className="text-4xl mb-2">🩺</div>
                  <div>No diagnoses yet</div>
                  <div className="text-sm">Start your first AI-powered diagnosis</div>
                </div>
              )}
            </div>
          </div>
        )

      case "diagnosis":
        return (
          <div className="space-y-6">
            {!currentSessionId && !isLoading && <DiagnosisForm patientId={patientId} />}
            {(currentSessionId || isLoading) && processingStatus && <ProcessingStatus />}
            {diagnosisResults && <DiagnosisResults />}
            {!processingStatus && !diagnosisResults && currentSessionId && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Processing Diagnosis...</h3>
                <div className="flex items-center space-x-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                  <span>Analyzing patient data with AI...</span>
                </div>
              </div>
            )}
          </div>
        )

      case "notes":
        return <ClinicalNotesView patientId={patientId} />

      case "visits":
        return <PatientVisitsView patientId={patientId} />

      case "chat":
        return <ChatInterface patientId={patientId} />

      default:
        return null
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg p-4 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <div className="flex-1">
            <button onClick={onBack} className="mb-2 text-blue-100 hover:text-white transition text-sm sm:text-base">
              ← Back to Patients
            </button>
            <h1 className="text-2xl sm:text-3xl font-bold">{patientData.patient_info.patient_name}</h1>
            <p className="text-blue-100 text-sm sm:text-base">Patient ID: {patientData.patient_info.patient_id}</p>
          </div>
          <div className="text-left sm:text-right">
            <div
              className={`inline-flex items-center space-x-2 px-3 py-2 sm:px-4 sm:py-2 rounded-full text-sm sm:text-base ${getRiskColor(patientData.concern_data?.current_risk_level || "low")}`}
            >
              <span className="text-lg sm:text-xl">
                {getRiskIcon(patientData.concern_data?.current_risk_level || "low")}
              </span>
              <span className="font-medium">
                {(patientData.concern_data?.current_risk_level || "low").toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow">
        <nav className="flex flex-wrap gap-1 p-1">
          {[
            { id: "overview", label: "Overview", icon: "📊", shortLabel: "Overview" },
            { id: "diagnosis", label: "AI Diagnosis", icon: "🩺", shortLabel: "Diagnosis" },
            { id: "notes", label: "Clinical Notes", icon: "📝", shortLabel: "Notes" },
            { id: "visits", label: "Patient Visits", icon: "🏥", shortLabel: "Visits" },
            { id: "chat", label: "AI Chat", icon: "💬", shortLabel: "Chat" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 min-w-0 flex items-center justify-center gap-1 sm:gap-2 py-3 sm:py-4 px-2 sm:px-4 rounded-lg font-medium transition text-xs sm:text-sm md:text-base ${
                activeTab === tab.id ? "bg-blue-600 text-white" : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <span className="text-sm sm:text-base">{tab.icon}</span>
              <span className="hidden sm:inline">{tab.label}</span>
              <span className="sm:hidden">{tab.shortLabel}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div>{renderTabContent()}</div>
    </div>
  )
}
