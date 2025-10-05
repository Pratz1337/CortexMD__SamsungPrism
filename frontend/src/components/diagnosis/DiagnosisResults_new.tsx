"use client"

import { useState, useEffect } from "react"
import {
  ChartBarIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ClockIcon,
  CpuChipIcon,
  BeakerIcon,
  BookOpenIcon,
  GlobeAltIcon,
  MagnifyingGlassIcon,
  ArrowDownTrayIcon,
  SparklesIcon,
  ShieldCheckIcon,
  DocumentTextIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  StarIcon,
  LightBulbIcon,
  FireIcon,
  PhotoIcon,
  CogIcon,
  ClipboardIcon,
  ClipboardDocumentCheckIcon,
  CalendarIcon,
  TagIcon
} from "@heroicons/react/24/outline"
import { useDiagnosisStore } from "@/store/diagnosisStore"
import { DiagnosisAPI } from "@/lib/api"
import ReportDownloadModal from "./ReportDownloadModal"
import toast from "react-hot-toast"

export function DiagnosisResults() {
  const { diagnosisResults, currentSessionId } = useDiagnosisStore()
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    primary: true,
    explanations: true,
    gradcam: true, // Add GradCAM section
    online_verification: true, // Add online verification section
    differential: true,
    recommendations: true,
    advanced: true, // Changed to true to show XAI features by default
  })
  const [showCelebration, setShowCelebration] = useState(true)
  const [showDownloadModal, setShowDownloadModal] = useState(false)

  useEffect(() => {
    // Auto-hide celebration after 3 seconds
    const timer = setTimeout(() => {
      setShowCelebration(false)
    }, 3000)
    return () => clearTimeout(timer)
  }, [])

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }))
  }

  const openDownloadModal = () => {
    setShowDownloadModal(true)
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-emerald-600"
    if (confidence >= 0.6) return "text-amber-600"
    return "text-red-500"
  }

  const getConfidenceBarColor = (confidence: number) => {
    if (confidence >= 0.8) return "bg-emerald-500"
    if (confidence >= 0.6) return "bg-amber-500"
    return "bg-red-500"
  }

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.8) return <CheckCircleIcon className="w-5 h-5 text-emerald-600" />
    if (confidence >= 0.6) return <ExclamationTriangleIcon className="w-5 h-5 text-amber-600" />
    return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />
  }

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case "critical":
        return "bg-red-50 text-red-800 border-red-200"
      case "high":
        return "bg-orange-50 text-orange-800 border-orange-200"
      case "medium":
        return "bg-amber-50 text-amber-800 border-amber-200"
      case "low":
        return "bg-emerald-50 text-emerald-800 border-emerald-200"
      default:
        return "bg-gray-50 text-gray-800 border-gray-200"
    }
  }

  if (!diagnosisResults) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="text-center">
          <div className="text-6xl mb-4">🔍</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">No Diagnosis Results</h2>
          <p className="text-gray-600">No results are available yet. Please run a diagnosis first.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-8">
      {/* Celebration Overlay */}
      {showCelebration && (
        <div className="fixed inset-0 bg-black bg-opacity-20 flex items-center justify-center z-50 pointer-events-none">
          <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md mx-4 animate-bounce">
            <div className="text-center">
              <div className="text-6xl mb-4">🎉</div>
              <h3 className="text-2xl font-bold text-green-600 mb-2">Diagnosis Complete!</h3>
              <p className="text-gray-600">AI analysis has been successfully completed</p>
            </div>
          </div>
        </div>
      )}

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6 md:space-y-8">
        {/* Hero Header */}
        <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 rounded-2xl shadow-xl text-white overflow-hidden">
          <div className="relative p-4 sm:p-6 md:p-8">
            <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between space-y-4 lg:space-y-0">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-4">
                  <SparklesIcon className="w-6 h-6 sm:w-8 sm:h-8" />
                  <h1 className="text-xl sm:text-2xl md:text-3xl font-bold">AI Diagnosis Complete</h1>
                </div>
                <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-6 text-blue-100">
                  <div className="flex items-center space-x-2">
                    <ClockIcon className="w-4 h-4" />
                    <span className="text-sm">
                      {new Date(diagnosisResults.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <DocumentTextIcon className="w-4 h-4" />
                    <span className="text-sm">Session: {currentSessionId}</span>
                  </div>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row lg:flex-col items-start sm:items-center lg:items-end space-y-4 sm:space-y-0 sm:space-x-4 lg:space-x-0 lg:space-y-4">
                <div className={`px-3 py-2 sm:px-4 sm:py-2 rounded-full border-2 font-semibold text-xs sm:text-sm ${getUrgencyColor(diagnosisResults.urgency_level)}`}>
                  <ExclamationTriangleIcon className="w-3 h-3 sm:w-4 sm:h-4 inline mr-2" />
                  {(diagnosisResults.urgency_level || "MEDIUM").toUpperCase()} PRIORITY
                </div>

                <button
                  onClick={openDownloadModal}
                  className="bg-white text-blue-600 px-4 py-2 sm:px-6 sm:py-3 rounded-xl hover:bg-blue-50 transition-all duration-200 flex items-center font-semibold shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 text-sm sm:text-base"
                >
                  <ArrowDownTrayIcon className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
                  <span className="hidden sm:inline">Download Report</span>
                  <span className="sm:hidden">Download</span>
                </button>
              </div>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-8">
              <div className="bg-white bg-opacity-20 rounded-xl p-4 backdrop-blur">
                <div className="text-center">
                  <div className="text-2xl font-bold mb-1">
                    {(diagnosisResults.processing_time || 0).toFixed(1)}s
                  </div>
                  <div className="text-sm text-blue-100">Processing Time</div>
                </div>
              </div>
              <div className="bg-white bg-opacity-20 rounded-xl p-4 backdrop-blur">
                <div className="text-center">
                  <div className="text-2xl font-bold mb-1">
                    {((diagnosisResults.confidence_metrics?.overall_confidence || 0) * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-blue-100">Confidence</div>
                </div>
              </div>
              <div className="bg-white bg-opacity-20 rounded-xl p-4 backdrop-blur">
                <div className="text-center">
                  <div className="text-2xl font-bold mb-1">
                    {((diagnosisResults.confidence_metrics?.data_quality || 0) * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-blue-100">Data Quality</div>
                </div>
              </div>
              <div className="bg-white bg-opacity-20 rounded-xl p-4 backdrop-blur">
                <div className="text-center">
                  <div className="text-2xl font-bold mb-1">
                    {((diagnosisResults.confidence_metrics?.source_reliability || 0) * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-blue-100">Source Reliability</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Primary Diagnosis - Spotlight */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
          <div className="bg-gradient-to-r from-emerald-50 to-blue-50 p-4 sm:p-6 border-b border-gray-100">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-emerald-100 rounded-xl">
                <ShieldCheckIcon className="w-6 h-6 sm:w-8 sm:h-8 text-emerald-600" />
              </div>
              <div>
                <h2 className="text-xl sm:text-2xl font-bold text-gray-800">Primary Diagnosis</h2>
                <p className="text-gray-600 text-sm sm:text-base">AI-powered medical analysis result</p>
              </div>
            </div>
          </div>

          <div className="p-4 sm:p-6 md:p-8">
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between mb-6 space-y-4 sm:space-y-0">
              <div className="flex-1">
                <h3 className="text-xl sm:text-2xl font-bold text-gray-800 mb-3">
                  {diagnosisResults.primary_diagnosis?.condition || "No condition specified"}
                </h3>
                {diagnosisResults.primary_diagnosis?.icd_code && (
                  <div className="inline-flex items-center bg-blue-50 text-blue-800 px-3 py-1 rounded-lg text-sm font-medium">
                    <DocumentTextIcon className="w-4 h-4 mr-2" />
                    ICD Code: {diagnosisResults.primary_diagnosis.icd_code}
                  </div>
                )}
              </div>

              <div className="flex items-center space-x-3">
                {getConfidenceIcon(diagnosisResults.primary_diagnosis?.confidence || 0)}
                <div className="text-right">
                  <div className={`text-lg sm:text-xl font-bold ${getConfidenceColor(diagnosisResults.primary_diagnosis?.confidence || 0)}`}>
                    {((diagnosisResults.primary_diagnosis?.confidence || 0) * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-gray-600">Confidence</div>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-xl p-4 sm:p-6">
              <p className="text-gray-700 leading-relaxed text-sm sm:text-base">
                {diagnosisResults.primary_diagnosis?.description || "No description available"}
              </p>
            </div>
          </div>
        </div>

        {/* FOL Verification XAI - Main MVP Feature */}
        {diagnosisResults.fol_verification && (
          <div className="bg-white rounded-2xl shadow-2xl overflow-hidden border-2 border-purple-200 mb-8 transform hover:scale-[1.01] transition-all duration-300">
            <div className="bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-700 p-8 text-white relative overflow-hidden">
              {/* Animated background pattern */}
              <div className="absolute inset-0 opacity-30">
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent transform -skew-x-12 animate-pulse"></div>
              </div>
              
              <div className="relative z-10">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-6 space-y-4 lg:space-y-0">
                  <div className="flex items-center space-x-4">
                    <div className="p-2 sm:p-3 bg-white bg-opacity-20 rounded-2xl backdrop-blur-sm shadow-lg">
                      <CpuChipIcon className="w-8 h-8 sm:w-10 sm:h-10 text-white drop-shadow-lg" />
                    </div>
                    <div>
                      <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-3 mb-2">
                        <h1 className="text-xl sm:text-2xl md:text-3xl font-bold">FOL Logic Verification XAI</h1>
                        <div className="px-2 py-1 sm:px-3 sm:py-1 bg-yellow-400 bg-opacity-20 rounded-full border border-yellow-300 self-start">
                          <span className="text-yellow-200 text-xs font-medium">MVP FEATURE</span>
                        </div>
                      </div>
                      <p className="text-purple-100 text-sm sm:text-base md:text-lg">First-Order Logic Explainable AI - Medical Reasoning Verification</p>
                    </div>
                  </div>
                  <div className="text-left lg:text-right">
                    <div className={`text-2xl sm:text-3xl md:text-4xl font-bold mb-1 ${
                      (diagnosisResults.fol_verification.overall_confidence || 0) > 0.8 ? 'text-green-300' :
                      (diagnosisResults.fol_verification.overall_confidence || 0) > 0.6 ? 'text-yellow-300' : 'text-red-300'
                    } drop-shadow-lg`}>
                      {((diagnosisResults.fol_verification.overall_confidence || 0) * 100).toFixed(0)}%
                    </div>
                    <div className="text-purple-200 text-sm font-medium flex items-center">
                      Logic Confidence
                      {(diagnosisResults.fol_verification as any)?.ai_service_used === 'Gemini' && (
                        <div className="ml-2 px-2 py-1 bg-green-400 bg-opacity-30 rounded-full flex items-center">
                          <div className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></div>
                          <span className="text-xs font-bold text-green-200">GEMINI ACTIVE</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20 hover:bg-opacity-25 transition-all">
                    <div className="text-2xl font-bold text-white">
                      {diagnosisResults.fol_verification.verified_predicates || diagnosisResults.fol_verification.verified_explanations || 0}
                    </div>
                    <div className="text-purple-100 text-sm">Verified Predicates</div>
                  </div>
                  <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20 hover:bg-opacity-25 transition-all">
                    <div className="text-2xl font-bold text-white">
                      {diagnosisResults.fol_verification.total_predicates || diagnosisResults.fol_verification.total_explanations || 0}
                    </div>
                    <div className="text-purple-100 text-sm">Total Predicates</div>
                  </div>
                  <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20 hover:bg-opacity-25 transition-all">
                    <div className="text-2xl font-bold text-white">
                      {((diagnosisResults.fol_verification.success_rate || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="text-purple-100 text-sm">Success Rate</div>
                  </div>
                  <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20 hover:bg-opacity-25 transition-all">
                    <div className={`text-2xl font-bold ${
                      diagnosisResults.fol_verification.status === 'VERIFIED' || diagnosisResults.fol_verification.status === 'FULLY_VERIFIED' 
                        ? 'text-green-300' : 'text-yellow-300'
                    }`}>
                      {diagnosisResults.fol_verification.status || 'VERIFIED'}
                    </div>
                    <div className="text-purple-100 text-sm">Status</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="p-8 space-y-6">
              {/* Progress Bar with Enhanced Visualization */}
              <div className="space-y-4">
                <div className="flex justify-between text-sm font-medium text-gray-700">
                  <span className="flex items-center">
                    <SparklesIcon className="w-4 h-4 mr-2 text-purple-600" />
                    Logic Verification Progress
                  </span>
                  <span className="font-bold text-purple-700">
                    {((diagnosisResults.fol_verification.success_rate || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="relative w-full bg-gray-200 rounded-full h-4 overflow-hidden shadow-inner">
                  <div 
                    className="bg-gradient-to-r from-purple-500 via-indigo-500 to-purple-600 h-4 rounded-full transition-all duration-2000 ease-out relative overflow-hidden"
                    style={{ width: `${(diagnosisResults.fol_verification.success_rate || 0) * 100}%` }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
                  </div>
                  {(diagnosisResults.fol_verification.success_rate || 0) > 0.8 && (
                    <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
                      <StarIcon className="w-3 h-3 text-yellow-400 fill-current" />
                    </div>
                  )}
                </div>
                <div className="text-center text-sm text-gray-600">
                  <span className="font-medium">
                    {diagnosisResults.fol_verification.verified_predicates || 0} out of {diagnosisResults.fol_verification.total_predicates || 0} predicates verified
                  </span>
                </div>
              </div>

              {/* Medical Reasoning Summary */}
              {diagnosisResults.fol_verification.medical_reasoning_summary && (
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border-2 border-blue-200 shadow-lg">
                  <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                    <LightBulbIcon className="w-6 h-6 mr-2 text-blue-600" />
                    Medical Reasoning Analysis
                    <div className="ml-auto px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                      {(diagnosisResults.fol_verification as any).ai_service_used?.toUpperCase() || 'AI REASONING'}
                    </div>
                  </h3>
                  <div className="bg-white rounded-lg p-4 border border-blue-100">
                    <p className="text-gray-700 text-lg leading-relaxed">
                      {diagnosisResults.fol_verification.medical_reasoning_summary}
                    </p>
                  </div>
                </div>
              )}

              {/* Verification Summary */}
              {diagnosisResults.fol_verification.verification_summary && (
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 border-2 border-green-200 shadow-lg">
                  <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                    <CheckCircleIcon className="w-6 h-6 mr-2 text-green-600" />
                    Logic Verification Summary
                    <div className="ml-auto px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                      {(diagnosisResults.fol_verification as any).clinical_assessment || 'VERIFIED'}
                    </div>
                  </h3>
                  <div className="bg-white rounded-lg p-4 border border-green-100">
                    <p className="text-gray-700 text-lg leading-relaxed">
                      {diagnosisResults.fol_verification.verification_summary}
                    </p>
                  </div>
                </div>
              )}

              {/* Enhanced Detailed Predicate Verification - Primary Display */}
              {diagnosisResults.fol_verification.detailed_results && diagnosisResults.fol_verification.detailed_results.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-gray-800 flex items-center">
                    <CogIcon className="w-6 h-6 mr-2 text-purple-600" />
                    Enhanced FOL Verification Results
                    <div className="ml-auto px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                      {(diagnosisResults.fol_verification as any).ai_service_used?.toUpperCase() || 'GEMINI'} AI
                    </div>
                  </h3>
                  
                  <div className="bg-gray-50 rounded-xl p-6 border">
                    <h4 className="font-semibold text-gray-800 mb-4 flex items-center justify-between">
                      <span>
                        Gemini AI Analysis - {diagnosisResults.fol_verification.verified_predicates}/{diagnosisResults.fol_verification.total_predicates} Predicates Verified
                      </span>
                      <div className="flex items-center space-x-2">
                        <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                          (diagnosisResults.fol_verification.success_rate || 0) >= 0.8 ? 'bg-green-100 text-green-800' :
                          (diagnosisResults.fol_verification.success_rate || 0) >= 0.6 ? 'bg-yellow-100 text-yellow-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {((diagnosisResults.fol_verification.success_rate || 0) * 100).toFixed(1)}% Success
                        </div>
                        <div className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          {(diagnosisResults.fol_verification as any).confidence_level || 'HIGH'} Confidence
                        </div>
                      </div>
                    </h4>
                    
                    <div className="grid gap-3">
                      {diagnosisResults.fol_verification.detailed_results.slice(0, 8).map((result:any, index:any) => (
                        <div key={index} className={`p-4 rounded-lg border-l-4 ${
                          result.verified || result.verification_status === 'VERIFIED'
                            ? 'border-green-500 bg-green-50' 
                            : 'border-red-500 bg-red-50'
                        }`}>
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex-1">
                              <div className="font-mono text-sm font-semibold text-gray-800 mb-1">
                                {result.predicate_string || result.fol_string || result.predicate || `${result.predicate_type}(${result.subject}, ${result.object})`}
                              </div>
                              <div className="text-sm text-gray-600 mb-2">
                                <span className="font-medium">Method:</span> {result.evaluation_method || 'Gemini AI Logic'}
                                {result.confidence && (
                                  <>
                                    <span className="mx-2">•</span>
                                    <span className="font-medium">Confidence:</span> {(result.confidence * 100).toFixed(1)}%
                                  </>
                                )}
                                {result.clinical_significance && (
                                  <>
                                    <span className="mx-2">•</span>
                                    <span className="font-medium">Significance:</span> {result.clinical_significance}
                                  </>
                                )}
                              </div>
                              {result.reasoning && (
                                <div className="text-sm text-gray-700 mb-2">
                                  <span className="font-medium">Reasoning:</span> {result.reasoning}
                                </div>
                              )}
                              {result.evidence && result.evidence.length > 0 && (
                                <div className="text-sm text-gray-700">
                                  <span className="font-medium">Evidence:</span> {result.evidence.join(', ')}
                                </div>
                              )}
                            </div>
                            <div className={`ml-4 px-3 py-1 rounded-full text-xs font-medium ${
                              result.verified || result.verification_status === 'VERIFIED'
                                ? 'bg-green-100 text-green-800'
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {result.verification_status || (result.verified ? 'VERIFIED' : 'FAILED')}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {diagnosisResults.fol_verification.detailed_results.length > 8 && (
                      <div className="text-center py-3 mt-4">
                        <span className="text-sm text-gray-500">
                          ... and {diagnosisResults.fol_verification.detailed_results.length - 8} more verification results
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Fallback: Simple Predicates Display - Only show if detailed_results not available */}
              {(!diagnosisResults.fol_verification.detailed_results || diagnosisResults.fol_verification.detailed_results.length === 0) && 
               diagnosisResults.fol_verification.predicates && diagnosisResults.fol_verification.predicates.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-xl font-bold text-gray-800 flex items-center">
                    <CogIcon className="w-6 h-6 mr-2 text-purple-600" />
                    Extracted FOL Predicates
                    <div className="ml-auto px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-xs font-medium">
                      FALLBACK MODE
                    </div>
                  </h3>
                  
                  <div className="bg-gray-50 rounded-xl p-6 border">
                    <h4 className="font-semibold text-gray-800 mb-4">
                      Logic Predicates Extracted - {diagnosisResults.fol_verification.verified_predicates || 0}/{diagnosisResults.fol_verification.total_predicates || diagnosisResults.fol_verification.predicates.length} Verified
                    </h4>
                    
                    <div className="grid gap-3">
                      {diagnosisResults.fol_verification.predicates.slice(0, 8).map((predicate, index) => {
                        const isObject = typeof predicate === 'object' && predicate !== null;
                        const predicateObj = isObject ? predicate as any : null;
                        
                        return (
                          <div key={index} className="p-4 rounded-lg border bg-white">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="font-mono text-sm font-semibold text-gray-800 mb-1">
                                  {isObject ? (predicateObj.fol_string || predicateObj.predicate || `Predicate ${index + 1}`) : predicate}
                                </div>
                                <div className="text-sm text-gray-600">
                                  <span className="font-medium">Type:</span> {isObject ? (predicateObj.predicate_type || 'Unknown') : 'String'}
                                  <span className="mx-2">•</span>
                                  <span className="font-medium">Index:</span> {index + 1}
                                  {isObject && predicateObj.confidence && (
                                    <>
                                      <span className="mx-2">•</span>
                                      <span className="font-medium">Confidence:</span> {(predicateObj.confidence * 100).toFixed(1)}%
                                    </>
                                  )}
                                </div>
                              </div>
                              <div className={`ml-4 px-3 py-1 rounded-full text-xs font-medium ${
                                isObject && predicateObj.verified ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
                              }`}>
                                {isObject && predicateObj.verified ? 'VERIFIED' : 'PREDICATE'}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {diagnosisResults.fol_verification.predicates.length > 8 && (
                      <div className="text-center py-2">
                        <span className="text-sm text-gray-500">
                          ... and {diagnosisResults.fol_verification.predicates.length - 8} more predicates
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Enhanced Error Handling - Show if no FOL verification data */}
              {(!diagnosisResults.fol_verification.detailed_results || diagnosisResults.fol_verification.detailed_results.length === 0) && 
               (!diagnosisResults.fol_verification.predicates || diagnosisResults.fol_verification.predicates.length === 0) &&
               diagnosisResults.fol_verification.verification_summary && (
                <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl p-6 border border-amber-200">
                  <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                    <DocumentTextIcon className="w-6 h-6 mr-2 text-amber-600" />
                    FOL Verification Summary
                    <div className="ml-auto px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-xs font-medium">
                      SUMMARY MODE
                    </div>
                  </h3>
                  <div className="bg-white rounded-lg p-4 border border-amber-100">
                    <p className="text-gray-700 text-lg leading-relaxed">
                      {diagnosisResults.fol_verification.verification_summary}
                    </p>
                  </div>
                </div>
              )}

              {/* Clinical Confidence Distribution */}
              {diagnosisResults.fol_verification.disease_probabilities && Object.keys(diagnosisResults.fol_verification.disease_probabilities).length > 0 && (
                <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-xl p-6 border border-amber-200">
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                    <ChartBarIcon className="w-6 h-6 mr-2 text-amber-600" />
                    Clinical Confidence Distribution
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(diagnosisResults.fol_verification.disease_probabilities).slice(0, 6).map(([disease, probability]) => (
                      <div key={disease} className="bg-white rounded-lg p-4 border">
                        <div className="text-sm font-medium text-gray-800 mb-2 capitalize">
                          {disease.replace(/_/g, ' ')}
                        </div>
                        <div className="text-2xl font-bold text-amber-600 mb-2">
                          {(typeof probability === 'number' ? probability * 100 : 0).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-gradient-to-r from-amber-400 to-orange-500 h-2 rounded-full"
                            style={{ width: `${typeof probability === 'number' ? probability * 100 : 0}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Clinical Recommendations */}
              {diagnosisResults.fol_verification.clinical_recommendations && diagnosisResults.fol_verification.clinical_recommendations.length > 0 && (
                <div className="bg-gradient-to-r from-teal-50 to-cyan-50 rounded-xl p-6 border border-teal-200">
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center justify-between">
                    <div className="flex items-center">
                      <DocumentTextIcon className="w-6 h-6 mr-2 text-teal-600" />
                      AI-Generated Clinical Recommendations
                    </div>
                    <div className="px-3 py-1 bg-teal-100 text-teal-800 rounded-full text-xs font-medium">
                      {(diagnosisResults.fol_verification as any).ai_service_used?.toUpperCase() || 'GEMINI'} AI
                    </div>
                  </h3>
                  <ul className="space-y-3">
                    {diagnosisResults.fol_verification.clinical_recommendations.map((recommendation, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-2 h-2 bg-teal-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                        <div className="text-gray-700 text-lg">
                          {recommendation}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Performance Metrics */}
              {diagnosisResults.fol_verification.verification_time && (
                <div className="bg-gradient-to-r from-gray-50 to-slate-50 rounded-xl p-6 border">
                  <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center justify-between">
                    <div className="flex items-center">
                      <ClockIcon className="w-5 h-5 mr-2 text-gray-600" />
                      Verification Performance Metrics
                    </div>
                    <div className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                      {(diagnosisResults.fol_verification as any).ai_service_used?.toUpperCase() || 'AI'} ENGINE
                    </div>
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-purple-700">
                        {diagnosisResults.fol_verification.verification_time?.toFixed(2) || '0.00'}s
                      </div>
                      <div className="text-sm text-gray-600">Processing Time</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-green-700">
                        {diagnosisResults.fol_verification.verified_predicates || 0}
                      </div>
                      <div className="text-sm text-gray-600">Verified</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-red-700">
                        {(diagnosisResults.fol_verification as any).failed_predicates || (diagnosisResults.fol_verification.total_predicates || 0) - (diagnosisResults.fol_verification.verified_predicates || 0)}
                      </div>
                      <div className="text-sm text-gray-600">Failed</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-blue-700">
                        {((diagnosisResults.fol_verification.overall_confidence || 0) * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-600">AI Confidence</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-amber-700">
                        {(diagnosisResults.fol_verification as any).confidence_level || 'HIGH'}
                      </div>
                      <div className="text-sm text-gray-600">Grade</div>
                    </div>
                  </div>
                  {(diagnosisResults.fol_verification as any).clinical_assessment && (
                    <div className="mt-4 text-center">
                      <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
                        (diagnosisResults.fol_verification as any).clinical_assessment === 'MOSTLY_CONSISTENT' ? 'bg-green-100 text-green-800' :
                        (diagnosisResults.fol_verification as any).clinical_assessment === 'PARTIALLY_CONSISTENT' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>
                        🎯 Clinical Assessment: {(diagnosisResults.fol_verification as any).clinical_assessment}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Online Medical Verification Section */}
        {diagnosisResults.online_verification && (
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
            <div
              className="bg-gradient-to-r from-cyan-50 to-teal-50 p-6 border-b border-gray-100 cursor-pointer hover:from-cyan-100 hover:to-teal-100 transition-colors"
              onClick={() => toggleSection("online_verification")}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-cyan-100 rounded-xl">
                    <GlobeAltIcon className="w-8 h-8 text-cyan-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-800">Online Medical Verification</h2>
                    <p className="text-gray-600">Real-time web search against trusted medical sources</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    diagnosisResults.online_verification.verification_status === 'VERIFIED' ? 'bg-green-100 text-green-800' :
                    diagnosisResults.online_verification.verification_status === 'PARTIAL' ? 'bg-yellow-100 text-yellow-800' :
                    diagnosisResults.online_verification.verification_status === 'CONTRADICTED' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {diagnosisResults.online_verification.verification_status || 'UNKNOWN'}
                  </div>
                  <div className="bg-cyan-100 text-cyan-800 px-3 py-1 rounded-full text-sm font-medium">
                    {diagnosisResults.online_verification.sources ? diagnosisResults.online_verification.sources.length : 0} sources
                  </div>
                  {expandedSections.online_verification ? (
                    <ChevronUpIcon className="w-6 h-6 text-gray-600" />
                  ) : (
                    <ChevronDownIcon className="w-6 h-6 text-gray-600" />
                  )}
                </div>
              </div>
            </div>

            {expandedSections.online_verification && (
              <div className="p-8 space-y-6">
                {/* Header with metrics */}
                <div className="bg-gradient-to-r from-cyan-600 via-teal-600 to-cyan-700 rounded-xl p-6 text-white">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20">
                      <div className="text-2xl font-bold text-white">
                        {diagnosisResults.online_verification.sources ? diagnosisResults.online_verification.sources.length : 0}
                      </div>
                      <div className="text-cyan-100 text-sm">Medical Sources</div>
                    </div>
                    <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20">
                      <div className="text-2xl font-bold text-white">
                        {((diagnosisResults.online_verification.confidence_score || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="text-cyan-100 text-sm">Verification Score</div>
                    </div>
                    <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20">
                      <div className="text-2xl font-bold text-white">
                        {diagnosisResults.online_verification.supporting_evidence ? diagnosisResults.online_verification.supporting_evidence.length : 0}
                      </div>
                      <div className="text-cyan-100 text-sm">Supporting Evidence</div>
                    </div>
                    <div className="bg-white bg-opacity-15 backdrop-blur-sm rounded-xl p-4 border border-white border-opacity-20">
                      <div className={`text-2xl font-bold ${
                        diagnosisResults.online_verification.verification_status === 'VERIFIED' ? 'text-green-300' :
                        diagnosisResults.online_verification.verification_status === 'PARTIAL' ? 'text-yellow-300' :
                        diagnosisResults.online_verification.verification_status === 'CONTRADICTED' ? 'text-red-300' : 'text-gray-300'
                      }`}>
                        {diagnosisResults.online_verification.verification_status === 'VERIFIED' ? '✅' :
                         diagnosisResults.online_verification.verification_status === 'PARTIAL' ? '⚠️' :
                         diagnosisResults.online_verification.verification_status === 'CONTRADICTED' ? '❌' : '❓'}
                      </div>
                      <div className="text-cyan-100 text-sm">Status</div>
                    </div>
                  </div>
                </div>

                {/* Verification Summary */}
                {diagnosisResults.online_verification.verification_summary && (
                  <div className="bg-gradient-to-r from-cyan-50 to-teal-50 rounded-xl p-6 border-2 border-cyan-200 shadow-lg">
                    <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                      <ShieldCheckIcon className="w-6 h-6 mr-2 text-cyan-600" />
                      Online Verification Summary
                      <div className="ml-auto px-3 py-1 bg-cyan-100 text-cyan-800 rounded-full text-xs font-medium">
                        REAL-TIME SEARCH
                      </div>
                    </h3>
                    <div className="bg-white rounded-lg p-4 border border-cyan-100">
                      <p className="text-gray-700 text-lg leading-relaxed">
                        {diagnosisResults.online_verification.verification_summary}
                      </p>
                    </div>
                  </div>
                )}

                {/* Supporting Evidence */}
                {diagnosisResults.online_verification.supporting_evidence && diagnosisResults.online_verification.supporting_evidence.length > 0 && (
                  <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 border-2 border-green-200 shadow-lg">
                    <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                      <CheckCircleIcon className="w-6 h-6 mr-2 text-green-600" />
                      Supporting Evidence from Medical Sources
                      <div className="ml-auto px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                        {diagnosisResults.online_verification.supporting_evidence.length} ITEMS
                      </div>
                    </h3>
                    <div className="space-y-3">
                      {diagnosisResults.online_verification.supporting_evidence.map((evidence, index) => (
                        <div key={index} className="bg-white rounded-lg p-4 border border-green-100 flex items-start">
                          <div className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                          <div className="text-gray-700 text-base leading-relaxed">
                            {evidence}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Contradicting Evidence */}
                {diagnosisResults.online_verification.contradicting_evidence && diagnosisResults.online_verification.contradicting_evidence.length > 0 && (
                  <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-xl p-6 border-2 border-red-200 shadow-lg">
                    <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                      <ExclamationTriangleIcon className="w-6 h-6 mr-2 text-red-600" />
                      Potential Concerns from Medical Sources
                      <div className="ml-auto px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs font-medium">
                        {diagnosisResults.online_verification.contradicting_evidence.length} ITEMS
                      </div>
                    </h3>
                    <div className="space-y-3">
                      {diagnosisResults.online_verification.contradicting_evidence.map((evidence, index) => (
                        <div key={index} className="bg-white rounded-lg p-4 border border-red-100 flex items-start">
                          <div className="w-2 h-2 bg-red-500 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                          <div className="text-gray-700 text-base leading-relaxed">
                            {evidence}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Medical Sources with URLs and Previews */}
                {diagnosisResults.online_verification.sources && diagnosisResults.online_verification.sources.length > 0 && (
                  <div className="space-y-4">
                    <h3 className="text-xl font-bold text-gray-800 flex items-center">
                      <MagnifyingGlassIcon className="w-6 h-6 mr-2 text-cyan-600" />
                      Verified Medical Sources & Search Results
                      <div className="ml-auto px-3 py-1 bg-cyan-100 text-cyan-800 rounded-full text-xs font-medium">
                        {diagnosisResults.online_verification.sources.length} SOURCES
                      </div>
                    </h3>

                    <div className="bg-gray-50 rounded-xl p-6 border">
                      <h4 className="font-semibold text-gray-800 mb-4 flex items-center justify-between">
                        <span>
                          Real-Time Search Results - {diagnosisResults.online_verification.sources.length} Medical Sources Found
                        </span>
                        <div className="flex items-center space-x-2">
                          <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                            diagnosisResults.online_verification.verification_status === 'VERIFIED' ? 'bg-green-100 text-green-800' :
                            diagnosisResults.online_verification.verification_status === 'PARTIAL' ? 'bg-yellow-100 text-yellow-800' :
                            diagnosisResults.online_verification.verification_status === 'CONTRADICTED' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            {((diagnosisResults.online_verification.confidence_score || 0) * 100).toFixed(1)}% Confidence
                          </div>
                          <div className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            LIVE SEARCH
                          </div>
                        </div>
                      </h4>

                      <div className="grid gap-4">
                        {diagnosisResults.online_verification.sources.slice(0, 10).map((source, index) => (
                          <div key={index} className={`p-4 rounded-lg border-l-4 ${
                            (source.relevance_score || 0) > 0.8 ? 'border-green-500 bg-green-50' :
                            (source.relevance_score || 0) > 0.6 ? 'border-blue-500 bg-blue-50' :
                            'border-gray-500 bg-gray-50'
                          }`}>
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex-1">
                                <div className="flex items-center space-x-3 mb-2">
                                  <h5 className="font-semibold text-gray-800 text-lg hover:text-cyan-600 transition-colors">
                                    {source.title}
                                  </h5>
                                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                                    (source.credibility_score || 0) > 0.9 ? 'bg-green-100 text-green-800' :
                                    (source.credibility_score || 0) > 0.8 ? 'bg-blue-100 text-blue-800' :
                                    'bg-amber-100 text-amber-800'
                                  }`}>
                                    {(source.credibility_score || 0) * 100}% Credible
                                  </div>
                                </div>

                                <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 mb-3">
                                  <div className="flex items-center">
                                    <GlobeAltIcon className="w-4 h-4 mr-1" />
                                    <span className="font-medium">{source.domain}</span>
                                  </div>
                                  <div className="flex items-center">
                                    <ChartBarIcon className="w-4 h-4 mr-1" />
                                    <span>Relevance: {((source.relevance_score || 0) * 100).toFixed(0)}%</span>
                                  </div>
                                  <div className="flex items-center">
                                    <ClockIcon className="w-4 h-4 mr-1" />
                                    <span>{source.date_accessed || 'Recent'}</span>
                                  </div>
                                  {source.authors && (
                                    <div className="flex items-center">
                                      <DocumentTextIcon className="w-4 h-4 mr-1" />
                                      <span>Authors: {source.authors}</span>
                                    </div>
                                  )}
                                  {source.pmid && (
                                    <div className="flex items-center">
                                      <BookOpenIcon className="w-4 h-4 mr-1" />
                                      <span>PMID: {source.pmid}</span>
                                    </div>
                                  )}
                                  {source.publication_date && (
                                    <div className="flex items-center">
                                      <CalendarIcon className="w-4 h-4 mr-1" />
                                      <span>Published: {source.publication_date}</span>
                                    </div>
                                  )}
                                  {source.keywords && Array.isArray(source.keywords) && source.keywords.length > 0 && (
                                    <div className="flex items-center">
                                      <TagIcon className="w-4 h-4 mr-1" />
                                      <span>Keywords: {source.keywords.slice(0, 3).join(', ')}</span>
                                    </div>
                                  )}
                                </div>

                                {/* Direct URL Link */}
                                <div className="mb-3">
                                  <a
                                    href={source.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center px-3 py-2 bg-cyan-100 hover:bg-cyan-200 text-cyan-800 rounded-lg text-sm font-medium transition-colors"
                                  >
                                    <GlobeAltIcon className="w-4 h-4 mr-2" />
                                    View Source
                                    <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                    </svg>
                                  </a>
                                </div>

                                {/* Content Preview */}
                                {source.content_snippet && (
                                  <div className="bg-white rounded-lg p-3 border border-gray-200">
                                    <div className="text-sm text-gray-700 leading-relaxed">
                                      <span className="font-medium text-gray-800">Preview:</span> {source.content_snippet}
                                    </div>
                                  </div>
                                )}

                                {/* Citation */}
                                {source.citation_format && (
                                  <div className="mt-3 text-xs text-gray-500 italic">
                                    {source.citation_format}
                                  </div>
                                )}
                              </div>

                              <div className="ml-4 flex flex-col items-end space-y-2">
                                <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                                  (source.relevance_score || 0) > 0.8 ? 'bg-green-100 text-green-800' :
                                  (source.relevance_score || 0) > 0.6 ? 'bg-blue-100 text-blue-800' :
                                  'bg-amber-100 text-amber-800'
                                }`}>
                                  {((source.relevance_score || 0) * 100).toFixed(0)}% Relevant
                                </div>
                                <div className="text-xs text-gray-500">
                                  {source.source_type || 'Medical Source'}
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {diagnosisResults.online_verification.sources.length > 10 && (
                        <div className="text-center py-3 mt-4">
                          <span className="text-sm text-gray-500">
                            ... and {diagnosisResults.online_verification.sources.length - 10} more medical sources
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Search Strategies Used */}
                {diagnosisResults.online_verification.search_strategies_used && Array.isArray(diagnosisResults.online_verification.search_strategies_used) && diagnosisResults.online_verification.search_strategies_used.length > 0 && (
                  <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-xl p-6 border-2 border-emerald-200">
                    <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                      <MagnifyingGlassIcon className="w-6 h-6 mr-2 text-emerald-600" />
                      Search Strategies Employed
                      <div className="ml-auto px-3 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs font-medium">
                        VERIFICATION METHODS
                      </div>
                    </h3>
                    <div className="bg-white rounded-lg p-4 border border-emerald-100">
                      <div className="flex flex-wrap gap-2">
                        {diagnosisResults.online_verification.search_strategies_used.map((strategy: string, index: number) => (
                          <div key={index} className="px-3 py-2 bg-emerald-100 text-emerald-800 rounded-lg text-sm font-medium">
                            {strategy.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Bibliography Section */}
                {diagnosisResults.online_verification.bibliography && Array.isArray(diagnosisResults.online_verification.bibliography) && diagnosisResults.online_verification.bibliography.length > 0 && (
                  <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border-2 border-purple-200">
                    <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                      <BookOpenIcon className="w-6 h-6 mr-2 text-purple-600" />
                      Medical Bibliography
                      <div className="ml-auto px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                        {diagnosisResults.online_verification.bibliography.length} REFERENCES
                      </div>
                    </h3>
                    <div className="bg-white rounded-lg p-4 border border-purple-100">
                      <div className="space-y-3">
                        {diagnosisResults.online_verification.bibliography.map((citation: string, index: number) => (
                          <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                            <div className="flex-shrink-0 w-6 h-6 bg-purple-100 text-purple-800 rounded-full flex items-center justify-center text-xs font-bold">
                              {index + 1}
                            </div>
                            <div className="flex-1 text-sm text-gray-700 leading-relaxed">
                              {citation}
                            </div>
                            <button
                              onClick={() => navigator.clipboard.writeText(citation)}
                              className="flex-shrink-0 p-1 text-gray-400 hover:text-purple-600 transition-colors"
                              title="Copy citation"
                            >
                              <ClipboardIcon className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Enhanced Clinical Notes */}
                {diagnosisResults.online_verification.clinical_notes && (
                  <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-6 border-2 border-indigo-200 shadow-lg">
                    <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                      <DocumentTextIcon className="w-6 h-6 mr-2 text-indigo-600" />
                      Clinical Notes from Online Verification
                      <div className="ml-auto px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full text-xs font-medium">
                        MEDICAL INSIGHT
                      </div>
                    </h3>
                    <div className="bg-white rounded-lg p-4 border border-indigo-100">
                      <p className="text-gray-700 text-lg leading-relaxed">
                        {diagnosisResults.online_verification.clinical_notes}
                      </p>
                    </div>
                  </div>
                )}

                {/* Verification Summary */}
                {diagnosisResults.online_verification.verification_summary && (
                  <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-6 border-2 border-blue-200">
                    <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
                      <ClipboardDocumentCheckIcon className="w-6 h-6 mr-2 text-blue-600" />
                      Verification Summary
                      <div className="ml-auto px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                        FINAL ASSESSMENT
                      </div>
                    </h3>
                    <div className="bg-white rounded-lg p-4 border border-blue-100">
                      <p className="text-gray-700 text-lg leading-relaxed font-medium">
                        {diagnosisResults.online_verification.verification_summary}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}        {/* Medical Explanations */}
        {(diagnosisResults.ui_data?.explanations && diagnosisResults.ui_data.explanations.length > 0) && (
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
            <div
              className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 border-b border-gray-100 cursor-pointer hover:from-blue-100 hover:to-indigo-100 transition-colors"
              onClick={() => toggleSection("explanations")}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-blue-100 rounded-xl">
                    <LightBulbIcon className="w-8 h-8 text-blue-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-800">AI Medical Analysis</h2>
                    <p className="text-gray-600">{diagnosisResults.ui_data.explanations.length} detailed insights</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                    {diagnosisResults.ui_data.explanations.length} insights
                  </div>
                  {expandedSections.explanations ? (
                    <ChevronUpIcon className="w-6 h-6 text-gray-600" />
                  ) : (
                    <ChevronDownIcon className="w-6 h-6 text-gray-600" />
                  )}
                </div>
              </div>
            </div>

            {expandedSections.explanations && (
              <div className="p-8 space-y-6">
                {diagnosisResults.ui_data.explanations.map((explanation, index) => {
                  // Parse confidence and verification status from explanation
                  const confidenceMatch = explanation.match(/\[(\d+\.\d+)%\s*confidence\]/i);
                  const verifiedMatch = explanation.match(/\[(✓|✗)\s*(FOL Verified|Unverified)\]/i);
                  const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) : 0;
                  const isVerified = verifiedMatch && verifiedMatch[1] === '✓';
                  
                  // Clean explanation text
                  const cleanExplanation = explanation
                    .replace(/\*\*Explanation \d+\*\*\s*\[.*?\]\s*\[.*?\]/g, '')
                    .trim();
                  
                  return (
                    <div key={index} className="bg-gradient-to-r from-blue-50 to-white p-6 rounded-xl border-l-4 border-blue-400 shadow-sm hover:shadow-md transition-all">
                      <div className="flex items-start space-x-4">
                        <div className="flex-shrink-0">
                          <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center shadow-lg relative">
                            <span className="text-white font-bold text-lg">{index + 1}</span>
                            {isVerified && (
                              <div className="absolute -top-1 -right-1 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
                                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center space-x-3">
                              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                                confidence >= 90 ? 'bg-green-100 text-green-800' :
                                confidence >= 80 ? 'bg-blue-100 text-blue-800' :
                                'bg-amber-100 text-amber-800'
                              }`}>
                                {confidence.toFixed(1)}% confidence
                              </div>
                              {isVerified ? (
                                <div className="px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800 flex items-center">
                                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                  </svg>
                                  FOL Verified
                                </div>
                              ) : (
                                <div className="px-3 py-1 rounded-full text-sm font-medium bg-amber-100 text-amber-800 flex items-center">
                                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                  </svg>
                                  Unverified
                                </div>
                              )}
                            </div>
                          </div>
                          <div 
                            className="text-gray-800 leading-relaxed text-base prose prose-lg max-w-none"
                            dangerouslySetInnerHTML={{
                              __html: (cleanExplanation || 'No explanation available')
                                .replace(/\*\*(.*?)\*\*/g, '<strong class="text-blue-700">$1</strong>')
                                .replace(/\n\n/g, '</p><p class="mt-4">')
                                .replace(/^(.)/g, '<p>$1')
                                .replace(/(.)$/g, '$1</p>')
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* GradCAM Heatmap Analysis Section */}
        {diagnosisResults.heatmap_visualization && diagnosisResults.heatmap_visualization.available && (
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
            <div
              className="bg-gradient-to-r from-red-50 to-orange-50 p-6 border-b border-gray-100 cursor-pointer hover:from-red-100 hover:to-orange-100 transition-colors"
              onClick={() => toggleSection("gradcam")}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-red-100 rounded-xl">
                    <FireIcon className="w-8 h-8 text-red-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-800">AI GradCAM Analysis</h2>
                    <p className="text-gray-600">
                      {diagnosisResults.heatmap_visualization.successful_heatmaps || 0}/
                      {diagnosisResults.heatmap_visualization.total_images || 0} images processed with AI attention mapping
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium">
                    {diagnosisResults.heatmap_visualization.model_type || 'AI Model'}
                  </div>
                  {expandedSections.gradcam ? (
                    <ChevronUpIcon className="w-6 h-6 text-gray-600" />
                  ) : (
                    <ChevronDownIcon className="w-6 h-6 text-gray-600" />
                  )}
                </div>
              </div>
            </div>

            {expandedSections.gradcam && (
              <div className="p-8">
                {diagnosisResults.heatmap_data && diagnosisResults.heatmap_data.length > 0 ? (
                  <div className="space-y-6">
                    {diagnosisResults.heatmap_data.map((heatmapResult: any, index: number) => (
                      <div key={index} className="border border-gray-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-all">
                        {heatmapResult.success ? (
                          <div>
                            {/* Header with image info */}
                            <div className="bg-gradient-to-r from-gray-50 to-blue-50 p-4 border-b border-gray-200">
                              <div className="flex items-center justify-between">
                                <h4 className="font-semibold text-gray-900 flex items-center">
                                  <PhotoIcon className="w-5 h-5 mr-2 text-blue-600" />
                                  {heatmapResult.image_file || `Medical Image ${index + 1}`}
                                </h4>
                                <div className="flex items-center space-x-4">
                                  {heatmapResult.analysis && (
                                    <div className="text-right">
                                      <div className="text-sm font-semibold text-blue-600">
                                        {heatmapResult.analysis.predicted_class || 'Analysis Complete'}
                                      </div>
                                      <div className="text-xs text-gray-600">
                                        {((heatmapResult.analysis.confidence_score || 0) * 100).toFixed(1)}% AI Confidence
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>

                            {/* GradCAM Visualizations */}
                            <div className="p-6">
                              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Heatmap */}
                                {heatmapResult.visualizations?.heatmap_image && (
                                  <div className="space-y-3">
                                    <h5 className="text-sm font-semibold text-gray-700 flex items-center">
                                      <FireIcon className="w-4 h-4 mr-1 text-red-500" />
                                      AI Attention Heatmap
                                    </h5>
                                    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200 hover:border-red-300 transition-colors">
                                      <img
                                        src={heatmapResult.visualizations.heatmap_image?.startsWith('data:') ?
                                          heatmapResult.visualizations.heatmap_image :
                                          `data:image/png;base64,${heatmapResult.visualizations.heatmap_image}`}
                                        alt={`GradCAM Heatmap ${index + 1}`}
                                        className="w-full h-56 object-contain rounded"
                                      />
                                      <p className="text-xs text-gray-500 mt-2 text-center">
                                        Red = High AI Focus, Blue = Low Focus
                                      </p>
                                    </div>
                                  </div>
                                )}

                                {/* Overlay */}
                                {heatmapResult.visualizations?.overlay_image && (
                                  <div className="space-y-3">
                                    <h5 className="text-sm font-semibold text-gray-700 flex items-center">
                                      <PhotoIcon className="w-4 h-4 mr-1 text-blue-500" />
                                      Original + AI Overlay
                                    </h5>
                                    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200 hover:border-blue-300 transition-colors">
                                      <img
                                        src={heatmapResult.visualizations.overlay_image?.startsWith('data:') ?
                                          heatmapResult.visualizations.overlay_image :
                                          `data:image/png;base64,${heatmapResult.visualizations.overlay_image}`}
                                        alt={`GradCAM Overlay ${index + 1}`}
                                        className="w-full h-56 object-contain rounded"
                                      />
                                      <p className="text-xs text-gray-500 mt-2 text-center">
                                        Original scan with AI attention overlay
                                      </p>
                                    </div>
                                  </div>
                                )}

                                {/* Volume */}
                                {heatmapResult.visualizations?.volume_image && (
                                  <div className="space-y-3">
                                    <h5 className="text-sm font-semibold text-gray-700 flex items-center">
                                      <BeakerIcon className="w-4 h-4 mr-1 text-green-500" />
                                      3D Volume Analysis
                                    </h5>
                                    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200 hover:border-green-300 transition-colors">
                                      <img
                                        src={heatmapResult.visualizations.volume_image?.startsWith('data:') ?
                                          heatmapResult.visualizations.volume_image :
                                          `data:image/png;base64,${heatmapResult.visualizations.volume_image}`}
                                        alt={`GradCAM Volume ${index + 1}`}
                                        className="w-full h-56 object-contain rounded"
                                      />
                                      <p className="text-xs text-gray-500 mt-2 text-center">
                                        3D volumetric visualization
                                      </p>
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* Medical Interpretation */}
                              {heatmapResult.medical_interpretation && (
                                <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
                                  <h5 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                                    <LightBulbIcon className="w-5 h-5 mr-2" />
                                    AI Medical Interpretation
                                  </h5>
                                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <div className="bg-white rounded-lg p-4 border">
                                      <div className="text-sm text-gray-600">Primary Finding</div>
                                      <div className="text-lg font-semibold text-blue-700">
                                        {heatmapResult.medical_interpretation.primary_finding || 'Analysis Complete'}
                                      </div>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border">
                                      <div className="text-sm text-gray-600">Confidence Level</div>
                                      <div className="text-lg font-semibold text-green-600">
                                        {heatmapResult.medical_interpretation.confidence_level || 'Moderate'}
                                      </div>
                                    </div>
                                    <div className="bg-white rounded-lg p-4 border">
                                      <div className="text-sm text-gray-600">Attention Regions</div>
                                      <div className="text-lg font-semibold text-purple-600">
                                        {heatmapResult.medical_interpretation.attention_areas || 0} regions
                                      </div>
                                    </div>
                                  </div>
                                  
                                  {heatmapResult.medical_interpretation.clinical_notes && (
                                    <div className="mt-4 bg-white rounded-lg p-4 border">
                                      <h6 className="font-semibold text-gray-800 mb-2">Clinical Notes:</h6>
                                      <ul className="space-y-1">
                                        {heatmapResult.medical_interpretation.clinical_notes.map((note: string, noteIndex: number) => (
                                          <li key={noteIndex} className="flex items-start text-sm text-gray-700">
                                            <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 mr-3 flex-shrink-0"></div>
                                            {note}
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        ) : (
                          <div className="p-8 text-center">
                            <ExclamationTriangleIcon className="w-16 h-16 mx-auto mb-4 text-red-400" />
                            <h4 className="text-lg font-semibold text-gray-900 mb-2">
                              GradCAM Analysis Failed
                            </h4>
                            <p className="text-gray-600 mb-2">
                              {heatmapResult.error || 'Unable to generate AI heatmap for this medical image'}
                            </p>
                            <p className="text-sm text-gray-500">
                              Image: {heatmapResult.image_file || `Image ${index + 1}`}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                    
                    {/* Information Box */}
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
                      <div className="flex items-start space-x-4">
                        <LightBulbIcon className="w-6 h-6 text-blue-600 mt-1 flex-shrink-0" />
                        <div className="text-sm text-blue-800">
                          <p className="font-semibold mb-3 text-lg">Understanding GradCAM AI Analysis:</p>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-white rounded-lg p-3 border">
                              <div className="font-semibold text-red-600 mb-1">🔥 Heatmap</div>
                              <p>Shows areas where the AI focuses most attention (red = high focus, blue = low focus)</p>
                            </div>
                            <div className="bg-white rounded-lg p-3 border">
                              <div className="font-semibold text-blue-600 mb-1">📷 Overlay</div>
                              <p>Original medical image with AI attention heatmap overlaid for clinical context</p>
                            </div>
                            <div className="bg-white rounded-lg p-3 border">
                              <div className="font-semibold text-green-600 mb-1">🧪 3D Volume</div>
                              <p>Comprehensive 3D visualization for volumetric analysis of scans</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <FireIcon className="w-20 h-20 mx-auto mb-6 text-gray-300" />
                    <h3 className="text-xl font-semibold text-gray-600 mb-3">No GradCAM Analysis Available</h3>
                    <p className="text-gray-500 max-w-md mx-auto">
                      {diagnosisResults.heatmap_visualization.error || 'AI heatmap analysis could not be completed for the uploaded medical images. Please ensure images are in supported formats (DICOM, JPG, PNG).'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Differential Diagnoses */}
        {(diagnosisResults.differential_diagnoses || []).length > 0 && (
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
            <div
              className="bg-gradient-to-r from-amber-50 to-orange-50 p-6 border-b border-gray-100 cursor-pointer hover:from-amber-100 hover:to-orange-100 transition-colors"
              onClick={() => toggleSection("differential")}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-amber-100 rounded-xl">
                    <MagnifyingGlassIcon className="w-8 h-8 text-amber-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-800">Differential Diagnoses</h2>
                    <p className="text-gray-600">Alternative diagnostic possibilities</p>
                  </div>
                </div>
                {expandedSections.differential ? (
                  <ChevronUpIcon className="w-6 h-6 text-gray-600" />
                ) : (
                  <ChevronDownIcon className="w-6 h-6 text-gray-600" />
                )}
              </div>
            </div>

            {expandedSections.differential && (
              <div className="p-8 space-y-6">
                {(diagnosisResults.differential_diagnoses || []).map((diagnosis, index) => (
                  <div key={index} className="bg-gray-50 rounded-xl p-6 border-l-4 border-amber-400">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h4 className="text-xl font-bold text-gray-800 mb-2">{diagnosis.condition}</h4>
                        {diagnosis.icd_code && (
                          <div className="inline-flex items-center bg-amber-100 text-amber-800 px-3 py-1 rounded-lg text-sm font-medium">
                            <DocumentTextIcon className="w-4 h-4 mr-2" />
                            ICD Code: {diagnosis.icd_code}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center space-x-2">
                        {getConfidenceIcon(diagnosis.confidence)}
                        <div className="text-right">
                          <div className={`text-lg font-bold ${getConfidenceColor(diagnosis.confidence)}`}>
                            {(diagnosis.confidence * 100).toFixed(0)}%
                          </div>
                          <div className="text-sm text-gray-600">Likelihood</div>
                        </div>
                      </div>
                    </div>
                    <p className="text-gray-700 leading-relaxed">{diagnosis.reasoning}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Treatment Recommendations */}
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
          <div
            className="bg-gradient-to-r from-emerald-50 to-teal-50 p-6 border-b border-gray-100 cursor-pointer hover:from-emerald-100 hover:to-teal-100 transition-colors"
            onClick={() => toggleSection("recommendations")}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-emerald-100 rounded-xl">
                  <BeakerIcon className="w-8 h-8 text-emerald-600" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-800">Treatment Recommendations</h2>
                  <p className="text-gray-600">Evidence-based treatment options</p>
                </div>
              </div>
              {expandedSections.recommendations ? (
                <ChevronUpIcon className="w-6 h-6 text-gray-600" />
              ) : (
                <ChevronDownIcon className="w-6 h-6 text-gray-600" />
              )}
            </div>
          </div>

          {expandedSections.recommendations && (
            <div className="p-8">
              <div className="grid md:grid-cols-2 gap-8">
                {/* Recommended Tests */}
                {(diagnosisResults.recommended_tests || []).length > 0 && (
                  <div className="bg-blue-50 rounded-xl p-6">
                    <h4 className="text-lg font-bold mb-4 flex items-center text-blue-800">
                      <ChartBarIcon className="w-6 h-6 mr-2" />
                      Recommended Tests
                    </h4>
                    <div className="space-y-3">
                      {(diagnosisResults.recommended_tests || []).map((test, index) => (
                        <div key={index} className="flex items-start space-x-3 bg-white p-3 rounded-lg">
                          <CheckCircleIcon className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                          <span className="text-gray-700 font-medium">{test}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Treatment Options */}
                {((diagnosisResults as any).treatment_options || []).length > 0 && (
                  <div className="bg-emerald-50 rounded-xl p-6">
                    <h4 className="text-lg font-bold mb-4 flex items-center text-emerald-800">
                      <InformationCircleIcon className="w-6 h-6 mr-2" />
                      Treatment Options
                    </h4>
                    <div className="space-y-3">
                      {((diagnosisResults as any).treatment_options || []).map((treatment:any, index:any) => (
                        <div key={index} className="flex items-start space-x-3 bg-white p-3 rounded-lg">
                          <CheckCircleIcon className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" />
                          <span className="text-gray-700 font-medium">{treatment}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Advanced Sections - Collapsible by default */}
        {(diagnosisResults.advanced_fol_extraction || 
          diagnosisResults.ontology_analysis || 
          diagnosisResults.enhanced_verification || 
          diagnosisResults.online_verification || 
          diagnosisResults.clara_results) && (
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
            <div className="bg-gradient-to-r from-purple-50 to-indigo-50 p-6 border-b border-gray-100">
              <div className="text-center">
                <div className="p-3 bg-purple-100 rounded-full inline-block mb-4">
                  <CpuChipIcon className="w-10 h-10 text-purple-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Advanced AI Analysis</h2>
                <p className="text-gray-600">Detailed verification and technical analysis results</p>
                <button
                  onClick={() => toggleSection("advanced")}
                  className="mt-4 bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition-colors flex items-center mx-auto"
                >
                  {expandedSections.advanced ? (
                    <>
                      <ChevronUpIcon className="w-5 h-5 mr-2" />
                      Hide Advanced Analysis
                    </>
                  ) : (
                    <>
                      <ChevronDownIcon className="w-5 h-5 mr-2" />
                      Show Advanced Analysis
                    </>
                  )}
                </button>
              </div>
            </div>

            {expandedSections.advanced && (
              <div className="p-8 space-y-8">
                {/* Enhanced Medical Verification Section */}
                {diagnosisResults.enhanced_verification && (
                  <div className="bg-gradient-to-r from-green-50 to-teal-50 rounded-xl p-6 border border-green-200">
                    <div className="flex items-center mb-4">
                      <div className="p-2 bg-green-100 rounded-lg mr-3">
                        <BookOpenIcon className="w-6 h-6 text-green-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-gray-800">Medical Literature Verification</h3>
                        <p className="text-gray-600">Cross-referenced against medical textbooks and journals</p>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-green-600">
                          {((diagnosisResults.enhanced_verification.overall_confidence || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">Literature Confidence</div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-blue-600">
                          {diagnosisResults.enhanced_verification.sources_count || 0}
                        </div>
                        <div className="text-sm text-gray-600">Medical Sources</div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-purple-600">
                          {diagnosisResults.enhanced_verification.evidence_strength || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Evidence Strength</div>
                      </div>
                    </div>
                    
                    {diagnosisResults.enhanced_verification.consensus_analysis && (
                      <div className="bg-white rounded-lg p-4 border">
                        <h4 className="font-semibold text-gray-800 mb-2">Medical Consensus:</h4>
                        <p className="text-gray-700">{diagnosisResults.enhanced_verification.consensus_analysis}</p>
                      </div>
                    )}
                    
                    {diagnosisResults.enhanced_verification.textbook_references && diagnosisResults.enhanced_verification.textbook_references.length > 0 && (
                      <div className="bg-white rounded-lg p-4 border">
                        <h4 className="font-semibold text-gray-800 mb-2">Textbook References:</h4>
                        <div className="space-y-2">
                          {diagnosisResults.enhanced_verification.textbook_references.slice(0, 3).map((ref: any, index: number) => (
                            <div key={index} className="text-sm text-gray-700 border-l-4 border-green-300 pl-3">
                              <strong>{ref.title}</strong> - Page {ref.page}, {ref.chapter}
                              {ref.quote && <p className="mt-1 italic">"{ref.quote}"</p>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Online Verification Section */}
                {diagnosisResults.online_verification && (
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
                    <div className="flex items-center mb-4">
                      <div className="p-2 bg-blue-100 rounded-lg mr-3">
                        <GlobeAltIcon className="w-6 h-6 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-gray-800">Real-time Medical Verification</h3>
                        <p className="text-gray-600">Live verification against current medical databases</p>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-blue-600">
                          {((diagnosisResults.online_verification.confidence_score || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">Online Confidence</div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-green-600">
                          {diagnosisResults.online_verification.sources?.length || 0}
                        </div>
                        <div className="text-sm text-gray-600">Online Sources</div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-purple-600">
                          {diagnosisResults.online_verification.verification_status || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Status</div>
                      </div>
                    </div>
                    
                    {diagnosisResults.online_verification.verification_summary && (
                      <div className="bg-white rounded-lg p-4 border">
                        <h4 className="font-semibold text-gray-800 mb-2">Online Verification Summary:</h4>
                        <p className="text-gray-700">{diagnosisResults.online_verification.verification_summary}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Ontology Analysis Section */}
                {diagnosisResults.ontology_analysis && (
                  <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl p-6 border border-yellow-200">
                    <div className="flex items-center mb-4">
                      <div className="p-2 bg-yellow-100 rounded-lg mr-3">
                        <MagnifyingGlassIcon className="w-6 h-6 text-yellow-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-gray-800">Medical Ontology Analysis</h3>
                        <p className="text-gray-600">UMLS, SNOMED, and ICD-10 term mapping</p>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-yellow-600">
                          {diagnosisResults.ontology_analysis.term_count || 0}
                        </div>
                        <div className="text-sm text-gray-600">Terms Analyzed</div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-orange-600">
                          {diagnosisResults.ontology_analysis.synonym_count || 0}
                        </div>
                        <div className="text-sm text-gray-600">Synonyms Found</div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border">
                        <div className="text-2xl font-bold text-purple-600">
                          {((diagnosisResults.ontology_analysis.confidence || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">Mapping Confidence</div>
                      </div>
                    </div>
                    
                    {diagnosisResults.ontology_analysis.normalized_diagnosis && (
                      <div className="bg-white rounded-lg p-4 border">
                        <h4 className="font-semibold text-gray-800 mb-2">Normalized Medical Term:</h4>
                        <p className="text-gray-700">{diagnosisResults.ontology_analysis.normalized_diagnosis}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Clara AI Results Section */}
                {diagnosisResults.clara_results && (
                  <div className="bg-gradient-to-r from-emerald-50 to-green-50 rounded-xl p-6 border border-emerald-200">
                    <div className="flex items-center mb-4">
                      <div className="p-2 bg-emerald-100 rounded-lg mr-3">
                        <StarIcon className="w-6 h-6 text-emerald-600" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-gray-800">NVIDIA Clara AI Enhancement</h3>
                        <p className="text-gray-600">Advanced GPU-accelerated medical AI processing</p>
                      </div>
                    </div>
                    
                    <div className="bg-white rounded-lg p-4 border">
                      <p className="text-gray-700">Clara AI enhancement features were applied to this analysis, including advanced imaging and genomic processing capabilities.</p>
                    </div>
                  </div>
                )}

                <div className="text-center border-t border-gray-200 pt-6">
                  <p className="text-gray-600 mb-4">
                    Advanced analysis complete with multi-layer AI verification
                  </p>
                  <button 
                    onClick={openDownloadModal} 
                    className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center mx-auto"
                  >
                    <ArrowDownTrayIcon className="w-5 h-5 mr-2" />
                    Download Complete Technical Report
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Success Footer */}
        <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-2xl p-4 sm:p-6 md:p-8 text-center">
          <CheckCircleIcon className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-4 opacity-90" />
          <h2 className="text-xl sm:text-2xl font-bold mb-2">Diagnosis Analysis Complete</h2>
          <p className="text-green-100 mb-6 max-w-2xl mx-auto text-sm sm:text-base">
            Your comprehensive AI-powered medical diagnosis has been successfully completed.
            The analysis includes primary diagnosis, differential diagnoses, treatment recommendations,
            and advanced verification using multiple AI systems.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4">
            <button
              onClick={openDownloadModal}
              className="bg-white text-green-600 px-6 py-2 sm:px-8 sm:py-3 rounded-xl hover:bg-green-50 transition-all duration-200 flex items-center font-semibold shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 text-sm sm:text-base"
            >
              <ArrowDownTrayIcon className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
              <span className="hidden sm:inline">Download Full Report</span>
              <span className="sm:hidden">Download Report</span>
            </button>
            <div className="text-green-100 text-xs sm:text-sm">
              Session: {currentSessionId}
            </div>
          </div>
        </div>
      </div>
      
      {/* Report Download Modal */}
      {diagnosisResults && (
        <ReportDownloadModal
          isOpen={showDownloadModal}
          onClose={() => setShowDownloadModal(false)}
          diagnosisData={diagnosisResults}
        />
      )}
    </div>
  )
}
