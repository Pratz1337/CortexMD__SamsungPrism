"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useForm } from "react-hook-form"
import { DiagnosisAPI, handleApiError, API_BASE_URL } from "@/lib/api"
import { useDiagnosisStore } from "@/store/diagnosisStore"
import type { PatientInput } from "@/types"
import { useSarvamVoice } from "@/hooks/useSarvamVoice"
import toast from "react-hot-toast"

interface DiagnosisFormProps {
  patientId?: string
}

interface FilePreview {
  file: File
  id: string
  previewUrl: string
}

export function DiagnosisForm({ patientId }: DiagnosisFormProps = {}) {
  const [filesPreviews, setFilesPreviews] = useState<FilePreview[]>([])
  const [videoPreviews, setVideoPreviews] = useState<FilePreview[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showFhirInput, setShowFhirInput] = useState(false)
  const [ontologySearchTerm, setOntologySearchTerm] = useState("")
  const [ontologySearchResults, setOntologySearchResults] = useState<any[]>([])
  const [isSearchingOntology, setIsSearchingOntology] = useState(false)
  const [showOntologyResults, setShowOntologyResults] = useState(false)
  const [showVideoInput, setShowVideoInput] = useState(false)
  const [claraOptions, setClaraOptions] = useState({
    dicom_processing: false,
    "3d_reconstruction": false,
    image_segmentation: false,
    genomic_analysis: false,
    variant_calling: false,
    multi_omics: false,
  })

  const { setCurrentSessionId, setIsLoading, setError, setProcessingStatus, setDiagnosisResults, isLoading } =
    useDiagnosisStore()

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
  } = useForm<PatientInput>()

  // Sarvam AI voice recording
  const sarvamVoice = useSarvamVoice({
    language: "en",
    model: "saarika:v1",
  })

  const generatePatientId = () => {
    return patientId || `WEB-${new Date().toISOString().slice(0, 19).replace(/[:-]/g, "")}`
  }

  // Auto-fill transcript when voice recording completes
  useEffect(() => {
    if (sarvamVoice.transcript && !sarvamVoice.isProcessing) {
      const currentSymptoms = watch("symptoms") || ""
      const newSymptoms = currentSymptoms + (currentSymptoms ? "\n\n" : "") + sarvamVoice.transcript
      setValue("symptoms", newSymptoms)
      toast.success("Voice recording transcribed and added to symptoms!")
    }
  }, [sarvamVoice.transcript, sarvamVoice.isProcessing, setValue, watch])

  useEffect(() => {
    return () => {
      filesPreviews.forEach((fp) => {
        if (fp.previewUrl.startsWith("blob:")) {
          URL.revokeObjectURL(fp.previewUrl)
        }
      })
    }
  }, [])

  const pollForResults = async (sessionId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await DiagnosisAPI.getProcessingStatus(sessionId)
        setProcessingStatus(status)

        if (status.status === "completed") {
          clearInterval(pollInterval)
          setIsLoading(false)

          // Fetch the results
          const results = await DiagnosisAPI.getDiagnosisResults(sessionId)
          setDiagnosisResults(results)
          toast.success("Diagnosis completed!")
        } else if (status.status === "error") {
          clearInterval(pollInterval)
          setIsLoading(false)
          setError("Diagnosis processing failed")
          toast.error("Diagnosis processing failed")
        }
      } catch (error) {
        console.error("Error polling status:", error)
        // Continue polling on error, don't break the process
      }
    }, 2000) // Poll every 2 seconds
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || [])

    selectedFiles.forEach((file) => {
      // Check if file already exists
      const exists = filesPreviews.some(
        (fp) => fp.file.name === file.name && fp.file.size === file.size && fp.file.lastModified === file.lastModified,
      )

      if (!exists) {
        const id = `${file.name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        const previewUrl = URL.createObjectURL(file)

        setFilesPreviews((prev) => [...prev, { file, id, previewUrl }])
      }
    })

    // Reset input to allow selecting the same file again if needed
    e.target.value = ""
  }

  const removeFile = (id: string) => {
    setFilesPreviews((prev) => {
      const fileToRemove = prev.find((fp) => fp.id === id)
      if (fileToRemove && fileToRemove.previewUrl.startsWith("blob:")) {
        URL.revokeObjectURL(fileToRemove.previewUrl)
      }
      return prev.filter((fp) => fp.id !== id)
    })
  }

  const onSubmit = async (data: PatientInput) => {
    console.log('🚀 Form submission started')
    console.log('📊 Form data:', data)
    console.log('📁 Files:', filesPreviews.length, 'images,', videoPreviews.length, 'videos')
    console.log('🏥 Patient ID:', patientId)
    console.log('🧠 Clara options:', claraOptions)
    console.log('✅ Form validation passed - proceeding with submission')
    
    setIsSubmitting(true)
    setError(null)

    try {
      const files = filesPreviews.map((fp) => fp.file)
      const videos = videoPreviews.map((vp) => vp.file)
      
      // Combine all media files
      const allFiles = [...files, ...videos]
      
      console.log('📤 About to call DiagnosisAPI.submitDiagnosis...')
      
      // Send directly to backend endpoint with Clara options and patient ID
      const result = await DiagnosisAPI.submitDiagnosis(data, allFiles, claraOptions, patientId)
      
      console.log('✅ DiagnosisAPI.submitDiagnosis successful:', result)
      
      setCurrentSessionId(result.session_id)
      setIsLoading(true)
      
      // Show context-aware success message
      if (patientId) {
        toast.success("Patient diagnosis submitted successfully! Using full medical history and context.")
      } else {
        toast.success("Diagnosis submitted successfully!")
      }

      // Start polling for status updates
      pollForResults(result.session_id)
    } catch (error) {
      console.error('❌ Form submission error:', error)
      const errorMessage = handleApiError(error)
      console.error('❌ Processed error message:', errorMessage)
      setError(errorMessage)
      toast.error(errorMessage)
    } finally {
      setIsSubmitting(false)
      console.log('🏁 Form submission finished (isSubmitting = false)')
    }
  }

  const handleOntologySearch = async () => {
    if (!ontologySearchTerm.trim()) {
      toast.error("Please enter a medical term to search")
      return
    }

    setIsSearchingOntology(true)
    setShowOntologyResults(false)

    try {
      const response = await fetch(`${API_BASE_URL}/ontology/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: ontologySearchTerm,
          limit: 10,
          search_type: "comprehensive",
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      if (data.results && data.results.length > 0) {
        setOntologySearchResults(data.results)
        setShowOntologyResults(true)
        toast.success(`Found ${data.results.length} medical concepts for "${ontologySearchTerm}"`)
      } else {
        setOntologySearchResults([])
        setShowOntologyResults(false)
        toast.error(`No medical concepts found for "${ontologySearchTerm}"`)
      }
    } catch (error) {
      console.error("Ontology search error:", error)
      toast.error("Failed to search medical ontology. Please try again.")
      setOntologySearchResults([])
      setShowOntologyResults(false)
    } finally {
      setIsSearchingOntology(false)
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg border-l-4 border-blue-600 p-8">
      <div className="mb-8 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg p-6">
        <h2 className="text-3xl font-bold mb-2 flex items-center">
          <svg className="w-8 h-8 mr-3" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M9.504 1.132a1 1 0 01.992 0l1.75 1a1 1 0 11-.992 1.736L10 3.152l-1.254.716a1 1 0 11-.992-1.736l1.75-1zM5.618 4.504a1 1 0 01-.372 1.364L5.016 6l.23.132a1 1 0 11-.992 1.736L3 7.723V8a1 1 0 01-2 0V6a.996.996 0 01.52-.878l1.734-.99a1 1 0 011.364.372zm8.764 0a1 1 0 011.364-.372l1.734.99A.996.996 0 0118 6v2a1 1 0 11-2 0v-.277l-1.254.145a1 1 0 11-.992-1.736L14.984 6l-.23-.132a1 1 0 01-.372-1.364zm-7 4a1 1 0 011.364-.372L10 8.848l1.254-.716a1 1 0 01.992 1.736L11 10.723V12a1 1 0 11-2 0v-1.277l-1.246-.855a1 1 0 01-.372-1.364zM3 11a1 1 0 011 1v1.277l1.246.855a1 1 0 01-.992 1.736L3 15.723V17a1 1 0 01-2 0v-5a1 1 0 011-1zm14 0a1 1 0 011 1v5a1 1 0 01-2 0v-1.277l-1.254-.145a1 1 0 01.992-1.736L16.984 15l-.23-.132A1 1 0 0117 11zm-8.5 4.5a1 1 0 011.364-.372l.254.145V16a1 1 0 112 0v-.727l.254-.145a1 1 0 11.992 1.736l-1.735.99a.995.995 0 01-1.022 0l-1.735-.99a1 1 0 01-.372-1.364z"
              clipRule="evenodd"
            />
          </svg>
          CortexMD - Medical Diagnosis Co-Pilot
        </h2>
        <p className="text-blue-100 flex items-center">
          <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
          Comprehensive AI-powered diagnosis with FOL verification and dynamic confidence scoring
        </p>
      </div>

      <form onSubmit={(e) => {
        console.log('📝 Form onSubmit event triggered')
        console.log('📋 Form errors:', errors)
        console.log('🔄 isSubmitting:', isSubmitting)
        return handleSubmit(onSubmit)(e)
      }} className="space-y-6">
        {/* Patient Information - Only show patient ID field if no patientId prop provided */}
        {!patientId && (
          <div className="bg-gray-50 rounded-lg p-4 md:p-6 border border-gray-200">
            <h3 className="text-lg md:text-xl font-semibold text-blue-700 mb-4 flex items-center">
              <svg className="w-5 h-5 md:w-6 md:h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                />
              </svg>
              Patient Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="md:col-span-3">
                <label className="block text-sm font-medium text-gray-700 mb-2">Patient ID</label>
                <input
                  type="text"
                  {...register("patient_id")}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors text-base"
                  placeholder="Auto-generated if empty"
                />
              </div>
              <div className="flex items-center space-x-2 pt-6 md:pt-6">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    {...register("anonymize")}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700">Anonymize PHI</span>
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Patient Context Information - Show when patientId is provided */}
        {patientId && (
          <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
            <h3 className="text-xl font-semibold text-blue-700 mb-4 flex items-center">
              <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                />
              </svg>
              Patient Context
            </h3>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-800">
                  🏥 Running diagnosis for patient: <span className="font-semibold">{patientId}</span>
                </p>
                <p className="text-xs text-blue-600 mt-1">
                  Patient demographics and history will be automatically included from their medical record.
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    {...register("anonymize")}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700">Anonymize PHI</span>
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Clinical Text */}
        <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
          <h3 className="text-xl font-semibold text-blue-700 mb-4 flex items-center">
            <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            Clinical Information
          </h3>
          <div className="relative">
            <textarea
              {...register("symptoms", { required: "Clinical information is required" })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors resize-none pr-12"
              rows={8}
              placeholder="Enter complete clinical presentation including:
• Patient demographics (age, gender)
• Chief complaint and history of present illness
• Physical examination findings
• Vital signs and measurements
• Laboratory results and diagnostic tests
• Imaging findings and reports
• Medical history and medications"
            />

            {/* Voice Recording Button */}
            <button
              type="button"
              onClick={sarvamVoice.isRecording ? sarvamVoice.stopRecording : sarvamVoice.startRecording}
              disabled={sarvamVoice.isProcessing || !sarvamVoice.isSupported}
              className={`absolute top-3 right-3 p-3 rounded-lg transition-all duration-200 ${
                sarvamVoice.isRecording
                  ? "text-red-600 bg-red-100 hover:bg-red-200 animate-pulse"
                  : sarvamVoice.isProcessing
                    ? "text-blue-600 bg-blue-100"
                    : "text-gray-600 hover:text-blue-600 hover:bg-blue-100"
              } ${!sarvamVoice.isSupported ? "opacity-50 cursor-not-allowed" : ""}`}
              title={
                !sarvamVoice.isSupported
                  ? "Voice recording not supported"
                  : sarvamVoice.isRecording
                    ? "Stop recording"
                    : sarvamVoice.isProcessing
                      ? "Processing with Sarvam AI..."
                      : "Record symptoms with voice"
              }
            >
              {sarvamVoice.isRecording ? (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a2 2 0 114 0v4a2 2 0 11-4 0V7z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : sarvamVoice.isProcessing ? (
                <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
              ) : (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                  />
                </svg>
              )}
            </button>
          </div>

          {/* Voice Recording Status */}
          {sarvamVoice.isRecording && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z"
                  clipRule="evenodd"
                />
              </svg>
              Recording... Speak your clinical observations
            </div>
          )}

          {sarvamVoice.isProcessing && (
            <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800 flex items-center">
              <div className="w-5 h-5 mr-2 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
              Processing audio with Sarvam AI...
            </div>
          )}

          {sarvamVoice.error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 flex items-center">
              <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
              {sarvamVoice.error}
            </div>
          )}

          {errors.symptoms && <p className="text-red-500 text-sm mt-2">{errors.symptoms.message}</p>}
          <p className="text-gray-600 text-sm mt-2">Provide comprehensive clinical data for accurate AI diagnosis</p>
        </div>

        {/* Medical Images */}
        <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
          <h3 className="text-xl font-semibold text-blue-700 mb-4 flex items-center">
            <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            Medical Images
          </h3>

          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-blue-500 transition-colors">
            <input
              type="file"
              multiple
              accept=".jpg,.jpeg,.png,.gif,.bmp,.tiff"
              onChange={handleFileChange}
              className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <p className="text-gray-600 text-sm mt-2">
              Upload X-rays, CT scans, MRI images, ultrasounds (max 16MB each)
            </p>

            {filesPreviews.length > 0 && (
              <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-green-800 text-sm font-medium flex items-center">
                  <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                  {filesPreviews.length} file(s) selected •{" "}
                  {(filesPreviews.reduce((acc, fp) => acc + fp.file.size, 0) / (1024 * 1024)).toFixed(1)} MB total
                </p>
              </div>
            )}
          </div>

          {filesPreviews.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-700 mb-3">Selected Images:</h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {filesPreviews.map((filePreview) => (
                  <div key={filePreview.id} className="relative group">
                    <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden border-2 border-gray-200 hover:border-blue-300 transition-colors">
                      <img
                        src={filePreview.previewUrl || "/placeholder.svg"}
                        alt={filePreview.file.name}
                        className="w-full h-full object-cover"
                      />
                    </div>

                    {/* Remove button */}
                    <button
                      type="button"
                      onClick={() => removeFile(filePreview.id)}
                      className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600 transition-colors shadow-md"
                      title="Remove image"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>

                    {/* File info overlay */}
                    <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white text-xs p-2 rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
                      <p className="truncate" title={filePreview.file.name}>
                        {filePreview.file.name}
                      </p>
                      <p className="text-gray-300">{(filePreview.file.size / (1024 * 1024)).toFixed(1)} MB</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

          {/* Medical Videos with XAI */}
          <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
            <h3 className="text-xl font-semibold text-blue-700 mb-4 flex items-center">
              <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              Medical Video Analysis (XAI)
            </h3>
            
            <button
              type="button"
              onClick={() => setShowVideoInput(!showVideoInput)}
              className="w-full px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center justify-center"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 16h4m10 0h4" />
              </svg>
              {showVideoInput ? 'Hide Video Input' : 'Add Medical Video'}
            </button>
            
            {showVideoInput && (
              <div className="mt-4">
                <div className="border-2 border-dashed border-purple-300 rounded-lg p-6 hover:border-purple-500 transition-colors bg-purple-50">
                  <input
                    type="file"
                    accept="video/*,.mp4,.avi,.mov,.webm"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) {
                        const id = `video-${Date.now()}`
                        const previewUrl = URL.createObjectURL(file)
                        setVideoPreviews([{ file, id, previewUrl }])
                        toast.success('Video selected for XAI analysis')
                      }
                    }}
                    className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-100 file:text-purple-700 hover:file:bg-purple-200"
                  />
                  <p className="text-gray-600 text-sm mt-2">
                    Upload ultrasound, endoscopy, fluoroscopy, or other medical videos (max 500MB)
                  </p>
                  
                  {videoPreviews.length > 0 && (
                    <div className="mt-4">
                      <div className="bg-purple-100 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-semibold text-purple-800">{videoPreviews[0].file.name}</p>
                            <p className="text-sm text-purple-600">
                              {(videoPreviews[0].file.size / (1024 * 1024)).toFixed(1)} MB
                            </p>
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              URL.revokeObjectURL(videoPreviews[0].previewUrl)
                              setVideoPreviews([])
                            }}
                            className="text-purple-600 hover:text-purple-800"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="mt-4 p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <h4 className="font-semibold text-purple-800 mb-2">XAI Video Analysis Features:</h4>
                  <ul className="space-y-1 text-sm text-gray-700">
                    <li className="flex items-center">
                      <svg className="w-4 h-4 mr-2 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      Temporal motion analysis
                    </li>
                    <li className="flex items-center">
                      <svg className="w-4 h-4 mr-2 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      Frame-by-frame attention mapping
                    </li>
                    <li className="flex items-center">
                      <svg className="w-4 h-4 mr-2 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      Explainable AI decision paths
                    </li>
                    <li className="flex items-center">
                      <svg className="w-4 h-4 mr-2 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                      Abnormality detection with confidence scores
                    </li>
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* FHIR Data */}
          <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
            <h3 className="text-xl font-semibold text-blue-700 mb-4 flex items-center">
              <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
                />
              </svg>
              FHIR Structured Data
            </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Upload FHIR JSON</label>
              <input
                type="file"
                accept=".json"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) {
                    const reader = new FileReader()
                    reader.onload = (e) => {
                      // Handle FHIR file
                    }
                    reader.readAsText(file)
                  }
                }}
                className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Manual FHIR Entry</label>
              <button
                type="button"
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors flex items-center"
                onClick={() => setShowFhirInput(!showFhirInput)}
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                  />
                </svg>
                Enter JSON
              </button>
            </div>
          </div>
          {showFhirInput && (
            <div className="mt-4">
              <textarea
                {...register("fhir_data")}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors resize-none"
                rows={4}
                placeholder='{"patient": {"age": 45, "gender": "male"}, "symptoms": ["chest pain"], "vital_signs": {...}}'
              />
            </div>
          )}
        </div>

        {/* NVIDIA Clara Integration */}
        <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 border border-green-200">
          <h3 className="text-xl font-semibold text-green-700 mb-4 flex items-center">
            <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            🚀 NVIDIA Clara AI Enhancement
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-lg font-semibold text-blue-600 mb-3 flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
                🔬 Clara Imaging
              </h4>
              <div className="space-y-3">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={claraOptions.dicom_processing}
                    onChange={(e) => setClaraOptions((prev) => ({ ...prev, dicom_processing: e.target.checked }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-3 text-sm text-gray-700">Enhanced DICOM Processing</span>
                </label>
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={claraOptions["3d_reconstruction"]}
                    onChange={(e) => setClaraOptions((prev) => ({ ...prev, "3d_reconstruction": e.target.checked }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-3 text-sm text-gray-700">3D Volume Reconstruction</span>
                </label>
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={claraOptions.image_segmentation}
                    onChange={(e) => setClaraOptions((prev) => ({ ...prev, image_segmentation: e.target.checked }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className="ml-3 text-sm text-gray-700">AI Image Segmentation</span>
                </label>
              </div>
            </div>
            <div>
              <h4 className="text-lg font-semibold text-purple-600 mb-3 flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                  />
                </svg>
                🧬 Clara Parabricks
              </h4>
              <div className="space-y-3">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={claraOptions.genomic_analysis}
                    onChange={(e) => setClaraOptions((prev) => ({ ...prev, genomic_analysis: e.target.checked }))}
                    className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                  />
                  <span className="ml-3 text-sm text-gray-700">Genomic Analysis</span>
                </label>
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={claraOptions.variant_calling}
                    onChange={(e) => setClaraOptions((prev) => ({ ...prev, variant_calling: e.target.checked }))}
                    className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                  />
                  <span className="ml-3 text-sm text-gray-700">Variant Calling</span>
                </label>
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={claraOptions.multi_omics}
                    onChange={(e) => setClaraOptions((prev) => ({ ...prev, multi_omics: e.target.checked }))}
                    className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                  />
                  <span className="ml-3 text-sm text-gray-700">Multi-omics Integration</span>
                </label>
              </div>
            </div>
          </div>
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-blue-800 text-sm flex items-center">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
              Clara features require NVIDIA GPU and Clara SDK installation.
            </p>
          </div>
        </div>

        {/* Medical Ontology Search */}
        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-6 border border-indigo-200">
          <h3 className="text-xl font-semibold text-indigo-700 mb-4 flex items-center">
            <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13m0-13C4.168 5.477 5.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
              />
            </svg>
            📚 Ontology Analysis & Medical Knowledge Graph
          </h3>
          <div className="space-y-4">
            <div>
              <h4 className="text-lg font-semibold text-blue-600 mb-3">Quick Ontology Search</h4>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={ontologySearchTerm}
                  onChange={(e) => setOntologySearchTerm(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && handleOntologySearch()}
                  placeholder="Enter medical term (e.g., sarcoma, myocardial infarction)"
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <button
                  type="button"
                  onClick={handleOntologySearch}
                  disabled={isSearchingOntology || !ontologySearchTerm.trim()}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200 flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSearchingOntology ? (
                    <>
                      <svg className="animate-spin w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24">
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        ></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                      </svg>
                      Searching...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                        />
                      </svg>
                      Search
                    </>
                  )}
                </button>
              </div>
              {showOntologyResults && ontologySearchResults.length > 0 && (
                <div className="mt-4 bg-white border border-gray-200 rounded-lg shadow-sm">
                  <div className="p-4">
                    <h6 className="text-green-600 font-semibold mb-2">Search Results for "{ontologySearchTerm}":</h6>
                    <div className="space-y-3">
                      {ontologySearchResults.map((result, index) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-md border">
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <h6 className="font-semibold text-gray-900">
                                {result.name || result.preferred_name || result.label}
                              </h6>
                              {result.description && <p className="text-sm text-gray-600 mt-1">{result.description}</p>}
                              {result.code && (
                                <span className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full mt-2">
                                  Code: {result.code}
                                </span>
                              )}
                              {result.source && (
                                <span className="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full mt-2 ml-2">
                                  {result.source}
                                </span>
                              )}
                            </div>
                            {result.confidence && (
                              <div className="text-right">
                                <span className="text-sm text-gray-500">
                                  Confidence: {(result.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-center">
          <button
            type="submit"
            disabled={isSubmitting}
            onClick={() => {
              console.log('🖱️ Submit button clicked')
              console.log('🔒 Button disabled:', isSubmitting)
              console.log('📋 Current form errors:', errors)
            }}
            className="w-full max-w-md px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105"
          >
            {isSubmitting ? (
              <div className="flex items-center justify-center">
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                {patientId ? 'Starting Patient Diagnosis...' : 'Starting Analysis...'}
              </div>
            ) : (
              <div className="flex items-center justify-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
                {patientId ? '🧠 Generate Patient Diagnosis with Full Context' : '🧠 Generate Comprehensive Diagnosis'}
              </div>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
