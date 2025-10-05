"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useForm } from "react-hook-form"
import {
  PaperAirplaneIcon,
  PhotoIcon,
  MicrophoneIcon,
  StopIcon,
  DocumentArrowDownIcon,
  TrashIcon,
} from "@heroicons/react/24/outline"
import { ChatAPI, handleApiError, api } from "@/lib/api"
import { useDiagnosisStore } from "@/store/diagnosisStore"
import type { ChatMessage } from "@/types"
import { useSarvamVoice } from "@/hooks/useSarvamVoice"
import { DetailedExplanations } from "./DetailedExplanations"
import { ConfidenceVisualization, ConfidenceIndicator } from "./ConfidenceVisualization"
import { ChatMarkdownText } from "@/utils/markdown"
import toast from "react-hot-toast"

interface ChatForm {
  message: string
}

interface ChatInterfaceProps {
  patientId?: string
}

export function ChatInterface({ patientId }: ChatInterfaceProps = {}) {
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImages, setSelectedImages] = useState<File[]>([])

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const { chatMessages, addChatMessage, clearChatMessages, currentSessionId } = useDiagnosisStore()
  const { register, handleSubmit, reset, watch, setValue } = useForm<ChatForm>()
  
  // Sarvam AI voice recording
  const sarvamVoice = useSarvamVoice({
    language: 'en',
    model: 'saarika:v1'
  })

  // Local audio recording state for manual MediaRecorder usage
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const [audioChunks, setAudioChunks] = useState<Blob[]>([])
  const [isRecording, setIsRecording] = useState(false)

  const messageText = watch("message", "")

  useEffect(() => {
    scrollToBottom()
  }, [chatMessages])

  // Auto-fill transcript when voice recording completes
  useEffect(() => {
    if (sarvamVoice.transcript && !sarvamVoice.isProcessing) {
      setValue('message', sarvamVoice.transcript)
      toast.success('Voice message transcribed! You can edit or send it.')
    }
  }, [sarvamVoice.transcript, sarvamVoice.isProcessing, setValue])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const handleSendMessage = async (data: ChatForm) => {
    if (!data.message.trim() && selectedImages.length === 0) return

    const messageId = Date.now().toString()
    const userMessage: ChatMessage = {
      id: messageId,
      content: data.message,
      sender: "user",
      timestamp: new Date().toISOString(),
      images: selectedImages.length > 0 ? selectedImages.map((f) => f.name) : undefined,
    }

    addChatMessage(userMessage)
    reset()
    setSelectedImages([])
    setIsLoading(true)

    try {
      let response;
      
      if (patientId) {
        // Use patient-specific chat API with full context
        console.log(`Sending patient-specific message for ${patientId}:`, data.message)
        const { data: resp } = await api.post(`/api/patients/${patientId}/chat`, { message: data.message })
        response = resp
        console.log('Patient chat response:', response)
      } else {
        // Use general chat API
        response = await ChatAPI.sendMessage(
          data.message,
          currentSessionId || undefined,
          undefined,
          selectedImages.length > 0 ? selectedImages : undefined,
          true // Enable explainability
        )
      }

      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: response.response ?? "",
        sender: "ai",
        timestamp: new Date().toISOString(),
        metrics: response.metrics,
        detailed_explanations: response.detailed_explanations,
        fol_verification: response.fol_verification,
        medical_reasoning: response.medical_reasoning,
        reasoning_steps: response.reasoning_steps,
        confidence_breakdown: response.confidence_breakdown,
        explainability_data: response.explainability_data
      }

      addChatMessage(aiMessage)
    } catch (error) {
      const errorMessage = handleApiError(error)
      toast.error(errorMessage)

      const errorAiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: `Sorry, I encountered an error: ${errorMessage}`,
        sender: "ai",
        timestamp: new Date().toISOString(),
      }
      addChatMessage(errorAiMessage)
    } finally {
      setIsLoading(false)
    }
  }

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    const imageFiles = files.filter((file) => file.type.startsWith("image/"))

    if (imageFiles.length !== files.length) {
      toast.error("Only image files are supported")
    }

    setSelectedImages((prev) => [...prev, ...imageFiles])
  }

  const removeImage = (index: number) => {
    setSelectedImages((prev) => prev.filter((_, i) => i !== index))
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setAudioChunks((prev) => [...prev, event.data])
        }
      }

      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
      }

      setMediaRecorder(recorder)
      setAudioChunks([])
      recorder.start()
      setIsRecording(true)
      toast.success("Recording started")
    } catch (error) {
      toast.error("Failed to start recording")
      console.error("Recording error:", error)
    }
  }

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop()
      setIsRecording(false)
      toast.success("Recording stopped")
    }
  }

  const downloadChat = () => {
    const chatData = {
      session_id: currentSessionId,
      messages: chatMessages,
      exported_at: new Date().toISOString(),
    }

    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `chat_history_${currentSessionId || "session"}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)

    toast.success("Chat history downloaded")
  }

  const clearChat = () => {
    if (window.confirm("Are you sure you want to clear the chat history?")) {
      clearChatMessages()
      toast.success("Chat history cleared")
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  return (
    <div className="flex flex-col h-[80vh] card overflow-hidden">
      {/* Chat Header */}
      <div className="card-header flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold">AI Medical Assistant</h2>
          <p className="opacity-90 text-sm">Ask questions about symptoms, treatments, or medical concepts</p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={downloadChat}
            disabled={chatMessages.length === 0}
            className="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors disabled:opacity-50"
            title="Download chat history"
          >
            <DocumentArrowDownIcon className="w-5 h-5" />
          </button>
          <button
            onClick={clearChat}
            disabled={chatMessages.length === 0}
            className="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors disabled:opacity-50"
            title="Clear chat"
          >
            <TrashIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
        {chatMessages.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🩺</div>
            <h3 className="text-lg font-semibold text-gray-600 mb-2">Welcome to AI Medical Assistant</h3>
            <p className="text-gray-500 max-w-md mx-auto">
              Ask me about symptoms, medical conditions, treatments, or upload medical images for analysis.
            </p>
          </div>
        ) : (
          <>
            {chatMessages.map((message) => (
              <div key={message.id} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl p-4 ${
                    message.sender === "user"
                      ? "bg-gradient-medical text-white rounded-br-md"
                      : "bg-white text-gray-800 shadow-md rounded-bl-md border"
                  }`}
                >
                  <div className="mb-2">
                    <ChatMarkdownText 
                      isAiMessage={message.sender === "ai"}
                    >
                      {message.content}
                    </ChatMarkdownText>
                  </div>

                  {message.images && message.images.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {message.images.map((imageName, index) => (
                        <span key={index} className="text-xs bg-white/20 px-2 py-1 rounded">
                          📷 {imageName}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Rich Explainability Section for AI Messages */}
                  {message.sender === "ai" && (
                    message.detailed_explanations || 
                    message.fol_verification || 
                    message.medical_reasoning ||
                    message.metrics ||
                    message.reasoning_steps
                  ) && (
                    <DetailedExplanations
                      explanations={message.detailed_explanations || []}
                      folVerification={message.fol_verification}
                      medicalReasoning={message.medical_reasoning}
                      confidenceMetrics={message.metrics}
                      reasoningSteps={message.reasoning_steps}
                    />
                  )}

                  <div
                    className={`text-xs mt-2 flex items-center justify-between ${
                      message.sender === "user" ? "text-white/80" : "text-gray-500"
                    }`}
                  >
                    <span>{formatTimestamp(message.timestamp)}</span>

                    {message.metrics && (
                      <div className="flex items-center space-x-2">
                        <ConfidenceIndicator confidence={message.metrics.confidence} />
                        <span
                          className={`px-2 py-1 rounded text-xs font-medium ${
                            message.metrics.fol_verified ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                          }`}
                        >
                          FOL {message.metrics.fol_verified ? "✓" : "✗"}
                        </span>
                        {message.metrics.explainability && (
                          <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded text-xs font-medium">
                            XAI: {(message.metrics.explainability * 100).toFixed(0)}%
                          </span>
                        )}
                        {message.metrics.medical_accuracy && (
                          <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-medium">
                            Med: {(message.metrics.medical_accuracy * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white rounded-2xl rounded-bl-md p-4 shadow-md border max-w-[70%]">
                  <div className="typing-indicator">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                  <div className="text-xs text-gray-500 mt-2">AI is thinking...</div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <div className="border-t bg-white p-4">
        {/* Selected Images Preview */}
        {selectedImages.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {selectedImages.map((file, index) => (
              <div key={index} className="flex items-center gap-2 bg-gray-100 rounded-lg p-2">
                <PhotoIcon className="w-4 h-4 text-gray-600" />
                <span className="text-sm text-gray-700">{file.name}</span>
                <button onClick={() => removeImage(index)} className="text-red-500 hover:text-red-700">
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit(handleSendMessage)} className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              {...register("message")}
              placeholder="Ask about symptoms, conditions, treatments..."
              className="w-full resize-none rounded-lg border-2 border-gray-200 p-3 focus:border-primary-500 focus:outline-none"
              rows={2}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit(handleSendMessage)()
                }
              }}
            />
          </div>

          <div className="flex items-center space-x-2">
            {/* Image Upload */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-3 text-gray-600 hover:text-primary-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Upload image"
            >
              <PhotoIcon className="w-5 h-5" />
            </button>

            {/* Voice Recording */}
            <button
              type="button"
              onClick={sarvamVoice.isRecording ? sarvamVoice.stopRecording : sarvamVoice.startRecording}
              disabled={sarvamVoice.isProcessing || !sarvamVoice.isSupported}
              className={`p-3 rounded-lg transition-colors ${
                sarvamVoice.isRecording
                  ? "text-red-600 bg-red-100 hover:bg-red-200 animate-pulse"
                  : sarvamVoice.isProcessing
                  ? "text-blue-600 bg-blue-100"
                  : "text-gray-600 hover:text-primary-600 hover:bg-gray-100"
              } ${!sarvamVoice.isSupported ? "opacity-50 cursor-not-allowed" : ""}`}
              title={
                !sarvamVoice.isSupported 
                  ? "Voice recording not supported"
                  : sarvamVoice.isRecording 
                  ? "Stop recording" 
                  : sarvamVoice.isProcessing
                  ? "Processing with Sarvam AI..."
                  : "Start voice recording"
              }
            >
              {sarvamVoice.isRecording ? (
                <StopIcon className="w-5 h-5" />
              ) : sarvamVoice.isProcessing ? (
                <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
              ) : (
                <MicrophoneIcon className="w-5 h-5" />
              )}
            </button>

            {/* Send Button */}
            <button
              type="submit"
              disabled={isLoading || (!messageText.trim() && selectedImages.length === 0)}
              className="btn-primary p-3 disabled:opacity-50"
              title="Send message"
            >
              <PaperAirplaneIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*"
            onChange={handleImageSelect}
            className="hidden"
          />
        </form>
      </div>
    </div>
  )
}
