"use client"

import { useState, useRef, useCallback } from 'react'
import { toast } from 'react-hot-toast'
import { VideoAnalysisService, VideoAnalysisResult } from '@/services/videoAnalysisService'
import { PlayIcon, PauseIcon, FilmIcon, CameraIcon, XMarkIcon } from '@heroicons/react/24/solid'

interface VideoInputProps {
  patientId: string
  onAnalysisComplete: (result: VideoAnalysisResult) => void
  onFramesExtracted?: (frames: Array<{ url: string; timestamp: number }>) => void
}

export function VideoInput({ patientId, onAnalysisComplete, onFramesExtracted }: VideoInputProps) {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string>('')
  const [analyzing, setAnalyzing] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [analysisType, setAnalysisType] = useState<'ultrasound' | 'endoscopy' | 'xray_motion' | 'mri_sequence' | 'general'>('general')
  const [showPreview, setShowPreview] = useState(false)
  const [extractedFrames, setExtractedFrames] = useState<Array<{ url: string; timestamp: number }>>([])
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (file.type.startsWith('video/')) {
        setVideoFile(file)
        setVideoUrl(URL.createObjectURL(file))
        setShowPreview(true)
      } else {
        toast.error('Please select a valid video file')
      }
    }
  }

  const handleAnalyze = async () => {
    if (!videoFile) {
      toast.error('Please select a video file first')
      return
    }

    setAnalyzing(true)
    setUploadProgress(0)

    try {
      // First extract key frames
      toast.loading('Extracting key frames...', { id: 'extract' })
      const frames = await VideoAnalysisService.extractKeyFrames(videoFile, {
        maxFrames: 10,
        motionThreshold: 0.3,
        includeFirst: true,
        includeLast: true
      })

      // Convert frames to URLs for display
      const frameUrls = frames.map(frame => ({
        url: URL.createObjectURL(frame.frame),
        timestamp: frame.timestamp
      }))
      setExtractedFrames(frameUrls)
      if (onFramesExtracted) {
        onFramesExtracted(frameUrls)
      }
      toast.dismiss('extract')
      toast.success(`Extracted ${frames.length} key frames`)

      // Analyze video
      toast.loading('Analyzing video with AI...', { id: 'analyze' })
      const result = await VideoAnalysisService.analyzeVideo(
        videoFile,
        patientId,
        analysisType
      )
      
      toast.dismiss('analyze')
      toast.success('Video analysis complete!')
      onAnalysisComplete(result)
    } catch (error) {
      console.error('Video analysis error:', error)
      toast.error('Failed to analyze video')
    } finally {
      setAnalyzing(false)
      setUploadProgress(0)
    }
  }

  const captureCurrentFrame = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas')
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0)
        canvas.toBlob((blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob)
            const timestamp = videoRef.current!.currentTime
            setExtractedFrames(prev => [...prev, { url, timestamp }])
            toast.success('Frame captured!')
          }
        }, 'image/png')
      }
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-gray-800 mb-2">Medical Video Analysis</h3>
        <p className="text-gray-600">Upload medical videos for AI-powered analysis with explainable insights</p>
      </div>

      {/* Video Type Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Video Type
        </label>
        <select
          value={analysisType}
          onChange={(e) => setAnalysisType(e.target.value as any)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="general">General Medical Video</option>
          <option value="ultrasound">Ultrasound</option>
          <option value="endoscopy">Endoscopy</option>
          <option value="xray_motion">X-Ray Motion Study</option>
          <option value="mri_sequence">MRI Sequence</option>
        </select>
      </div>

      {/* File Input */}
      <div className="mb-6">
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="hidden"
        />
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
        >
          <FilmIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600 mb-2">Click to upload medical video</p>
          <p className="text-sm text-gray-500">Supported formats: MP4, AVI, MOV, WEBM</p>
          <p className="text-sm text-gray-500">Max file size: 500MB</p>
        </div>
      </div>

      {/* Video Preview */}
      {videoUrl && showPreview && (
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <h4 className="font-semibold text-gray-700">Video Preview</h4>
            <button
              onClick={() => {
                setShowPreview(false)
                setVideoUrl('')
                setVideoFile(null)
                setExtractedFrames([])
              }}
              className="text-red-500 hover:text-red-700"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>
          <div className="relative bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full max-h-96"
            />
            <button
              onClick={captureCurrentFrame}
              className="absolute bottom-4 right-4 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition flex items-center space-x-2"
            >
              <CameraIcon className="w-5 h-5" />
              <span>Capture Frame</span>
            </button>
          </div>
        </div>
      )}

      {/* Extracted Frames */}
      {extractedFrames.length > 0 && (
        <div className="mb-6">
          <h4 className="font-semibold text-gray-700 mb-2">Key Frames</h4>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
            {extractedFrames.map((frame, index) => (
              <div key={index} className="relative group">
                <img
                  src={frame.url}
                  alt={`Frame at ${frame.timestamp.toFixed(1)}s`}
                  className="w-full h-24 object-cover rounded border border-gray-300 cursor-pointer hover:border-blue-500"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white text-xs p-1 text-center opacity-0 group-hover:opacity-100 transition-opacity">
                  {frame.timestamp.toFixed(1)}s
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Progress Bar */}
      {analyzing && (
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>Processing video...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={!videoFile || analyzing}
        className={`w-full py-3 px-6 rounded-lg font-semibold transition-all ${
          !videoFile || analyzing
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'bg-blue-600 text-white hover:bg-blue-700'
        }`}
      >
        {analyzing ? (
          <span className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
            Analyzing Video...
          </span>
        ) : (
          'Analyze Video with XAI'
        )}
      </button>

      {/* XAI Info */}
      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-800">
          <strong>Explainable AI (XAI) Features:</strong> Our analysis provides transparent insights including attention maps, 
          temporal analysis, and detailed explanations for each finding.
        </p>
      </div>
    </div>
  )
}
