"use client"

import { useState, useRef, useCallback } from 'react'
import { api } from '@/lib/api'
import { toast } from 'react-hot-toast'

interface SarvamVoiceConfig {
  apiKey?: string
  language?: string
  model?: string
}

interface VoiceRecordingState {
  isRecording: boolean
  isProcessing: boolean
  transcript: string
  error: string | null
  audioBlob: Blob | null
}

export function useSarvamVoice(config: SarvamVoiceConfig = {}) {
  const [state, setState] = useState<VoiceRecordingState>({
    isRecording: false,
    isProcessing: false,
    transcript: '',
    error: null,
    audioBlob: null
  })

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  const startRecording = useCallback(async () => {
    try {
      // Reset state
      setState(prev => ({
        ...prev,
        isRecording: false,
        isProcessing: false,
        transcript: '',
        error: null,
        audioBlob: null
      }))

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        }
      })

      streamRef.current = stream
      audioChunksRef.current = []

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        
        setState(prev => ({
          ...prev,
          isRecording: false,
          isProcessing: true,
          audioBlob
        }))

        // Process with Sarvam AI
        await processAudioWithSarvam(audioBlob)
      }

      mediaRecorder.start(1000) // Collect data every second

      setState(prev => ({
        ...prev,
        isRecording: true,
        error: null
      }))

      toast.success('🎤 Recording started')

    } catch (error) {
      console.error('Error starting recording:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to start recording'
      
      setState(prev => ({
        ...prev,
        error: errorMessage
      }))
      
      toast.error(`Recording failed: ${errorMessage}`)
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && state.isRecording) {
      mediaRecorderRef.current.stop()
      
      // Stop all tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }
      
      toast('🔄 Processing audio...')
    }
  }, [state.isRecording])

  const processAudioWithSarvam = useCallback(async (audioBlob: Blob) => {
    try {
      // Send audio as multipart/form-data to backend STT endpoint
      const formData = new FormData()
      const file = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
      formData.append('audio', file)

      const { data } = await api.post('/audio/transcribe', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      const result = data
      const transcript = result?.transcription?.transcription || result?.transcript

      if (transcript) {
        setState(prev => ({
          ...prev,
          transcript,
          isProcessing: false,
          error: null
        }))
        
        toast.success('✅ Audio transcribed successfully!')
        return transcript
      } else {
        throw new Error('No transcript received from Sarvam AI')
      }

    } catch (error) {
      console.error('Error processing audio with Sarvam:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to process audio'
      
      setState(prev => ({
        ...prev,
        isProcessing: false,
        error: errorMessage
      }))
      
      toast.error(`Transcription failed: ${errorMessage}`)
      return null
    }
  }, [config])

  const resetRecording = useCallback(() => {
    // Stop recording if active
    if (state.isRecording) {
      stopRecording()
    }

    // Clean up
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    setState({
      isRecording: false,
      isProcessing: false,
      transcript: '',
      error: null,
      audioBlob: null
    })
  }, [state.isRecording, stopRecording])

  return {
    ...state,
    startRecording,
    stopRecording,
    resetRecording,
    isSupported: typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
  }
}
