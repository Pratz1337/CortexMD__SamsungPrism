"use client"

import { useState, useEffect } from "react"
import { DiagnosisAPI } from "@/lib/api"
import { AddNoteForm } from "@/components/concern/AddNoteForm"
import { toast } from "react-hot-toast"
import { MedicalMarkdownText } from "@/utils/markdown"
import ScanClinicalNote from "@/components/notes/ScanClinicalNote"
import ScannedNotesList from "@/components/notes/ScannedNotesList"
import { HealthcareLoadingScreen } from "@/components/ui/HealthcareLoadingScreen"

interface ClinicalNote {
  note_id: string
  patient_id: string
  nurse_id: string
  content: string
  location?: string
  shift?: string
  note_type: string
  timestamp: string
}

interface ClinicalNotesViewProps {
  patientId: string
}

export function ClinicalNotesView({ patientId }: ClinicalNotesViewProps) {
  const [notes, setNotes] = useState<ClinicalNote[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [showScanForm, setShowScanForm] = useState(false)
  const [activeView, setActiveView] = useState<'manual' | 'scanned'>('manual')

  const loadNotes = async () => {
    try {
      setLoading(true)
      console.log('🔍 Loading clinical notes for patient:', patientId)
      const response = await DiagnosisAPI.getPatientClinicalNotes(patientId)
      console.log('📋 Clinical notes API response:', response)
      
      if (response.success) {
        const notesData = response.notes || []
        console.log('📝 Setting notes data:', notesData)
        setNotes(notesData)
        
        if (notesData.length === 0) {
          console.log('ℹ️ No clinical notes found for patient')
        } else {
          console.log(`✅ Loaded ${notesData.length} clinical notes`)
        }
      } else {
        console.warn('⚠️ API returned success: false', response)
        toast.error(`Failed to load clinical notes: ${response.error || 'Unknown error'}`)
      }
    } catch (error: any) {
      console.error('❌ Error loading clinical notes:', error)
      console.error('❌ Error details:', error?.response?.data)
      toast.error(`Failed to load clinical notes: ${error?.message || 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadNotes()
  }, [patientId])

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getRiskColor = (content: string) => {
    const lowerContent = content.toLowerCase()
    if (lowerContent.includes('critical') || lowerContent.includes('emergency') || lowerContent.includes('urgent')) {
      return 'border-l-red-500 bg-red-50'
    }
    if (lowerContent.includes('concern') || lowerContent.includes('monitoring') || lowerContent.includes('watch')) {
      return 'border-l-yellow-500 bg-yellow-50'
    }
    return 'border-l-green-500 bg-green-50'
  }

  if (showAddForm) {
    return (
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Add Clinical Note</h2>
          <button
            onClick={() => setShowAddForm(false)}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 transition"
          >
            ← Back to Notes
          </button>
        </div>
        <AddNoteForm 
          patientId={patientId} 
          onSuccess={() => {
            setShowAddForm(false)
            loadNotes()
          }}
        />
      </div>
    )
  }

  if (showScanForm) {
    return (
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">📷 Scan Clinical Note with AR</h2>
          <button
            onClick={() => setShowScanForm(false)}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 transition"
          >
            ← Back to Notes
          </button>
        </div>
        <ScanClinicalNote 
          patientId={patientId} 
          onAdded={() => {
            setShowScanForm(false)
            setActiveView('scanned')
            loadNotes()
          }}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Clinical Notes</h2>
          <p className="text-gray-600">Patient ID: {patientId}</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowScanForm(true)}
            className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-medium flex items-center gap-2"
          >
            📷 AR Scan Note
          </button>
          <button
            onClick={() => setShowAddForm(true)}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
          >
            + Add Note
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveView('manual')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeView === 'manual'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            📝 Manual Notes
          </button>
          <button
            onClick={() => setActiveView('scanned')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeView === 'scanned'
                ? 'border-purple-500 text-purple-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            📷 Scanned Notes (AR)
          </button>
        </nav>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-8">
          <HealthcareLoadingScreen 
            variant="heartbeat" 
            message="Loading clinical notes..." 
            className="min-h-0"
          />
        </div>
      )}

      {/* Content based on active view */}
      {!loading && activeView === 'manual' && (
        <div className="space-y-4">
          {notes.length === 0 ? (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <div className="text-gray-400 text-4xl mb-4">📝</div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Manual Notes Yet</h3>
              <p className="text-gray-600 mb-4">Start documenting patient observations and care notes.</p>
              <button
                onClick={() => setShowAddForm(true)}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
              >
                Add First Note
              </button>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <span className="text-sm text-gray-600">
                    {notes.length} note{notes.length !== 1 ? 's' : ''} found
                  </span>
                  <div className="text-xs text-gray-400 mt-1">
                    Patient ID: {patientId} | Last updated: {new Date().toLocaleString()}
                  </div>
                </div>
                <button
                  onClick={loadNotes}
                  className="text-sm text-blue-600 hover:text-blue-800 transition flex items-center gap-1"
                >
                  🔄 Refresh
                </button>
              </div>

              {notes.map((note) => (
                <div
                  key={note.note_id}
                  className={`p-6 bg-white border-l-4 rounded-lg shadow-sm ${getRiskColor(note.content)}`}
                >
                  {/* Note Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <div className="flex items-center gap-3">
                        <span className="font-semibold text-gray-900">
                          {note.nurse_id}
                        </span>
                        {note.location && (
                          <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                            {note.location}
                          </span>
                        )}
                        {note.shift && (
                          <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                            {note.shift} Shift
                          </span>
                        )}
                      </div>
                      <div className="text-sm text-gray-600 mt-1">
                        {formatDate(note.timestamp)}
                      </div>
                    </div>
                    <div className="text-xs text-gray-500">
                      {note.note_type}
                    </div>
                  </div>

                  {/* Note Content */}
                  <div className="prose prose-sm max-w-none">
                    {/* Debug info */}
                    {process.env.NODE_ENV === 'development' && (
                      <div className="text-xs text-gray-400 mb-2 p-2 bg-gray-50 rounded">
                        <strong>Debug:</strong> Note ID: {note.note_id} | Content length: {note.content?.length || 0}
                      </div>
                    )}
                    
                    {/* Note content with fallback */}
                    {note.content ? (
                      <MedicalMarkdownText className="text-gray-800">
                        {note.content}
                      </MedicalMarkdownText>
                    ) : (
                      <div className="text-gray-400 italic p-4 border-2 border-dashed border-gray-200 rounded">
                        ℹ️ No content available for this note
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Scanned Notes View */}
      {!loading && activeView === 'scanned' && (
        <ScannedNotesList patientId={patientId} />
      )}
    </div>
  )
}
