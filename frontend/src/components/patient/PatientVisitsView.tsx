"use client"

import { useState, useEffect } from "react"
import { DiagnosisAPI } from "@/lib/api"
import { AddVisitForm } from "@/components/concern/AddVisitForm"
import { toast } from "react-hot-toast"
import { HealthcareLoadingScreen } from "@/components/ui/HealthcareLoadingScreen"

interface PatientVisit {
  visit_id: string
  patient_id: string
  nurse_id: string
  location: string
  visit_type: string
  duration_minutes: number
  notes?: string
  timestamp: string
}

interface PatientVisitsViewProps {
  patientId: string
}

export function PatientVisitsView({ patientId }: PatientVisitsViewProps) {
  const [visits, setVisits] = useState<PatientVisit[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)

  const loadVisits = async () => {
    try {
      setLoading(true)
      const response = await DiagnosisAPI.getPatientVisits(patientId)
      if (response.success) {
        setVisits(response.visits || [])
      }
    } catch (error) {
      console.error('Error loading patient visits:', error)
      toast.error('Failed to load patient visits')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadVisits()
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

  const getVisitTypeColor = (visitType: string) => {
    switch (visitType.toLowerCase()) {
      case 'urgent':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'emergency':
        return 'bg-red-200 text-red-900 border-red-300'
      case 'routine':
        return 'bg-green-100 text-green-800 border-green-200'
      default:
        return 'bg-blue-100 text-blue-800 border-blue-200'
    }
  }

  if (showAddForm) {
    return (
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Record Patient Visit</h2>
          <button
            onClick={() => setShowAddForm(false)}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 transition"
          >
            ← Back to Visits
          </button>
        </div>
        <AddVisitForm 
          patientId={patientId} 
          onSuccess={() => {
            setShowAddForm(false)
            loadVisits()
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
          <h2 className="text-2xl font-bold text-gray-900">Patient Visits</h2>
          <p className="text-gray-600">Patient ID: {patientId}</p>
        </div>
        <button
          onClick={() => setShowAddForm(true)}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
        >
          + Record Visit
        </button>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-8">
          <HealthcareLoadingScreen 
            variant="pulse" 
            message="Loading patient visits..." 
            className="min-h-0"
          />
        </div>
      )}

      {/* Visits List */}
      {!loading && (
        <div className="space-y-4">
          {visits.length === 0 ? (
            <div className="text-center py-12 bg-gray-50 rounded-lg">
              <div className="text-gray-400 text-4xl mb-4">🏥</div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Patient Visits Yet</h3>
              <p className="text-gray-600 mb-4">Start recording nurse visits and patient interactions.</p>
              <button
                onClick={() => setShowAddForm(true)}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
              >
                Record First Visit
              </button>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm text-gray-600">
                  {visits.length} visit{visits.length !== 1 ? 's' : ''} found
                </span>
                <button
                  onClick={loadVisits}
                  className="text-sm text-blue-600 hover:text-blue-800 transition"
                >
                  🔄 Refresh
                </button>
              </div>

              {visits.map((visit) => (
                <div
                  key={visit.visit_id}
                  className="p-6 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition"
                >
                  {/* Visit Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <span className="font-semibold text-gray-900">
                          {visit.nurse_id}
                        </span>
                        <span className={`px-3 py-1 rounded-full border text-sm font-medium ${getVisitTypeColor(visit.visit_type)}`}>
                          {visit.visit_type}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-gray-600">
                        <span>📍 {visit.location}</span>
                        <span>⏱️ {visit.duration_minutes} min</span>
                        <span>🕐 {formatDate(visit.timestamp)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Visit Notes */}
                  {visit.notes && (
                    <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Visit Notes:</h4>
                      <p className="text-gray-800 whitespace-pre-wrap text-sm">
                        {visit.notes}
                      </p>
                    </div>
                  )}
                </div>
              ))}

              {/* Visit Summary */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-6">
                <h3 className="font-semibold text-blue-900 mb-2">Visit Summary</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-blue-600 font-medium">Total Visits:</span>
                    <span className="ml-2 text-blue-900">{visits.length}</span>
                  </div>
                  <div>
                    <span className="text-blue-600 font-medium">Total Time:</span>
                    <span className="ml-2 text-blue-900">
                      {visits.reduce((total, visit) => total + visit.duration_minutes, 0)} minutes
                    </span>
                  </div>
                  <div>
                    <span className="text-blue-600 font-medium">Latest Visit:</span>
                    <span className="ml-2 text-blue-900">
                      {visits.length > 0 ? formatDate(visits[0].timestamp) : 'None'}
                    </span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
