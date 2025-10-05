"use client"

import { useState } from "react"
import { DiagnosisAPI } from "@/lib/api"
import { toast } from "react-hot-toast"

interface AddVisitFormProps {
  patientId?: string
  onSuccess?: () => void
}

export function AddVisitForm({ patientId, onSuccess }: AddVisitFormProps = {}) {
  const [formData, setFormData] = useState({
    patient_id: patientId || '',
    nurse_id: '',
    location: 'Ward A',
    visit_type: 'routine',
    duration_minutes: 5
  })
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.patient_id || !formData.nurse_id) {
      toast.error('Please fill in all required fields')
      return
    }

    setSubmitting(true)
    try {
      const data = await DiagnosisAPI.addPatientVisit(formData)
      toast.success('Patient visit recorded successfully')
      
      // Reset form
      setFormData({
        patient_id: '',
        nurse_id: '',
        location: 'Ward A',
        visit_type: 'routine',
        duration_minutes: 5
      })

      // Show analysis results if available
      if (data.patient_dashboard) {
        const score = (data.patient_dashboard.current_concern_score * 100).toFixed(1)
        const level = data.patient_dashboard.current_risk_level
        toast.success(`CONCERN Analysis: ${score}% risk (${level.toUpperCase()})`, {
          duration: 6000
        })
      }

      // Call onSuccess callback if provided
      if (onSuccess) {
        onSuccess()
      }

    } catch (error) {
      console.error('Error adding visit:', error)
      toast.error('Failed to record patient visit')
    } finally {
      setSubmitting(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name === 'duration_minutes' ? parseInt(value) || 5 : value
    }))
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Record Patient Visit</h2>
          {patientId && (
            <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                🏥 Recording visit for: <span className="font-semibold">{patientId}</span>
              </p>
            </div>
          )}
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Patient ID */}
          <div>
            <label htmlFor="patient_id" className="block text-sm font-medium text-gray-700 mb-2">
              Patient ID *
            </label>
            <input
              type="text"
              id="patient_id"
              name="patient_id"
              value={formData.patient_id}
              onChange={handleInputChange}
              className={`w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500 ${
                patientId ? 'bg-gray-100' : ''
              }`}
              placeholder="e.g., PATIENT_001"
              readOnly={!!patientId}
              required
            />
            {patientId && (
              <p className="mt-1 text-sm text-gray-500">
                Pre-filled for current patient
              </p>
            )}
          </div>

          {/* Nurse ID */}
          <div>
            <label htmlFor="nurse_id" className="block text-sm font-medium text-gray-700 mb-2">
              Nurse ID *
            </label>
            <input
              type="text"
              id="nurse_id"
              name="nurse_id"
              value={formData.nurse_id}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
              placeholder="e.g., NURSE_A"
              required
            />
          </div>

          {/* Location */}
          <div>
            <label htmlFor="location" className="block text-sm font-medium text-gray-700 mb-2">
              Location *
            </label>
            <select
              id="location"
              name="location"
              value={formData.location}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
              required
            >
              <option value="Ward A">Ward A</option>
              <option value="Ward B">Ward B</option>
              <option value="Ward C">Ward C</option>
              <option value="ICU">ICU</option>
              <option value="Emergency">Emergency</option>
              <option value="Recovery">Recovery</option>
              <option value="Operating Room">Operating Room</option>
              <option value="Radiology">Radiology</option>
            </select>
          </div>

          {/* Visit Type */}
          <div>
            <label htmlFor="visit_type" className="block text-sm font-medium text-gray-700 mb-2">
              Visit Type
            </label>
            <select
              id="visit_type"
              name="visit_type"
              value={formData.visit_type}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
            >
              <option value="routine">Routine Check</option>
              <option value="urgent">Urgent Check</option>
              <option value="emergency">Emergency Response</option>
              <option value="medication">Medication Administration</option>
              <option value="assessment">Assessment</option>
              <option value="comfort">Comfort Care</option>
            </select>
          </div>

          {/* Duration */}
          <div>
            <label htmlFor="duration_minutes" className="block text-sm font-medium text-gray-700 mb-2">
              Duration (minutes)
            </label>
            <input
              type="number"
              id="duration_minutes"
              name="duration_minutes"
              value={formData.duration_minutes}
              onChange={handleInputChange}
              min="1"
              max="120"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
            />
            <p className="mt-1 text-sm text-gray-500">
              Typical visit durations: Routine (5-10 min), Urgent (10-20 min), Emergency (20+ min)
            </p>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={submitting}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition font-medium"
            >
              {submitting ? 'Recording Visit...' : 'Record Patient Visit'}
            </button>
          </div>
        </form>

        {/* Visit Type Guide */}
        <div className="mt-8 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-900 mb-3">Visit Type Guidelines</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-green-400 rounded-full"></span>
                <span><strong>Routine:</strong> Regular scheduled checks, vital signs</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-blue-400 rounded-full"></span>
                <span><strong>Medication:</strong> Administering prescribed medications</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-purple-400 rounded-full"></span>
                <span><strong>Assessment:</strong> Detailed patient evaluation</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-yellow-400 rounded-full"></span>
                <span><strong>Urgent:</strong> Responding to patient concerns</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-red-400 rounded-full"></span>
                <span><strong>Emergency:</strong> Critical situation response</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="w-3 h-3 bg-indigo-400 rounded-full"></span>
                <span><strong>Comfort:</strong> Patient comfort and support</span>
              </div>
            </div>
          </div>
        </div>

        {/* CONCERN System Info */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <div className="flex items-start space-x-3">
            <span className="text-blue-500 text-xl">ℹ️</span>
            <div>
              <h4 className="font-semibold text-blue-900">CONCERN Analysis</h4>
              <p className="text-blue-800 text-sm">
                Each visit is automatically analyzed for patterns that may indicate patient deterioration. 
                Frequent visits, off-hours activity, and longer durations contribute to the concern score.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
