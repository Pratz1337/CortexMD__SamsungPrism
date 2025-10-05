"use client"

import { useState } from "react"
import { DiagnosisAPI } from "@/lib/api"
import { toast } from "react-hot-toast"

interface AddNoteFormProps {
  patientId?: string
  onSuccess?: () => void
}

export function AddNoteForm({ patientId, onSuccess }: AddNoteFormProps = {}) {
  const [formData, setFormData] = useState({
    patient_id: patientId || '',
    nurse_id: '',
    content: '',
    location: 'Ward A',
    shift: 'Day'
  })
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.patient_id || !formData.nurse_id || !formData.content) {
      toast.error('Please fill in all required fields')
      return
    }

    setSubmitting(true)
    try {
      const data = await DiagnosisAPI.addClinicalNote(formData)
      toast.success('Clinical note added successfully')
      
      // Reset form
      setFormData({
        patient_id: '',
        nurse_id: '',
        content: '',
        location: 'Ward A',
        shift: 'Day'
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
      console.error('Error adding note:', error)
      toast.error('Failed to add clinical note')
    } finally {
      setSubmitting(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Add Clinical Note</h2>
          {patientId && (
            <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800">
                📝 Adding note for: <span className="font-semibold">{patientId}</span>
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
              Location
            </label>
            <select
              id="location"
              name="location"
              value={formData.location}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
            >
              <option value="Ward A">Ward A</option>
              <option value="Ward B">Ward B</option>
              <option value="Ward C">Ward C</option>
              <option value="ICU">ICU</option>
              <option value="Emergency">Emergency</option>
              <option value="Recovery">Recovery</option>
            </select>
          </div>

          {/* Shift */}
          <div>
            <label htmlFor="shift" className="block text-sm font-medium text-gray-700 mb-2">
              Shift
            </label>
            <select
              id="shift"
              name="shift"
              value={formData.shift}
              onChange={handleInputChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
            >
              <option value="Day">Day</option>
              <option value="Night">Night</option>
              <option value="Evening">Evening</option>
            </select>
          </div>

          {/* Clinical Note Content */}
          <div>
            <label htmlFor="content" className="block text-sm font-medium text-gray-700 mb-2">
              Clinical Note *
            </label>
            <textarea
              id="content"
              name="content"
              value={formData.content}
              onChange={handleInputChange}
              rows={6}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
              placeholder="Enter clinical observations, patient status, medications, concerns, etc."
              required
            />
            <p className="mt-2 text-sm text-gray-500">
              Include any observations about patient condition, medication changes, behavioral changes, or concerns.
            </p>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={submitting}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition font-medium"
            >
              {submitting ? 'Adding Note...' : 'Add Clinical Note'}
            </button>
          </div>
        </form>

        {/* Sample Notes */}
        <div className="mt-8 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-900 mb-3">Sample Clinical Notes</h3>
          <div className="space-y-2 text-sm">
            <div className="p-3 bg-white rounded border-l-4 border-green-400">
              <strong>Normal:</strong> "Patient stable, vitals within normal range. Medications administered as scheduled. No concerns noted."
            </div>
            <div className="p-3 bg-white rounded border-l-4 border-yellow-400">
              <strong>Concerning:</strong> "Patient appears more lethargic today, color slightly pale. Increased monitoring frequency."
            </div>
            <div className="p-3 bg-white rounded border-l-4 border-red-400">
              <strong>Critical:</strong> "Hold blood pressure medication - BP reading 90/60. Patient refused morning medications, seems confused."
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
