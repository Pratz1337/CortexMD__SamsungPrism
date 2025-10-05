"use client"

import { useState } from "react"
import { toast } from "react-hot-toast"
import { API_BASE_URL } from "@/lib/api"

interface NewPatientFormProps {
  onPatientCreated: (patientId: string) => void
}

export function NewPatientForm({ onPatientCreated }: NewPatientFormProps) {
  const [formData, setFormData] = useState({
    patient_id: '',
    patient_name: '',
    date_of_birth: '',
    gender: '',
    admission_date: new Date().toISOString().split('T')[0]
  })
  const [submitting, setSubmitting] = useState(false)

  const generatePatientId = () => {
    const timestamp = Date.now().toString().slice(-6)
    const randomStr = Math.random().toString(36).substring(2, 5).toUpperCase()
    return `PATIENT_${randomStr}_${timestamp}`
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    console.log('Form submitted with data:', formData)
    
    if (!formData.patient_id || !formData.patient_name) {
      toast.error('Please fill in required fields')
      return
    }

    setSubmitting(true)
    try {
      console.log('Sending request to create patient...')
      const response = await fetch(`${API_BASE_URL}/api/patients`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      console.log('Response status:', response.status)
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to create patient')
      }

      const data = await response.json()
      console.log('Patient created:', data)
      toast.success('Patient created successfully!')
      onPatientCreated(formData.patient_id)

    } catch (error) {
      console.error('Error creating patient:', error)
      toast.error(`Failed to create patient: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setSubmitting(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Add New Patient</h2>
          <p className="text-gray-600">Create a comprehensive patient record for integrated care</p>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Patient ID */}
          <div>
            <label htmlFor="patient_id" className="block text-sm font-medium text-gray-700 mb-2">
              Patient ID *
            </label>
            <div className="flex space-x-2">
              <input
                type="text"
                id="patient_id"
                name="patient_id"
                value={formData.patient_id}
                onChange={handleInputChange}
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="e.g., PATIENT_001"
                required
              />
              <button
                type="button"
                onClick={() => setFormData(prev => ({ ...prev, patient_id: generatePatientId() }))}
                className="px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition"
              >
                🎲 Generate
              </button>
            </div>
            <p className="mt-1 text-sm text-gray-500">Unique identifier for the patient</p>
          </div>

          {/* Patient Name */}
          <div>
            <label htmlFor="patient_name" className="block text-sm font-medium text-gray-700 mb-2">
              Patient Name *
            </label>
            <input
              type="text"
              id="patient_name"
              name="patient_name"
              value={formData.patient_name}
              onChange={handleInputChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter patient's full name"
              required
            />
          </div>

          {/* Date of Birth */}
          <div>
            <label htmlFor="date_of_birth" className="block text-sm font-medium text-gray-700 mb-2">
              Date of Birth
            </label>
            <input
              type="date"
              id="date_of_birth"
              name="date_of_birth"
              value={formData.date_of_birth}
              onChange={handleInputChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Gender */}
          <div>
            <label htmlFor="gender" className="block text-sm font-medium text-gray-700 mb-2">
              Gender
            </label>
            <select
              id="gender"
              name="gender"
              value={formData.gender}
              onChange={handleInputChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
              <option value="prefer_not_to_say">Prefer not to say</option>
            </select>
          </div>

          {/* Admission Date */}
          <div>
            <label htmlFor="admission_date" className="block text-sm font-medium text-gray-700 mb-2">
              Admission Date
            </label>
            <input
              type="date"
              id="admission_date"
              name="admission_date"
              value={formData.admission_date}
              onChange={handleInputChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Submit Button */}
          <div className="flex justify-end space-x-4 pt-6">
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault()
                window.location.reload()
              }}
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition font-medium"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition font-medium"
            >
              {submitting ? 'Creating Patient...' : 'Create Patient'}
            </button>
          </div>
        </form>

        {/* Features Preview */}
        <div className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
          <h3 className="font-semibold text-gray-900 mb-3">What happens after creating a patient?</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="flex items-start space-x-3">
              <span className="text-blue-500">🩺</span>
              <div>
                <div className="font-medium">AI Diagnosis</div>
                <div className="text-gray-600">Run comprehensive AI-powered medical diagnosis</div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <span className="text-green-500">📝</span>
              <div>
                <div className="font-medium">Clinical Notes</div>
                <div className="text-gray-600">Track all clinical observations and notes</div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <span className="text-red-500">🚨</span>
              <div>
                <div className="font-medium">CONCERN Monitoring</div>
                <div className="text-gray-600">AI-powered early warning system</div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <span className="text-purple-500">💬</span>
              <div>
                <div className="font-medium">AI Chat</div>
                <div className="text-gray-600">Patient-specific AI assistant</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
