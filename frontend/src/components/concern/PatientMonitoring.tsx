"use client"

import { useState, useEffect } from "react"
import { api } from "@/lib/api"
import { toast } from "react-hot-toast"

interface Patient {
  patient_id: string
  concern_score: number
  risk_level: string
  last_updated: string
}

export function PatientMonitoring() {
  const [patients, setPatients] = useState<Patient[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedPatient, setSelectedPatient] = useState<string>("")
  const [analyzing, setAnalyzing] = useState(false)

  const fetchPatients = async () => {
    try {
      // Use fast mode to avoid hanging on CONCERN calculations  
      const { data } = await api.get('/api/concern/patients?fast=true')
      setPatients(data.patients || data?.patients || [])
    } catch (error) {
      console.error('Error fetching patients:', error)
      toast.error('Failed to load patients')
    } finally {
      setLoading(false)
    }
  }

  const analyzePatient = async () => {
    if (!selectedPatient) {
      toast.error('Please select a patient')
      return
    }

    setAnalyzing(true)
    try {
      const { data } = await api.post(`/api/concern/analyze/${selectedPatient}`)
      toast.success(`Analysis complete. Risk level: ${data.risk_level || data?.current_risk_level || 'updated'}`)
      fetchPatients() // Refresh the list
    } catch (error) {
      console.error('Error analyzing patient:', error)
      toast.error('Failed to analyze patient')
    } finally {
      setAnalyzing(false)
    }
  }

  useEffect(() => {
    fetchPatients()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchPatients, 30000)
    return () => clearInterval(interval)
  }, [])

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical': return 'bg-red-500'
      case 'high': return 'bg-orange-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical': return '🚨'
      case 'high': return '⚠️'
      case 'medium': return '⚡'
      case 'low': return '✅'
      default: return '❓'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-red-600"></div>
        <span className="ml-3">Loading patients...</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Real-time Monitoring Header */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-900">Real-time Patient Monitoring</h2>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Live Updates</span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={selectedPatient}
            onChange={(e) => setSelectedPatient(e.target.value)}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500"
          >
            <option value="">Select a patient to analyze</option>
            {patients.map((patient) => (
              <option key={patient.patient_id} value={patient.patient_id}>
                {patient.patient_id} - {patient.risk_level.toUpperCase()}
              </option>
            ))}
          </select>
          
          <button
            onClick={analyzePatient}
            disabled={!selectedPatient || analyzing}
            className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {analyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {/* Patient Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {patients.map((patient) => (
          <div key={patient.patient_id} className="bg-white rounded-lg shadow hover:shadow-lg transition">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">{patient.patient_id}</h3>
                <span className="text-2xl">{getRiskIcon(patient.risk_level)}</span>
              </div>
              
              {/* Risk Level Indicator */}
              <div className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-600">Risk Level</span>
                  <span className="text-sm font-medium">{patient.risk_level.toUpperCase()}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${getRiskColor(patient.risk_level)}`}
                    style={{
                      width: `${Math.max(patient.concern_score * 100, 5)}%`
                    }}
                  ></div>
                </div>
              </div>
              
              {/* Concern Score */}
              <div className="mb-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-gray-900">
                    {(patient.concern_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Concern Score</div>
                </div>
              </div>
              
              {/* Last Updated */}
              <div className="text-xs text-gray-500 text-center">
                Last updated: {patient.last_updated ? new Date(patient.last_updated).toLocaleString() : 'Never'}
              </div>
              
              {/* Quick Actions */}
              <div className="mt-4 pt-4 border-t flex space-x-2">
                <button
                  onClick={() => setSelectedPatient(patient.patient_id)}
                  className="flex-1 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition"
                >
                  Select
                </button>
                <button
                  onClick={() => {
                    setSelectedPatient(patient.patient_id)
                    setTimeout(analyzePatient, 100)
                  }}
                  className="flex-1 px-3 py-2 text-sm bg-red-600 text-white rounded hover:bg-red-700 transition"
                >
                  Analyze
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {patients.length === 0 && (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <div className="text-4xl mb-4">👥</div>
          <h3 className="text-lg font-semibold mb-2">No Patients Found</h3>
          <p className="text-gray-600 mb-4">No patients are currently being monitored by the CONCERN system.</p>
          <p className="text-sm text-gray-500">Add clinical notes or patient visits to start monitoring.</p>
        </div>
      )}
      
      {/* Alert Banner for High Risk Patients */}
      {patients.some(p => ['high', 'critical'].includes(p.risk_level)) && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <span className="text-red-500 text-xl mr-3">🚨</span>
            <div>
              <h4 className="font-semibold text-red-800">High Risk Patients Detected</h4>
              <p className="text-red-700 text-sm">
                {patients.filter(p => ['high', 'critical'].includes(p.risk_level)).length} patients 
                require immediate attention based on CONCERN analysis.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
