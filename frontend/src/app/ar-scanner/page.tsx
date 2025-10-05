"use client"

import { useState } from 'react'
import ScanClinicalNote from '@/components/notes/ScanClinicalNote'
import ScannedNotesList from '@/components/notes/ScannedNotesList'

export default function ARScannerPage() {
  const [patientId, setPatientId] = useState('PATIENT_001')
  const [refreshKey, setRefreshKey] = useState(0)

  const handleScanSuccess = () => {
    // Refresh the scanned notes list
    setRefreshKey(prev => prev + 1)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-50">
      <div className="container mx-auto px-4 py-8">
        {/* Enhanced Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full mb-6 shadow-lg">
            <span className="text-3xl">🏥</span>
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
            CortexMD AR Scanner
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Revolutionary AI-powered medical note scanning with advanced OCR and intelligent entity extraction
          </p>
          
          {/* Camera Test Link */}
          <div className="mt-6">
            <a 
              href="/test-camera"
              className="inline-flex items-center px-4 py-2 bg-yellow-100 text-yellow-800 rounded-lg hover:bg-yellow-200 transition-colors text-sm font-medium"
            >
              📷 Test Camera Setup
            </a>
          </div>
        </div>

        {/* Enhanced Patient ID Selector */}
        <div className="max-w-lg mx-auto mb-10">
          <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
            <div className="flex items-center mb-4">
              <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full flex items-center justify-center mr-3">
                <span className="text-white font-semibold">👤</span>
              </div>
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-1">
                  Patient Identification
                </label>
                <p className="text-xs text-gray-500">Enter the patient ID for AR scanning</p>
              </div>
            </div>
            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all duration-200 text-gray-700 font-medium"
              placeholder="e.g., PATIENT_001"
            />
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Left Column - Scanner */}
          <div>
            <ScanClinicalNote 
              patientId={patientId} 
              onAdded={handleScanSuccess}
            />
          </div>

          {/* Right Column - Scanned Notes List */}
          <div>
            <ScannedNotesList 
              key={refreshKey}
              patientId={patientId} 
            />
          </div>
        </div>

        {/* Enhanced Features Section */}
        <div className="mt-16 max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-4">
              ✨ Revolutionary AR Scanner Features
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Powered by cutting-edge AI technology and medical-grade processing
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 border border-gray-100 hover:border-indigo-200 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">📷</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Smart Capture</h3>
              <p className="text-gray-600 leading-relaxed">
                Advanced mobile camera integration with intelligent document detection, auto-focus, and real-time image enhancement
              </p>
            </div>
            <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 border border-gray-100 hover:border-purple-200 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">🤖</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">AI Processing</h3>
              <p className="text-gray-600 leading-relaxed">
                Powered by Gemini 2.5 Pro for intelligent medical text extraction, context-aware summarization, and clinical reasoning
              </p>
            </div>
            <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 border border-gray-100 hover:border-green-200 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-teal-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">🏥</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Medical Entities</h3>
              <p className="text-gray-600 leading-relaxed">
                Automatic extraction of medications, diagnoses, allergies, vital signs, and clinical findings with medical terminology validation
              </p>
            </div>
            <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 border border-gray-100 hover:border-red-200 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-r from-red-500 to-pink-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">💾</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Secure Storage</h3>
              <p className="text-gray-600 leading-relaxed">
                HIPAA-compliant storage with PostgreSQL database, encrypted data transmission, and comprehensive audit trails
              </p>
            </div>
            <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 border border-gray-100 hover:border-yellow-200 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-r from-yellow-500 to-orange-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Confidence Scoring</h3>
              <p className="text-gray-600 leading-relaxed">
                Advanced confidence metrics for OCR accuracy, AI processing reliability, and data quality assessment
              </p>
            </div>
            <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 border border-gray-100 hover:border-cyan-200 transform hover:-translate-y-1">
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <span className="text-2xl">🔍</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Full-Text Search</h3>
              <p className="text-gray-600 leading-relaxed">
                Powerful search capabilities across all scanned notes with advanced filtering, date ranges, and medical term matching
              </p>
            </div>
          </div>
        </div>

        {/* Enhanced Instructions */}
        <div className="mt-16 max-w-5xl mx-auto">
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-3xl p-8 border border-indigo-100 shadow-xl">
            <div className="text-center mb-8">
              <h3 className="text-3xl font-bold text-gray-900 mb-2">🚀 Quick Start Guide</h3>
              <p className="text-gray-600">Get started with AR scanning in just 5 simple steps</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full flex items-center justify-center mb-4 text-white font-bold text-lg">1</div>
                <h4 className="font-semibold text-gray-900 mb-2">Enter Patient ID</h4>
                <p className="text-sm text-gray-600">Input the patient identification number for accurate record association</p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full flex items-center justify-center mb-4 text-white font-bold text-lg">2</div>
                <h4 className="font-semibold text-gray-900 mb-2">Choose/Capture Image</h4>
                <p className="text-sm text-gray-600">Select a medical note image or use camera for direct capture</p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="w-12 h-12 bg-gradient-to-r from-pink-500 to-red-600 rounded-full flex items-center justify-center mb-4 text-white font-bold text-lg">3</div>
                <h4 className="font-semibold text-gray-900 mb-2">AI Processing</h4>
                <p className="text-sm text-gray-600">Click "Scan & Analyze" to process with advanced AI algorithms</p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100 md:col-span-2 lg:col-span-1">
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-orange-600 rounded-full flex items-center justify-center mb-4 text-white font-bold text-lg">4</div>
                <h4 className="font-semibold text-gray-900 mb-2">Review Results</h4>
                <p className="text-sm text-gray-600">Examine extracted data, AI summary, and medical entities</p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100 md:col-span-2 lg:col-span-2">
                <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-yellow-600 rounded-full flex items-center justify-center mb-4 text-white font-bold text-lg">5</div>
                <h4 className="font-semibold text-gray-900 mb-2">Secure Storage</h4>
                <p className="text-sm text-gray-600">All scanned notes are automatically saved and appear in the patient record list</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
