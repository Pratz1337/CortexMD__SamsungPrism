import React, { useRef, useState } from 'react';
import { DiagnosisAPI } from '@/lib/api';
import EnhancedARScanner from './EnhancedARScanner';
import ARCameraTest from './ARCameraTest';
import CameraTest from './CameraTest';

interface Props {
  patientId: string;
  onAdded?: () => void;
}

interface ScanResult {
  success: boolean;
  patient_id: string;
  scanned_note_id: string;
  clinical_note_id: string;
  parsed_data: any;
  ai_summary: string;
  extracted_entities: any;
  ocr_confidence: number;
  ai_confidence: number;
  text_length: number;
  word_count: number;
  preview_image: string;
  patient_dashboard: any;
  processing_timestamp: string;
}

const ScanClinicalNote: React.FC<Props> = ({ patientId, onAdded }) => {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [activeTab, setActiveTab] = useState<'live' | 'upload' | 'test'>('live');

  const handlePick = () => fileRef.current?.click();

  const handleChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0] || null;
    setSelectedFile(f);
    setResult(null);
    setError(null);
    setShowDetails(false);
  };

  const submit = async () => {
    if (!selectedFile) {
      setError('Please select or capture a note image first.');
      return;
    }
    try {
      setIsSubmitting(true);
      setError(null);
      
      console.log('Starting AR scan for patient:', patientId);
      console.log('Selected file:', selectedFile.name, 'Size:', selectedFile.size);
      
      const data = await DiagnosisAPI.submitClinicalNoteScan(patientId, selectedFile, {
        nurseId: 'AR_SCANNER',
        location: 'Ward',
        shift: 'Day'
      });
      
      console.log('AR scan response:', data);
      setResult(data);
      if (onAdded) onAdded();
    } catch (e: any) {
      console.error('AR scan error:', e);
      console.error('Error response:', e?.response?.data);
      console.error('Error message:', e?.message);
      setError(e?.response?.data?.error || e?.message || 'Failed to scan note');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
      {/* Enhanced Tab Navigation */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200">
        <nav className="flex px-6">
          <button
            onClick={() => setActiveTab('live')}
            className={`flex-1 py-4 px-4 font-semibold text-sm transition-all duration-300 rounded-tl-xl relative ${
              activeTab === 'live'
                ? 'text-indigo-600 bg-white shadow-sm'
                : 'text-gray-600 hover:text-indigo-500 hover:bg-white/50'
            }`}
          >
            <div className="flex items-center justify-center space-x-2">
              <span className="text-lg">🎯</span>
              <span>Enhanced AR</span>
            </div>
            {activeTab === 'live' && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-t"></div>
            )}
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex-1 py-4 px-4 font-semibold text-sm transition-all duration-300 relative ${
              activeTab === 'upload'
                ? 'text-blue-600 bg-white shadow-sm'
                : 'text-gray-600 hover:text-blue-500 hover:bg-white/50'
            }`}
          >
            <div className="flex items-center justify-center space-x-2">
              <span className="text-lg">📁</span>
              <span>File Upload</span>
            </div>
            {activeTab === 'upload' && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-t"></div>
            )}
          </button>
          <button
            onClick={() => setActiveTab('test')}
            className={`flex-1 py-4 px-4 font-semibold text-sm transition-all duration-300 rounded-tr-xl relative ${
              activeTab === 'test'
                ? 'text-green-600 bg-white shadow-sm'
                : 'text-gray-600 hover:text-green-500 hover:bg-white/50'
            }`}
          >
            <div className="flex items-center justify-center space-x-2">
              <span className="text-lg">💬</span>
              <span>Camera Test</span>
            </div>
            {activeTab === 'test' && (
              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-green-500 to-teal-600 rounded-t"></div>
            )}
          </button>
        </nav>
      </div>

      {/* Enhanced AR Scanner Tab */}
      {activeTab === 'live' && (
        <div className="p-0">
          <EnhancedARScanner patientId={patientId} onAdded={onAdded} />
        </div>
      )}

      {/* Camera Test Tab */}
      {activeTab === 'test' && (
        <div className="p-0">
          <CameraTest />
        </div>
      )}

      {/* File Upload Tab */}
      {activeTab === 'upload' && (
        <div className="p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl text-white">📁</span>
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">File Upload Scanner</h3>
            <p className="text-gray-600">Upload medical note images for AI-powered analysis</p>
          </div>

          <div className="max-w-md mx-auto space-y-6">
            {/* File Selection Area */}
            <div
              onClick={handlePick}
              className="border-2 border-dashed border-gray-300 rounded-2xl p-8 text-center hover:border-indigo-400 hover:bg-indigo-50/50 transition-all duration-300 cursor-pointer group"
            >
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={handleChange}
              />
              <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-indigo-100 transition-colors">
                <span className="text-2xl text-gray-400 group-hover:text-indigo-500">📎</span>
              </div>
              <p className="text-lg font-medium text-gray-700 mb-1">Choose or Capture Image</p>
              <p className="text-sm text-gray-500">Click to select a file or use camera</p>
              <p className="text-xs text-gray-400 mt-2">Supported formats: PNG, JPG, JPEG</p>
            </div>

            {/* Selected File Display */}
            {selectedFile && (
              <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-xl p-4">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold">📄</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{selectedFile.name}</p>
                    <p className="text-xs text-gray-600">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                  </div>
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="w-8 h-8 bg-red-100 hover:bg-red-200 rounded-full flex items-center justify-center transition-colors"
                  >
                    <span className="text-red-600 font-bold text-sm">×</span>
                  </button>
                </div>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                    <span className="text-red-600">⚠️</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-red-800">Error</p>
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-3">
              <button
                onClick={handlePick}
                className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-3 px-6 rounded-xl transition-all duration-200 hover:shadow-md"
              >
                Choose File
              </button>
              <button
                onClick={submit}
                disabled={!selectedFile || isSubmitting}
                className={`flex-1 font-semibold py-3 px-6 rounded-xl transition-all duration-200 ${
                  !selectedFile || isSubmitting
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                }`}
              >
                {isSubmitting ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Processing…</span>
                  </div>
                ) : (
                  'Scan & Analyze'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results Section - Only show when there's a result */}
      {result && activeTab === 'upload' && (
        <div className="p-8 border-t border-gray-100">
          <div className="space-y-6">
            {/* Success Message */}
            <div className="bg-green-50 border border-green-200 rounded-xl p-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                  <span className="text-green-600">✅</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-green-800">Success! Note scanned and analyzed with AI</p>
                  <p className="text-xs text-green-700 mt-1">
                    Scanned Note ID: {result.scanned_note_id} | Clinical Note ID: {result.clinical_note_id}
                  </p>
                </div>
              </div>
            </div>

            {/* AI Summary */}
            {result.ai_summary && (
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center mr-3">
                    <span className="text-white text-lg">🤖</span>
                  </div>
                  <h4 className="text-lg font-bold text-blue-800">AI Summary</h4>
                </div>
                <p className="text-blue-700 leading-relaxed">{result.ai_summary}</p>
              </div>
            )}

            {/* Preview and Details Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Image Preview */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <span className="text-xl mr-2">📷</span>
                  Scanned Image Preview
                </h4>
                {result.preview_image && (
                  <img
                    src={result.preview_image}
                    alt="Scanned Note Preview"
                    className="rounded-lg border border-gray-200 max-w-full shadow-md"
                  />
                )}
              </div>

              {/* Processing Statistics */}
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <span className="text-xl mr-2">📊</span>
                  Processing Statistics
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600">OCR Confidence:</span>
                    <span className={`font-bold text-lg ${
                      result.ocr_confidence > 80 ? 'text-green-600' :
                      result.ocr_confidence > 60 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {result.ocr_confidence?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600">AI Confidence:</span>
                    <span className={`font-bold text-lg ${
                      result.ai_confidence > 0.8 ? 'text-green-600' :
                      result.ai_confidence > 0.6 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {(result.ai_confidence * 100)?.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600">Text Length:</span>
                    <span className="font-medium text-gray-900">{result.text_length} characters</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600">Word Count:</span>
                    <span className="font-medium text-gray-900">{result.word_count} words</span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-gray-600">Processed:</span>
                    <span className="font-medium text-gray-900 text-sm">
                      {new Date(result.processing_timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Extracted Medical Entities */}
            {result.extracted_entities && Object.keys(result.extracted_entities).length > 0 && (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6">
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl flex items-center justify-center mr-3">
                    <span className="text-white text-lg">🏥</span>
                  </div>
                  <h4 className="text-lg font-bold text-green-800">Extracted Medical Entities</h4>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(result.extracted_entities).map(([key, value]: [string, any]) => (
                    <div key={key} className="bg-white rounded-lg p-3 border border-green-200">
                      <span className="font-semibold text-green-700 capitalize text-sm block mb-1">
                        {key.replace('_', ' ')}:
                      </span>
                      <div className="text-green-600 text-sm">
                        {Array.isArray(value) ? (
                          value.length > 0 ? value.join(', ') : 'None found'
                        ) : (
                          value || 'Not found'
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Toggle Details Button */}
            <div className="text-center">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold py-2 px-6 rounded-xl transition-all duration-200 hover:shadow-md"
              >
                {showDetails ? 'Hide' : 'Show'} Detailed Parsed Data
              </button>
            </div>

            {/* Detailed Parsed Data */}
            {showDetails && (
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-6">
                <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                  <span className="text-xl mr-2">🔍</span>
                  Detailed Parsed Data
                </h4>
                <pre className="bg-white p-4 rounded-lg border border-gray-200 text-xs overflow-auto max-h-96 text-gray-800">
                  {JSON.stringify(result.parsed_data || {}, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ScanClinicalNote;
