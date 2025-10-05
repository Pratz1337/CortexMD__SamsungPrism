import React, { useState, useEffect } from 'react';
import { DiagnosisAPI } from '@/lib/api';

interface Props {
  patientId: string;
}

interface ScannedNote {
  note_id: string;
  patient_id: string;
  nurse_id: string;
  image_mime_type: string;
  image_size: number;
  thumbnail_data: string;
  ocr_text: string;
  ocr_confidence: number;
  parsed_data: any;
  ai_summary: string;
  ai_extracted_entities: any;
  ai_confidence_score: number;
  scan_location: string;
  scan_shift: string;
  scan_timestamp: string;
  processing_status: string;
}

const ScannedNotesList: React.FC<Props> = ({ patientId }) => {
  const [scannedNotes, setScannedNotes] = useState<ScannedNote[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNote, setSelectedNote] = useState<ScannedNote | null>(null);
  const [showImageModal, setShowImageModal] = useState(false);

  useEffect(() => {
    fetchScannedNotes();
  }, [patientId]);

  const fetchScannedNotes = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await DiagnosisAPI.getScannedNotes(patientId, 20);
      
      if (data.success) {
        setScannedNotes(data.scanned_notes);
      } else {
        setError(data.error || 'Failed to fetch scanned notes');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch scanned notes');
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 80) return 'text-green-600';
    if (confidence > 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const openImageModal = (note: ScannedNote) => {
    setSelectedNote(note);
    setShowImageModal(true);
  };

  const closeImageModal = () => {
    setSelectedNote(null);
    setShowImageModal(false);
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading scanned notes...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="alert alert-warning">
            <strong>Error:</strong> {error}
            <button 
              className="btn btn-sm btn-outline ml-2"
              onClick={fetchScannedNotes}
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (scannedNotes.length === 0) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <div className="text-gray-400 text-4xl mb-4">📄</div>
          <h3 className="text-lg font-medium text-gray-600 mb-2">No Scanned Notes</h3>
          <p className="text-gray-500">No scanned clinical notes found for this patient.</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="card">
        <div className="card-header flex items-center justify-between">
          <h3 className="text-lg font-semibold">📄 Scanned Clinical Notes</h3>
          <div className="text-sm text-gray-500">
            {scannedNotes.length} note{scannedNotes.length !== 1 ? 's' : ''} found
          </div>
        </div>
        <div className="card-body">
          <div className="space-y-4">
            {scannedNotes.map((note) => (
              <div key={note.note_id} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-sm font-medium text-gray-900">
                        Note #{note.note_id.slice(-8)}
                      </span>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        note.processing_status === 'completed' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {note.processing_status}
                      </span>
                    </div>
                    
                    {note.ai_summary && (
                      <p className="text-sm text-gray-700 mb-2 line-clamp-2">
                        {note.ai_summary}
                      </p>
                    )}
                    
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>📅 {formatDate(note.scan_timestamp)}</span>
                      <span>👩‍⚕️ {note.nurse_id}</span>
                      <span>📍 {note.scan_location}</span>
                      <span>🕐 {note.scan_shift}</span>
                      <span>📏 {formatFileSize(note.image_size)}</span>
                    </div>
                    
                    <div className="flex items-center space-x-4 mt-2 text-xs">
                      <span className={`font-medium ${getConfidenceColor(note.ocr_confidence)}`}>
                        OCR: {note.ocr_confidence?.toFixed(1)}%
                      </span>
                      <span className={`font-medium ${getConfidenceColor(note.ai_confidence_score * 100)}`}>
                        AI: {(note.ai_confidence_score * 100)?.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 ml-4">
                    {note.thumbnail_data && (
                      <button
                        onClick={() => openImageModal(note)}
                        className="btn btn-sm btn-outline"
                      >
                        👁️ View
                      </button>
                    )}
                    <button
                      onClick={() => openImageModal(note)}
                      className="btn btn-sm btn-primary"
                    >
                      📷 Image
                    </button>
                  </div>
                </div>
                
                {/* Extracted Entities Preview */}
                {note.ai_extracted_entities && Object.keys(note.ai_extracted_entities).length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="text-xs text-gray-600 mb-1">🏥 Extracted Entities:</div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(note.ai_extracted_entities).map(([key, value]: [string, any]) => {
                        if (Array.isArray(value) && value.length > 0) {
                          return value.slice(0, 2).map((item, index) => (
                            <span key={`${key}-${index}`} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                              {item}
                            </span>
                          ));
                        }
                        return null;
                      })}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Image Modal */}
      {showImageModal && selectedNote && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl max-h-full overflow-auto">
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="text-lg font-semibold">Scanned Note Details</h3>
              <button
                onClick={closeImageModal}
                className="text-gray-400 hover:text-gray-600 text-2xl"
              >
                ×
              </button>
            </div>
            
            <div className="p-4 space-y-4">
              {/* Image */}
              <div className="text-center">
                <img
                  src={`/api/concern/scanned-note/${selectedNote.note_id}/image`}
                  alt="Scanned Clinical Note"
                  className="max-w-full max-h-96 mx-auto rounded border shadow-sm"
                />
              </div>
              
              {/* Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">📊 Processing Details</h4>
                  <div className="space-y-1 text-sm">
                    <div>Note ID: {selectedNote.note_id}</div>
                    <div>Scanned: {formatDate(selectedNote.scan_timestamp)}</div>
                    <div>Nurse: {selectedNote.nurse_id}</div>
                    <div>Location: {selectedNote.scan_location}</div>
                    <div>Shift: {selectedNote.scan_shift}</div>
                    <div>File Size: {formatFileSize(selectedNote.image_size)}</div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">🎯 Confidence Scores</h4>
                  <div className="space-y-1 text-sm">
                    <div className={`${getConfidenceColor(selectedNote.ocr_confidence)}`}>
                      OCR Confidence: {selectedNote.ocr_confidence?.toFixed(1)}%
                    </div>
                    <div className={`${getConfidenceColor(selectedNote.ai_confidence_score * 100)}`}>
                      AI Confidence: {(selectedNote.ai_confidence_score * 100)?.toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
              
              {/* AI Summary */}
              {selectedNote.ai_summary && (
                <div>
                  <h4 className="font-medium mb-2">🤖 AI Summary</h4>
                  <p className="text-sm text-gray-700 bg-blue-50 p-3 rounded">
                    {selectedNote.ai_summary}
                  </p>
                </div>
              )}
              
              {/* Extracted Entities */}
              {selectedNote.ai_extracted_entities && Object.keys(selectedNote.ai_extracted_entities).length > 0 && (
                <div>
                  <h4 className="font-medium mb-2">🏥 Extracted Medical Entities</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    {Object.entries(selectedNote.ai_extracted_entities).map(([key, value]: [string, any]) => (
                      <div key={key} className="bg-green-50 p-3 rounded">
                        <span className="font-medium text-green-800 capitalize">
                          {key.replace('_', ' ')}:
                        </span>
                        <div className="text-green-700 mt-1">
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
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ScannedNotesList;
