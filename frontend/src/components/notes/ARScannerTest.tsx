import React, { useState } from 'react';
import { DiagnosisAPI, API_BASE_URL } from '@/lib/api';

const ARScannerTest: React.FC = () => {
  const [testResult, setTestResult] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const testConnection = async () => {
    setIsLoading(true);
    setTestResult('Testing connection...');
    
    try {
      // Test basic API connection
      const response = await fetch(`${API_BASE_URL}/api/health`);
      const data = await response.json();
      setTestResult(`✅ Backend connection successful: ${JSON.stringify(data, null, 2)}`);
    } catch (error: any) {
      setTestResult(`❌ Backend connection failed: ${error.message}`);
    }
    
    setIsLoading(false);
  };

  const testScanEndpoint = async () => {
    setIsLoading(true);
    setTestResult('Testing scan endpoint...');
    
    try {
      // Create a test file
      const testFile = new File(['test content'], 'test.txt', { type: 'text/plain' });
      
      const result = await DiagnosisAPI.submitClinicalNoteScan('PATIENT_001', testFile, {
        nurseId: 'TEST_USER',
        location: 'Test Ward',
        shift: 'Test Shift'
      });
      
      setTestResult(`✅ Scan endpoint test successful: ${JSON.stringify(result, null, 2)}`);
    } catch (error: any) {
      setTestResult(`❌ Scan endpoint test failed: ${error.message}`);
    }
    
    setIsLoading(false);
  };

  return (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
      <h3 className="font-semibold text-yellow-800 mb-2">🧪 AR Scanner Debug Test</h3>
      <div className="space-x-2 mb-3">
        <button
          onClick={testConnection}
          disabled={isLoading}
          className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50"
        >
          Test Backend Connection
        </button>
        <button
          onClick={testScanEndpoint}
          disabled={isLoading}
          className="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600 disabled:opacity-50"
        >
          Test Scan Endpoint
        </button>
      </div>
      {testResult && (
        <div className="bg-white p-3 rounded border text-sm">
          <pre className="whitespace-pre-wrap">{testResult}</pre>
        </div>
      )}
    </div>
  );
};

export default ARScannerTest;
