"use client"

import React, { useState, useEffect } from 'react';
import {
  CogIcon,
  CubeIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowUpTrayIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';

interface ModelStatus {
  has_custom_model: boolean;
  model_path: string | null;
  model_filename: string;
  framework: string;
  status: string;
  available_models: Record<string, {
    filename: string;
    framework: string;
    size_mb: number;
  }>;
  total_models_found: number;
}

interface ModelConfigurationProps {
  className?: string;
  onModelChange?: (modelInfo: ModelStatus) => void;
}

export function ModelConfiguration({ className, onModelChange }: ModelConfigurationProps) {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [selecting, setSelecting] = useState(false);

  // Fetch current model status
  const fetchModelStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/heatmap/model_status');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setModelStatus(data);
      onModelChange?.(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch model status');
    } finally {
      setLoading(false);
    }
  };

  const selectModel = async (modelName: string) => {
    setSelecting(true);
    setError(null);

    try {
      const response = await fetch('/api/heatmap/select_model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_name: modelName }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Refresh model status
      await fetchModelStatus();
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to select model');
    } finally {
      setSelecting(false);
    }
  };

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const handleModelUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const allowedExtensions = ['.h5', '.keras', '.pth', '.pt'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedExtensions.includes(fileExtension)) {
      setError(`Unsupported file format. Allowed: ${allowedExtensions.join(', ')}`);
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('model', file);

      const response = await fetch('/api/heatmap/set_model', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Refresh model status
      await fetchModelStatus();
      
      // Reset file input
      event.target.value = '';
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload model');
    } finally {
      setUploading(false);
    }
  };

  const getStatusIcon = () => {
    if (!modelStatus) return <CogIcon className="w-5 h-5 text-gray-400" />;
    
    if (modelStatus.has_custom_model) {
      return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
    } else {
      return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
    }
  };

  const getStatusColor = () => {
    if (!modelStatus) return 'text-gray-600';
    
    return modelStatus.has_custom_model ? 'text-green-600' : 'text-yellow-600';
  };

  return (
    <div className={`bg-white rounded-lg shadow-md border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <CubeIcon className="w-6 h-6 text-blue-600" />
          <div>
            <h3 className="text-lg font-semibold text-gray-900">GradCAM Model Configuration</h3>
            <p className="text-sm text-gray-600">Manage AI models for heatmap generation</p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Current Status */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium text-gray-900">Current Model Status</h4>
            <button
              onClick={fetchModelStatus}
              disabled={loading}
              className="text-sm text-blue-600 hover:text-blue-800 disabled:text-gray-400"
            >
              {loading ? 'Checking...' : 'Refresh'}
            </button>
          </div>

          {modelStatus && (
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                {getStatusIcon()}
                <span className={`font-medium ${getStatusColor()}`}>
                  {modelStatus.status}
                </span>
              </div>
              
              {modelStatus.has_custom_model ? (
                <div className="text-sm text-gray-600">
                  <p><strong>Model:</strong> {modelStatus.model_filename}</p>
                  <p><strong>Type:</strong> Custom uploaded model</p>
                </div>
              ) : (
                <div className="text-sm text-gray-600">
                  <p><strong>Model:</strong> 3d_image_classification.h5</p>
                  <p><strong>Type:</strong> Default demo model</p>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
              <div className="flex items-center space-x-2">
                <ExclamationTriangleIcon className="w-4 h-4 text-red-500" />
                <span className="text-sm text-red-700">{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Model Upload */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
          <div className="text-center">
            <ArrowUpTrayIcon className="mx-auto h-12 w-12 text-gray-400" />
            <div className="mt-4">
              <label htmlFor="model-upload" className="cursor-pointer">
                <span className="mt-2 block text-sm font-medium text-gray-900">
                  Upload Custom Model
                </span>
                <span className="mt-1 block text-sm text-gray-600">
                  {uploading ? 'Uploading...' : 'Select .h5, .keras, .pth, or .pt file'}
                </span>
              </label>
              <input
                id="model-upload"
                name="model-upload"
                type="file"
                className="sr-only"
                accept=".h5,.keras,.pth,.pt"
                onChange={handleModelUpload}
                disabled={uploading}
              />
            </div>
            
            {!uploading && (
              <div className="mt-4">
                <button
                  onClick={() => document.getElementById('model-upload')?.click()}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  <ArrowUpTrayIcon className="w-4 h-4 mr-2" />
                  Choose File
                </button>
              </div>
            )}

            {uploading && (
              <div className="mt-4">
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span className="text-sm text-gray-600">Uploading model...</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Available Models Info */}
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-start space-x-2">
            <InformationCircleIcon className="w-5 h-5 text-blue-600 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-medium mb-1">Available Models:</p>
              <ul className="space-y-1 text-xs">
                <li>• <strong>3d_image_classification.h5</strong> - Default 3D medical imaging model</li>
                <li>• <strong>GlobalNet_pretrain20_T_0.8497.pth</strong> - PyTorch pretrained model</li>
                <li>• Upload your own custom model for specialized analysis</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
