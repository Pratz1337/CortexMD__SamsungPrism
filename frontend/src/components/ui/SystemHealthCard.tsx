'use client';

import { useEffect, useState } from 'react';
import { DiagnosisAPI } from '@/lib/api';
import { SystemHealth } from '@/types';

export function SystemHealthCard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const healthData = await DiagnosisAPI.getSystemHealth();
        setHealth(healthData);
      } catch (error) {
        console.error('Failed to fetch system health:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchHealth();
    // Refresh every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="card p-4">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-2"></div>
          <div className="h-3 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (!health) {
    return (
      <div className="card p-4 border-l-4 border-yellow-500">
        <div className="flex items-center">
          <span className="text-yellow-600">⚠️</span>
          <span className="ml-2 text-sm text-gray-600">
            System health unavailable
          </span>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational':
        return 'text-green-600 border-green-500';
      case 'partial':
        return 'text-yellow-600 border-yellow-500';
      case 'down':
        return 'text-red-600 border-red-500';
      default:
        return 'text-gray-600 border-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'operational':
        return '✅';
      case 'partial':
        return '⚠️';
      case 'down':
        return '❌';
      default:
        return '⚪';
    }
  };

  const statusColor = health.status === 'healthy' ? 'text-green-600 border-green-500' : 'text-yellow-600 border-yellow-500';
  const statusIcon = health.status === 'healthy' ? '✅' : '⚠️';

  return (
    <div className={`card p-4 border-l-4 ${statusColor}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <span className="text-xl">{statusIcon}</span>
          <div>
            <h3 className="font-semibold text-gray-800">
              System Status: {health.status.toUpperCase()}
            </h3>
            <p className="text-sm text-gray-600">
              Last checked: {new Date(health.timestamp).toLocaleTimeString()}
            </p>
          </div>
        </div>
        
        <div className="flex space-x-4">
          <div className="text-center">
            <div className={`text-sm ${health.api_configured ? 'text-green-600' : 'text-gray-400'}`}>
              {health.api_configured ? '✓' : '○'} AI API
            </div>
          </div>
          <div className="text-center">
            <div className={`text-sm ${health.features?.ontology_mapping ? 'text-green-600' : 'text-gray-400'}`}>
              {health.features?.ontology_mapping ? '✓' : '○'} Ontology
            </div>
          </div>
          <div className="text-center">
            <div className={`text-sm ${health.features?.fol_verification ? 'text-green-600' : 'text-gray-400'}`}>
              {health.features?.fol_verification ? '✓' : '○'} FOL
            </div>
          </div>
          <div className="text-center">
            <div className={`text-sm ${health.features?.chatbot_interface ? 'text-green-600' : 'text-gray-400'}`}>
              {health.features?.chatbot_interface ? '✓' : '○'} Chat
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
