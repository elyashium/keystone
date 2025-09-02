import React, { useState, useEffect } from 'react';
import { Shield, ShieldCheck, Loader2, AlertCircle } from 'lucide-react';
import { useApp } from '@/context/AppContext';
import { apiService } from '@/services/apiService';

interface PrivacyModeToggleProps {
  className?: string;
}

export function PrivacyModeToggle({ className = '' }: PrivacyModeToggleProps) {
  const { isPrivacyMode, setIsPrivacyMode } = useApp();
  const [isInitializing, setIsInitializing] = useState(false);
  const [backendHealth, setBackendHealth] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Check backend health on mount and when privacy mode changes
  useEffect(() => {
    if (isPrivacyMode) {
      checkBackendHealth();
    }
  }, [isPrivacyMode]);

  const checkBackendHealth = async () => {
    try {
      const health = await apiService.checkLocalBackendHealth();
      setBackendHealth(health);
      
      if (health.status === 'error') {
        setError(health.error || 'Backend health check failed');
      } else {
        setError(null);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to check backend health';
      setError(errorMsg);
      setBackendHealth({ status: 'error', error: errorMsg });
    }
  };

  const handleToggle = async () => {
    if (!isPrivacyMode) {
      // Turning ON privacy mode
      setIsInitializing(true);
      setError(null);
      
      try {
        // First set the API service to privacy mode
        await apiService.setPrivacyMode(true);
        
        // Initialize the local backend services
        const result = await apiService.initializeLocalBackend();
        
        if (result.success) {
          setIsPrivacyMode(true);
          await checkBackendHealth();
        } else {
          throw new Error(result.error || 'Failed to initialize local backend');
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Failed to enable privacy mode';
        setError(errorMsg);
        await apiService.setPrivacyMode(false);
      } finally {
        setIsInitializing(false);
      }
    } else {
      // Turning OFF privacy mode
      setIsPrivacyMode(false);
      await apiService.setPrivacyMode(false);
      setBackendHealth(null);
      setError(null);
    }
  };

  const getStatusColor = () => {
    if (error) return 'text-red-500';
    if (isPrivacyMode && backendHealth?.status === 'ok') return 'text-green-500';
    if (isPrivacyMode && !backendHealth?.initialized) return 'text-yellow-500';
    return 'text-gray-400';
  };

  const getStatusText = () => {
    if (error) return 'Error';
    if (isInitializing) return 'Initializing...';
    if (isPrivacyMode && backendHealth?.status === 'ok') {
      if (backendHealth.initialized) return 'Local AI Ready';
      return 'Backend Started (AI Loading...)';
    }
    if (isPrivacyMode) return 'Privacy Mode (Starting...)';
    return 'Cloud Mode';
  };

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      <button
        onClick={handleToggle}
        disabled={isInitializing}
        className={`
          flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all duration-200
          ${isPrivacyMode 
            ? 'bg-green-50 border-green-300 text-green-800 hover:bg-green-100' 
            : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
          }
          ${isInitializing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        {isInitializing ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : isPrivacyMode ? (
          <ShieldCheck className="w-4 h-4" />
        ) : (
          <Shield className="w-4 h-4" />
        )}
        <span className="font-medium">
          {isPrivacyMode ? 'Privacy Mode' : 'Cloud Mode'}
        </span>
      </button>

      <div className="flex items-center space-x-2">
        <div className={`w-2 h-2 rounded-full ${getStatusColor()} ${error ? 'bg-red-500' : isPrivacyMode && backendHealth?.status === 'ok' ? 'bg-green-500' : isPrivacyMode ? 'bg-yellow-500' : 'bg-gray-400'}`} />
        <span className={`text-sm ${getStatusColor()}`}>
          {getStatusText()}
        </span>
        
        {error && (
          <AlertCircle className="w-4 h-4 text-red-500" title={error} />
        )}
      </div>

      {isPrivacyMode && backendHealth && (
        <div className="text-xs text-gray-500">
          <details>
            <summary className="cursor-pointer hover:text-gray-700">Status</summary>
            <div className="mt-1 p-2 bg-gray-50 rounded border text-left">
              <div>Mode: {backendHealth.mode || 'local'}</div>
              <div>Initialized: {backendHealth.initialized ? '✓' : '✗'}</div>
              {backendHealth.services && (
                <div className="mt-1">
                  <div>LLM: {backendHealth.services.llm ? '✓' : '✗'}</div>
                  <div>Embeddings: {backendHealth.services.embeddings ? '✓' : '✗'}</div>
                  <div>Vector Store: {backendHealth.services.vectorStore ? '✓' : '✗'}</div>
                </div>
              )}
            </div>
          </details>
        </div>
      )}

      {error && isPrivacyMode && (
        <div className="text-xs text-red-600 max-w-md">
          <details>
            <summary className="cursor-pointer hover:text-red-800">Error Details</summary>
            <div className="mt-1 p-2 bg-red-50 rounded border text-left">
              {error}
              {error.includes('model file') && (
                <div className="mt-2 text-xs">
                  <p className="font-medium">To enable local AI:</p>
                  <ol className="list-decimal list-inside mt-1 space-y-1">
                    <li>Download a GGUF model file (e.g., phi-3-mini-4k-instruct-q4.gguf)</li>
                    <li>Place it in the app's models directory</li>
                    <li>Restart the app and try again</li>
                  </ol>
                </div>
              )}
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
