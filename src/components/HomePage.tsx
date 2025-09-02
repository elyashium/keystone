import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileUpload } from './FileUpload';
import { PrivacyModeToggle } from './PrivacyModeToggle';
import { useApp } from '@/context/AppContext';
import { apiService } from '@/services/apiService';
import keystoneLogo from '@/media/keystone.png';

/**
 * Home page with central upload interface
 */
export function HomePage() {
  const navigate = useNavigate();
  const { setCurrentDocumentId, setCurrentGraphData, isOnline, isPrivacyMode } = useApp();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Update API service mode
  React.useEffect(() => {
    apiService.setOnlineMode(isOnline);
  }, [isOnline]);

  // Update API service privacy mode
  React.useEffect(() => {
    apiService.setPrivacyMode(isPrivacyMode);
  }, [isPrivacyMode]);

  const handleFileSelect = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await apiService.processDocument(file);
      setCurrentDocumentId(result.documentId);
      
      // Store graph data if available
      if (result.graphData) {
        setCurrentGraphData(result.graphData);
      }
      
      navigate(`/graph/${result.documentId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process file');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUrlSubmit = async (url: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await apiService.processDocument(url);
      setCurrentDocumentId(result.documentId);
      
      // Store graph data if available
      if (result.graphData) {
        setCurrentGraphData(result.graphData);
      }
      
      navigate(`/graph/${result.documentId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process URL');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-keystone-primary flex flex-col">
      {/* Privacy Mode Toggle at top */}
      <div className="p-4 flex justify-end">
        <PrivacyModeToggle />
      </div>
      
      {/* Main content */}
      <div className="flex-1 flex items-center px-8 py-12">
        <div className="max-w-6xl mx-auto w-full">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            {/* Left Side - Logo/Title */}
            <div className="text-center lg:text-left animate-fade-in">
              <img src={keystoneLogo} alt="Keystone" className="max-w-full h-auto" />
            </div>

            {/* Right Side - Upload Options */}
            <div className="animate-slide-up">
              <FileUpload
                onFileSelect={handleFileSelect}
                onUrlSubmit={handleUrlSubmit}
                isLoading={isLoading}
                error={error}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}