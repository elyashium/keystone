import type { GraphData, ProcessDocumentResponse, AskQuestionResponse } from '@/types';

/**
 * API service abstraction layer
 * Handles routing between offline (Electron IPC) and online (HTTP) modes
 */
class ApiService {
  private isOnline: boolean = false;
  private isPrivacyMode: boolean = false;
  private baseUrl: string = this.getBaseUrl();
  private localBackendUrl: string | null = null;

  private getBaseUrl(): string {
    // Check if running in Electron
    const isElectron = typeof window !== 'undefined' && window.electronAPI;
    
    // Check if running in development mode in browser
    const isDevelopmentBrowser = import.meta.env.DEV && !isElectron;
    
    if (isDevelopmentBrowser) {
      // Use proxy in browser development
      return '';
    } else {
      // Use full URL for Electron or production builds
      return 'https://hybrid-ai-tutor-1.onrender.com';
    }
  }

  /**
   * Set the online/offline mode
   */
  setOnlineMode(online: boolean): void {
    this.isOnline = online;
  }

  /**
   * Set privacy mode (local processing)
   */
  async setPrivacyMode(enabled: boolean): Promise<void> {
    this.isPrivacyMode = enabled;
    
    if (enabled && !this.localBackendUrl) {
      // Get local backend URL from Electron
      if (typeof window !== 'undefined' && window.electronAPI) {
        this.localBackendUrl = await window.electronAPI.getLocalBackendUrl();
        console.log('Local backend URL:', this.localBackendUrl);
      }
    }
  }

  /**
   * Get current privacy mode status
   */
  isPrivacyModeEnabled(): boolean {
    return this.isPrivacyMode;
  }

  /**
   * Initialize local backend services
   */
  async initializeLocalBackend(): Promise<{ success: boolean; error?: string }> {
    if (typeof window !== 'undefined' && window.electronAPI) {
      return await window.electronAPI.initializeLocalBackend();
    }
    return { success: false, error: 'Electron API not available' };
  }

  /**
   * Check local backend health
   */
  async checkLocalBackendHealth(): Promise<any> {
    if (typeof window !== 'undefined' && window.electronAPI) {
      return await window.electronAPI.checkBackendHealth();
    }
    return { status: 'error', error: 'Electron API not available' };
  }

  /**
   * Process a document (PDF file or YouTube URL)
   */
  async processDocument(source: File | string): Promise<ProcessDocumentResponse> {
    try {
      if (this.isPrivacyMode) {
        return this.processDocumentLocal(source);
      } else if (this.isOnline) {
        return this.processDocumentOnline(source);
      } else {
        return this.processDocumentOffline(source);
      }
    } catch (error) {
      console.error('Document processing error:', error);
      throw new Error('Failed to process document');
    }
  }

  /**
   * Fetch knowledge graph data for a document
   */
  async fetchGraphData(documentId: string): Promise<GraphData> {
    try {
      if (this.isPrivacyMode) {
        // In privacy mode, graph data is returned with document processing
        throw new Error('Graph data is included in the document processing response for privacy mode');
      } else if (this.isOnline) {
        return this.fetchGraphDataOnline(documentId);
      } else {
        return this.fetchGraphDataOffline(documentId);
      }
    } catch (error) {
      console.error('Graph fetch error:', error);
      throw new Error('Failed to fetch graph data');
    }
  }

  /**
   * Ask a question about the document content
   */
  async askQuestion(documentId: string, question: string): Promise<AskQuestionResponse> {
    try {
      if (this.isPrivacyMode) {
        return this.askQuestionLocal(documentId, question);
      } else if (this.isOnline) {
        return this.askQuestionOnline(documentId, question);
      } else {
        return this.askQuestionOffline(documentId, question);
      }
    } catch (error) {
      console.error('Question answering error:', error);
      throw new Error('Failed to get answer');
    }
  }

  /**
   * Get a summary for a specific topic from the document
   */
  async getSummary(documentId: string, topic: string): Promise<{ summary: string }> {
    try {
      if (this.isPrivacyMode) {
        return this.getSummaryLocal(documentId, topic);
      } else if (this.isOnline) {
        return this.getSummaryOnline(documentId, topic);
      } else {
        throw new Error('Summary functionality not available in offline mode');
      }
    } catch (error) {
      console.error('Summary generation error:', error);
      throw new Error('Failed to generate summary');
    }
  }

  // Private methods for local backend (Privacy Mode)

  private async processDocumentLocal(source: File | string): Promise<ProcessDocumentResponse> {
    if (!this.localBackendUrl) {
      throw new Error('Local backend not available');
    }

    if (typeof source === 'string') {
      throw new Error('YouTube processing not available in privacy mode');
    }
    
    // File upload to local backend
    const formData = new FormData();
    formData.append('file', source);

    const response = await fetch(`${this.localBackendUrl}/api/process-pdf`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Local backend error! status: ${response.status} - ${errorText}`);
    }

    const backendResponse = await response.json();
    
    // Transform backend response to match frontend expectations
    return {
      documentId: backendResponse.index_name || 'nerv',
      title: source.name || 'Processed Document',
      status: 'success' as const,
      graphData: backendResponse.graph_data
    };
  }

  private async askQuestionLocal(documentId: string, question: string): Promise<AskQuestionResponse> {
    if (!this.localBackendUrl) {
      throw new Error('Local backend not available');
    }

    const response = await fetch(`${this.localBackendUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        chat_id: documentId,
        index_name: 'nerv',
        user_message: question,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Local backend error! status: ${response.status} - ${errorText}`);
    }

    const backendResponse = await response.json();
    
    return {
      answer: backendResponse.ai_response,
      sources: []
    };
  }

  private async getSummaryLocal(documentId: string, topic: string): Promise<{ summary: string }> {
    if (!this.localBackendUrl) {
      throw new Error('Local backend not available');
    }

    const response = await fetch(`${this.localBackendUrl}/api/get-summary`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        topic: topic,
        index_name: 'nerv',
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Local backend error! status: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  // Private methods for offline mode (Electron IPC)

  private async processDocumentOffline(source: File | string): Promise<ProcessDocumentResponse> {
    let filePath: string;

    if (typeof source === 'string') {
      // Handle YouTube URL
      throw new Error('YouTube processing not implemented in offline mode');
    } else {
      // Handle file upload
      filePath = source.path || source.name;
    }

    const result = await window.electronAPI.processDocument(filePath);
    return result;
  }

  private async fetchGraphDataOffline(documentId: string): Promise<GraphData> {
    return window.electronAPI.fetchGraphData(documentId);
  }

  private async askQuestionOffline(documentId: string, question: string): Promise<AskQuestionResponse> {
    return window.electronAPI.askQuestion(documentId, question);
  }

  // Private methods for online mode (HTTP API)

  private async processDocumentOnline(source: File | string): Promise<ProcessDocumentResponse> {
    if (typeof source === 'string') {
      // YouTube URL processing not available in current backend
      throw new Error('YouTube processing is not available with the current backend');
    }
    
    // File upload to /api/process-pdf endpoint
    const formData = new FormData();
    formData.append('file', source);

    const response = await fetch(`${this.baseUrl}/api/process-pdf`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
    }

    const backendResponse = await response.json();
    
    // Transform backend response to match frontend expectations
    return {
      documentId: backendResponse.index_name || 'nerv',
      title: source.name || 'Processed Document',
      status: 'success' as const,
      graphData: backendResponse.graph_data
    };
  }

  private async fetchGraphDataOnline(documentId: string): Promise<GraphData> {
    // For now, graph data is returned with the process-pdf response
    // This method is kept for compatibility but should be enhanced 
    // if the backend provides a separate graph endpoint
    throw new Error('Graph data is included in the document processing response. Use processDocument instead.');
  }

  private async askQuestionOnline(documentId: string, question: string): Promise<AskQuestionResponse> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        chat_id: documentId,
        index_name: 'nerv', // Using the fixed index name from backend
        user_message: question,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
    }

    const backendResponse = await response.json();
    
    // Transform backend response to match frontend expectations
    return {
      answer: backendResponse.ai_response,
      sources: [] // Backend doesn't return sources in current implementation
    };
  }

  private async getSummaryOnline(documentId: string, topic: string): Promise<{ summary: string }> {
    const response = await fetch(`${this.baseUrl}/api/get-summary`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        topic: topic,
        index_name: 'nerv', // Using the fixed index name from backend
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
    }

    return response.json();
  }
}

// Export singleton instance
export const apiService = new ApiService();