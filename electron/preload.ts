import { contextBridge, ipcRenderer } from 'electron';

/**
 * Electron API exposed to the renderer process
 */
const electronAPI = {
  // Document processing
  processDocument: (filePath: string) => ipcRenderer.invoke('document:process', filePath),
  
  // Graph data
  fetchGraphData: (documentId: string) => ipcRenderer.invoke('graph:fetch', documentId),
  
  // LLM interactions
  askQuestion: (documentId: string, question: string) => 
    ipcRenderer.invoke('llm:ask', documentId, question),
  
  // File dialog
  selectFile: () => ipcRenderer.invoke('dialog:selectFile'),
  
  // Local backend
  getLocalBackendUrl: () => ipcRenderer.invoke('backend:getLocalUrl'),
  initializeLocalBackend: () => ipcRenderer.invoke('backend:initialize'),
  checkBackendHealth: () => ipcRenderer.invoke('backend:health'),
  
  // Platform info
  platform: process.platform,
};

contextBridge.exposeInMainWorld('electronAPI', electronAPI);

export type ElectronAPI = typeof electronAPI;