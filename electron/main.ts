import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync } from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { LocalBackend } from './services/LocalBackend';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Keep a global reference of the window object
let mainWindow: BrowserWindow;
let localBackend: LocalBackend;

const isDevelopment = process.env.NODE_ENV === 'development';

/**
 * Creates the main application window
 */
function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: join(__dirname, 'preload.js'),
    },
    titleBarStyle: 'hiddenInset',
    show: false,
  });

  const startUrl = isDevelopment 
    ? 'http://localhost:5500' 
    : `file://${join(__dirname, '../dist/index.html')}`;

  mainWindow.loadURL(startUrl);

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    if (isDevelopment) {
      mainWindow.webContents.openDevTools();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null as any;
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(async () => {
  // Initialize local backend
  localBackend = new LocalBackend();
  
  try {
    await localBackend.start();
    console.log(`Local backend started on port ${localBackend.getPort()}`);
  } catch (error) {
    console.error('Failed to start local backend:', error);
  }
  
  createWindow();
});

app.on('window-all-closed', async () => {
  // Clean up local backend
  if (localBackend) {
    await localBackend.cleanup();
  }
  
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC Handlers for offline mode functionality

/**
 * Mock document processing for offline mode
 * In a real implementation, this would process PDFs with a local LLM
 */
ipcMain.handle('document:process', async (event, filePath: string) => {
  try {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const documentId = uuidv4();
    
    // Mock response - in real implementation, would extract content and generate knowledge graph
    return {
      documentId,
      title: 'Sample Document',
      status: 'success'
    };
  } catch (error) {
    console.error('Document processing error:', error);
    throw new Error('Failed to process document');
  }
});

/**
 * Mock graph data generation for offline mode
 */
ipcMain.handle('graph:fetch', async (event, documentId: string) => {
  try {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Mock knowledge graph data
    const nodes = [
      { id: 1, label: 'Machine Learning', level: 0 },
      { id: 2, label: 'Neural Networks', level: 1 },
      { id: 3, label: 'Deep Learning', level: 1 },
      { id: 4, label: 'Backpropagation', level: 2 },
      { id: 5, label: 'Gradient Descent', level: 2 },
      { id: 6, label: 'Convolutional Networks', level: 2 },
    ];
    
    const edges = [
      { from: 1, to: 2 },
      { from: 1, to: 3 },
      { from: 2, to: 4 },
      { from: 2, to: 5 },
      { from: 3, to: 6 },
    ];
    
    return { nodes, edges };
  } catch (error) {
    console.error('Graph fetch error:', error);
    throw new Error('Failed to fetch graph data');
  }
});

/**
 * Mock LLM question answering for offline mode
 */
ipcMain.handle('llm:ask', async (event, documentId: string, question: string) => {
  try {
    // Simulate LLM processing time
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Mock response - in real implementation, would query local LLM
    const mockResponses = [
      "Based on the document content, this concept relates to the fundamental principles of machine learning.",
      "The document explains this in the context of neural network architectures and their applications.",
      "According to the material, this is a key component in understanding deep learning algorithms.",
    ];
    
    const answer = mockResponses[Math.floor(Math.random() * mockResponses.length)];
    
    return { answer };
  } catch (error) {
    console.error('LLM ask error:', error);
    throw new Error('Failed to get answer from LLM');
  }
});

/**
 * Handle file selection dialog
 */
ipcMain.handle('dialog:selectFile', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'PDF Files', extensions: ['pdf'] },
    ],
  });
  
  if (result.canceled) {
    return null;
  }
  
  return result.filePaths[0];
});

/**
 * Get local backend URL
 */
ipcMain.handle('backend:getLocalUrl', async () => {
  if (localBackend) {
    return `http://127.0.0.1:${localBackend.getPort()}`;
  }
  return null;
});

/**
 * Initialize local AI services
 */
ipcMain.handle('backend:initialize', async () => {
  if (localBackend) {
    try {
      await localBackend.initializeServices();
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : String(error) 
      };
    }
  }
  return { success: false, error: 'Local backend not available' };
});

/**
 * Check local backend health
 */
ipcMain.handle('backend:health', async () => {
  if (localBackend) {
    try {
      // Make a health check request to our own backend
      const response = await fetch(`http://127.0.0.1:${localBackend.getPort()}/health`);
      const health = await response.json();
      return health;
    } catch (error) {
      return { 
        status: 'error', 
        error: error instanceof Error ? error.message : String(error) 
      };
    }
  }
  return { status: 'error', error: 'Local backend not available' };
});