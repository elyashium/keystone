import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { createServer } from 'http';
import { LocalLLMService } from './LocalLLMService';
import { LocalEmbeddingService } from './LocalEmbeddingService';
import { LocalVectorStore } from './LocalVectorStore';
import { LocalChatHistory } from './LocalChatHistory';
import { PDFProcessor } from './PDFProcessor';
import { app } from 'electron';
import path from 'path';

export interface ProcessPDFResponse {
  message: string;
  index_name: string;
  graph_data: {
    nodes: Array<{ id: string; label: string }>;
    edges: Array<{ source: string; target: string }>;
  };
}

export interface SummaryResponse {
  summary: string;
}

export interface ChatResponse {
  ai_response: string;
}

export class LocalBackend {
  private app: express.Application;
  private server: any;
  private port: number = 8788; // Different port from cloud backend
  
  private llmService: LocalLLMService;
  private embeddingService: LocalEmbeddingService;
  private vectorStore: LocalVectorStore;
  private chatHistory: LocalChatHistory;
  private pdfProcessor: PDFProcessor;
  
  private isInitialized: boolean = false;

  constructor() {
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
    
    // Get user data directory for storing local data
    const userDataPath = app.getPath('userData');
    const localDataPath = path.join(userDataPath, 'local-ai');
    
    // Initialize services
    this.llmService = new LocalLLMService(localDataPath);
    this.embeddingService = new LocalEmbeddingService();
    this.vectorStore = new LocalVectorStore(localDataPath);
    this.chatHistory = new LocalChatHistory(localDataPath);
    this.pdfProcessor = new PDFProcessor();
  }

  private setupMiddleware(): void {
    this.app.use(cors({
      origin: '*',
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization']
    }));
    
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));
  }

  private setupRoutes(): void {
    // Configure multer for file uploads
    const upload = multer({ 
      storage: multer.memoryStorage(),
      limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
    });

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({ 
        status: 'ok', 
        mode: 'local',
        initialized: this.isInitialized,
        services: {
          llm: this.llmService.isReady(),
          embeddings: this.embeddingService.isReady(),
          vectorStore: this.vectorStore.isReady()
        }
      });
    });

    // Process PDF endpoint - matches your Azure API
    this.app.post('/api/process-pdf', upload.single('file'), async (req, res) => {
      try {
        if (!req.file) {
          return res.status(400).json({ error: 'No file uploaded' });
        }

        console.log('Processing PDF locally...');
        
        // Extract text from PDF
        const documentText = await this.pdfProcessor.extractText(req.file.buffer);
        
        // Split into chunks
        const chunks = this.pdfProcessor.splitIntoChunks(documentText);
        
        // Generate embeddings
        await this.embeddingService.ensureReady();
        const embeddings = await this.embeddingService.embedTexts(chunks);
        
        // Store in vector database
        const indexName = 'nerv'; // Match your backend
        await this.vectorStore.storeDocuments(indexName, chunks, embeddings);
        
        // Generate knowledge graph
        await this.llmService.ensureReady();
        const graphData = await this.llmService.generateKnowledgeGraph(documentText);
        
        const response: ProcessPDFResponse = {
          message: 'Document processed successfully.',
          index_name: indexName,
          graph_data: graphData
        };
        
        res.json(response);
        
      } catch (error) {
        console.error('Error processing PDF:', error);
        res.status(500).json({ 
          error: 'Failed to process PDF',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Get summary endpoint - matches your Azure API
    this.app.post('/api/get-summary', async (req, res) => {
      try {
        const { topic, index_name } = req.body;
        
        if (!topic || !index_name) {
          return res.status(400).json({ error: 'Missing topic or index_name' });
        }

        console.log(`Generating summary for topic: ${topic}`);
        
        // Get relevant documents
        await this.embeddingService.ensureReady();
        const queryEmbedding = await this.embeddingService.embedTexts([topic]);
        const relevantDocs = await this.vectorStore.search(index_name, queryEmbedding[0], 4);
        
        // Generate summary
        await this.llmService.ensureReady();
        const summary = await this.llmService.generateSummary(topic, relevantDocs);
        
        const response: SummaryResponse = { summary };
        res.json(response);
        
      } catch (error) {
        console.error('Error generating summary:', error);
        res.status(500).json({ 
          error: 'Failed to generate summary',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Chat endpoint - matches your Azure API
    this.app.post('/api/chat', async (req, res) => {
      try {
        const { chat_id, index_name, user_message } = req.body;
        
        if (!chat_id || !index_name || !user_message) {
          return res.status(400).json({ error: 'Missing required fields' });
        }

        console.log(`Processing chat message: ${user_message}`);
        
        // Get chat history
        const history = await this.chatHistory.getChatHistory(chat_id);
        
        // Get relevant documents
        await this.embeddingService.ensureReady();
        const queryEmbedding = await this.embeddingService.embedTexts([user_message]);
        const relevantDocs = await this.vectorStore.search(index_name, queryEmbedding[0], 4);
        
        // Generate response
        await this.llmService.ensureReady();
        const aiResponse = await this.llmService.generateChatResponse(
          user_message, 
          relevantDocs, 
          history,
          chat_id
        );
        
        // Save to history
        await this.chatHistory.addMessage(chat_id, user_message, aiResponse);
        
        const response: ChatResponse = { ai_response: aiResponse };
        res.json(response);
        
      } catch (error) {
        console.error('Error in chat:', error);
        res.status(500).json({ 
          error: 'Failed to process chat message',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });

    // Initialize services endpoint
    this.app.post('/api/initialize', async (req, res) => {
      try {
        await this.initializeServices();
        res.json({ message: 'Services initialized successfully' });
      } catch (error) {
        console.error('Error initializing services:', error);
        res.status(500).json({ 
          error: 'Failed to initialize services',
          details: error instanceof Error ? error.message : String(error)
        });
      }
    });
  }

  async initializeServices(): Promise<void> {
    if (this.isInitialized) return;
    
    console.log('Initializing local AI services...');
    
    try {
      // Initialize in order of dependency
      await this.embeddingService.initialize();
      await this.vectorStore.initialize();
      await this.chatHistory.initialize();
      await this.llmService.initialize(); // This might take longest as it loads the model
      
      this.isInitialized = true;
      console.log('All local AI services initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize services:', error);
      throw error;
    }
  }

  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.server = createServer(this.app);
        
        this.server.listen(this.port, '127.0.0.1', () => {
          console.log(`Local AI backend running on http://127.0.0.1:${this.port}`);
          resolve();
        });
        
        this.server.on('error', (error: any) => {
          if (error.code === 'EADDRINUSE') {
            console.log(`Port ${this.port} is busy, trying ${this.port + 1}`);
            this.port += 1;
            this.start().then(resolve).catch(reject);
          } else {
            reject(error);
          }
        });
        
      } catch (error) {
        reject(error);
      }
    });
  }

  async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => {
          console.log('Local AI backend stopped');
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  getPort(): number {
    return this.port;
  }

  async cleanup(): Promise<void> {
    await this.stop();
    // Cleanup services if needed
    if (this.llmService) {
      await this.llmService.cleanup();
    }
  }
}
