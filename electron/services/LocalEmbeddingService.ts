import { pipeline, Pipeline } from '@xenova/transformers';

export class LocalEmbeddingService {
  private model: Pipeline | null = null;
  private isInitializing: boolean = false;
  private modelName: string = 'Xenova/all-MiniLM-L6-v2';
  
  constructor() {
    // Configure environment for Electron
    if (typeof window === 'undefined') {
      // Set environment variables for Node.js/Electron
      process.env.TRANSFORMERS_CACHE = require('path').join(require('electron').app.getPath('userData'), 'transformers-cache');
    }
  }

  async initialize(): Promise<void> {
    if (this.model || this.isInitializing) {
      return;
    }

    this.isInitializing = true;
    
    try {
      console.log('Loading embedding model...');
      
      // Create feature extraction pipeline
      this.model = await pipeline('feature-extraction', this.modelName, {
        quantized: true, // Use quantized model for better performance
        cache_dir: process.env.TRANSFORMERS_CACHE
      });
      
      console.log('Embedding model loaded successfully');
      
    } catch (error) {
      console.error('Failed to load embedding model:', error);
      throw new Error(`Failed to initialize embedding model: ${error}`);
    } finally {
      this.isInitializing = false;
    }
  }

  async ensureReady(): Promise<void> {
    if (!this.model && !this.isInitializing) {
      await this.initialize();
    }
    
    // Wait for initialization to complete
    while (this.isInitializing) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    if (!this.model) {
      throw new Error('Embedding model failed to initialize');
    }
  }

  async embedTexts(texts: string[]): Promise<number[][]> {
    await this.ensureReady();
    
    if (!this.model) {
      throw new Error('Embedding model not initialized');
    }

    try {
      const embeddings: number[][] = [];
      
      // Process texts in batches to avoid memory issues
      const batchSize = 10;
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        
        for (const text of batch) {
          // Clean and prepare text
          const cleanText = this.preprocessText(text);
          
          // Generate embedding
          const result = await this.model(cleanText, {
            pooling: 'mean', // Use mean pooling
            normalize: true   // Normalize the embeddings
          });
          
          // Extract the tensor data and convert to array
          let embedding: number[];
          if (result.data) {
            embedding = Array.from(result.data);
          } else if (result.tolist) {
            embedding = result.tolist();
          } else if (Array.isArray(result)) {
            embedding = result;
          } else {
            // Fallback: try to convert tensor to array
            embedding = Array.from(result);
          }
          
          embeddings.push(embedding);
        }
        
        // Small delay between batches to prevent blocking
        if (i + batchSize < texts.length) {
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      }
      
      console.log(`Generated embeddings for ${texts.length} texts, dimension: ${embeddings[0]?.length || 0}`);
      return embeddings;
      
    } catch (error) {
      console.error('Error generating embeddings:', error);
      throw new Error(`Failed to generate embeddings: ${error}`);
    }
  }

  async embedSingleText(text: string): Promise<number[]> {
    const embeddings = await this.embedTexts([text]);
    return embeddings[0];
  }

  private preprocessText(text: string): string {
    // Clean and normalize text for better embedding quality
    return text
      .trim()
      .replace(/\s+/g, ' ') // Normalize whitespace
      .replace(/[^\w\s.,!?;:()-]/g, '') // Remove special characters except punctuation
      .slice(0, 512); // Truncate to max model length (adjust as needed)
  }

  isReady(): boolean {
    return this.model !== null;
  }

  getEmbeddingDimension(): number {
    // all-MiniLM-L6-v2 produces 384-dimensional embeddings
    return 384;
  }

  getModelName(): string {
    return this.modelName;
  }

  async cleanup(): Promise<void> {
    // Clean up model resources if needed
    this.model = null;
  }
}
