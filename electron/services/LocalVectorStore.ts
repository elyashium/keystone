import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

export interface VectorDocument {
  id: string;
  content: string;
  embedding: number[];
  metadata: Record<string, any>;
}

export interface SearchResult {
  content: string;
  metadata: Record<string, any>;
  similarity: number;
}

export class LocalVectorStore {
  private db: Database.Database | null = null;
  private dbPath: string;
  private isInitialized: boolean = false;

  constructor(dataPath: string) {
    // Ensure data directory exists
    if (!fs.existsSync(dataPath)) {
      fs.mkdirSync(dataPath, { recursive: true });
    }
    
    this.dbPath = path.join(dataPath, 'vector_store.db');
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      console.log('Initializing local vector store...');
      
      this.db = new Database(this.dbPath);
      
      // Enable WAL mode for better performance
      this.db.pragma('journal_mode = WAL');
      
      // Create tables
      this.createTables();
      
      this.isInitialized = true;
      console.log('Local vector store initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize vector store:', error);
      throw new Error(`Failed to initialize vector store: ${error}`);
    }
  }

  private createTables(): void {
    if (!this.db) throw new Error('Database not initialized');

    // Create documents table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        index_name TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        metadata TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create index for faster searching
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_documents_index_name 
      ON documents(index_name)
    `);
  }

  async storeDocuments(indexName: string, texts: string[], embeddings: number[][]): Promise<void> {
    if (!this.db) throw new Error('Vector store not initialized');
    
    if (texts.length !== embeddings.length) {
      throw new Error('Number of texts and embeddings must match');
    }

    try {
      console.log(`Storing ${texts.length} documents in index: ${indexName}`);
      
      // Prepare insert statement
      const insertStmt = this.db.prepare(`
        INSERT OR REPLACE INTO documents (id, index_name, content, embedding, metadata)
        VALUES (?, ?, ?, ?, ?)
      `);

      // Use transaction for better performance
      const insertMany = this.db.transaction((documents: VectorDocument[]) => {
        for (const doc of documents) {
          insertStmt.run(
            doc.id,
            indexName,
            doc.content,
            this.serializeEmbedding(doc.embedding),
            JSON.stringify(doc.metadata)
          );
        }
      });

      // Prepare documents
      const documents: VectorDocument[] = texts.map((text, index) => ({
        id: `${indexName}_${Date.now()}_${index}`,
        content: text,
        embedding: embeddings[index],
        metadata: { chunkIndex: index }
      }));

      // Insert all documents
      insertMany(documents);
      
      console.log(`Successfully stored ${documents.length} documents`);
      
    } catch (error) {
      console.error('Error storing documents:', error);
      throw new Error(`Failed to store documents: ${error}`);
    }
  }

  async search(indexName: string, queryEmbedding: number[], limit: number = 4): Promise<SearchResult[]> {
    if (!this.db) throw new Error('Vector store not initialized');

    try {
      // Get all documents from the index
      const selectStmt = this.db.prepare(`
        SELECT content, embedding, metadata 
        FROM documents 
        WHERE index_name = ?
      `);
      
      const rows = selectStmt.all(indexName);
      
      if (rows.length === 0) {
        console.log(`No documents found in index: ${indexName}`);
        return [];
      }

      // Calculate similarities
      const results: SearchResult[] = [];
      
      for (const row of rows) {
        const docEmbedding = this.deserializeEmbedding(row.embedding as Buffer);
        const similarity = this.cosineSimilarity(queryEmbedding, docEmbedding);
        
        results.push({
          content: row.content as string,
          metadata: JSON.parse(row.metadata as string),
          similarity
        });
      }

      // Sort by similarity (highest first) and limit results
      const sortedResults = results
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit);

      console.log(`Found ${sortedResults.length} similar documents (top similarity: ${sortedResults[0]?.similarity.toFixed(3) || 'N/A'})`);
      
      return sortedResults;
      
    } catch (error) {
      console.error('Error searching documents:', error);
      throw new Error(`Failed to search documents: ${error}`);
    }
  }

  async getDocumentCount(indexName: string): Promise<number> {
    if (!this.db) throw new Error('Vector store not initialized');

    const stmt = this.db.prepare(`
      SELECT COUNT(*) as count 
      FROM documents 
      WHERE index_name = ?
    `);
    
    const result = stmt.get(indexName) as { count: number };
    return result.count;
  }

  async deleteIndex(indexName: string): Promise<void> {
    if (!this.db) throw new Error('Vector store not initialized');

    const stmt = this.db.prepare(`
      DELETE FROM documents 
      WHERE index_name = ?
    `);
    
    const result = stmt.run(indexName);
    console.log(`Deleted ${result.changes} documents from index: ${indexName}`);
  }

  async listIndexes(): Promise<string[]> {
    if (!this.db) throw new Error('Vector store not initialized');

    const stmt = this.db.prepare(`
      SELECT DISTINCT index_name 
      FROM documents 
      ORDER BY index_name
    `);
    
    const rows = stmt.all() as { index_name: string }[];
    return rows.map(row => row.index_name);
  }

  private serializeEmbedding(embedding: number[]): Buffer {
    // Convert float array to binary format for efficient storage
    const buffer = Buffer.allocUnsafe(embedding.length * 4); // 4 bytes per float
    
    for (let i = 0; i < embedding.length; i++) {
      buffer.writeFloatLE(embedding[i], i * 4);
    }
    
    return buffer;
  }

  private deserializeEmbedding(buffer: Buffer): number[] {
    const embedding: number[] = [];
    const length = buffer.length / 4; // 4 bytes per float
    
    for (let i = 0; i < length; i++) {
      embedding.push(buffer.readFloatLE(i * 4));
    }
    
    return embedding;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    
    if (magnitude === 0) {
      return 0;
    }

    return dotProduct / magnitude;
  }

  isReady(): boolean {
    return this.isInitialized && this.db !== null;
  }

  async cleanup(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
    this.isInitialized = false;
  }
}
