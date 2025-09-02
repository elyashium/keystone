import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export class LocalChatHistory {
  private db: Database.Database | null = null;
  private dbPath: string;
  private isInitialized: boolean = false;

  constructor(dataPath: string) {
    // Ensure data directory exists
    if (!fs.existsSync(dataPath)) {
      fs.mkdirSync(dataPath, { recursive: true });
    }
    
    this.dbPath = path.join(dataPath, 'chat_history.db');
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      console.log('Initializing local chat history...');
      
      this.db = new Database(this.dbPath);
      
      // Enable WAL mode for better performance
      this.db.pragma('journal_mode = WAL');
      
      // Create tables
      this.createTables();
      
      this.isInitialized = true;
      console.log('Local chat history initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize chat history:', error);
      throw new Error(`Failed to initialize chat history: ${error}`);
    }
  }

  private createTables(): void {
    if (!this.db) throw new Error('Database not initialized');

    // Create chat_messages table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create index for faster searching by chat_id
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id 
      ON chat_messages(chat_id, created_at)
    `);
  }

  async addMessage(chatId: string, userMessage: string, aiResponse: string): Promise<void> {
    if (!this.db) throw new Error('Chat history not initialized');

    try {
      const insertStmt = this.db.prepare(`
        INSERT INTO chat_messages (chat_id, role, content)
        VALUES (?, ?, ?)
      `);

      // Use transaction to ensure both messages are added atomically
      const addMessages = this.db.transaction(() => {
        insertStmt.run(chatId, 'user', userMessage);
        insertStmt.run(chatId, 'assistant', aiResponse);
      });

      addMessages();
      
      console.log(`Added chat messages for chat_id: ${chatId}`);
      
    } catch (error) {
      console.error('Error adding chat messages:', error);
      throw new Error(`Failed to add chat messages: ${error}`);
    }
  }

  async getChatHistory(chatId: string, limit: number = 50): Promise<ChatMessage[]> {
    if (!this.db) throw new Error('Chat history not initialized');

    try {
      const selectStmt = this.db.prepare(`
        SELECT role, content, created_at
        FROM chat_messages 
        WHERE chat_id = ?
        ORDER BY created_at ASC
        LIMIT ?
      `);
      
      const rows = selectStmt.all(chatId, limit);
      
      const messages: ChatMessage[] = rows.map((row: any) => ({
        role: row.role as 'user' | 'assistant',
        content: row.content,
        timestamp: new Date(row.created_at)
      }));

      console.log(`Retrieved ${messages.length} messages for chat_id: ${chatId}`);
      
      return messages;
      
    } catch (error) {
      console.error('Error getting chat history:', error);
      throw new Error(`Failed to get chat history: ${error}`);
    }
  }

  async getChatSessions(): Promise<Array<{ chatId: string; lastMessage: Date; messageCount: number }>> {
    if (!this.db) throw new Error('Chat history not initialized');

    try {
      const selectStmt = this.db.prepare(`
        SELECT 
          chat_id,
          MAX(created_at) as last_message,
          COUNT(*) as message_count
        FROM chat_messages
        GROUP BY chat_id
        ORDER BY last_message DESC
      `);
      
      const rows = selectStmt.all();
      
      return rows.map((row: any) => ({
        chatId: row.chat_id,
        lastMessage: new Date(row.last_message),
        messageCount: row.message_count
      }));
      
    } catch (error) {
      console.error('Error getting chat sessions:', error);
      throw new Error(`Failed to get chat sessions: ${error}`);
    }
  }

  async deleteChatHistory(chatId: string): Promise<void> {
    if (!this.db) throw new Error('Chat history not initialized');

    try {
      const deleteStmt = this.db.prepare(`
        DELETE FROM chat_messages 
        WHERE chat_id = ?
      `);
      
      const result = deleteStmt.run(chatId);
      
      console.log(`Deleted ${result.changes} messages for chat_id: ${chatId}`);
      
    } catch (error) {
      console.error('Error deleting chat history:', error);
      throw new Error(`Failed to delete chat history: ${error}`);
    }
  }

  async clearAllHistory(): Promise<void> {
    if (!this.db) throw new Error('Chat history not initialized');

    try {
      const deleteStmt = this.db.prepare(`DELETE FROM chat_messages`);
      const result = deleteStmt.run();
      
      console.log(`Cleared all chat history (${result.changes} messages deleted)`);
      
    } catch (error) {
      console.error('Error clearing chat history:', error);
      throw new Error(`Failed to clear chat history: ${error}`);
    }
  }

  async getMessageCount(chatId?: string): Promise<number> {
    if (!this.db) throw new Error('Chat history not initialized');

    try {
      let stmt;
      let result;
      
      if (chatId) {
        stmt = this.db.prepare(`
          SELECT COUNT(*) as count 
          FROM chat_messages 
          WHERE chat_id = ?
        `);
        result = stmt.get(chatId) as { count: number };
      } else {
        stmt = this.db.prepare(`
          SELECT COUNT(*) as count 
          FROM chat_messages
        `);
        result = stmt.get() as { count: number };
      }
      
      return result.count;
      
    } catch (error) {
      console.error('Error getting message count:', error);
      throw new Error(`Failed to get message count: ${error}`);
    }
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
