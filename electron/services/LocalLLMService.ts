import { LlamaModel, LlamaContext, LlamaChatSession } from 'node-llama-cpp';
import path from 'path';
import fs from 'fs';
import { ChatMessage } from './LocalChatHistory';
import { SearchResult } from './LocalVectorStore';

export interface KnowledgeGraphData {
  nodes: Array<{ id: string; label: string }>;
  edges: Array<{ source: string; target: string }>;
}

export class LocalLLMService {
  private model: LlamaModel | null = null;
  private context: LlamaContext | null = null;
  private isInitializing: boolean = false;
  private modelsPath: string;
  private modelFile: string = '';
  
  constructor(dataPath: string) {
    this.modelsPath = path.join(dataPath, 'models');
    
    // Ensure models directory exists
    if (!fs.existsSync(this.modelsPath)) {
      fs.mkdirSync(this.modelsPath, { recursive: true });
    }
  }

  async initialize(): Promise<void> {
    if (this.model || this.isInitializing) {
      return;
    }

    this.isInitializing = true;
    
    try {
      console.log('Initializing local LLM...');
      
      // Check for existing model files
      await this.findOrDownloadModel();
      
      // Load the model
      this.model = new LlamaModel({
        modelPath: this.modelFile,
        gpuLayers: 0, // Start with CPU-only, can be configured later
      });

      // Create context
      this.context = new LlamaContext({
        model: this.model,
        contextSize: 4096, // Adjust based on available memory
        batchSize: 128,
      });

      console.log('Local LLM initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize LLM:', error);
      throw new Error(`Failed to initialize LLM: ${error}`);
    } finally {
      this.isInitializing = false;
    }
  }

  private async findOrDownloadModel(): Promise<void> {
    // Check for common model file patterns
    const modelPatterns = [
      'phi-3-mini-4k-instruct-q4.gguf',
      'llama-3-8b-instruct-q4_k_m.gguf',
      'mistral-7b-instruct-v0.3.Q4_K_M.gguf',
      '*.gguf'
    ];

    for (const pattern of modelPatterns) {
      if (pattern === '*.gguf') {
        // Find any GGUF file
        const files = fs.readdirSync(this.modelsPath).filter(f => f.endsWith('.gguf'));
        if (files.length > 0) {
          this.modelFile = path.join(this.modelsPath, files[0]);
          console.log(`Found model file: ${files[0]}`);
          return;
        }
      } else {
        const modelPath = path.join(this.modelsPath, pattern);
        if (fs.existsSync(modelPath)) {
          this.modelFile = modelPath;
          console.log(`Found model file: ${pattern}`);
          return;
        }
      }
    }

    // If no model found, throw an error with instructions
    const message = `
No GGUF model file found in ${this.modelsPath}.

To use the local LLM feature, please download a compatible GGUF model file:

Recommended models:
1. Phi-3-mini (small, fast): https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
2. Llama 3 8B (better quality): https://huggingface.co/Meta-Llama/Meta-Llama-3-8B-Instruct-GGUF

Download a .gguf file and place it in: ${this.modelsPath}

Example files:
- phi-3-mini-4k-instruct-q4.gguf (recommended for most computers)
- Meta-Llama-3-8B-Instruct-Q4_K_M.gguf (for powerful computers)
`;

    throw new Error(message);
  }

  async ensureReady(): Promise<void> {
    if (!this.model && !this.isInitializing) {
      await this.initialize();
    }
    
    // Wait for initialization to complete
    while (this.isInitializing) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    if (!this.model || !this.context) {
      throw new Error('LLM failed to initialize');
    }
  }

  async generateKnowledgeGraph(documentText: string): Promise<KnowledgeGraphData> {
    await this.ensureReady();
    
    if (!this.context) {
      throw new Error('LLM context not initialized');
    }

    try {
      // Truncate text if too long to avoid context overflow
      const maxLength = 8000;
      const truncatedText = documentText.length > maxLength 
        ? documentText.slice(0, maxLength) + '...'
        : documentText;

      const prompt = `Based on the following text, identify the main topics and their relationships.
Generate a JSON object with two keys: "nodes" and "edges".
- "nodes" should be a list of objects, each with an "id" and "label".
- "edges" should be a list of objects, each with a "source" and "target" id.

IMPORTANT: Your response MUST be ONLY the JSON object. Do not include any extra text,
explanations, or markdown formatting. The response must start with a '{'
and end with a '}'.

Here is the text:
---
${truncatedText}
---

JSON response:`;

      const session = new LlamaChatSession({
        context: this.context,
      });

      const response = await session.prompt(prompt, {
        maxTokens: 1000,
        temperature: 0.1, // Low temperature for more consistent JSON output
        stopOnAbortSignal: true,
      });

      // Extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No valid JSON found in LLM response');
      }

      const graphData = JSON.parse(jsonMatch[0]);
      
      // Validate the structure
      if (!graphData.nodes || !graphData.edges || 
          !Array.isArray(graphData.nodes) || !Array.isArray(graphData.edges)) {
        throw new Error('Invalid graph data structure');
      }

      console.log(`Generated knowledge graph with ${graphData.nodes.length} nodes and ${graphData.edges.length} edges`);
      
      return graphData;
      
    } catch (error) {
      console.error('Error generating knowledge graph:', error);
      
      // Return a fallback graph
      const fallbackGraph: KnowledgeGraphData = {
        nodes: [
          { id: "1", label: "Main Topic" },
          { id: "2", label: "Subtopic A" },
          { id: "3", label: "Subtopic B" }
        ],
        edges: [
          { source: "1", target: "2" },
          { source: "1", target: "3" }
        ]
      };
      
      console.log('Using fallback knowledge graph');
      return fallbackGraph;
    }
  }

  async generateSummary(topic: string, relevantDocs: SearchResult[]): Promise<string> {
    await this.ensureReady();
    
    if (!this.context) {
      throw new Error('LLM context not initialized');
    }

    try {
      const contextText = relevantDocs
        .map(doc => `Source: ${doc.content}`)
        .join('\n\n');

      const prompt = `Based *only* on the following text, write a concise summary of the topic: '${topic}'.

Text:
---
${contextText}
---

Summary:`;

      const session = new LlamaChatSession({
        context: this.context,
      });

      const summary = await session.prompt(prompt, {
        maxTokens: 500,
        temperature: 0.3,
        stopOnAbortSignal: true,
      });

      console.log(`Generated summary for topic: ${topic}`);
      return summary.trim();
      
    } catch (error) {
      console.error('Error generating summary:', error);
      throw new Error(`Failed to generate summary: ${error}`);
    }
  }

  async generateChatResponse(
    userMessage: string,
    relevantDocs: SearchResult[],
    chatHistory: ChatMessage[],
    chatId: string
  ): Promise<string> {
    await this.ensureReady();
    
    if (!this.context) {
      throw new Error('LLM context not initialized');
    }

    try {
      const contextText = relevantDocs
        .map(doc => `Source: ${doc.content}`)
        .join('\n\n');

      const historyText = chatHistory
        .slice(-10) // Keep last 10 messages for context
        .map(msg => `${msg.role}: ${msg.content}`)
        .join('\n');

      const systemPrompt = `You are an expert AI tutor for the topic of "${chatId}".
Your goal is to provide the best possible answer. Base your answer on the user's conversation history and the relevant context from the document provided below. Prioritize the document's information.

CONTEXT FROM DOCUMENT:
---
${contextText}
---

CONVERSATION HISTORY:
${historyText}

Provide a helpful, accurate response based on the context and conversation history.`;

      const session = new LlamaChatSession({
        context: this.context,
      });

      const fullPrompt = `${systemPrompt}

User: ${userMessage}

Assistant:`;

      const response = await session.prompt(fullPrompt, {
        maxTokens: 1000,
        temperature: 0.7,
        stopOnAbortSignal: true,
      });

      console.log(`Generated chat response for: ${userMessage.slice(0, 50)}...`);
      return response.trim();
      
    } catch (error) {
      console.error('Error generating chat response:', error);
      throw new Error(`Failed to generate chat response: ${error}`);
    }
  }

  isReady(): boolean {
    return this.model !== null && this.context !== null;
  }

  getModelsPath(): string {
    return this.modelsPath;
  }

  async cleanup(): Promise<void> {
    // Clean up model resources if needed
    this.model = null;
    this.context = null;
  }
}
