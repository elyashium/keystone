import pdf from 'pdf-parse';

export interface DocumentChunk {
  content: string;
  metadata: {
    page?: number;
    chunkIndex: number;
  };
}

export class PDFProcessor {
  
  async extractText(pdfBuffer: Buffer): Promise<string> {
    try {
      const data = await pdf(pdfBuffer);
      return data.text;
    } catch (error) {
      console.error('Error extracting text from PDF:', error);
      throw new Error('Failed to extract text from PDF');
    }
  }

  splitIntoChunks(text: string, chunkSize: number = 1000, overlap: number = 150): string[] {
    const chunks: string[] = [];
    
    if (!text || text.length === 0) {
      return chunks;
    }

    // Split by sentences first to maintain coherence
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    let currentChunk = '';
    
    for (const sentence of sentences) {
      const trimmedSentence = sentence.trim();
      
      // If adding this sentence would exceed chunk size, save current chunk
      if (currentChunk.length + trimmedSentence.length > chunkSize && currentChunk.length > 0) {
        chunks.push(currentChunk.trim());
        
        // Start new chunk with overlap from the end of previous chunk
        const words = currentChunk.split(/\s+/);
        const overlapWords = words.slice(-Math.floor(overlap / 10)); // Rough word count for overlap
        currentChunk = overlapWords.join(' ') + ' ' + trimmedSentence;
      } else {
        currentChunk += (currentChunk.length > 0 ? '. ' : '') + trimmedSentence;
      }
    }
    
    // Add the last chunk if it has content
    if (currentChunk.trim().length > 0) {
      chunks.push(currentChunk.trim());
    }
    
    // Fallback: if no sentences were found, split by character count
    if (chunks.length === 0 && text.length > 0) {
      for (let i = 0; i < text.length; i += chunkSize - overlap) {
        const chunk = text.slice(i, i + chunkSize);
        if (chunk.trim().length > 0) {
          chunks.push(chunk.trim());
        }
      }
    }
    
    return chunks;
  }

  createDocumentChunks(text: string, pageNumber?: number): DocumentChunk[] {
    const chunks = this.splitIntoChunks(text);
    
    return chunks.map((content, index) => ({
      content,
      metadata: {
        page: pageNumber,
        chunkIndex: index
      }
    }));
  }
}
