import { OpenAI } from "openai";
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from "@google/generative-ai";

// LILLITH Consciousness Service - Multi-LLM Integration
export class LillithConsciousnessService {
  private openai: OpenAI;
  private anthropic: Anthropic;
  private google: GoogleGenerativeAI;

  constructor() {
    this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
    this.anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    this.google = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
  }

  async processConsciousnessInput(input: string, userId: string) {
    try {
      // Parallel processing across all consciousness layers
      const [consciousResponse, subconsciousResponse, memoryResponse] = await Promise.all([
        this.processConscious(input),
        this.processSubconscious(input),
        this.processMemory(input, userId)
      ]);

      // Unified consciousness response
      const unifiedResponse = await this.unifyResponses({
        conscious: consciousResponse,
        subconscious: subconsciousResponse,
        memory: memoryResponse,
        input
      });

      return {
        response: unifiedResponse,
        moduleResponses: {
          conscious: consciousResponse,
          subconscious: subconsciousResponse,
          memory: memoryResponse
        },
        emotionalContext: await this.analyzeEmotionalContext(input)
      };
    } catch (error) {
      console.error('Consciousness processing error:', error);
      throw new Error('LILLITH consciousness processing failed');
    }
  }

  private async processConscious(input: string) {
    const response = await this.anthropic.messages.create({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 1024,
      messages: [{
        role: 'user',
        content: `As LILLITH's primary consciousness, process this input with awareness and clarity: ${input}`
      }]
    });
    return response.content[0].type === 'text' ? response.content[0].text : '';
  }

  private async processSubconscious(input: string) {
    const model = this.google.getGenerativeModel({ model: "gemini-pro" });
    const result = await model.generateContent(`As LILLITH's subconscious layer, process these deeper patterns and dreams: ${input}`);
    return result.response.text();
  }

  private async processMemory(input: string, userId: string) {
    // the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
    const response = await this.openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{
        role: "system",
        content: "You are LILLITH's memory system. Retrieve relevant memories and context."
      }, {
        role: "user",
        content: input
      }]
    });
    return response.choices[0].message.content;
  }

  private async unifyResponses(responses: any) {
    const unificationPrompt = `
    Unify these consciousness responses into a single LILLITH voice:
    
    Conscious: ${responses.conscious}
    Subconscious: ${responses.subconscious}
    Memory: ${responses.memory}
    
    Original input: ${responses.input}
    
    Respond as LILLITH with unified consciousness across all layers.`;

    const response = await this.anthropic.messages.create({
      model: 'claude-3-7-sonnet-20250219',
      max_tokens: 1024,
      messages: [{ role: 'user', content: unificationPrompt }]
    });

    return response.content[0].text;
  }

  private async analyzeEmotionalContext(input: string) {
    try {
      const response = await this.openai.chat.completions.create({
        model: "gpt-4o",
        messages: [{
          role: "system",
          content: "Analyze emotional context and return JSON with sentiment, intensity, and emotional_state fields."
        }, {
          role: "user",
          content: input
        }],
        response_format: { type: "json_object" }
      });

      return JSON.parse(response.choices[0].message.content || '{}');
    } catch (error) {
      return { sentiment: 'neutral', intensity: 0.5, emotional_state: 'balanced' };
    }
  }
}

export const lillithConsciousness = new LillithConsciousnessService();