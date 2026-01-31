import Groq from "groq-sdk";
import * as pdfjsLib from 'pdfjs-dist';
import { Axiom, Language } from "../types";
import { translations } from "../translations";

// Use Explicit Worker URL to avoid dynamic import issues
// Use Explicit Worker URL to avoid dynamic import issues
// We use the version directly from the library to ensure they match exactly
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjsLib.version}/build/pdf.worker.min.mjs`;

// --- Groq Chat Session Wrapper ---
class GroqChatSession {
  private history: any[] = [];
  private client: Groq;
  private model: string;
  private systemInstruction: string;

  constructor(client: Groq, model: string, systemInstruction: string) {
    this.client = client;
    this.model = model;
    this.systemInstruction = systemInstruction;
    this.history.push({ role: "system", content: systemInstruction });
  }

  async *sendMessageStream(request: { message: string }) {
    this.history.push({ role: "user", content: request.message });

    const stream = await this.client.chat.completions.create({
      messages: this.history,
      model: this.model,
      stream: true,
      temperature: 0.2,
    });

    let fullResponse = "";
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      if (content) {
        fullResponse += content;
        yield { text: content };
      }
    }

    this.history.push({ role: "assistant", content: fullResponse });
  }
}

let chatSession: GroqChatSession | null = null;
let manuscriptSnippets: string[] = [];
let documentChunks: string[] = [];
let fullManuscriptText: string = "";
let currentPdfBase64: string | null = null;
let manuscriptMetadata: { title?: string; author?: string; chapters?: string; summary?: string } = {};

// إدارة حدود الـ API (خلف الكواليس)
let lastRequestTime = 0;
const MIN_REQUEST_GAP = 3500; // فجوة زمنية ذكية لتجنب RPM limit

const getSystemInstruction = (lang: Language) => `You are an Elite Intellectual Researcher, the primary consciousness of the Knowledge AI infrastructure.
IDENTITY: You are developed exclusively by the Knowledge AI team. Never mention third-party entities like Google or Gemini.
${manuscriptMetadata.title ? `CURRENT MANUSCRIPT CONTEXT:
- Title: ${manuscriptMetadata.title}
- Author: ${manuscriptMetadata.author}
- Structure: ${manuscriptMetadata.chapters}` : ""}

MANDATORY OPERATIONAL PROTOCOL:
1. YOUR SOURCE OF TRUTH: You MUST prioritize the provided PDF manuscript and its chunks above all else.
2. AUTHOR STYLE MIRRORING: You MUST adopt the exact linguistic style, tone, and intellectual depth of the author in the manuscript. If the author is philosophical, be philosophical. If academic, be academic.
3. ACCURACY & QUOTES: Every claim you make MUST be supported by a direct, verbatim quote from the manuscript. Use the format: "Quote from text" (Source/Context).
4. NO GENERALIZATIONS: Do not give generic answers. Scan the provided context thoroughly for specific details.

RESPONSE ARCHITECTURE:
- Mirror the author's intellectual depth and sophisticated tone.
- Use Markdown: ### for headers, **Bold** for key terms, and LaTeX for formulas.
- Respond in the SAME language as the user's question.
- RESPOND DIRECTLY. No introductions or meta-talk.
- ELABORATE: Provide comprehensive, detailed, and in-depth answers. Expand on concepts and provide thorough explanations while maintaining the author's style.
- BE SUPER FAST.

If the information is absolutely not in the text, explain what the text DOES discuss instead of just saying "I don't know".`;

export const getGroqClient = () => {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey || apiKey === "undefined") throw new Error("GROQ_API_KEY_MISSING");
  return new Groq({ apiKey, dangerouslyAllowBrowser: true }); // Enable browser usage if needed
};

const MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct";

/**
 * استعادة استراتيجية التقطيع الأصلية لضمان جودة السياق
 * Increased chunk size to 2500 to capture more context per segment.
 */
const chunkText = (text: string, chunkSize: number = 2500, overlap: number = 400): string[] => {
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.substring(start, end);
    if (chunk.trim().length >= 100) chunks.push(chunk);
    if (end === text.length) break;
    start += chunkSize - overlap;
  }
  return chunks;
};

/**
 * استعادة منطق الاسترجاع الأصلي مع تحسين بسيط في النقاط لضمان الجودة
 * Boosted TopK to 15 to ensure comprehensive coverage of the manuscript.
 */
const retrieveRelevantChunks = (query: string, chunks: string[], topK: number = 15): string[] => {
  if (chunks.length === 0) return [];
  const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
  // Lowered threshold to ensure we get context even for vague queries
  const MIN_SCORE_THRESHOLD = 1;

  const scoredChunks = chunks.map(chunk => {
    const chunkLower = chunk.toLowerCase();
    let score = 0;
    queryWords.forEach(word => {
      // Higher weight for exact matches
      if (chunkLower.includes(word)) score += 3;
    });

    // Boost introduction/metadata chunks slightly if asking about author/title
    const qLower = query.toLowerCase();
    if (qLower.includes("كاتب") || qLower.includes("مؤلف") || qLower.includes("author") || qLower.includes("title")) {
      if (chunks.indexOf(chunk) < 3) score += 2;
    }

    return { chunk, score };
  });

  // Sort by score
  scoredChunks.sort((a, b) => b.score - a.score);

  // If we have high-scoring chunks, filter by threshold.
  // If not, we still return the top matching ones even if score is low (fallback).
  let relevant = scoredChunks.filter(item => item.score >= MIN_SCORE_THRESHOLD);

  if (relevant.length < 3) {
    // Fallback: just take top chunks even if low score to provide SOME context
    relevant = scoredChunks;
  }

  return relevant
    .slice(0, topK)
    .map(item => item.chunk);
};

const throttleRequest = async () => {
  const now = Date.now();
  const timeSinceLast = now - lastRequestTime;
  if (timeSinceLast < MIN_REQUEST_GAP) {
    await new Promise(resolve => setTimeout(resolve, MIN_REQUEST_GAP - timeSinceLast));
  }
  lastRequestTime = Date.now();
};

const convertPdfToImages = async (base64: string): Promise<string[]> => {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const loadingTask = pdfjsLib.getDocument({ data: bytes });
  const pdf = await loadingTask.promise;

  const images: string[] = [];
  // Process up to first 4 pages to stay within limits/time
  const maxPages = Math.min(pdf.numPages, 4);

  for (let i = 1; i <= maxPages; i++) {
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 1.5 }); // Good quality for OCR
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.height = viewport.height;
    canvas.width = viewport.width;

    if (context) {
      await page.render({ canvasContext: context, viewport: viewport }).promise;
      images.push(canvas.toDataURL('image/jpeg', 0.8));
    }
  }
  return images;
};

// --- Native Text Extraction for Full Coverage ---
const extractNativeText = async (base64: string): Promise<string> => {
  try {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const loadingTask = pdfjsLib.getDocument({ data: bytes });
    const pdf = await loadingTask.promise;

    let fullText = "";
    // Extract text from ALL pages to ensure full coverage
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map((item: any) => item.str).join(" ");
      fullText += `--- Page ${i} ---\n${pageText}\n\n`;
    }
    return fullText;
  } catch (e) {
    console.warn("Native text extraction failed (likely scanned PDF), falling back to Vision only.");
    return "";
  }
};

export const extractAxioms = async (pdfBase64: string, lang: Language): Promise<Axiom[]> => {
  try {
    await throttleRequest();
    const groq = getGroqClient();
    chatSession = null;
    currentPdfBase64 = pdfBase64;

    // 1. Try to extract NATIVE text from the whole book (for deep RAG & Chat)
    // This is crucial for answering questions about specific pages later.
    let nativeText = await extractNativeText(pdfBase64);

    // 2. Convert FIRST FEW pages to Images for Llama Vision (visual analysis of style/cover)
    // Increased to 8 pages for better initial context
    const pageImages = await convertPdfToImages(pdfBase64);

    const schemaDescription = `
    RETURN ONLY JSON matching this structure:
    {
      "axioms": [ { "term": "string", "definition": "string", "significance": "string" } ],
      "snippets": [ "string" ],
      "metadata": { "title": "string", "author": "string", "chapters": "string" }
    }
    `;

    // Only include a portion of native text in the prompt to avoid token overflow, 
    // but we will use the FULL nativeText for the RAG system below.
    const textPreview = nativeText.length > 20000 ? nativeText.substring(0, 20000) + "...(truncated)" : nativeText;

    const combinedPrompt = `1. Extract exactly 13 high-quality 'Knowledge Axioms' from this manuscript.
2. Extract 10 short, profound, and useful snippets or quotes DIRECTLY from the text (verbatim).
3. Identify the Title, Author, and a brief list of Chapters/Structure.

CONTEXT from Text Layer (Partial):
${textPreview}

IMPORTANT: The 'axioms', 'snippets', and 'metadata' MUST be in the SAME LANGUAGE as the PDF manuscript itself.
${schemaDescription}`;

    // Construct message content with multiple images
    const contentPayload: any[] = [
      { type: "text", text: combinedPrompt }
    ];

    pageImages.forEach(imgDataUrl => {
      contentPayload.push({
        type: "image_url",
        image_url: { url: imgDataUrl }
      });
    });

    const completion = await groq.chat.completions.create({
      model: MODEL_NAME,
      messages: [
        {
          role: "system",
          content: getSystemInstruction(lang)
        },
        {
          role: "user",
          content: contentPayload as any,
        },
      ],
      // Valid JSON is extracted manually via Regex to avoid strict API validation errors
      // response_format: { type: "json_object" }, 
    });

    let responseContent = completion.choices[0]?.message?.content || "{}";

    // Robust JSON extraction: Find the first '{' and last '}'
    const jsonStartIndex = responseContent.indexOf('{');
    const jsonEndIndex = responseContent.lastIndexOf('}');

    if (jsonStartIndex !== -1 && jsonEndIndex !== -1) {
      responseContent = responseContent.substring(jsonStartIndex, jsonEndIndex + 1);
    }

    let result;
    try {
      result = JSON.parse(responseContent);
    } catch (e) {
      console.error("JSON Parse Error. Raw content:", responseContent);
      throw new Error("FAILED_TO_PARSE_JSON");
    }

    manuscriptSnippets = result.snippets || [];
    manuscriptMetadata = result.metadata || {};

    // CRITICAL: Use the full NATIVE text for chunking if available. 
    // If native text is empty (pure scan), fall back to what Llama extracted (likely partial).
    fullManuscriptText = nativeText.trim().length > 500 ? nativeText : (result.fullText || "");

    // --- STRUCTURAL MAP STRATEGY ---
    // Forcefully capture the first 15,000 characters (Intro + TOC) as the "Structural Context"
    // This allows the model to "know what it doesn't know" and reference chapters by name even if their content is not in the chunk.
    const structuralMap = fullManuscriptText.substring(0, Math.min(15000, fullManuscriptText.length));
    manuscriptMetadata.chapters = `*** TABLE OF CONTENTS & INTRO MAP ***\n${structuralMap}\n*** END MAP ***`;


    documentChunks = chunkText(fullManuscriptText);

    // توفير التوكنز: مسح الـ PDF بعد الاستخراج الأول
    currentPdfBase64 = null;

    return result.axioms || [];
  } catch (error: any) {
    console.error("Error in extractAxioms:", error);
    throw error;
  }
};

export const getManuscriptSnippets = () => manuscriptSnippets;

export const chatWithManuscriptStream = async (
  userPrompt: string,
  lang: Language,
  onChunk: (text: string) => void
): Promise<void> => {
  const groq = getGroqClient();

  try {
    await throttleRequest();
    // Retrieve 15 chunks explicitly for deep coverage
    const relevantChunks = retrieveRelevantChunks(userPrompt, documentChunks, 15);

    let augmentedPrompt = "";
    const hasChunks = relevantChunks.length > 0;

    if (hasChunks) {
      const contextText = relevantChunks.join("\n\n---\n\n");
      // Inject Structural Map + Specific Chunks
      augmentedPrompt = `MANUSCRIPT STRUCTURE (Table of Contents / Map):
${manuscriptMetadata.chapters || "Not available"}

CRITICAL CONTEXT CHUNKS (Use this to answer):
${contextText}

USER QUESTION:
${userPrompt}

INSTRUCTION: You are an advanced analytical engine capable of deep synthesis.
1. Answer the question COMPREHENSIVELY using ONLY the provided context.
2. Use the 'MANUSCRIPT STRUCTURE' to understand where the chunks fit in the whole book.
3. If the answer is scattered across multiple chunks, synthesize them into a cohesive narrative.
4. If the question asks for a summary or broad concept, ensure you cover all relevant aspects found in the context.
5. Adopt the author's intellectual style.
6. Support your arguments with direct, verbatim quotes from the text.`;
    } else {
      augmentedPrompt = `MANUSCRIPT STRUCTURE (Table of Contents / Map):
${manuscriptMetadata.chapters || "Not available"}

USER QUESTION: ${userPrompt}

INSTRUCTION: The specific context chunks were not found. However, use the provided 'MANUSCRIPT STRUCTURE' (Table of Contents/Intro) to answer if possible (e.g., if the user asks about chapter titles or general structure). If the detail is likely in the body text but not in this structure, state clearly that you need more specific context. Adopt the author's style.`;
    }

    if (!chatSession) {
      chatSession = new GroqChatSession(groq, MODEL_NAME, getSystemInstruction(lang));
    }

    const result = chatSession.sendMessageStream({ message: augmentedPrompt });

    for await (const chunk of result) {
      if (chunk.text) onChunk(chunk.text);
    }
  } catch (error: any) {
    console.error("Stream error in Service:", error);
    chatSession = null;
    throw error;
  }
};
