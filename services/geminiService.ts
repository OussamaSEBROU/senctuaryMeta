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
 */
const chunkText = (text: string, chunkSize: number = 1800, overlap: number = 250): string[] => {
  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.substring(start, end);
    if (chunk.trim().length >= 200) chunks.push(chunk);
    if (end === text.length) break;
    start += chunkSize - overlap;
  }
  return chunks;
};

/**
 * استعادة منطق الاسترجاع الأصلي مع تحسين بسيط في النقاط لضمان الجودة
 */
const retrieveRelevantChunks = (query: string, chunks: string[], topK: number = 2): string[] => {
  if (chunks.length === 0) return [];
  const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
  const MIN_SCORE_THRESHOLD = 4;

  const scoredChunks = chunks.map(chunk => {
    const chunkLower = chunk.toLowerCase();
    let score = 0;
    queryWords.forEach(word => { if (chunkLower.includes(word)) score += 2; });
    const qLower = query.toLowerCase();
    if (qLower.includes("كاتب") || qLower.includes("مؤلف") || qLower.includes("author")) {
      if (chunks.indexOf(chunk) === 0) score += 5;
    }
    return { chunk, score };
  });

  return scoredChunks
    .sort((a, b) => b.score - a.score)
    .filter(item => item.score >= MIN_SCORE_THRESHOLD)
    .slice(0, topK)
    .map(item => item.chunk); // إرسال القطعة كاملة دون ضغط لضمان الجودة الأصلية
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

export const extractAxioms = async (pdfBase64: string, lang: Language): Promise<Axiom[]> => {
  try {
    await throttleRequest();
    const groq = getGroqClient();
    chatSession = null;
    currentPdfBase64 = pdfBase64;

    // Convert PDF pages to Images for Llama Vision
    const pageImages = await convertPdfToImages(pdfBase64);

    const schemaDescription = `
    RETURN ONLY JSON matching this structure:
    {
      "axioms": [ { "term": "string", "definition": "string", "significance": "string" } ],
      "snippets": [ "string" ],
      "metadata": { "title": "string", "author": "string", "chapters": "string" },
      "fullText": "string"
    }
    `;

    const combinedPrompt = `1. Extract exactly 13 high-quality 'Knowledge Axioms' from this manuscript.
2. Extract 10 short, profound, and useful snippets or quotes DIRECTLY from the text (verbatim).
3. Extract the FULL TEXT of this PDF accurately.
4. Identify the Title, Author, and a brief list of Chapters/Structure.

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
      response_format: { type: "json_object" },
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
    fullManuscriptText = result.fullText || "";
    manuscriptMetadata = result.metadata || {};
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
    const relevantChunks = retrieveRelevantChunks(userPrompt, documentChunks);

    let augmentedPrompt = "";
    const hasChunks = relevantChunks.length > 0;

    if (hasChunks) {
      const contextText = relevantChunks.join("\n\n---\n\n");
      augmentedPrompt = `CRITICAL CONTEXT FROM MANUSCRIPT:
${contextText}

USER QUESTION:
${userPrompt}

INSTRUCTION: You MUST answer based on the provided context. Adopt the author's style. Support your answer with direct quotes.`;
    } else {
      augmentedPrompt = `USER QUESTION: ${userPrompt}
INSTRUCTION: Scan the entire manuscript to find the answer. Adopt the author's style. Be specific and provide quotes.`;
    }

    if (!chatSession) {
      chatSession = new GroqChatSession(groq, MODEL_NAME, getSystemInstruction(lang));
    }

    // إرسال الطلب مع الحفاظ على جودة الإجابة الأصلية
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
