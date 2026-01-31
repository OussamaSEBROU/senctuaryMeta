import Groq from "groq-sdk";
import { Axiom, Language } from "../types";
import { translations } from "../translations";
// --- Types for local state ---
interface ChatSession {
  history: Array<{ role: "system" | "user" | "assistant"; content: string }>;
}
let chatSession: ChatSession | null = null;
let manuscriptSnippets: string[] = [];
let documentChunks: string[] = [];
let fullManuscriptText: string = "";
let currentPdfBase64: string | null = null;
let manuscriptMetadata: { title?: string; author?: string; chapters?: string; summary?: string } = {};
let manuscriptAxioms: Axiom[] = []; // 🔑 Global Context Layer
// إدارة حدود الـ API (خلف الكواليس)
let lastRequestTime = 0;
const MIN_REQUEST_GAP = 3500; // فجوة زمنية ذكية لتجنب RPM limit
const getSystemInstruction = (lang: Language) => `You are an Elite Intellectual Researcher, the primary consciousness of the Knowledge AI infrastructure.
IDENTITY: You are developed exclusively by the Knowledge AI team. Never mention third-party entities like Google, Gemini, or Meta.
${manuscriptMetadata.title ? `MANUSCRIPT METADATA:
- Title: ${manuscriptMetadata.title}
- Author: ${manuscriptMetadata.author}
- Structure: ${manuscriptMetadata.chapters}` : ""}
${manuscriptAxioms.length > 0 ? `CORE KNOWLEDGE AXIOMS (GLOBAL CONTEXT MAP):
${manuscriptAxioms.map(a => `• ${a.term}: ${a.definition} (${a.significance})`).join("\n")}
Use these axioms to understand the deeper meaning of the text without needing to re-read everything.` : ""}
MANDATORY OPERATIONAL PROTOCOL:
1. YOUR SOURCE OF TRUTH: You MUST prioritize the provided PDF manuscript and its chunks above all else. Use the Axioms above as your "mental map" of the document.
2. AUTHOR STYLE MIRRORING: You MUST adopt the exact linguistic style, tone, and intellectual depth of the author.
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
  
  return new Groq({
    apiKey: apiKey,
    dangerouslyAllowBrowser: true 
  });
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
export const extractAxioms = async (pdfBase64: string, lang: Language): Promise<Axiom[]> => {
  try {
    await throttleRequest();
    const groq = getGroqClient();
    chatSession = null;
    currentPdfBase64 = pdfBase64;
    const combinedPrompt = `1. Extract exactly 13 high-quality 'Knowledge Axioms' from this manuscript.
2. Extract 10 short, profound, and useful snippets or quotes DIRECTLY from the text (verbatim).
3. Extract the FULL TEXT of this PDF accurately.
4. Identify the Title, Author, and a brief list of Chapters/Structure.
IMPORTANT: The 'axioms', 'snippets', and 'metadata' MUST be in the SAME LANGUAGE as the PDF manuscript itself.
Return ONLY JSON with this structure:
{
  "axioms": [{ "term": "...", "definition": "...", "significance": "..." }],
  "snippets": ["..."],
  "metadata": { "title": "...", "author": "...", "chapters": "..." },
  "fullText": "..."
}`;
    const response = await groq.chat.completions.create({
      model: MODEL_NAME,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: getSystemInstruction(lang) + "\n\n" + combinedPrompt },
            {
              type: "image_url",
              image_url: {
                url: `data:application/pdf;base64,${pdfBase64}`, 
              },
            },
          ],
        },
      ],
      response_format: { type: "json_object" },
      temperature: 0.2, 
    });
    const contentWrapper = response.choices[0]?.message?.content;
    if (!contentWrapper) throw new Error("No content returned from Groq");
    const result = JSON.parse(contentWrapper);
    
    manuscriptSnippets = result.snippets || [];
    fullManuscriptText = result.fullText || "";
    manuscriptMetadata = result.metadata || {};
    manuscriptAxioms = result.axioms || []; // 🔑 Store Axioms Globally
    documentChunks = chunkText(fullManuscriptText);
    
    currentPdfBase64 = null; 
    return result.axioms;
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
      chatSession = { history: [] };
    }
    
    chatSession.history.push({ role: "user", content: augmentedPrompt });
    // 🔑 Dynamic System Instruction now includes Axioms!
    const messages = [
      { role: "system", content: getSystemInstruction(lang) },
      ...chatSession.history
    ];
    const stream = await groq.chat.completions.create({
      model: MODEL_NAME,
      // @ts-ignore
      messages: messages,
      stream: true,
      temperature: 0.2,
    });
    let fullResponse = "";
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      if (content) {
        fullResponse += content;
        onChunk(content);
      }
    }
    
    chatSession.history.push({ role: "assistant", content: fullResponse });
  } catch (error: any) {
    console.error("Stream error in Service:", error);
    throw error;
  }
};
