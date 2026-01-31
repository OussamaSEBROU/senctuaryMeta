
export type Language = 'en' | 'ar';

export interface Axiom {
  term: string;
  definition: string;
  significance: string;
}

export interface Message {
  role: 'user' | 'model';
  content: string;
}

export interface PDFData {
  base64: string;
  name: string;
}
