
import React, { useState, useRef, useEffect } from 'react';
import { Axiom, PDFData, Language } from './types';
import { extractAxioms } from './services/geminiService';
import AxiomCard from './components/AxiomCard';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import ManuscriptViewer from './components/ManuscriptViewer';
import { translations } from './translations';

// مصفوفة المقولات المختارة بعناية من المصادر المحددة
const quotes = {
  en: [
    "The present is theirs; the future, for which I really worked, is mine. — Nikola Tesla",
    "I do not think you can name many great inventions that have been made by married men. — Nikola Tesla",
    "Our virtues and our failings are inseparable, like force and matter. — Nikola Tesla",
    "The scientists of today think deeply instead of clearly. — Nikola Tesla",
    "The gift of mental power comes from God, Divine Being. — Nikola Tesla",
    "Imagination is more important than knowledge. — Albert Einstein",
    "The only real valuable thing is intuition. — Albert Einstein",
    "Science without religion is lame, religion without science is blind. — Albert Einstein",
    "A person who never made a mistake never tried anything new. — Albert Einstein",
    "The important thing is not to stop questioning. — Albert Einstein",
    "Technology is nothing. What's important is that you have a faith in people. — Steve Jobs",
    "Design is not just what it looks like and feels like. Design is how it works. — Steve Jobs",
    "The people who are crazy enough to think they can change the world are the ones who do. — Steve Jobs",
    "Innovation distinguishes between a leader and a follower. — Steve Jobs",
    "Stay hungry, stay foolish. — Steve Jobs",
    "The best way to predict the future is to invent it. — Alan Kay",
    "The computer is the most remarkable tool that we've ever come up with. — Steve Jobs",
    "Simplicity is the ultimate sophistication. — Leonardo da Vinci",
    "Nature is the source of all true knowledge. — Leonardo da Vinci",
    "Learning never exhausts the mind. — Leonardo da Vinci",
    "The unexamined life is not worth living. — Socrates",
    "Turn your wounds into wisdom. — Oprah Winfrey",
    "You must be the change you wish to see in the world. — Mahatma Gandhi",
    "The only way to do great work is to love what you do. — Steve Jobs",
    "Innovation distinguishes between a leader and a follower. — Steve Jobs"
  ],
  ar: [
    "إن الحضارة لا تباع ولا تشترى، وإنما هي نتاج جهد فكري وعملي دؤوب. — مالك بن نبي (شروط النهضة)",
    "الأفكار هي التي تصنع التاريخ، والعمل هو الذي يجسدها. — مالك بن نبي (وجهة العالم الإسلامي)",
    "التكنولوجيا بلا روح هي أداة للتدمير، أما التكنولوجيا الموجهة بالقيم فهي وسيلة للتحرر. — مالك بن نبي",
    "إن مشكلة كل شعب هي في جوهرها مشكلة حضارته. — مالك بن نبي",
    "لا يمكن لعالم أن ينهض إذا كان يستهلك أفكار غيره دون تمحيص. — مالك بن نبي",
    "الحرية ليست مجرد غياب القيود، بل هي القدرة على اختيار الخير. — علي عزت بيجوفيتش (الإسلام بين الشرق والغرب)",
    "الصلاة لا تغير العالم، ولكنها تغير الإنسان الذي سيغير العالم. — علي عزت بيجوفيتش",
    "إن المجتمع الذي لا يقرأ هو مجتمع لا يفكر، والمجتمع الذي لا يفكر لا يمكنه أن يبني حضارة. — علي عزت بيجوفيتش",
    "التكنولوجيا يجب أن تخدم الإنسان، لا أن تستعبده. — علي عزت بيجوفيتش",
    "الإيمان هو الذي يعطي للحياة معنى، والعمل هو الذي يعطي للإيمان قيمة. — علي عزت بيجوفيتش",
    "العلم بلا أخلاق هو دمار للبشرية. — البشير الإبراهيمي",
    "إن الأمة التي تنسى تاريخها لا يمكنها أن نبني مستقبلها. — البشير الإبراهيمي",
    "اللغة هي وعاء الفكر، فإذا فسد الوعاء فسد ما فيه. — البشير الإبراهيمي",
    "العمل هو المقياس الحقيقي لقيمة الإنسان. — البشير الإبراهيمي",
    "الاستعمار الفكري أخطر من الاستعمار العسكري. — البشير الإبراهيمي",
    "الوهم نصف الداء، والاطمئنان نصف الدواء، والصبر أول خطوات الشفاء. — ابن سينا (القانون في الطب)",
    "العقل البشري هو أعظم هبة من الله، وبه ندرك الحقائق. — ابن سينا",
    "العلم هو معرفة الأشياء بحقائقها. — ابن سينا",
    "المعرفة هي القوة التي تمكننا من فهم الكون. — ابن سينا",
    "الطب هو حفظ الصحة وشفاء المرض. — ابن سينا",
    "مقاصد الشريعة هي حفظ مصالح العباد في الدارين. — ابن عاشور (مقاصد الشريعة الإسلامية)",
    "الاجتهاد هو روح الشريعة، وبه تبقى صالحة لكل زمان ومكان. — ابن عاشور",
    "التعليم هو الأساس الذي تبنى عليه الأمم. — ابن عاشور",
    "الحرية هي حق أصيل لكل إنسان، ولا يجوز سلبها إلا بحق. — ابن عاشور",
    "العدل هو ميزان الله في الأرض. — ابن عاشور",
    "العمل هو تجسيد للمنطق، والمنطق هو روح العمل. — طه عبد الرحمن (روح الحداثة)",
    "لا حداثة بلا أخلاق، ولا أخلاق بلا إيمان. — طه عبد الرحمن",
    "التكنولوجيا هي تجلٍ للعقل، ولكن يجب أن تخضع للقيم. — طه عبد الرحمن",
    "الإنسان هو كائن أخلاقي بامتياز، وعمله يجب أن يعكس ذلك. — طه عبد الرحمن",
    "الحوار هو السبيل الوحيد للتفاهم بين الحضارات. — طه عبد الرحمن",
    "إن الإنسان في غاية الحاجة لمركزية 'منطق العمل' كما هو في حاجة لمركزية 'منطق الفكرة'. — طه عبد الرحمن",
    "التقنية ليست مجرد أدوات، بل هي نمط من الوجود يتطلب وعياً أخلاقياً. — طه عبد الرحمن"
  ]
};

// نصوص الحالة التقنية
const statusMessages = {
  en: [
    "Analyzing manuscript structure...",
    "Deep thinking in progress...",
    "Extracting axiomatic wisdom...",
    "Synthesizing neural connections...",
    "Mapping intellectual framework...",
    "Decoding author's logic...",
    "Finalizing neural sync..."
  ],
  ar: [
    "يتم تحليل بنية المخطوط...",
    "تفكير معمق في الأفكار المركزية...",
    "استخراج الحكمة الأكسيومية...",
    "توليف الروابط العصبية...",
    "رسم الإطار الفكري...",
    "فك شفرة منطق المؤلف...",
    "إتمام المزامنة العصبية..."
  ]
};

const App: React.FC = () => {
  const [pdf, setPdf] = useState<PDFData | null>(null);
  const [axioms, setAxioms] = useState<Axiom[]>([]);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lang, setLang] = useState<Language>('en');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [currentQuoteIndex, setCurrentQuoteIndex] = useState(0);
  const [currentStatusIndex, setCurrentStatusIndex] = useState(0);
  
  const [flowStep, setFlowStep] = useState<'axioms' | 'chat'>('axioms');
  const [showViewer, setShowViewer] = useState(false);
  const carouselRef = useRef<HTMLDivElement>(null);

  const t = translations[lang];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSynthesizing) {
      setCurrentQuoteIndex(Math.floor(Math.random() * quotes[lang].length));
      setCurrentStatusIndex(0);
      
      interval = setInterval(() => {
        setCurrentQuoteIndex(Math.floor(Math.random() * quotes[lang].length));
        setCurrentStatusIndex(prev => (prev + 1) % statusMessages[lang].length);
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isSynthesizing, lang]);

  useEffect(() => {
    if (axioms.length > 0 && carouselRef.current) {
      carouselRef.current.scrollTo({ left: 0, behavior: 'smooth' });
    }
  }, [axioms]);

  const handleSynthesis = async (base64: string, currentLang: Language) => {
    setIsSynthesizing(true);
    setError(null);
    setAxioms([]);
    setFlowStep('axioms');

    try {
      const extracted = await extractAxioms(base64, currentLang);
      if (extracted && extracted.length > 0) {
        setAxioms(extracted);
      } else {
        throw new Error("EMPTY_RESULT");
      }
    } catch (err: any) {
      console.error("Synthesis error:", err);
      let errorMsg = currentLang === 'ar' ? "فشل التحليل العصبي للمخطوط." : "Synthesis failed.";
      setError(errorMsg);
      setPdf(null);
    } finally {
      setIsSynthesizing(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.type !== 'application/pdf') {
      setError(lang === 'ar' ? "يرجى رفع ملف PDF فقط." : "Please upload a PDF file only.");
      return;
    }
    
    const reader = new FileReader();
    reader.onload = async () => {
      const result = reader.result as string;
      const base64 = result.substring(result.indexOf(',') + 1);
      setPdf({ base64, name: file.name });
      handleSynthesis(base64, lang);
    };
    reader.readAsDataURL(file);
  };

  const handleNewChat = () => {
    setPdf(null);
    setAxioms([]);
    setFlowStep('axioms');
    setShowViewer(false);
    setError(null);
  };

  return (
    <div className={`fixed inset-0 flex flex-col bg-[#020202] text-white overflow-hidden ${lang === 'ar' ? 'rtl font-academic' : 'ltr font-sans'}`} dir={lang === 'ar' ? 'rtl' : 'ltr'}>
      <style>{`
        @keyframes shiningText {
          0% { opacity: 0; transform: translateY(15px); filter: blur(8px); }
          15% { opacity: 1; transform: translateY(0); filter: blur(0); }
          85% { opacity: 1; transform: translateY(0); filter: blur(0); }
          100% { opacity: 0; transform: translateY(-15px); filter: blur(8px); }
        }
        .shining-quote {
          animation: shiningText 5s ease-in-out infinite;
          background: linear-gradient(90deg, #fff, #a34a28, #fff);
          background-size: 200% auto;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          animation: shiningText 5s ease-in-out infinite, shine 4s linear infinite;
        }
        @keyframes shine {
          to { background-position: 200% center; }
        }
      `}</style>

      {isSynthesizing && (
        <div className="fixed inset-0 bg-black z-[100] flex flex-col items-center justify-center p-6 text-center animate-in fade-in duration-1000">
          <div className="spinner-arc mb-16 w-24 h-24 border-t-orange-600"></div>
          
          {/* نص الحالة التقنية */}
          <div className="mb-4 h-6">
            <p className="text-orange-500/60 text-[10px] font-black tracking-[0.4em] uppercase animate-pulse">
              {statusMessages[lang][currentStatusIndex]}
            </p>
          </div>

          <h2 className="text-white text-xl font-black tracking-[0.6em] mb-12 uppercase opacity-30">{t.synthesis}</h2>
          
          <div className="h-32 flex items-center justify-center">
            <p key={currentQuoteIndex} className="shining-quote text-xl md:text-3xl font-medium italic max-w-3xl leading-relaxed px-6">
              {quotes[lang][currentQuoteIndex]}
            </p>
          </div>
        </div>
      )}

      <Sidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)} 
        lang={lang} 
        setLang={setLang}
        onNewChat={handleNewChat}
      />

      <header className="h-14 md:h-16 px-4 md:px-8 flex items-center justify-between border-b border-white/5 bg-black/40 backdrop-blur-3xl z-[60] shrink-0">
        <div className="flex items-center gap-4">
          <button onClick={() => setIsSidebarOpen(true)} className="p-2 hover:bg-white/5 rounded-xl transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4 6h16M4 12h16M4 18h16" /></svg>
          </button>
          <h1 className="text-[10px] font-black tracking-[0.4em] text-white/40 uppercase hidden sm:block">{translations.en.title}</h1>
        </div>
        
        {pdf && (
          <button 
            onClick={() => setShowViewer(!showViewer)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all border text-[10px] font-black tracking-widest uppercase ${showViewer ? 'bg-white border-white text-black' : 'bg-white/5 border-white/5 text-white/40 hover:text-white'}`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
            <span className="hidden md:inline">{showViewer ? (lang === 'ar' ? 'إغلاق العارض' : 'Close Viewer') : (lang === 'ar' ? 'فتح المخطوط' : 'Open Manuscript')}</span>
          </button>
        )}
      </header>

      <main className="flex-1 overflow-hidden relative z-10 flex flex-col">
        {!pdf ? (
          <div className="flex-1 overflow-y-auto flex flex-col items-center justify-center p-6 text-center touch-auto" dir="ltr">
            <h2 className="text-6xl md:text-9xl font-black mb-4 select-none text-white tracking-tighter uppercase font-sans">
              {translations.en.sanctuary}
            </h2>
            <p className="mb-12 text-sm md:text-2xl font-black tracking-tight text-glow-orange max-w-2xl leading-tight font-sans">
              {lang === 'ar' ? translations.ar.introText : translations.en.introText}
            </p>
            <label className="w-full max-w-sm group relative block aspect-[1.3/1] border border-dashed border-white/10 rounded-[3rem] hover:border-[#a34a28]/40 transition-all cursor-pointer bg-white/[0.01]">
              <input type="file" className="hidden" accept="application/pdf" onChange={handleFileUpload} />
              <div className="absolute inset-0 flex flex-col items-center justify-center space-y-6">
                 <div className="w-16 h-16 rounded-[2rem] bg-white/5 border border-white/10 flex items-center justify-center group-hover:bg-[#a34a28]/10 transition-all duration-500 shadow-[0_0_20px_rgba(163,74,40,0)] group-hover:shadow-[0_0_20px_rgba(163,74,40,0.2)]">
                    <svg className="w-8 h-8 text-white/10 group-hover:text-[#a34a28] transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                 </div>
                 <span className="text-[10px] md:text-xs font-black tracking-[0.4em] text-white/20 uppercase group-hover:text-white/60">
                   {translations.en.upload}
                 </span>
              </div>
            </label>
            {error && (
              <div className="mt-8 max-w-md mx-auto">
                <p className="text-red-500 text-[10px] font-black uppercase tracking-[0.3em] bg-red-500/10 px-6 py-2 rounded-full border border-red-500/20 shadow-lg">
                  {error}
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="flex-1 flex flex-col lg:flex-row relative overflow-hidden">
            <div className={`flex-1 flex flex-col transition-all duration-700 ease-in-out overflow-hidden ${showViewer ? 'lg:w-1/2 opacity-100' : 'lg:w-full'}`}>
              {flowStep === 'axioms' && (
                <div className="flex-1 overflow-y-auto flex flex-col items-center justify-center p-4 touch-auto">
                   <h3 className="text-2xl md:text-5xl font-black mb-8 uppercase text-center text-white/90 tracking-widest">{t.axiomsTitle}</h3>
                   <div ref={carouselRef} className="w-full flex gap-6 px-4 md:px-[5%] overflow-x-auto snap-x scrollbar-none pb-10 touch-pan-x">
                      {axioms.length > 0 ? axioms.map((ax, i) => (
                        <div key={i} className="min-w-[280px] md:min-w-[400px] snap-center">
                          <AxiomCard axiom={ax} index={i} />
                        </div>
                      )) : (
                        <div className="w-full flex justify-center py-20 opacity-20">
                           <div className="w-10 h-10 border-2 border-white/10 border-t-white/40 rounded-full animate-spin"></div>
                        </div>
                      )}
                   </div>
                   <button 
                    onClick={() => { setFlowStep('chat'); if(window.innerWidth > 1024) setShowViewer(true); }} 
                    className="px-12 py-5 bg-[#a34a28] rounded-full font-black text-xs tracking-[0.4em] uppercase hover:bg-orange-800 transition-all shadow-[0_0_30px_rgba(163,74,40,0.3)] mt-4 active:scale-95"
                   >
                     {t.deepChatBtn}
                   </button>
                </div>
              )}
              {flowStep === 'chat' && (
                <div className="flex-1 bg-[#080808] overflow-hidden">
                  <ChatInterface pdf={pdf} lang={lang} />
                </div>
              )}
            </div>

            {showViewer && (
              <div className={`fixed inset-0 lg:relative lg:inset-auto lg:w-1/2 bg-black z-[70] lg:z-10 animate-in slide-in-from-right duration-500 border-l border-white/10 flex flex-col shadow-[-20px_0_50px_rgba(0,0,0,0.8)] overflow-hidden`}>
                <div className="flex lg:hidden items-center justify-between p-4 bg-[#1a1a1a] border-b border-white/10">
                   <h4 className="text-[10px] font-black tracking-widest uppercase text-white/40">{t.viewer}</h4>
                   <button onClick={() => setShowViewer(false)} className="p-2 bg-white/5 rounded-full text-white/60">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12" strokeWidth="2.5" strokeLinecap="round"/></svg>
                   </button>
                </div>
                <ManuscriptViewer pdf={pdf} lang={lang} />
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
