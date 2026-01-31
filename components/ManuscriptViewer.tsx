import React, { useEffect, useRef, useState, useCallback } from 'react';
import { PDFData, Language } from '../types';
import { translations } from '../translations';

interface ManuscriptViewerProps {
  pdf: PDFData;
  lang: Language;
}

interface Bubble {
  id: number;
  x: number;
  y: number;
  size: number;
  speed: number;
  opacity: number;
}

export const ManuscriptViewer: React.FC<ManuscriptViewerProps> = ({ pdf, lang }) => {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [jumpPage, setJumpPage] = useState<string>('');
  const [zoom, setZoom] = useState<number>(1.0);
  const [readingTime, setReadingTime] = useState<number>(0); // بالثواني
  const [showAchievement, setShowAchievement] = useState<string | null>(null);
  const [bubbles, setBubbles] = useState<Bubble[]>([]);
  const [isCelebrating, setIsCelebrating] = useState(false);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const pdfDocRef = useRef<any>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const t = translations[lang];

  const storageKeyTime = `reading_time_${pdf.name}`;
  const storageKeyPage = `current_page_${pdf.name}`;

  // استعادة الوقت والصفحة
  useEffect(() => {
    const savedTime = localStorage.getItem(storageKeyTime);
    const savedPage = localStorage.getItem(storageKeyPage);
    if (savedTime) setReadingTime(parseInt(savedTime));
    if (savedPage) setCurrentPage(parseInt(savedPage));
  }, [storageKeyTime, storageKeyPage]);

  // تتبع الوقت
  useEffect(() => {
    const timer = setInterval(() => {
      setReadingTime(prev => {
        const newTime = prev + 1;
        localStorage.setItem(storageKeyTime, newTime.toString());
        return newTime;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [storageKeyTime]);

  const getStars = (seconds: number) => {
    const mins = seconds / 60;
    if (mins >= 120) return 6;
    if (mins >= 90) return 5;
    if (mins >= 60) return 4;
    if (mins >= 40) return 3;
    if (mins >= 15) return 2;
    if (mins >= 5) return 1;
    return 0;
  };

  const getNextMilestone = (seconds: number) => {
    const mins = seconds / 60;
    if (mins < 5) return { time: 5, stars: 1 };
    if (mins < 15) return { time: 15, stars: 2 };
    if (mins < 40) return { time: 40, stars: 3 };
    if (mins < 60) return { time: 60, stars: 4 };
    if (mins < 90) return { time: 90, stars: 5 };
    if (mins < 120) return { time: 120, stars: 6 };
    return null;
  };

  const stars = getStars(readingTime);
  const next = getNextMilestone(readingTime);

  // منطق الاحتفال والفقاعات
  const startCelebration = useCallback(() => {
    setIsCelebrating(true);
    
    // تشغيل صوت الاحتفال
    if (!audioRef.current) {
      audioRef.current = new Audio('https://assets.mixkit.co/active_storage/sfx/2013/2013-preview.mp3');
    }
    audioRef.current.play().catch(e => console.log("Audio play failed", e));

    // توليد الفقاعات
    const newBubbles: Bubble[] = Array.from({ length: 40 }).map((_, i) => ({
      id: Math.random(),
      x: Math.random() * 100,
      y: 110,
      size: Math.random() * 40 + 20,
      speed: Math.random() * 2 + 1,
      opacity: Math.random() * 0.5 + 0.3
    }));
    setBubbles(newBubbles);

    // إخفاء الاحتفال بعد 10 ثوانٍ
    setTimeout(() => {
      setIsCelebrating(false);
      setBubbles([]);
      setShowAchievement(null);
    }, 10000);
  }, []);

  const popBubble = (id: number) => {
    setBubbles(prev => prev.filter(b => b.id !== id));
    // صوت فرقعة بسيط
    const popAudio = new Audio('https://assets.mixkit.co/active_storage/sfx/3005/3005-preview.mp3');
    popAudio.volume = 0.4;
    popAudio.play().catch(() => {});
  };

  // تحديث حركة الفقاعات
  useEffect(() => {
    if (!isCelebrating) return;
    const interval = setInterval(() => {
      setBubbles(prev => prev.map(b => ({
        ...b,
        y: b.y - b.speed,
        x: b.x + Math.sin(b.y / 20) * 0.5
      })).filter(b => b.y > -20));
    }, 30);
    return () => clearInterval(interval);
  }, [isCelebrating]);

  // مراقبة النجوم لإطلاق الاحتفال
  const lastStarsRef = useRef(stars);
  useEffect(() => {
    if (stars > lastStarsRef.current) {
      const messages = lang === 'ar' ? [
        "رائع! استمر في الغوص في أعماق المعرفة",
        "إنجاز مذهل! عقلك يزداد نورا",
        "أنت بطل القراءة! لا تتوقف الآن",
        "نجمة جديدة تضيء مسيرتك العلمية",
        "مستوى جديد من التركيز! أحسنت",
        "القراءة هي مفتاح العظمة، وأنت تملكه"
      ] : [
        "Amazing! Keep diving deep into knowledge",
        "Incredible achievement! Your mind is glowing",
        "You're a reading champion! Don't stop now",
        "A new star lights up your academic journey",
        "New level of focus unlocked! Well done",
        "Reading is the key to greatness, and you hold it"
      ];
      
      setShowAchievement(messages[stars - 1] || messages[0]);
      startCelebration();
    }
    lastStarsRef.current = stars;
  }, [stars, lang, startCelebration]);

  useEffect(() => {
    if (!pdf.base64) return;
    const loadPdf = async () => {
      setLoading(true);
      setError(null);
      try {
        // @ts-ignore
        const pdfjsLib = await import('https://esm.sh/pdfjs-dist@4.10.38');
        pdfjsLib.GlobalWorkerOptions.workerSrc = `https://esm.sh/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs`;
        const binaryString = atob(pdf.base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
        const loadingTask = pdfjsLib.getDocument({ data: bytes });
        const pdfDoc = await loadingTask.promise;
        pdfDocRef.current = pdfDoc;
        setNumPages(pdfDoc.numPages);
        setLoading(false);
        const savedPage = localStorage.getItem(storageKeyPage);
        if (savedPage) {
          const pageNum = parseInt(savedPage);
          setTimeout(() => {
            if (containerRef.current) {
              const width = containerRef.current.clientWidth;
              containerRef.current.scrollTo({ left: (pageNum - 1) * width, behavior: 'instant' });
            }
          }, 100);
        }
      } catch (err) {
        setError(lang === 'ar' ? "فشل تحميل المحتوى البصري للمخطوط." : "Failed to load manuscript visual core.");
        setLoading(false);
      }
    };
    loadPdf();
    return () => { if (pdfDocRef.current) pdfDocRef.current.destroy(); };
  }, [pdf.base64, lang, storageKeyPage]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handleScroll = () => {
      const page = Math.round(container.scrollLeft / container.clientWidth) + 1;
      if (page !== currentPage && page > 0 && page <= numPages) {
        setCurrentPage(page);
        localStorage.setItem(storageKeyPage, page.toString());
      }
    };
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [currentPage, numPages, storageKeyPage]);

  const goToPage = (e: React.FormEvent) => {
    e.preventDefault();
    const pageNum = parseInt(jumpPage);
    if (pageNum > 0 && pageNum <= numPages && containerRef.current) {
      containerRef.current.scrollTo({ left: (pageNum - 1) * containerRef.current.clientWidth, behavior: 'smooth' });
      setJumpPage('');
      localStorage.setItem(storageKeyPage, pageNum.toString());
    }
  };

  const handleZoom = (delta: number) => {
    setZoom(prev => parseFloat(Math.min(Math.max(prev + delta, 0.5), 3.0).toFixed(1)));
  };

  return (
    <div className="w-full h-full flex flex-col bg-[#050505] overflow-hidden select-none relative">
      <style>{`
        @keyframes shine-3d {
          0% { transform: perspective(1000px) rotateX(0deg) rotateY(0deg) scale(1); text-shadow: 0 0 10px rgba(255,255,255,0.5); }
          50% { transform: perspective(1000px) rotateX(10deg) rotateY(10deg) scale(1.1); text-shadow: 0 0 30px rgba(255,165,0,0.8), 0 0 50px rgba(255,165,0,0.4); }
          100% { transform: perspective(1000px) rotateX(0deg) rotateY(0deg) scale(1); text-shadow: 0 0 10px rgba(255,255,255,0.5); }
        }
        @keyframes bubble-float {
          0% { transform: translateY(0) scale(1); }
          50% { transform: translateY(-10px) scale(1.05); }
          100% { transform: translateY(0) scale(1); }
        }
        .achievement-popup {
          animation: shine-3d 2s ease-in-out infinite;
          background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,165,0,0.3));
          backdrop-filter: blur(15px);
          border: 1px solid rgba(255,165,0,0.5);
          box-shadow: 0 20px 60px rgba(0,0,0,0.6), inset 0 0 30px rgba(255,165,0,0.3);
          z-index: 100;
        }
        .bubble {
          position: absolute;
          border-radius: 50%;
          background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.4), rgba(255,165,0,0.1));
          border: 1px solid rgba(255,255,255,0.2);
          backdrop-filter: blur(2px);
          cursor: pointer;
          transition: transform 0.1s;
          z-index: 90;
        }
        .bubble:hover { transform: scale(1.2); }
        .celebration-overlay {
          position: absolute;
          inset: 0;
          background: rgba(0,0,0,0.3);
          backdrop-filter: blur(4px);
          z-index: 80;
          pointer-events: none;
          transition: opacity 0.5s;
        }
      `}</style>

      {/* واجهة الاحتفال */}
      {isCelebrating && (
        <>
          <div className="celebration-overlay" />
          {bubbles.map(bubble => (
            <div
              key={bubble.id}
              className="bubble"
              style={{
                left: `${bubble.x}%`,
                top: `${bubble.y}%`,
                width: `${bubble.size}px`,
                height: `${bubble.size}px`,
                opacity: bubble.opacity
              }}
              onClick={() => popBubble(bubble.id)}
            />
          ))}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-[110] pointer-events-none w-full max-w-lg px-6">
            <div className="achievement-popup px-8 py-10 rounded-[2.5rem] text-center">
              <div className="text-6xl mb-4 animate-bounce">🌟</div>
              <h2 className="text-2xl md:text-3xl font-black text-white mb-4 tracking-tight leading-tight">
                {showAchievement}
              </h2>
              <div className="flex justify-center gap-2 mb-4">
                {Array.from({ length: stars }).map((_, i) => (
                  <span key={i} className="text-2xl drop-shadow-[0_0_10px_rgba(255,165,0,1)]">⭐</span>
                ))}
              </div>
              <p className="text-[12px] uppercase tracking-[0.4em] text-orange-400 font-black">
                {lang === 'ar' ? 'إنجاز معرفي استثنائي' : 'Exceptional Knowledge Achievement'}
              </p>
            </div>
          </div>
        </>
      )}

      {/* شريط الأدوات العلوي */}
      <div className="h-14 bg-black/95 border-b border-white/10 flex flex-col z-30 shrink-0">
        <div className="flex items-center justify-between px-3 h-8 border-b border-white/5">
          <span className="text-[9px] font-medium text-white/30 truncate max-w-[150px]">{pdf.name}</span>
          <div className="flex items-center gap-2">
             <form onSubmit={goToPage} className="flex items-center bg-white/5 rounded-md border border-white/10 overflow-hidden h-5 px-1">
                <input type="number" value={jumpPage} onChange={(e) => setJumpPage(e.target.value)} placeholder={lang === 'ar' ? 'صفحة' : 'Page'} className="bg-transparent text-[8px] text-white w-8 outline-none placeholder:text-white/20" />
                <button type="submit" className="text-[8px] text-orange-500 font-bold ml-1">GO</button>
             </form>
             <div className="flex items-center bg-white/5 rounded-md border border-white/10 overflow-hidden h-5">
                <button onClick={() => handleZoom(-0.2)} className="px-1.5 text-white/40 hover:text-white"><svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M20 12H4" /></svg></button>
                <span className="text-[8px] font-mono text-white/40 px-1 border-x border-white/10">{Math.round(zoom * 100)}%</span>
                <button onClick={() => handleZoom(0.2)} className="px-1.5 text-white/40 hover:text-white"><svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M12 4v16m8-8H4" /></svg></button>
             </div>
             {!loading && !error && (
                <div className="flex items-center gap-1 text-[9px] text-white/40">
                  <span className="text-white font-bold">{currentPage}</span>
                  <span className="opacity-30">/</span>
                  <span>{numPages}</span>
                </div>
             )}
          </div>
        </div>
        <div className="flex-1 flex items-center justify-center px-4 bg-gradient-to-r from-transparent via-white/5 to-transparent">
          <div className="flex items-center gap-3">
            <div className="flex gap-0.5">
              {Array.from({ length: 6 }).map((_, i) => (
                <span key={i} className={`text-xs transition-all duration-500 ${i < stars ? 'scale-110 drop-shadow-[0_0_5px_rgba(255,165,0,0.8)]' : 'grayscale opacity-10 scale-90'}`}>🌟</span>
              ))}
            </div>
            <div className="h-4 w-[1px] bg-white/10"></div>
            <div className="flex flex-col justify-center">
              <div className="flex items-center gap-1.5">
                <span className="text-[9px] font-black text-orange-500 uppercase tracking-tighter">
                  {Math.floor(readingTime / 60)} {lang === 'ar' ? 'دقيقة مطالعة' : 'mins read'}
                </span>
                {next && (
                  <span className="text-[8px] font-bold text-white/40 animate-pulse">
                    • {lang === 'ar' ? `بقي ${Math.ceil(next.time - readingTime/60)}د للنجمة ${next.stars}` : `${Math.ceil(next.time - readingTime/60)}m to Star ${next.stars}`}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* منطقة عرض الصفحات */}
      <div ref={containerRef} className="flex-1 overflow-x-auto overflow-y-hidden snap-x snap-mandatory bg-black flex flex-row items-center scrollbar-none scroll-smooth relative">
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4 z-50 bg-black">
            <div className="spinner-arc w-12 h-12 border-t-orange-600"></div>
            <p className="text-[#a34a28] text-[9px] font-black uppercase tracking-[0.3em] animate-pulse">{lang === 'ar' ? 'جاري الاستحضار...' : 'Summoning...'}</p>
          </div>
        )}
        {error && (
           <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center bg-black">
             <p className="text-[10px] text-red-500/60 font-black uppercase tracking-widest">{error}</p>
           </div>
        )}
        {!loading && !error && Array.from({ length: numPages }, (_, i) => (
          <div key={i} className="w-full h-full flex-shrink-0 flex items-center justify-center snap-center overflow-auto scrollbar-none">
            <PageRenderer pdfDoc={pdfDocRef.current} pageNum={i + 1} zoom={zoom} />
          </div>
        ))}
      </div>
    </div>
  );
};

const PageRenderer: React.FC<{ pdfDoc: any, pageNum: number, zoom: number }> = ({ pdfDoc, pageNum, zoom }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderTaskRef = useRef<any>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [isRendered, setIsRendered] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setIsVisible(entry.isIntersecting), { threshold: 0.1, rootMargin: '0px 400px 0px 400px' });
    if (canvasRef.current) observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isVisible || !pdfDoc || !canvasRef.current) return;
    const renderPage = async () => {
      try {
        if (renderTaskRef.current) renderTaskRef.current.cancel();
        const page = await pdfDoc.getPage(pageNum);
        const viewport = page.getViewport({ scale: 2.0 * zoom });
        const canvas = canvasRef.current!;
        const context = canvas.getContext('2d');
        if (!context) return;
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        renderTaskRef.current = page.render({ canvasContext: context, viewport });
        await renderTaskRef.current.promise;
        setIsRendered(true);
      } catch (err: any) {
        if (err.name !== 'RenderingCancelledException') console.error(`Error rendering page ${pageNum}:`, err);
      }
    };
    renderPage();
    return () => { if (renderTaskRef.current) renderTaskRef.current.cancel(); };
  }, [isVisible, pdfDoc, pageNum, zoom]);

  return (
    <div className="relative flex items-center justify-center min-h-full min-w-full p-2 md:p-4">
      <div className="relative shadow-2xl bg-white transition-transform duration-300">
        {!isRendered && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900">
             <div className="w-6 h-6 border-2 border-white/5 border-t-white/20 rounded-full animate-spin"></div>
          </div>
        )}
        <canvas ref={canvasRef} className={`block object-contain transition-opacity duration-500 ${isRendered ? 'opacity-100' : 'opacity-0'}`} style={{ width: 'auto', height: zoom > 1.2 ? 'auto' : 'calc(100vh - 80px)', maxWidth: '100%', maxHeight: zoom > 1.2 ? 'none' : 'calc(100vh - 80px)' }} />
      </div>
    </div>
  );
};

export default ManuscriptViewer;
