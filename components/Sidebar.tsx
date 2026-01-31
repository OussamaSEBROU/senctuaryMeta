
import React from 'react';
import { Language } from '../types';
import { translations } from '../translations';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  lang: Language;
  setLang: (l: Language) => void;
  onNewChat: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose, lang, setLang, onNewChat }) => {
  const t = translations[lang];

  return (
    <>
      <div 
        className={`fixed inset-0 bg-black/90 backdrop-blur-md z-[60] transition-opacity duration-300 ${isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
        onClick={onClose}
      />
      <aside className={`fixed top-0 bottom-0 ${lang === 'ar' ? 'right-0' : 'left-0'} w-[300px] bg-[#050505] border-${lang === 'ar' ? 'l' : 'r'} border-white/5 z-[70] transition-transform duration-500 cubic-bezier(0.4, 0, 0.2, 1) transform ${isOpen ? 'translate-x-0' : (lang === 'ar' ? 'translate-x-full' : '-translate-x-full')} p-8 flex flex-col shadow-2xl`}>
        <div className="flex justify-between items-center mb-12">
          <h2 className="font-black tracking-[0.4em] text-[10px] uppercase text-white/60">{t.title}</h2>
          <button onClick={onClose} className="text-white/20 hover:text-white transition-colors text-xl">✕</button>
        </div>

        <nav className="flex-1 space-y-10 overflow-y-auto scrollbar-none">
          <section>
            <button 
              onClick={() => { onNewChat(); onClose(); }}
              className="w-full flex items-center gap-3 p-4 bg-white/5 border border-white/10 rounded-2xl hover:bg-white/10 transition-all"
            >
              <div className="w-6 h-6 rounded-full bg-orange-600/20 flex items-center justify-center text-orange-500">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M12 4v16m8-8H4" /></svg>
              </div>
              <span className="text-[10px] font-black tracking-widest uppercase text-white/80">{t.newSanctuary}</span>
            </button>
          </section>

          <section>
            <p className="text-[9px] font-black tracking-[0.4em] text-white/20 uppercase mb-4">{t.language}</p>
            <div className="grid grid-cols-2 gap-2 bg-white/5 p-1 rounded-xl">
              <button onClick={() => setLang('en')} className={`py-2 text-[9px] font-black rounded-lg transition-all ${lang === 'en' ? 'bg-[#a34a28] text-white' : 'text-white/30'}`}>ENGLISH</button>
              <button onClick={() => setLang('ar')} className={`py-2 text-[9px] font-black rounded-lg transition-all ${lang === 'ar' ? 'bg-[#a34a28] text-white' : 'text-white/30'}`}>العربية</button>
            </div>
          </section>

          <section className="space-y-6">
            <div>
              <p className="text-[9px] font-black tracking-[0.4em] text-white/20 uppercase mb-2">{t.about}</p>
              <p className="text-[10px] text-white/40 leading-relaxed italic">{t.aboutText}</p>
            </div>
            <div>
              <p className="text-[9px] font-black tracking-[0.4em] text-white/20 uppercase mb-2">{t.help}</p>
              <p className="text-[10px] text-white/40 leading-relaxed">{t.helpText}</p>
            </div>
          </section>
        </nav>

        <div className="pt-8 border-t border-white/5 mt-auto">
           <span className="text-[8px] text-white/10 font-black tracking-widest uppercase block">Knowledge AI Core: V3.5 Neural</span>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
