
import React, { useState } from 'react';
import { Axiom } from '../types';

interface AxiomCardProps {
  axiom: Axiom;
  index: number;
}

const AxiomCard: React.FC<AxiomCardProps> = ({ axiom, index }) => {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div 
      className="perspective w-full h-[280px] md:h-[340px] cursor-pointer group"
      onClick={() => setIsFlipped(!isFlipped)}
    >
      <div className={`relative w-full h-full preserve-3d duration-700 ${isFlipped ? 'rotate-y-180' : ''}`}>
        {/* Front */}
        <div className="absolute inset-0 backface-hidden glass rounded-[2rem] p-6 md:p-8 flex flex-col items-center justify-center text-center border-white/10 group-hover:border-indigo-500/40 transition-all bg-white/[0.02] shadow-2xl">
          <span className="text-[8px] font-black tracking-[0.4em] text-indigo-400/40 uppercase mb-4">Axiom {index + 1}</span>
          <h3 className="text-lg md:text-2xl font-black text-white leading-tight tracking-tight px-4 transition-all group-hover:scale-105">
            {axiom.term}
          </h3>
          <div className="w-5 h-[1px] bg-white/10 mt-6 group-hover:w-10 group-hover:bg-indigo-500/40 transition-all duration-500"></div>
          <p className="mt-6 text-[7px] text-white/20 font-black uppercase tracking-[0.3em] opacity-0 group-hover:opacity-100 transition-opacity">Touch to Reveal</p>
        </div>

        {/* Back */}
        <div className="absolute inset-0 backface-hidden rotate-y-180 glass rounded-[2rem] p-6 md:p-8 flex flex-col justify-between border-white/20 bg-[#050505] shadow-2xl">
          <div className="overflow-y-auto scrollbar-none">
            <h4 className="text-[8px] font-black tracking-[0.4em] text-indigo-400 uppercase mb-3 border-b border-white/5 pb-2">Synthesis</h4>
            <p className="text-white font-serif text-sm md:text-lg leading-relaxed italic pr-2">
              {axiom.definition}
            </p>
          </div>
          <div className="pt-4 mt-2 border-t border-white/5">
            <p className="text-[7px] uppercase tracking-[0.2em] text-white/30 font-black mb-1">Significance</p>
            <p className="text-[10px] text-white/50 leading-relaxed font-sans line-clamp-2">
              {axiom.significance}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AxiomCard;
