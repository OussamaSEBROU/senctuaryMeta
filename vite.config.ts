
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // تحميل متغيرات البيئة من النظام (Render)
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [react()],
    define: {
      // تمرير API_KEY فقط بدلاً من كامل process.env لتجنب أخطاء البناء
      'process.env.GROQ_API_KEY': JSON.stringify(env.GROQ_API_KEY)
    },
    build: {
      outDir: 'dist',
      sourcemap: false
    }
  };
});
