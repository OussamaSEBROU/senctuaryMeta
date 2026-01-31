 
     

export const translations = {
  en: {
    title: "KNOWLEDGE AI",
    sanctuary: "Sanctuary",
    introText: "Neural extraction of axiomatic wisdom from deep manuscripts",
    upload: "Upload Manuscript",
    uploadDesc: "PDF manuscripts only",
    newSanctuary: "New Sanctuary",
    axiomsTitle: "Knowledge Axioms",
    dialogue: "The Dialogue",
    dialogueDesc: "Interrogate the text's deeper essence through the neural sanctuary.",
    language: "Language",
    viewer: "Manuscript Viewer",
    synthesis: "Neural Synthesis...",
    covenant: "The Sanctuary Covenant: Direct reading and personal comprehension are the only paths to wisdom.",
    about: "About Us",
    help: "Help",
    aboutText: "We are an elite neural research collective dedicated to the extraction of axiomatic truths from historical manuscripts.",
    helpText: "Upload a PDF manuscript to begin. The system will extract core axioms, which you can then explore through a deep neural dialogue.",
    placeholder: "Interrogate the author's logic...",
    deepChatBtn: "Deep Knowledge Chat",
    extractionPrompt: (lang: string) => `Extract 6 core 'Knowledge Axioms' from this manuscript. The entire response must be in Arabic. Each axiom should include a profound term, a scholarly definition, and its overarching significance.`
  },
  ar: {
    title: "KNOWLEDGE AI",
    sanctuary: "Sanctuary",
    introText: "إنّ الباحث عن المعرفة ينبغي أن يعي (منطق العمل) كما يعي (منطق الفكرة) فهذا سبيله نحو الحِكمّة",
    upload: "رفع المخطوط",
    uploadDesc: "وثائق PDF فقط",
    newSanctuary: "محراب جديد",
    axiomsTitle:  "مقاصد المخطُوط",
    dialogue:"بُنيّة الفكرة",
    dialogueDesc: "سبر أغوار الجوهر المعرفي للمخطوط عبر المحراب",
    language: "اللغة",
    viewer: "المخطوط",
    synthesis: "توليف معرفي جاري...",
    covenant: "ميثاق المحراب: القراءة المباشرة والمطالعة المعمقة من بطون الكتب ومُعاينة المعرفة هي السبيل الوحيد للوصول الى الفهم و الحكمة.",
    about: "عن المنظومة",
    help: "الإرشاد المعرفي",
    aboutText: "نحن منظومة بحثية نخبوية مكرسة لاستخراج الحقائق البديهية من المخطوطات التاريخية باستخدام الذكاء الاصطناعي.",
    helpText: "قم برفع المخطوط بصيغة PDF للبدء. سيقوم النظام باستخراج البديهيات الأساسية، والتي يمكنك استكشافها بعمق من خلال حوار معرفي.",
    placeholder: "استجوب منطق المؤلف وعمقه...",
    deepChatBtn: "الغوص في الحوار المعرفي",
    extractionPrompt: (lang: string) => `استخرج 6 'بديهيات معرفية' أساسية من هذه المخطوطة. يجب أن تكون الاستجابة كاملة باللغة العربية. يجب أن تتضمن كل بديهية مصطلحاً عميقاً، تعريفاً أكاديمياً، وأهميتها الشاملة.`
  }
};
