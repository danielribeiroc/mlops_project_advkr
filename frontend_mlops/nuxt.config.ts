export default defineNuxtConfig({
  modules: ['@nuxtjs/tailwindcss'],
  runtimeConfig: {
    public: {
      ragENV: process.env.RAG_ENV || "http://localhost:8000/api/v1",
      loraENV: process.env.LORA_ENV || "http://localhost:3002",
    },
  },
  compatibilityDate: '2024-11-06',
});

console.log("ragENV:", process.env.RAG_ENV);
console.log("loraENV:", process.env.LORA_ENV);
