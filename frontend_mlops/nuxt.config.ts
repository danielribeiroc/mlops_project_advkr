export default defineNuxtConfig({
  modules: ['@nuxtjs/tailwindcss'],
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || "http://localhost:8000/api/v1",
    },
  },
  compatibilityDate: '2024-11-06',
});

console.log("NUXT_PUBLIC_API_BASE:", process.env.NUXT_PUBLIC_API_BASE);
