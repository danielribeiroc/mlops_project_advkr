export default defineNuxtConfig({
  modules: ['@nuxtjs/tailwindcss'],
  runtimeConfig: {
    public: {
      apiBase: 'http://localhost:8000/api/v1'
    }
  },
  compatibilityDate: '2024-11-06'
})