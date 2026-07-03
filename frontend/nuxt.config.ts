export default defineNuxtConfig({
  compatibilityDate: "2025-01-01",
  runtimeConfig: {
    public: {
      apiBase: process.env.API_BASE || "http://localhost:8080",
    },
  },
  css: ["~/assets/css/main.css"],
  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },
  modules: ["@nuxtjs/tailwindcss"],
})
