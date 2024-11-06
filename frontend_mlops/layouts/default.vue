<template>
    <div class="min-h-screen flex flex-col">
      <!-- Header Section -->
      <header class="bg-blue-500 text-white p-6">
        <AppHeader />
        <p class="text-xl mt-2"><strong>MLOPS - ChatBot - RAG vs LLM LoRa</strong></p>
      </header>
  
      <!-- Main Content Section -->
      <main class="flex-grow p-6">
        <!-- Slot for page content -->
        <slot />
      </main>
  
      <!-- Footer Section -->
      <footer class="bg-gray-800 text-white p-6">
        <AppFooter />
        <p class="text-xl mt-2">Mozerski Alex, Terceiro Ruben, Ruffieux Killian, Rominger Victor, Ribeiro Cabral Daniel</p>
      </footer>
    </div>
  </template>
  