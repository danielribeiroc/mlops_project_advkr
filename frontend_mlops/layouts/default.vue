<template>
  <div class="min-h-screen flex flex-col">
    <!-- Header Section -->
    <header class="bg-blue-500 text-white p-6 flex items-center justify-between">
      <!-- Project Name -->
      <p class="text-xl font-bold">MLOPS - ChatBot - RAG vs LLM LoRa</p>
      
      <!-- Tab Navigation -->
      <nav class="space-x-6">
        <NuxtLink 
          to="/" 
          class="text-gray-200 font-semibold p-3 rounded-lg transition-all hover:bg-indigo-600 hover:text-white hover:shadow-lg active:bg-indigo-800 active:text-white"
          active-class="bg-indigo-800 text-white shadow-lg"
        >
          ChatBot
        </NuxtLink>
        <NuxtLink 
          to="/train-params" 
          class="text-gray-200 font-semibold p-3 rounded-lg transition-all hover:bg-indigo-600 hover:text-white hover:shadow-lg active:bg-indigo-800 active:text-white"
          active-class="bg-indigo-800 text-white shadow-lg"
        >
          Train Model
        </NuxtLink>
      </nav>
    </header>

    <!-- Main Content Section -->
    <main class="flex-grow p-6">
      <!-- Slot for page content -->
      <slot />
    </main>

    <!-- Footer Section -->
    <footer class="bg-gray-800 text-white p-6">
      <p class="text-xl mt-2">Mozerski Alex, Terceiro Ruben, Ruffieux Killian, Rominger Victor, Ribeiro Cabral Daniel</p>
    </footer>
  </div>
</template>

<script>
export default {
  components: {}
};
</script>
