<template>
  <div class="h-[75vh] flex bg-gray-50 p-3 overflow-hidden">
    <!-- Chat Comparisons Section -->
    <div class="w-1/2 bg-white shadow-lg rounded-lg flex flex-col mr-3">
      <!-- Chat RAG Header -->
      <header class="bg-blue-500 text-white p-4 rounded-t-lg flex items-center">
        <img src="../assets/images/technical-support.png" alt="RAG Icon" class="w-10 h-10 rounded-full mr-3" />
        <h1 class="text-lg font-bold">RAG Bot</h1>
      </header>

      <!-- RAG Chat History -->
      <div class="p-4 overflow-y-auto flex-grow">
        <div v-for="(message, index) in messages.rag" :key="index" class="mb-3">
          <!-- User Message -->
          <div v-if="message.sender === 'user'" class="flex justify-end">
            <div class="bg-blue-500 text-white p-3 rounded-lg max-w-sm">
              {{ message.text }}
            </div>
          </div>

          <!-- Bot Message -->
          <div v-else class="flex justify-start">
            <div class="bg-gray-200 text-gray-800 p-3 rounded-lg max-w-sm">
              {{ message.text }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="w-1/2 bg-white shadow-lg rounded-lg flex flex-col">
      <!-- Chat LORA Header -->
      <header class="bg-green-500 text-white p-4 rounded-t-lg flex items-center">
        <img src="../assets/images/technical-support.png" alt="LORA Icon" class="w-10 h-10 rounded-full mr-3" />
        <h1 class="text-lg font-bold">LORA Bot</h1>
      </header>

      <div class="p-4 overflow-y-auto flex-grow">
        <div v-for="(message, index) in messages.lora" :key="index" class="mb-3">
          <!-- User Message -->
          <div v-if="message.sender === 'user'" class="flex justify-end">
            <div class="bg-green-500 text-white p-3 rounded-lg max-w-sm">
              {{ message.text }}
            </div>
          </div>

          <div v-else class="flex justify-start">
            <div class="bg-gray-200 text-gray-800 p-3 rounded-lg max-w-sm">
              {{ message.text }}
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Chat Input Bar -->
    <div class="flex items-center p-3 border-t border-gray-200 w-full absolute bottom-0 left-0 bg-gray-50">
      <input
          v-model="userMessage"
          @keydown.enter="sendMessage"
          type="text"
          placeholder="Type your message..."
          class="flex-grow p-3 border border-gray-300 rounded-lg mr-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <button
          @click="sendMessage"
          class="bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-600 transition"
      >
        Send
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";

const userMessage = ref("");
const messages = ref({
  rag: [],
  lora: [],
});

async function sendMessage() {
  if (userMessage.value.trim() !== "") {
    const newMessage = { text: userMessage.value, sender: "user" };

    // Ajouter un message utilisateur avec RAg et Lora
    messages.value.rag.push(newMessage);
    messages.value.lora.push(newMessage);

    userMessage.value = "";
    try {
      // Envoyer un message au llm avec le sys RAG
      const ragResponse = await $fetch(`${useRuntimeConfig().public.ragENV}/generate`, {
        method: "POST",
        body: {
          prompt: newMessage.text,
          "max_length": 100,
          "temperature": 0.9
        },
      });
      messages.value.rag.push({ text: ragResponse.generated_text, sender: "ai" });

      // Envoyer un message au llm LORA
      const loraResponse = await $fetch(`${useRuntimeConfig().public.loraENV}/generate`, {
        method: "POST",
        body: {
          "prompt": newMessage.text,
          "max_length": 100,
          "temperature": 0.9
        },
      });
      messages.value.lora.push({ text: loraResponse.generated_text, sender: "ai" });
    } catch (error) {
      console.error("Error sending message:", error);
      messages.value.rag.push({ text: "Error processing message (RAG bot).", sender: "ai" });
      messages.value.lora.push({ text: "Error processing message (LORA bot).", sender: "ai" });
    }
  }
}
</script>

<style scoped>
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-thumb {
  background-color: #cbd5e0;
  border-radius: 4px;
}

::-webkit-scrollbar-track {
  background-color: #f7fafc;
}
</style>