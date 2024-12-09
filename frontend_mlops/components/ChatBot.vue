<template>
  <div class="h-[75vh] flex bg-gray-50 p-3 overflow-hidden">
    <!-- Bot Selection List -->
    <div class="w-1/4 bg-white shadow-lg rounded-lg mr-6 p-4 overflow-y-auto">
      <h2 class="text-lg font-bold mb-4">Select a Bot</h2>
      <ul>
        <li v-for="bot in bots" :key="bot.id" @click="selectBot(bot.id)"
          :class="{ 'bg-blue-100': selectedBot === bot.id }"
          class="p-3 rounded-lg cursor-pointer hover:bg-blue-200 transition mb-2">
          {{ bot.name }}
        </li>
      </ul>
    </div>

    <!-- Chatbot Container -->
    <div class="w-3/4 bg-white shadow-lg rounded-lg flex flex-col">
      <!-- Chat Header -->
      <header class="bg-blue-500 text-white p-4 rounded-t-lg flex items-center">
        <img src="../assets/images/technical-support.png" alt="Bot Icon" class="w-10 h-10 rounded-full mr-3" />
        <h1 class="text-lg font-bold">{{ selectedBotName }}</h1>
      </header>

      <!-- Chat History -->
      <div class="p-4 overflow-y-auto flex-grow">
        <div v-for="(message, index) in currentMessages" :key="index" class="mb-3">
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

      <!-- Input Area -->
      <div class="flex items-center p-3 border-t border-gray-200">
        <input v-model="userMessage" @keydown.enter="sendMessage" type="text" placeholder="Type your message..."
          class="flex-grow p-3 border border-gray-300 rounded-lg mr-2 focus:outline-none focus:ring-2 focus:ring-blue-500" />
        <button @click="sendMessage" class="bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-600 transition">
          Send
        </button>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, computed } from 'vue';

const bots = ref([
  { id: 'rag', name: 'RAG Bot' },
  { id: 'llm', name: 'LLM Bot' }
]);
const selectedBot = ref('rag');
const userMessage = ref('');
const messages = ref({
  rag: [],
  llm: []
});

const currentMessages = computed(() => messages.value[selectedBot.value]);

const selectedBotName = computed(() => {
  const bot = bots.value.find(bot => bot.id === selectedBot.value);
  return bot ? bot.name : 'ChatBot';
});

function selectBot(botId) {
  selectedBot.value = botId;
}

function doNothing(){
  console.log("do nothing");
}

async function sendMessage() {
  if (userMessage.value.trim() !== '') {
    const newMessage = { text: userMessage.value, sender: 'user' };
    messages.value[selectedBot.value].push(newMessage);
    userMessage.value = '';

    try {
      // Use $fetch to send the message to the backend API
      const response = await $fetch(`${useRuntimeConfig().public.apiBase}/messages`, {
        method: 'POST',
        body: {
          sender: 'user',
          text: newMessage.text
        }
      });

      // Append the response from the backend to the chat
      messages.value[selectedBot.value].push({ text: response.text, sender: 'ai' });
    } catch (error) {
      console.error("Error sending message:", error);
      messages.value[selectedBot.value].push({ text: "Error processing message.", sender: 'ai' });
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
