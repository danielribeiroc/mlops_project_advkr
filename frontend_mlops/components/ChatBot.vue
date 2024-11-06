<template>
    <div class="flex flex-col items-center p-6">
      <!-- Chatbot Header -->
      <h1 class="text-2xl font-bold mb-4">Chatbot</h1>
  
      <!-- Chat History (User's questions and AI responses) -->
      <div class="w-full max-w-lg bg-gray-100 p-4 rounded-lg overflow-y-auto mb-4" style="height: 300px;">
        <div v-for="(message, index) in messages" :key="index" class="mb-2">
          <div :class="{'text-right': message.sender === 'user', 'text-left': message.sender === 'ai'}">
            <p :class="{'bg-blue-500 text-white p-2 rounded-lg': message.sender === 'user', 'bg-gray-300 p-2 rounded-lg': message.sender === 'ai'}">
              {{ message.text }}
            </p>
          </div>
        </div>
      </div>
  
      <!-- Input Area -->
      <div class="w-full max-w-lg flex">
        <input 
          v-model="userMessage"
          @keydown.enter="sendMessage"
          type="text"
          placeholder="Type your message..."
          class="w-full p-2 border border-gray-300 rounded-lg mr-2"
        />
        <button @click="sendMessage" class="bg-blue-500 text-white p-2 rounded-lg">Send</button>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        userMessage: '', // The message typed by the user
        messages: [ // Array to store the conversation history
          { sender: 'ai', text: 'Hello! How can I assist you today?' }
        ]
      }
    },
    methods: {
      sendMessage() {
        if (this.userMessage.trim() !== '') {
          // Add user's message to the conversation
          this.messages.push({ sender: 'user', text: this.userMessage });
  
          // Generate AI response (for simplicity, just echo the message here)
          this.messages.push({ sender: 'ai', text: `You said: "${this.userMessage}"` });
  
          // Clear the input field after sending
          this.userMessage = '';
        }
      }
    }
  }
  </script>
  
  <style scoped>
  /* TailwindCSS classes are used here for layout and styling */
  </style>
  