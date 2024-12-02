<template>
  <div>
    <h2 class="text-2xl font-bold mb-4">Train Model</h2>

    <!-- Drag-and-Drop Zone -->
    <div
      @dragover.prevent="handleDragOver"
      @dragenter.prevent="handleDragEnter"
      @dragleave="handleDragLeave"
      @drop.prevent="handleFileDrop"
      :class="[
        'w-full p-6 border-dashed border-2 rounded-lg flex items-center justify-center mb-4 transition-all duration-300',
        isDragging ? 'bg-blue-100 border-blue-500' : 'bg-gray-50 border-gray-300'
      ]"
    >
      <p class="text-gray-500 text-center">
        Drag and drop .txt files here, or click to select files.
      </p>
      <input
        ref="fileInput"
        type="file"
        accept=".txt"
        multiple
        @change="handleFileSelect"
        class="hidden"
      />
    </div>

    <!-- Uploaded Files Display -->
    <div v-if="files.length" class="flex flex-wrap gap-4 mb-4">
      <div
        v-for="(file, index) in files"
        :key="index"
        class="flex items-center p-2 border border-gray-300 rounded-lg bg-white shadow-md"
      >
        <!-- File Icon -->
        <div class="mr-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            class="h-6 w-6 text-gray-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 16v1a2 2 0 002 2h10a2 2 0 002-2v-1m-6 0v4m0-10v4M4 8h16"
            />
          </svg>
        </div>
        <!-- File Name -->
        <p class="text-gray-800 flex-1 truncate">{{ file.name }}</p>
        <!-- Delete Button -->
        <button
          @click="removeFile(index)"
          class="ml-2 text-red-500 hover:text-red-700"
        >
          ✖
        </button>
      </div>
    </div>

    <!-- Upload Button -->
    <div>
      <button
        @click="uploadFiles"
        :disabled="!files.length"
        class="bg-blue-500 text-white p-3 rounded-lg"
      >
        Upload & Train Model
      </button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      files: [], // Store multiple files
      isDragging: false, // Dragging state for animation
    };
  },
  methods: {
    handleDragOver() {
      // Prevent default browser behavior
    },
    handleDragEnter() {
      this.isDragging = true;
    },
    handleDragLeave() {
      this.isDragging = false;
    },
    handleFileDrop(event) {
      this.isDragging = false; // Reset dragging state

      const droppedFiles = Array.from(event.dataTransfer.files);
      const validFiles = droppedFiles.filter((file) => file.type === "text/plain");

      if (validFiles.length) {
        this.files = [...this.files, ...validFiles];
      } else {
        alert("Please upload valid .txt files.");
      }
    },
    handleFileSelect(event) {
      const selectedFiles = Array.from(event.target.files);
      const validFiles = selectedFiles.filter((file) => file.type === "text/plain");

      if (validFiles.length) {
        this.files = [...this.files, ...validFiles];
      } else {
        alert("Please upload valid .txt files.");
      }
    },
    removeFile(index) {
      this.files.splice(index, 1); // Remove file from the list
    },
    async uploadFiles() {
      if (!this.files.length) {
        alert("No files selected!");
        return;
      }

      const formData = new FormData();
      this.files.forEach((file) => formData.append("files", file));

      console.log(`${useRuntimeConfig().public.apiBase}/train-model`);
      // Send the files to the backend
      const response = await $fetch(`${useRuntimeConfig().public.apiBase}/train-model`, {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          this.files = []; // Clear the file list after successful upload
          alert(response.message);
        })
        .catch((error) => {
          console.error("Error uploading files:", error);
          alert("An error occurred while uploading the files.");
        });
    },
  },
};
</script>

<style scoped>
/* Add animations for the drag-and-drop zone */
.w-full {
  transition: background-color 0.3s ease, border-color 0.3s ease;
}
.flex-wrap {
  display: flex;
  flex-wrap: wrap;
}
</style>
