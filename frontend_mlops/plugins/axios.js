// plugins/axios.js
import axios from 'axios';

export default defineNuxtPlugin((nuxtApp) => {
  const axiosInstance = axios.create({
    baseURL: 'http://localhost:8000/api/v1/'
  });

  // Optionally, you can add interceptors for requests or responses
  axiosInstance.interceptors.response.use(
    response => response,
    error => {
      return Promise.reject(error);
    }
  );

  nuxtApp.provide('axios', axiosInstance);
});
