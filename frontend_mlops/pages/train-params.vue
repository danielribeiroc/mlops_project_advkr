<template>
    <div>
      <h2 class="text-2xl font-bold mb-4">Train Parameters</h2>
      
      <!-- Form to Add Employee Data -->
      <div class="mb-6">
        <input 
          v-model="employeeName" 
          type="text" 
          placeholder="Employee Name" 
          class="w-full p-3 border border-gray-300 rounded-lg mb-2"
        />
        <textarea 
          v-model="employeeData" 
          placeholder="Enter data related to the employee..." 
          class="w-full p-3 border border-gray-300 rounded-lg mb-2"
        ></textarea>
        <button @click="addEmployeeData" class="bg-blue-500 text-white p-3 rounded-lg">Add</button>
      </div>
  
      <!-- Display Added Employee Data -->
      <div v-if="employees.length" class="mt-4">
        <h3 class="text-xl font-semibold mb-2">Employee Data List</h3>
        <ul>
          <li v-for="(employee, index) in employees" :key="index" class="mb-2">
            <strong>{{ employee.name }}</strong>: {{ employee.data }}
          </li>
        </ul>
      </div>
  
      <!-- Button to Trigger Training -->
      <div class="mt-6">
        <button @click="trainModel" class="bg-green-500 text-white p-3 rounded-lg">Train Model</button>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        employeeName: '',
        employeeData: '',
        employees: []
      };
    },
    methods: {
      addEmployeeData() {
        if (this.employeeName && this.employeeData) {
          this.employees.push({ name: this.employeeName, data: this.employeeData });
          this.employeeName = '';
          this.employeeData = '';
        } else {
          alert("Please fill in both the name and data fields.");
        }
      },
      trainModel() {
        // Logic to send employee data for training
        alert("Training model with the provided employee data...");
      }
    }
  };
  </script>
  