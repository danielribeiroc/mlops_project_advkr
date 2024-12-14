# Documentation

## 1. **Prerequisites**

Ensure you have the following tools installed:

- **Node.js** (v16 or later)
- **npm** (comes with Node.js) or **Yarn** (optional)
- **Docker** and **Docker Compose** (if running in Docker)

---

## 2. **Setting Up the Project Locally**

### **Step 1: Clone the Repository**
```bash
# Clone the project repository
git clone <repository_url>

# Navigate into the project directory
cd <project_directory>
```

### **Step 2: Install Dependencies**
```bash
# Install dependencies
npm install
```

### **Step 3: Create and Configure the `.env` File**
Create a `.env` file in the root directory. This file contains the environment variables required for the project.

#### Example `.env` File:
```env
NUXT_PUBLIC_API_BASE=http://localhost:8000/api/v1
```

> **Note:** Replace the `NUXT_PUBLIC_API_BASE` value with your API base URL if different.

### **Step 4: Start the Development Server**
Run the following command to start the server in development mode:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

#### **Debugging Environment Variables**
To debug environment variables, you can print them in the `nuxt.config.ts`:
```ts
console.log("Environment Variables:", process.env);
```
Restart the server after making changes to the `.env` file.

---

## 3. **Setting Up the Project with Docker**

### **Step 1: Create a `.env` File**
Create a `.env` file in the root directory with the necessary variables (as described earlier).

### **Step 2: Build and Run the Container**
Run the following commands to build and start the container:

```bash
# Build the Docker image
docker build -t nuxt-app .

# Run the container
docker run --rm -p 3000:3000 --env-file .env nuxt-app
```

The application will be accessible at `http://localhost:3000`.

---

## 5. **Key Commands**

- **Install dependencies:** `npm install`
- **Run development server:** `npm run dev`
- **Build production assets:** `npm run build`
- **Start production server:** `npm run start`
- **Run in Docker:** `docker run --rm -p 3000:3000 --env-file .env nuxt-app`
- **Use Docker Compose:** `docker-compose up --build`

---

### **Environment Variables Not Loading**
- Ensure the `.env` file exists and is in the root directory.
- Ensure variables are prefixed with `NUXT_PUBLIC_` for public access.
- Restart the server after modifying the `.env` file.


