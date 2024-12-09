# Use the official Node.js image as the base image
FROM node:18 AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the package.json and package-lock.json (if you have it)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the entire project into the container
COPY . .

# Build the Nuxt.js application for production
RUN npm run build

# Use a smaller production image for the final container
FROM node:18-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=build /app /app

# Expose the port the app will run on
EXPOSE 3000

# Command to run the Nuxt.js app
CMD [ "npm", "run", "start" ]