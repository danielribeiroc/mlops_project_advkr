# Documentation to Run the Project with Docker Compose

This guide will help you launch both the frontend and backend using Docker Compose.

## Prerequisites
- **Docker**: Make sure Docker and Docker Compose are installed on your machine.

## Step 1: Clone the Project
Clone the repository to your local machine:
```bash
git clone <REPO-URL>
cd <PROJECT-DIRECTORY-NAME>
```

## Step 2: Build and Start the Containers with Docker Compose
1. **Build and Start Containers:**
In the project directory, run the following command to build and start both the frontend and backend containers:
```bash
docker-compose up --build
```
This command will:
- Build the Docker images for both the frontend and backend.
- Start the containers for both services.

2. **Access the Application:**
Once the containers are running, you can access the application:
- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend**: [http://localhost:8000](http://localhost:8000)

## Step 3: Stop the Containers
To stop the containers, simply run:
```bash
docker-compose down
```

This will stop and remove the containers.

# Important point

The two Docker containers in the Docker Compose setup function locally but currently return the same response for all queries. This limitation arises from challenges encountered during the deployment of BentoML in the cloud. As a result, the routes are functional only with the backend running locally and provide static responses.

However, the application has been tested successfully with BentoML running locally, where the responses are displayed correctly.