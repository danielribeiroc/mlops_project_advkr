# Documentation to Run the FastAPI Backend

## Prerequisites
### 1. Install the Necessary Tools:
- **Python 3.10 or later**: Ensure Python is installed on your machine.
- **Pip**: Managed automatically when installing Python.
- **Docker**: If you wish to run the project with Docker, download and install Docker.

## Step 1: Run Locally with Python
1. **Clone the Project:**
Clone this repository to your local machine:
```bash
   git clone <REPO-URL>
   cd <PROJECT-DIRECTORY-NAME>
```

2. **Set Up a Virtual Environment:**
It's recommended to use a virtual environment to manage dependencies. Here are the commands to create and activate a virtual environment:
On Windows:
```bash
   python -m venv venv
   venv\Scripts\activate
```
On macOS/Linux:
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. **Install Dependencies:**
Once the virtual environment is activated, install the dependencies from the `requirements.txt` file:
```bash
   pip install -r requirements.txt
```

4. **Start the Application:**
Use the following command to start the application with Uvicorn (included in the dependencies):
```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
```

5. **Access the API:**
Once the application is running, you can access the API at the following address: [http://127.0.0.1:8000]()

### Example Request:
```bash
curl -X POST http://localhost:8000/api/v1/messages-rag \
-H "Content-Type: application/json" \
-d '{
    "sender": "user",
    "text": "hello"
}'
```

## Step 2: Run with Docker
1. **Clone the Project:**
Clone this repository as instructed in the previous step and navigate to the project folder:
```bash
   git clone <REPO-URL>
   cd <PROJECT-DIRECTORY-NAME>
```

2. **Build the Docker Image:**
Use the following command to build a Docker image from the `Dockerfile` included in the project:
```bash
   docker build -t backend .
```
Replace `backend` with a descriptive name for your image.

3. **Start the Docker Container:**
Once the image is built, start a container based on that image:
```bash
   docker run -p 8000:8000 backend
```
- The `-p 8000:8000` flag maps port 8000 on your local machine to port 8000 in the Docker container.
- Ensure port 8000 is free on your machine.

4. **Access the API:**
Access the API at the following address: [http://127.0.0.1:8000]()
