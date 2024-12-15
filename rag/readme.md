# Retrieval-Augmented Generation System (RAG)

A Retrieval-Augmented Generation (RAG) system designed to provide intelligent and context-aware responses by combining document retrieval and text generation capabilities.

## Features
- **Document Retrieval:** System retrieval of relevant documents based on user queries using FAISS and HuggingFace embeddings (model *all-MiniLM-L6-v2*).
- **Text Generation:** Natural language responses generated using the model *meta-llama/Llama-3.2-1B*. 
- **Google Cloud Integration:** Supports cloud storage for data and vectorstore management.
- **Customizable Workflow:** Easily extendable to include new data and models.

---

## Project Structure
```plaintext
rag/
├── .env                             # Environment variables (not included in the repository, must be added manually)
├── config/
│   └── google-credentials.json      # Google Cloud credentials (not included in the repository)
├── data/                            # Contains sample text files for document retrieval
├── faiss_index/                     # Stores the FAISS index and vectorstore files
├── models/
│   ├── generator.py                 # Defines the text generation logic
│   ├── retriever.py                 # Handles document retrieval using FAISS and embeddings
│   └── rag_system.py                # Combines the generator and retriever into the RAG system
├── rag_service.py                   # Main BentoML service definition
├── bentofile.yaml                   # BentoML configuration file
├── requirements.txt                 # Python dependencies
```

## Prerequisites

1. **Python Environment**
   - Ensure Python 3.11 is installed.
   - Use a virtual environment for dependency management.

2. **Google Cloud Setup**
   - Create a Google Cloud project.
   - Generate a service account and download the credentials file (`google-credentials.json`).

3. **HuggingFace Setup**
   - Create a HuggingFace account.
   - Obtain a token from [HuggingFace](https://huggingface.co/settings/tokens).

4. **Environment Variables**
   - Add a `.env` file at the root with the following variables:
     ```plaintext
     HUGGING_FACE_TOKEN=<your-huggingface-token>
     EMBEDDING_MODEL=<your-embedding-model-name>
     GENERATOR_MODEL=<your-generator-model-name>
     GC_ACCESS_CREDENTIALS=<path-to-google-credentials>
     GC_BUCKET_NAME=<your-google-cloud-bucket-name>
     ```

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up FAISS Index**
   - If a pre-existing index is not available, the retriever will build one on the first run using data from the `data/` directory.

---

## Local Deployment

1. **Run the Service**
   ```bash
   bentoml serve rag_service.py:RAGService --port 3002
   ```
   - Access the service at `http://localhost:3002`.

2. **API Endpoints**
   - `POST /ask`: Query the RAG system.
   - `POST /add_document`: Add a single document to the retriever.
   - `POST /add_documents`: Add multiple documents to the retriever.

3. **Sample Queries**
   - Ask a question:
     ```bash
     curl -X POST "http://localhost:3002/ask" \
          -H "Content-Type: application/json" \
          -d '{"query": "What is the role of employee X?"}'
     ```
   - Add a document:
     ```bash
     curl -X POST "http://localhost:3002/add_document" \
          -H "Content-Type: application/json" \
          -d '{"filepath": "path/to/your/document.txt"}'
     ```

---

## Advanced Deployment with Docker

1. **Build a BentoML Bundle**
   ```bash
   bentoml build
   ```

2. **Containerize the Service**
   ```bash
   bentoml containerize <service-name>:<version-tag>
   ```

3. **Run the Docker Container**
   ```bash
   docker run -p 3002:3002 <container-tag>
   ```

---

## Notes

- **Security**
  - `.env` and `google-credentials.json` are not included in the repository for security reasons.
  - Use secure methods to manage sensitive credentials.

- **Customizing Models**
  - Update the `.env` file to point to different embedding or generation models.
