# README: BentoML Text Generation Service

This guide explains the workflow for setting up and deploying a text generation service using BentoML with a fine-tuned LLaMA model.

Module : MLOps
Authors: alex.mozerski, daniel.ribeirocabral, victor.rominger, killian.ruffieux, ruben.terceiro
Date: 05.12.2024
---

## **Workflow Overview**
1. **Model Saving (`bento_save_model.py`)**: Save your fine-tuned model and tokenizer to the BentoML model store.
2. **Service Definition (`service.py`)**: Define the API and text generation logic.
3. **Configuration (`bentofile.yaml`)**: Specify dependencies and packaging details.
4. **Run/Deploy Service**: Serve locally or deploy via Docker.
5. **Client Interaction (`client.py`)**: Send requests to the service.

---

## **Steps**

### 1. Save the Model
Run `bento_save_model.py` to save the fine-tuned LLaMA model and tokenizer.

```bash
python bento_save_model.py
```

- **Saved Model**: `llama_generation`
- **Commands**:
  - List Models: `bentoml models list`
  - Inspect Model: `bentoml models get llama_generation:latest`

---

### 2. Define the Service
Use `service.py` to:
- Load the model/tokenizer.
- Define the `/generate` API.
- Include parameters like `prompt`, `max_length`, and `temperature`.

---

### 3. Configure BentoML
`bentofile.yaml` specifies:
- **Service**: `service:svc`
- **Dependencies**: `torch`, `transformers`, `bentoml`
- **Docker Base Image**: `bentoml/model-server:latest`

---

### 4. Serve the Service
#### Locally
```bash
bentoml serve service.py:svc --port 3002 --debug
```

#### Build & Run with Docker
```bash
bentoml build
docker run -it --rm -p 3002:3002 bentoml-text-generator:latest
```

---

### 5. Test with Client
Run `client.py` to send a request.

```bash
python client.py
```

- **Service URL**: `http://0.0.0.0:3002/generate`
- **Payload**:
  ```json
  {
      "prompt": "Who is Alex Mozerski?",
      "max_length": 100
  }
  ```

---

## **Commands Recap**
- Serve: `bentoml serve service.py:svc --port 3002`
- Build: `bentoml build`
- Run Docker: `docker run -p 3002:3002 bentoml-text-generator:latest`
- List Models: `bentoml models list`

This concise guide ensures you can set up, deploy, and interact with your BentoML-based service efficiently.