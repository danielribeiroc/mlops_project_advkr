from fastapi import FastAPI
from routers import chat, train_model
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from services import rag_functions

# environnement variables
load_dotenv()

# Load environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID")

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init pipe --> one time at start
#pipeline_rag = rag_functions.init_pipe(MODEL_ID)

app.include_router(chat.router, prefix="/api/v1")
#app.include_router(train_model.router, prefix="/api/v1")
