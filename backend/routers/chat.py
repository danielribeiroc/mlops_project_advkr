from fastapi import APIRouter, HTTPException, File, UploadFile
from models.message import Message
from typing import List
from services.message_service import process_message

router = APIRouter()

@router.post("/messages-rag")
async def send_message(message: Message):
    #from main import pipeline_rag

    response = process_message(message)
    if not response:
        raise HTTPException(status_code=500, detail="Error processing message")
    return response

@router.post("/messages-lora")
async def send_message(message: Message):
    #from main import pipeline_rag

    response = process_message(message)
    if not response:
        raise HTTPException(status_code=500, detail="Error processing message")
    return response
@router.get("/test")
async def test_endpoint():
    return {"message": "All working fine! v1"}
