from fastapi import APIRouter, HTTPException, File, UploadFile
from models.message import Message
from typing import List
from services.message_service import lora_call, process_message

router = APIRouter()

@router.post("/generate")
async def send_message(message: Message):
    #from main import pipeline_rag

    response = process_message(message)
    if not response:
        raise HTTPException(status_code=500, detail="Error processing message")
    return response

@router.post("/messages-lora")
async def send_message(message: Message):
    response = lora_call(message)
    if not response:
        raise HTTPException(status_code=500, detail="Error processing message")
    return response


@router.get("/test")
async def test_endpoint():
    return {"message": "All working fine! v1"}

@router.get("/train")
async def test_endpoint():
    return {"message": "Train not done yet, needs to be implemented! v1"}