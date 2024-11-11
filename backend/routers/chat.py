from fastapi import APIRouter, HTTPException
from models.message import Message
from services.message_service import process_message

router = APIRouter()

@router.post("/messages")
async def send_message(message: Message):
    response = process_message(message)
    if not response:
        raise HTTPException(status_code=500, detail="Error processing message")
    return response

@router.get("/test")
async def test_endpoint():
    return {"message": "All working fine!"}

