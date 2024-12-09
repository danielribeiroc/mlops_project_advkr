from pydantic import BaseModel
from datetime import datetime

class Message(BaseModel):
    sender: str
    text: str
    timestamp: datetime = None
    
    class Config:
        schema_extra = {
            "example": {
                "sender": "user",
                "text": "Hello, how are you?",
                "timestamp": "2024-11-07T12:00:00"
            }
        }
