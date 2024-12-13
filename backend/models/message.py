from pydantic import BaseModel
from datetime import datetime

class Message(BaseModel):
    prompt: str
    max_length: int
    temperature: float
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Who is this person?",
                "max_length": "100",
                "temperature": "0.8"
            }
        }
