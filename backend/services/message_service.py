from models.message import Message
from datetime import datetime

def process_message(message: Message):
    if not message.timestamp:
        message.timestamp = datetime.utcnow()
        
    if "hello" in message.text.lower():
        return {"sender": "bot", "text": "Hello! How can I assist you today?", "timestamp": datetime.utcnow()}
    elif "bye" in message.text.lower():
        return {"sender": "bot", "text": "Goodbye! Have a great day!", "timestamp": datetime.utcnow()}
    else:
        return {"sender": "bot", "text": "I'm here to help! Ask me anything.", "timestamp": datetime.utcnow()}
