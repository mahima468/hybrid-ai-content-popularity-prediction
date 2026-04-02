"""
Chatbot API Route
Uses HuggingFace InferenceClient to avoid CORS and URL-format issues
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are an AI assistant embedded in the HybridAI Content Analytics dashboard. "
    "You help users understand sentiment analysis, fake engagement detection, and content popularity prediction. "
    "Be concise, helpful, and focus on content analytics topics."
)

_executor = ThreadPoolExecutor(max_workers=2)


class Message(BaseModel):
    role: str  # "user" or "bot"
    text: str


class ChatRequest(BaseModel):
    messages: List[Message]


def _call_hf(messages: List[Message]) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=HF_MODEL, token=HF_API_KEY)

    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        if msg.role == "user":
            openai_messages.append({"role": "user", "content": msg.text})
        elif msg.role == "bot":
            openai_messages.append({"role": "assistant", "content": msg.text})

    response = client.chat_completion(
        messages=openai_messages,
        max_tokens=300,
        temperature=0.7,
        top_p=0.9,
    )
    return response.choices[0].message.content or ""


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        loop = asyncio.get_event_loop()
        bot_text = await loop.run_in_executor(
            _executor, _call_hf, request.messages
        )

        if not bot_text.strip():
            bot_text = "I'm not sure about that. Could you rephrase your question?"

        return {"reply": bot_text.strip()}

    except Exception as e:
        err_str = str(e)
        logger.error(f"Chat error: {err_str}")
        if "loading" in err_str.lower() or "503" in err_str:
            raise HTTPException(status_code=503, detail="Model is loading, please retry in ~20 seconds.")
        raise HTTPException(status_code=500, detail=err_str[:300])
