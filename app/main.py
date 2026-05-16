from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat_stub():
    # Stub — will be implemented in Phase 3
    return {
        "reply": "Service is up. Full agent coming soon.",
        "recommendations": [],
        "end_of_conversation": False,
    }
