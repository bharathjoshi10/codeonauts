import os
import base64
import asyncio
import struct
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="ALM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


class AudioRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000
    question: str = "What is happening in this audio scene? Describe everything you hear."


class ALMResponse(BaseModel):
    transcript: str
    claude_response: str


def build_wav_header(pcm_length: int, sample_rate: int = 16000) -> bytes:
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    chunk_size = 36 + pcm_length
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", pcm_length
    )


async def transcribe_with_groq(wav_bytes: bytes) -> str:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": "whisper-large-v3"},
            )
            if response.status_code == 200:
                return response.json().get("text", "").strip()
            return f"(transcription error: {response.text})"
    except Exception as e:
        return f"(transcription failed: {e})"


async def ask_groq(transcript: str, question: str) -> str:
    prompt = f"""[TRANSCRIPT]
{transcript if transcript else '(no speech detected)'}

[USER QUESTION]
{question}"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an audio-aware AI assistant called ALM. "
                            "You Listen, Think, and Understand audio scenes. "
                            "You receive a transcript of what was recorded. "
                            "Give a rich, natural, insightful response about what is happening. "
                            "Be concise but complete."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 512,
            },
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise Exception(f"Groq error: {response.text}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=ALMResponse)
async def analyze(req: AudioRequest):
    try:
        pcm_bytes = base64.b64decode(req.audio_base64)
    except Exception as e:
        raise HTTPException(400, f"Audio decode failed: {e}")

    wav_bytes = build_wav_header(len(pcm_bytes), req.sample_rate) + pcm_bytes

    transcript = await transcribe_with_groq(wav_bytes)
    groq_response = await ask_groq(transcript, req.question)

    return ALMResponse(
        transcript=transcript,
        claude_response=groq_response,
    )
