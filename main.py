import os
import base64
import asyncio
import struct
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI(title="ALM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DB_PATH = "/tmp/alm_data.db"


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            transcript  TEXT,
            response    TEXT,
            question    TEXT,
            rating      INTEGER DEFAULT 0  -- 0=unrated, 1=thumbsup, -1=thumbsdown
        )
    """)
    conn.commit()
    conn.close()

init_db()


def get_conn():
    return sqlite3.connect(DB_PATH)


def save_session(transcript: str, response: str, question: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO sessions (timestamp, transcript, response, question) VALUES (?,?,?,?)",
        (datetime.utcnow().isoformat(), transcript, response, question)
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def get_top_examples(limit: int = 5) -> list[dict]:
    """Get highest-rated responses to use as few-shot examples."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT transcript, response FROM sessions
           WHERE rating = 1 AND transcript != '' AND response != ''
           ORDER BY id DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [{"transcript": r[0], "response": r[1]} for r in rows]


def get_recent_history(limit: int = 3) -> list[dict]:
    """Get recent conversations for context window."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT transcript, response FROM sessions
           WHERE transcript != '' AND response != ''
           ORDER BY id DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    return [{"transcript": r[0], "response": r[1]} for r in reversed(rows)]


# ── Pydantic models ────────────────────────────────────────────────────────────

class AudioRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000
    question: str = "What is happening in this audio scene? Describe everything you hear."


class ALMResponse(BaseModel):
    session_id: int
    transcript: str
    claude_response: str


class RatingRequest(BaseModel):
    session_id: int
    rating: int  # 1 = thumbs up, -1 = thumbs down


class RatingResponse(BaseModel):
    success: bool
    message: str


# ── Audio helpers ──────────────────────────────────────────────────────────────

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
    # Build few-shot examples from highly-rated past responses
    top_examples = get_top_examples(limit=3)
    recent_history = get_recent_history(limit=3)

    messages = []

    # System prompt
    messages.append({
        "role": "system",
        "content": (
            "You are an audio-aware AI assistant called ALM. "
            "You Listen, Think, and Understand audio scenes. "
            "You receive a transcript of what was recorded. "
            "Give a rich, natural, insightful response about what is happening. "
            "Be concise but complete. "
            "Learn from the highly-rated example responses provided — they represent "
            "the style and quality users prefer."
        )
    })

    # Few-shot examples from highly rated responses
    if top_examples:
        messages.append({
            "role": "user",
            "content": "Here are some highly-rated example responses to learn from:\n" +
                "\n---\n".join(
                    f"Transcript: {ex['transcript']}\nResponse: {ex['response']}"
                    for ex in top_examples
                )
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I will match the style and quality of those highly-rated responses."
        })

    # Recent conversation history for context
    for entry in recent_history:
        messages.append({"role": "user", "content": f"[TRANSCRIPT]\n{entry['transcript']}\n\n[QUESTION]\n{question}"})
        messages.append({"role": "assistant", "content": entry["response"]})

    # Current request
    messages.append({
        "role": "user",
        "content": f"[TRANSCRIPT]\n{transcript if transcript else '(no speech detected)'}\n\n[QUESTION]\n{question}"
    })

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 512,
            },
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise Exception(f"Groq error: {response.text}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    rated = conn.execute("SELECT COUNT(*) FROM sessions WHERE rating != 0").fetchone()[0]
    conn.close()
    return {"status": "ok", "total_sessions": count, "rated_sessions": rated}


@app.post("/analyze", response_model=ALMResponse)
async def analyze(req: AudioRequest):
    try:
        pcm_bytes = base64.b64decode(req.audio_base64)
    except Exception as e:
        raise HTTPException(400, f"Audio decode failed: {e}")

    wav_bytes = build_wav_header(len(pcm_bytes), req.sample_rate) + pcm_bytes
    transcript = await transcribe_with_groq(wav_bytes)
    groq_response = await ask_groq(transcript, req.question)

    session_id = save_session(transcript, groq_response, req.question)

    return ALMResponse(
        session_id=session_id,
        transcript=transcript,
        claude_response=groq_response,
    )


@app.post("/rate", response_model=RatingResponse)
async def rate(req: RatingRequest):
    if req.rating not in [1, -1]:
        raise HTTPException(400, "Rating must be 1 (thumbs up) or -1 (thumbs down)")
    conn = get_conn()
    conn.execute("UPDATE sessions SET rating = ? WHERE id = ?", (req.rating, req.session_id))
    conn.commit()
    conn.close()
    return RatingResponse(success=True, message="Rating saved!")


@app.get("/dataset")
def export_dataset():
    """Export all rated sessions as a dataset for fine-tuning."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, timestamp, transcript, response, question, rating FROM sessions ORDER BY id"
    ).fetchall()
    conn.close()
    return {
        "total": len(rows),
        "data": [
            {
                "id": r[0], "timestamp": r[1], "transcript": r[2],
                "response": r[3], "question": r[4], "rating": r[5]
            }
            for r in rows
        ]
    }


@app.get("/stats")
def stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    thumbs_up = conn.execute("SELECT COUNT(*) FROM sessions WHERE rating = 1").fetchone()[0]
    thumbs_down = conn.execute("SELECT COUNT(*) FROM sessions WHERE rating = -1").fetchone()[0]
    unrated = conn.execute("SELECT COUNT(*) FROM sessions WHERE rating = 0").fetchone()[0]
    conn.close()
    return {
        "total_sessions": total,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "unrated": unrated,
        "approval_rate": f"{round(thumbs_up / total * 100)}%" if total > 0 else "N/A"
    }


# ── File upload endpoint ───────────────────────────────────────────────────────

from fastapi import UploadFile, File, Form

@app.post("/analyze_file", response_model=ALMResponse)
async def analyze_file(
    file: UploadFile = File(...),
    question: str = Form(default="Analyze this music or audio file. Describe the genre, mood, instruments, tempo, vocals if any, and overall feel.")
):
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(400, "Empty file uploaded")

    # Send file directly to Groq Whisper (supports mp3, mp4, wav, m4a, ogg, flac, webm)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (file.filename or "audio", file_bytes, file.content_type or "audio/mpeg")},
                data={"model": "whisper-large-v3"},
            )
            if response.status_code == 200:
                transcript = response.json().get("text", "").strip()
            else:
                transcript = f"(transcription error: {response.text})"
    except Exception as e:
        transcript = f"(transcription failed: {e})"

    groq_response = await ask_groq(transcript, question)
    session_id = save_session(transcript, groq_response, question)

    return ALMResponse(
        session_id=session_id,
        transcript=transcript,
        claude_response=groq_response,
    )
