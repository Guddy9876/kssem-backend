from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Load local model
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lower threshold = stricter (flags more as OOC), higher = more lenient
THRESHOLD = 0.38
sessions = {}   # session_id -> {"topic": str, "history": [str]}


class Message(BaseModel):
    session_id: str
    text: str


def get_embedding(text: str):
    return model.encode(text.strip())


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def split_sentences(text: str):
    """Split text into sentences on . ! ? while preserving content."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Also split on plain '.' without trailing space
    result = []
    for part in parts:
        sub = [s.strip() for s in part.split('.') if s.strip()]
        result.extend(sub)
    return result if result else [text.strip()]


def detect(topic: str, history: list, text: str):
    sentences = split_sentences(text)

    # Always embed the topic — it is the single source of truth
    topic_vec = get_embedding(topic)

    # Blend topic (heavily weighted) with recent history
    if history:
        recent_vecs = [get_embedding(h) for h in history[-3:]]
        # 70% topic, 30% recent history average
        history_mean = np.mean(recent_vecs, axis=0)
        context_vec  = 0.70 * topic_vec + 0.30 * history_mean
    else:
        context_vec = topic_vec

    result = []
    for s in sentences:
        score = cosine(context_vec, get_embedding(s))
        result.append({
            "text": s,
            "score": round(score, 3),
            "out_of_context": score < THRESHOLD,
        })

    return result


@app.post("/analyze")
async def analyze(msg: Message):
    sid  = msg.session_id
    text = msg.text.strip()

    data = sessions.get(sid)

    # ── First message always becomes the TOPIC ──
    if not data:
        sessions[sid] = {"topic": text, "history": []}
        return {
            "chunks": [{"text": text, "out_of_context": False, "score": 1.0}],
            "is_topic": True,
        }

    topic   = data["topic"]
    history = data["history"]

    result = detect(topic, history, text)

    # Append to history (keep last 10)
    data["history"].append(text)
    data["history"] = data["history"][-10:]

    return {"chunks": result, "is_topic": False}


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}


class TopicUpdate(BaseModel):
    topic: str

@app.put("/sessions/{session_id}/topic")
def update_topic(session_id: str, payload: TopicUpdate):
    if session_id in sessions:
        sessions[session_id]["topic"] = payload.topic
    else:
        sessions[session_id] = {"topic": payload.topic, "history": []}
    return {"status": "updated", "topic": payload.topic}


@app.get("/health")
def health():
    return {"status": "running", "sessions": len(sessions)}


@app.get("/")
def home():
    return {"status": "GroupMind API running 🚀"}