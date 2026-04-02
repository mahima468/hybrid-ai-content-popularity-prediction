"""
Authentication Routes — SQLite + JWT
"""

import sqlite3
import hashlib
import hmac
import os
import time
import base64
import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

DB_PATH = "auth.db"
SECRET_KEY = "hybridai_secret_2024_xK9mP2qR"
TOKEN_EXPIRY = 60 * 60 * 24 * 7  # 7 days


# ---------- DB setup ----------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            label TEXT,
            confidence REAL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ---------- Helpers ----------

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        raw = base64.b64decode(stored.encode())
        salt, dk = raw[:16], raw[16:]
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(dk, check)
    except Exception:
        return False


def create_token(user_id: int, email: str) -> str:
    payload = {"sub": user_id, "email": email, "exp": int(time.time()) + TOKEN_EXPIRY}
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig_input = f"{header}.{body}".encode()
    sig = hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return f"{header}.{body}.{sig_b64}"


def decode_token(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("bad token")
        header, body, sig = parts
        sig_input = f"{header}.{body}".encode()
        expected_sig = hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
        expected_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")
        if not hmac.compare_digest(sig, expected_b64):
            raise ValueError("invalid signature")
        # re-pad base64
        padding = 4 - len(body) % 4
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * padding))
        if payload["exp"] < int(time.time()):
            raise ValueError("token expired")
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return decode_token(credentials.credentials)


# ---------- Schemas ----------

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class LogAnalysisRequest(BaseModel):
    type: str       # 'sentiment' | 'engagement' | 'prediction'
    label: str      # e.g. 'Positive', 'Authentic', 'High Potential'
    confidence: float


# ---------- Routes ----------

@router.post("/register")
def register(req: RegisterRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    if len(req.name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Name must be at least 2 characters.")
    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email=?", (req.email.lower(),)).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered.")
        pw_hash = hash_password(req.password)
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?,?,?,?)",
            (req.name.strip(), req.email.lower(), pw_hash, int(time.time()))
        )
        conn.commit()
        user_id = cursor.lastrowid
        token = create_token(user_id, req.email.lower())
        return {"token": token, "user": {"id": user_id, "name": req.name.strip(), "email": req.email.lower()}}
    finally:
        conn.close()


@router.post("/login")
def login(req: LoginRequest):
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM users WHERE email=?", (req.email.lower(),)).fetchone()
        if not row or not verify_password(req.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = create_token(row["id"], row["email"])
        return {"token": token, "user": {"id": row["id"], "name": row["name"], "email": row["email"]}}
    finally:
        conn.close()


@router.get("/me")
def get_me(current=Depends(get_current_user)):
    conn = get_db()
    try:
        row = conn.execute("SELECT id, name, email, created_at FROM users WHERE id=?", (current["sub"],)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found.")
        return {"id": row["id"], "name": row["name"], "email": row["email"]}
    finally:
        conn.close()


@router.post("/log-analysis")
def log_analysis(req: LogAnalysisRequest, current=Depends(get_current_user)):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO analyses (user_id, type, label, confidence, created_at) VALUES (?,?,?,?,?)",
            (current["sub"], req.type, req.label, req.confidence, int(time.time()))
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@router.get("/my-stats")
def my_stats(current=Depends(get_current_user)):
    conn = get_db()
    try:
        user_id = current["sub"]
        rows = conn.execute(
            "SELECT type, label, confidence, created_at FROM analyses WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()

        total = len(rows)
        sentiment_count = sum(1 for r in rows if r["type"] == "sentiment")
        engagement_count = sum(1 for r in rows if r["type"] == "engagement")
        prediction_count = sum(1 for r in rows if r["type"] == "prediction")
        avg_confidence = (sum(r["confidence"] for r in rows) / total) if total > 0 else 0.0

        # Sentiment distribution from user's own sentiment analyses
        sentiment_rows = [r for r in rows if r["type"] == "sentiment"]
        dist = {"positive": 0, "negative": 0, "neutral": 0}
        for r in sentiment_rows:
            key = r["label"].lower() if r["label"] else "neutral"
            if key in dist:
                dist[key] += 1

        # Weekly analyses — last 7 days bucketed by day
        now = int(time.time())
        week_buckets = [0] * 7
        for r in rows:
            day_ago = (now - r["created_at"]) // 86400
            if 0 <= day_ago <= 6:
                week_buckets[6 - day_ago] += 1

        # Recent activity — last 6 entries with human-readable time
        recent = []
        for r in list(rows)[:6]:
            diff = now - r["created_at"]
            if diff < 60:
                t = "just now"
            elif diff < 3600:
                t = f"{diff // 60} min ago"
            elif diff < 86400:
                t = f"{diff // 3600}h ago"
            else:
                t = f"{diff // 86400}d ago"
            recent.append({
                "type": r["type"],
                "label": r["label"],
                "confidence": r["confidence"],
                "time": t,
            })

        return {
            "total_analyses": total,
            "sentiment_count": sentiment_count,
            "engagement_count": engagement_count,
            "prediction_count": prediction_count,
            "avg_confidence": avg_confidence,
            "sentiment_distribution": dist,
            "weekly_analyses": week_buckets,
            "recent_activity": recent,
        }
    finally:
        conn.close()
