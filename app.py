"""
Multi-Model Chat App
====================
Run:  streamlit run app.py
Deps: pip install streamlit torch sentencepiece gdown
"""

import json
import os
from datetime import datetime
from pathlib import Path

import gdown
import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════

MODEL_REGISTRY = {
    "V2":   (st.secrets.get("MODEL_V2_ID", os.environ.get("MODEL_V2_ID", "")), "models/ckpt_step042455.pt"),
    "V2.1": (st.secrets.get("MODEL_V21_ID", os.environ.get("MODEL_V21_ID", "")), "models/v2-1-sft_epoch3.pt"),
    "V2.2": (st.secrets.get("MODEL_V22_ID", os.environ.get("MODEL_V22_ID", "")), "models/v2-2-sft_epoch3.pt"),
}

TOKENIZER_PATH = "v2_tokenizer.model"
SESSIONS_FILE  = "chat_sessions.json"

MAX_NEW_TOKENS = 100
TEMPERATURE    = 0.8
TOP_K          = 50

# ══════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════

class SimpleSeqModel(nn.Module):
    def __init__(self, vocab_size: int = 32000, d_model: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb        = nn.Embedding(vocab_size, d_model)
        self.gru        = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
        self.fc         = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.gru(e)
        return self.fc(h)

# ══════════════════════════════════════════════
# TOKENIZER
# ══════════════════════════════════════════════

@st.cache_resource
def load_tokenizer():
    if not Path(TOKENIZER_PATH).exists():
        return None
    try:
        sp = spm.SentencePieceProcessor()
        sp.Load(TOKENIZER_PATH)
        return sp
    except Exception as e:
        st.warning(f"Tokenizer error: {e}")
        return None

def encode(tokenizer, text):
    try:
        return tokenizer.Encode(text, out_type=int)
    except:
        return []

def decode(tokenizer, ids):
    try:
        return tokenizer.Decode(ids)
    except:
        return ""

# ══════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════

@st.cache_resource
def load_model(model_key: str):
    file_id, path = MODEL_REGISTRY[model_key]

    if not file_id:
        st.error(f"No Drive ID found for {model_key}. Check your secrets.")
        return None

    os.makedirs("models", exist_ok=True)

    if not Path(path).exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"⬇️ Downloading {model_key} from Google Drive… (first time only)"):
            try:
                gdown.download(url, path, quiet=False)
            except Exception as e:
                st.error(f"Download failed for {model_key}: {e}")
                return None

    try:
        checkpoint = torch.load(path, map_location="cpu")
        vocab_size = 32000
        state_dict = None

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            cfg        = checkpoint.get("config", {})
            vocab_size = cfg.get("vocab_size", vocab_size)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            return checkpoint

        for key in ("emb.weight", "tok_emb.weight", "embed.weight",
                    "embeddings.word_embeddings.weight"):
            if state_dict and key in state_dict:
                vocab_size = state_dict[key].shape[0]
                break

        model = SimpleSeqModel(vocab_size=vocab_size)
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Could not load {model_key}: {e}")
        return None

# ══════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════

def _sample(logits, temperature, top_k):
    logits = logits.float() / max(temperature, 1e-8)
    if top_k > 0:
        top_k     = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def generate_response(model_key, model, tokenizer, user_input):
    if model is None or tokenizer is None:
        reason = "model unavailable" if model is None else "tokenizer not found"
        return f"[{model_key} · offline — {reason}]\nYou said: {user_input}"
    try:
        input_ids = encode(tokenizer, user_input)
        if not input_ids:
            return f"[{model_key}] Tokenisation returned empty — check tokenizer."

        generated = list(input_ids)
        eos_id    = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else -1
        seen      = set(generated)

        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                x           = torch.tensor([generated])
                logits      = model(x)
                next_logits = logits[0, -1].clone()
                for tid in seen:
                    if 0 <= tid < next_logits.size(-1):
                        next_logits[tid] *= 0.9
                next_id = _sample(next_logits, TEMPERATURE, TOP_K)
                if next_id == eos_id:
                    break
                generated.append(next_id)
                seen.add(next_id)

        new_ids = generated[len(input_ids):]
        output  = decode(tokenizer, new_ids).strip()
        return output if output else f"[{model_key}] (empty output)"

    except Exception as e:
        return f"[{model_key}] Inference error: {e}"

# ══════════════════════════════════════════════
# SESSIONS
# ══════════════════════════════════════════════

def load_sessions() -> dict:
    try:
        if not Path(SESSIONS_FILE).exists():
            return {}
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_sessions(sessions: dict) -> None:
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except:
        pass

def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def session_title(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            txt = m["content"]
            return txt[:40] + ("…" if len(txt) > 40 else "")
    return "New Chat"

# ══════════════════════════════════════════════
# INIT STATE
# ══════════════════════════════════════════════

st.set_page_config(page_title="Multi-Model Chat", page_icon="💬", layout="wide")

if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()

if "current_session" not in st.session_state:
    sid = new_session_id()
    st.session_state.sessions[sid] = []
    st.session_state.current_session = sid

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

with st.sidebar:
    st.title("💬 Multi-Model Chat")
    st.divider()

    model_key = st.selectbox("🤖 Select Model", list(MODEL_REGISTRY.keys()))
    model     = load_model(model_key)
    tokenizer = load_tokenizer()

    if model:
        st.success(f"✓ {model_key} loaded")
    else:
        st.warning(f"⚠ {model_key} — unavailable")

    st.divider()

    if st.button("✏️ New Chat", use_container_width=True):
        sid = new_session_id()
        st.session_state.sessions[sid] = []
        st.session_state.current_session = sid
        save_sessions(st.session_state.sessions)
        st.rerun()

    st.markdown("#### 🕘 Previous Chats")

    all_sids = sorted(st.session_state.sessions.keys(), reverse=True)
    for sid in all_sids:
        msgs      = st.session_state.sessions[sid]
        title     = session_title(msgs) if msgs else "Empty Chat"
        is_active = sid == st.session_state.current_session

        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"**{title}**" if is_active else title
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                st.session_state.current_session = sid
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{sid}", help="Delete"):
                del st.session_state.sessions[sid]
                if st.session_state.current_session == sid:
                    remaining = sorted(st.session_state.sessions.keys(), reverse=True)
                    if remaining:
                        st.session_state.current_session = remaining[0]
                    else:
                        new_sid = new_session_id()
                        st.session_state.sessions[new_sid] = []
                        st.session_state.current_session = new_sid
                save_sessions(st.session_state.sessions)
                st.rerun()

# ══════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════

current_msgs = st.session_state.sessions.get(st.session_state.current_session, [])

for msg in current_msgs:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type a message…")

if user_input:
    current_msgs.append({
        "role":
