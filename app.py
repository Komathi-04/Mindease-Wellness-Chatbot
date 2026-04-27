"""
Mental Wellness Chatbot — Streamlit Web App
============================================
Run with: streamlit run app.py
"""

import streamlit as st
import datetime
import json
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindEase · Wellness Chatbot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background: #f5f0eb; }

.chat-bubble-user {
    background: #2d5016;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 0.95rem;
}
.chat-bubble-bot {
    background: white;
    color: #1a1a1a;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    border-left: 4px solid #7fb069;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.emotion-badge {
    display: inline-block;
    background: #e8f5e3;
    color: #2d5016;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 6px;
}
.crisis-box {
    background: #fff3cd;
    border: 2px solid #ff6b6b;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ─── Import Core ──────────────────────────────────────────────────────────────
from chatbot_core import (
    get_emotion, is_crisis, build_empathy_response,
    log_emotion, load_mood_log, get_mood_summary,
    CRISIS_RESPONSE, CALMING_RESPONSES,
)

# ─── Model Loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading emotion model...")
def load_models():
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    emotion_clf = pipeline(
        "text-classification",
        model="nateraw/bert-base-uncased-emotion",
        top_k=None,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    gpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return emotion_clf, tokenizer, gpt_model

# ─── Session State Init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 MindEase")
    st.markdown("*Your mental wellness companion*")
    st.divider()

    page = st.radio("Navigate", ["💬 Chat", "📊 Mood Dashboard"], label_visibility="collapsed")
    st.divider()

    st.markdown("#### 🆘 Crisis Resources")
    st.markdown("""
- **iCall:** 9152987821
- **Vandrevala:** 1860-2662-345
- **International:** [findahelpline.com](https://findahelpline.com)
    """)
    st.divider()
    if st.button("🗑 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history_ids = None
        st.session_state.emotion_history = []
        st.rerun()

# ─── Emotion Emoji Map ────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "sadness": "😔", "joy": "😊", "love": "❤️",
    "anger": "😤", "fear": "😨", "surprise": "😮", "neutral": "😐",
}
EMOTION_COLOR = {
    "sadness": "#6ba3d6", "joy": "#f9c74f", "love": "#f4a0a0",
    "anger": "#e05252", "fear": "#9b72cf", "surprise": "#f3a04a", "neutral": "#8ab4a1",
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
if "Chat" in page:
    st.markdown("# 💬 How are you feeling today?")
    st.caption("This is a safe space. Share freely — I'm listening.")

    # Load models
    try:
        emotion_clf, tokenizer, gpt_model = load_models()
        models_ready = True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        models_ready = False

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                emotion = msg.get("emotion", "neutral")
                emoji = EMOTION_EMOJI.get(emotion, "🤖")
                st.markdown(
                    f'<div class="chat-bubble-bot">{msg["content"]}'
                    f'<br><span class="emotion-badge">{emoji} {emotion} · {msg.get("confidence", 0):.0%}</span></div>',
                    unsafe_allow_html=True
                )

    st.divider()

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Your message",
                placeholder="Type how you're feeling...",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send 💬", use_container_width=True)

    if submitted and user_input.strip() and models_ready:
        user_text = user_input.strip()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_text})

        # Crisis check
        if is_crisis(user_text):
            bot_reply = CRISIS_RESPONSE
            emotion, confidence, scores = "fear", 1.0, {}
        else:
            # Emotion detection
            emotion, confidence, scores = get_emotion(user_text, emotion_clf)

            # Build empathy response
            empathy = build_empathy_response(emotion, confidence)

            # GPT conversational layer
            from chatbot_core import build_gpt_response
            gpt_reply, st.session_state.chat_history_ids = build_gpt_response(
                user_text, st.session_state.chat_history_ids, tokenizer, gpt_model
            )

            bot_reply = f"{empathy}\n\n*{gpt_reply}*" if gpt_reply and len(gpt_reply) > 5 else empathy

        # Log & store
        log_emotion(user_text, emotion, confidence, scores)
        st.session_state.emotion_history.append({
            "time": datetime.datetime.now().isoformat(),
            "emotion": emotion,
            "confidence": confidence,
        })
        st.session_state.messages.append({
            "role": "bot",
            "content": bot_reply,
            "emotion": emotion,
            "confidence": confidence,
        })
        st.rerun()

    # Emotion scores live preview
    if st.session_state.emotion_history:
        latest = st.session_state.emotion_history[-1]
        st.markdown(f"**Last detected emotion:** {EMOTION_EMOJI.get(latest['emotion'], '❓')} `{latest['emotion']}` · confidence `{latest['confidence']:.0%}`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MOOD DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("# 📊 Mood Analytics Dashboard")
    st.caption("Insights from your wellness conversations")

    entries = load_mood_log()

    if not entries:
        st.info("No mood data yet. Start chatting to see your analytics!")
    else:
        df = pd.DataFrame(entries)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour

        summary = get_mood_summary(entries)

        # ── KPI Row ─────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Sessions", summary["total_sessions"])
        with c2:
            st.metric("Most Frequent Mood", f"{EMOTION_EMOJI.get(summary['most_frequent'], '')} {summary['most_frequent']}")
        with c3:
            pos = summary["emotion_counts"].get("joy", 0) + summary["emotion_counts"].get("love", 0)
            st.metric("Positive Moments", pos)
        with c4:
            last_dt = pd.to_datetime(summary["latest"]).strftime("%b %d, %H:%M")
            st.metric("Last Check-in", last_dt)

        st.divider()

        col1, col2 = st.columns(2)

        # ── Donut Chart ──────────────────────────────────────────────────────
        with col1:
            st.markdown("#### Emotion Breakdown")
            counts = summary["emotion_counts"]
            colors = [EMOTION_COLOR.get(e, "#cccccc") for e in counts.keys()]
            fig_pie = go.Figure(go.Pie(
                labels=[f"{EMOTION_EMOJI.get(k,'')} {k}" for k in counts.keys()],
                values=list(counts.values()),
                hole=0.5,
                marker_colors=colors,
                textinfo="percent+label",
            ))
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Timeline ─────────────────────────────────────────────────────────
        with col2:
            st.markdown("#### Mood Over Time")
            daily = df.groupby(["date", "emotion"]).size().reset_index(name="count")
            fig_line = px.bar(
                daily, x="date", y="count", color="emotion",
                color_discrete_map=EMOTION_COLOR,
                labels={"count": "Messages", "date": "Date"},
            )
            fig_line.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title="Emotion",
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # ── Hour Heatmap ─────────────────────────────────────────────────────
        st.markdown("#### When Do You Check In?")
        hour_counts = df.groupby("hour").size().reset_index(name="sessions")
        fig_hour = px.bar(
            hour_counts, x="hour", y="sessions",
            labels={"hour": "Hour of Day", "sessions": "Sessions"},
            color="sessions",
            color_continuous_scale="Greens",
        )
        fig_hour.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_hour, use_container_width=True)

        # ── Raw Log Table ────────────────────────────────────────────────────
        with st.expander("📋 View Raw Session Log"):
            display_df = df[["timestamp", "emotion", "confidence", "message"]].copy()
            display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.0%}")
            display_df = display_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, height=300)
