"""
Mental Wellness Chatbot - Core Module
======================================
Modular, production-ready mental wellness chatbot with:
- Multi-label emotion detection (7 emotions)
- Crisis detection & safety routing
- Conversation memory
- Mood logging for analytics
"""

import random
import datetime
import os
import json
from pathlib import Path

# ─── Emotion Response Library ──────────────────────────────────────────────────

CALMING_RESPONSES = {
    "sadness": [
        "It's completely okay to feel sad. I'm right here with you. 💚",
        "Sometimes emotions need space. Let them flow — I'm listening.",
        "You're not alone in this. Things do get better, one moment at a time.",
        "I hear you. Your feelings are real and they matter.",
    ],
    "joy": [
        "That's wonderful! Hold onto that feeling — you deserve it. 😊",
        "I'm genuinely happy you're feeling this way!",
        "Celebrate those good moments. They're yours.",
        "That spark in your words made me smile too!",
    ],
    "love": [
        "Love is such a powerful feeling. Thank you for sharing that. ❤️",
        "That's heartwarming. You deserve kindness and love in return.",
        "Cherish those connections — they're what makes life meaningful.",
    ],
    "anger": [
        "Your anger is valid. I'm here — tell me what's going on.",
        "It's okay to be upset. Let's work through this together.",
        "Take a breath with me. You're safe here, and I'm listening.",
        "That sounds really frustrating. I want to understand.",
    ],
    "fear": [
        "Fear is natural and human. You're safe right now. 🌿",
        "Let's ground you — can you name 5 things you can see around you?",
        "I believe in your strength, even when you can't feel it.",
        "You don't have to face this alone. I'm here.",
    ],
    "surprise": [
        "Wow, that sounds unexpected! How are you feeling about it?",
        "Life really does throw curveballs. Want to talk it through?",
        "That must have caught you off guard. Take your time.",
    ],
    "neutral": [
        "I'm here and listening. Tell me more about what's on your mind.",
        "How are you feeling today overall?",
        "I'm glad you're talking to me. What's going on?",
    ],
}

STRESS_TIPS = [
    "Try box breathing: inhale 4 sec → hold 4 sec → exhale 4 sec → hold 4 sec.",
    "Drink a glass of water slowly. Hydration affects mood more than we think.",
    "Step outside for even 5 minutes. Fresh air resets the nervous system.",
    "Repeat quietly: 'I am calm. I am safe. This will pass.'",
    "Tense every muscle for 5 seconds, then release. Notice the relief.",
    "Put on one song you love and do nothing else — just listen.",
    "Write down 3 things you can control right now.",
]

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "don't want to live",
    "want to die", "self harm", "hurt myself", "cutting", "no reason to live",
    "can't go on", "give up on life", "worthless", "hopeless", "nobody cares",
]

CRISIS_RESPONSE = """🆘 I'm really concerned about what you've shared, and I want you to know you matter deeply.

Please reach out to a crisis helpline right now:
• **iCall (India):** 9152987821
• **Vandrevala Foundation:** 1860-2662-345 (24/7)
• **International:** findahelpline.com

You don't have to face this alone. A trained counselor is ready to listen.
I'm here with you too — please tell me you're safe. 💚"""


# ─── Emotion Detection ─────────────────────────────────────────────────────────

def load_emotion_classifier():
    """Load the emotion classifier pipeline (cached after first load)."""
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="nateraw/bert-base-uncased-emotion",
        top_k=None,  # Return all scores
    )


def get_emotion(text: str, classifier) -> tuple[str, float, dict]:
    """
    Detect emotion from text.
    Returns: (top_emotion, confidence, all_scores_dict)
    """
    results = classifier(text)
    # results is a list of lists when top_k=None
    scores = results[0] if isinstance(results[0], list) else results
    scores_dict = {item["label"]: round(item["score"], 3) for item in scores}
    top = max(scores, key=lambda x: x["score"])
    return top["label"], round(top["score"], 3), scores_dict


def is_crisis(text: str) -> bool:
    """Detect if user message contains crisis signals."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)


# ─── Response Generation ───────────────────────────────────────────────────────

def build_empathy_response(emotion: str, confidence: float) -> str:
    """Build a context-aware empathetic response."""
    base = random.choice(CALMING_RESPONSES.get(emotion, CALMING_RESPONSES["neutral"]))
    # Add a wellness tip for distress emotions
    if emotion in ("sadness", "anger", "fear") and confidence > 0.5:
        tip = random.choice(STRESS_TIPS)
        base += f"\n\n💡 *Wellness tip:* {tip}"
    return base


def build_gpt_response(user_input: str, chat_history_ids, tokenizer, model) -> tuple[str, object]:
    """Generate a conversational reply using DialoGPT."""
    import torch
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    if chat_history_ids is not None:
        # Keep only last 512 tokens to avoid overflow
        combined = torch.cat([chat_history_ids, input_ids], dim=-1)
        if combined.shape[-1] > 512:
            combined = combined[:, -512:]
        input_ids = combined

    output = model.generate(
        input_ids,
        max_length=min(input_ids.shape[-1] + 100, 600),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.85,
        repetition_penalty=1.3,
    )
    reply = tokenizer.decode(
        output[:, input_ids.shape[-1]:][0], skip_special_tokens=True
    )
    return reply.strip() or "I'm here with you.", output


# ─── Mood Logging ──────────────────────────────────────────────────────────────

LOG_FILE = Path("mood_log.jsonl")


def log_emotion(text: str, emotion: str, confidence: float, scores: dict) -> None:
    """Append a mood entry as JSON Lines for structured analytics."""
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "message": text,
        "emotion": emotion,
        "confidence": confidence,
        "all_scores": scores,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_mood_log() -> list[dict]:
    """Load all mood log entries."""
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ─── Analytics ─────────────────────────────────────────────────────────────────

def get_mood_summary(entries: list[dict]) -> dict:
    """Compute summary stats from mood log."""
    if not entries:
        return {}
    from collections import Counter
    emotions = [e["emotion"] for e in entries]
    counts = Counter(emotions)
    total = len(entries)
    return {
        "total_sessions": total,
        "emotion_counts": dict(counts),
        "emotion_percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
        "most_frequent": counts.most_common(1)[0][0],
        "latest": entries[-1]["timestamp"],
    }
