# 🌿 MindEase — Mental Wellness Chatbot

**Enhanced Edition** for Internship Project

---

## Features

| Feature | Original | Enhanced |
|---|---|---|
| Emotion classes | 6 | 7 (+ neutral) |
| Emotion confidence scores | ❌ | ✅ Live % display |
| All emotion scores | ❌ | ✅ Top-3 shown |
| Crisis detection | ❌ | ✅ Keyword + helplines |
| Mood log format | Plain text `.txt` | Structured `.jsonl` |
| Analytics charts | 1 (matplotlib) | 4 (Plotly interactive) |
| GPT memory management | ❌ (overflows) | ✅ 512-token sliding window |
| Repetition guard | ❌ | ✅ `repetition_penalty=1.3` |
| Web UI | ❌ | ✅ Streamlit app |
| Code structure | Single file | Modular (`core` + `app`) |

---

## File Structure

```
wellness_bot/
├── chatbot_core.py          # All logic, functions, constants
├── app.py                   # Streamlit web UI
├── MindEase_Enhanced.ipynb  # Google Colab notebook
├── requirements.txt
└── README.md
```

---

## Setup

### Google Colab
Open `MindEase_Enhanced.ipynb` and run all cells.

### Local / Streamlit Web App
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Running the Chatbot (Notebook)

- Type your message and press Enter
- Type `log` to see how many entries are saved
- Type `bye` to exit and see your session summary

---

## Crisis Safety

The chatbot automatically detects distress keywords and responds with:
- Empathetic acknowledgment
- Indian helplines: iCall (9152987821), Vandrevala (1860-2662-345)
- International: findahelpline.com
