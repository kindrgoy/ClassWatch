# ClassWatch — Proto

## Struktur
```
classwatch/
├── main.py              # FastAPI backend
├── templates/
│   └── index.html       # Dashboard frontend
├── static/              # (kosong, untuk aset nanti)
└── requirements.txt
```

## Cara Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 3. Buka browser
# http://localhost:8000
```

## Endpoint
- `GET /`            → Dashboard HTML
- `WS  /ws`          → WebSocket stream (frame + activity data)
- `GET /api/summary` → Summary sesi (JSON)

## Cara Integrasi Model AI
Edit fungsi `classify_frame()` di `main.py`:

```python
def classify_frame(frame: np.ndarray) -> dict:
    # Ganti bagian ini dengan inferensi model kamu
    result = your_model.predict(frame)
    return {
        "timestamp": time.time(),
        "detected": result.person_count,
        "activities": {
            "memperhatikan": result.attention_pct,
            "menulis":       result.writing_pct,
            "membaca":       result.reading_pct,
            "tidur":         result.sleeping_pct,
            "bermain_hp":    result.phone_pct,
        },
        "engagement_index": result.engagement_score,
    }
```
