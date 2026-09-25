import os
import cv2
import base64
import asyncio
import time
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from camera import VideoStream

# ── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("classwatch")

# ── Path config ───────────────────────────────────────
CUR_DIR       = os.path.dirname(os.path.realpath(__file__))
STATIC_PATH   = os.path.join(CUR_DIR, "static")
TEMPLATE_PATH = os.path.join(CUR_DIR, "templates", "index.html")
SESSIONS_DIR  = os.path.join(os.path.dirname(CUR_DIR), "Data", "logs")

os.makedirs(SESSIONS_DIR, exist_ok=True)

# ── Session state ─────────────────────────────────────
session_history: deque  = deque(maxlen=5400)
session_start:   Optional[float] = None
session_active:  bool   = False


# ── Placeholder classifier ────────────────────────────
def classify_frame(frame) -> dict:
    """
    PLACEHOLDER — kembalikan data dummy.
    Ganti isi fungsi ini dengan inferensi model AI kamu.
    Input  : np.ndarray (frame BGR)
    Output : dict dengan keys detected, engagement_index, activities
    """
    return {
        "detected": 0,
        "engagement_index": 0,
        "activities": {
            "memperhatikan": 0,
            "menulis":       0,
            "membaca":       0,
            "angkat_tangan": 0,
            "tidur":         0,
            "bermain_hp":    0,
        }
    }


# ── Helper: simpan session ke JSON ───────────────────
def save_session(history: deque, start: float) -> str:
    if not history:
        return ""

    keys = ["memperhatikan", "menulis", "membaca", "angkat_tangan", "tidur", "bermain_hp"]
    avg_activities = {
        k: round(sum(h["activities"][k] for h in history) / len(history))
        for k in keys
    }
    avg_engagement = round(sum(h["engagement_index"] for h in history) / len(history))
    learning_pct   = sum(avg_activities[k] for k in ["memperhatikan","menulis","membaca","angkat_tangan"])
    duration_sec   = round(time.time() - start)

    label    = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{label}.json"
    filepath = os.path.join(SESSIONS_DIR, filename)

    payload = {
        "session_label":     label,
        "started_at":        datetime.fromtimestamp(start).isoformat(),
        "ended_at":          datetime.now().isoformat(),
        "duration_seconds":  duration_sec,
        "total_frames":      len(history),
        "avg_engagement":    avg_engagement,
        "avg_activities":    avg_activities,
        "learning_time_pct": learning_pct,
        "peak_engagement":   max(history, key=lambda h: h["engagement_index"]),
        "frames":            list(history),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info(f"Session disimpan: {filepath}")
    return filename


# ── Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== ClassWatch dimulai ===")

    cam = VideoStream(src=0).start()
    if not cam.ret:
        log.warning("Kamera tidak dapat dibuka — periksa koneksi device")

    app.state.camera = cam
    yield

    log.info("=== ClassWatch dimatikan ===")
    app.state.camera.stop()


# ── App ───────────────────────────────────────────────
app = FastAPI(title="ClassWatch", version="0.1.0", lifespan=lifespan)

if os.path.exists(STATIC_PATH):
    app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
else:
    log.warning(f"Folder static tidak ditemukan: {STATIC_PATH}")


# ── Routes ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        log.error(f"index.html tidak ditemukan: {TEMPLATE_PATH}")
        return HTMLResponse("<h1>Error: index.html tidak ditemukan</h1>", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info(f"WebSocket terhubung: {websocket.client}")

    camera: VideoStream = websocket.app.state.camera

    if camera is None or not camera.ret:
        log.error("Kamera tidak tersedia saat WebSocket connect")
        await websocket.send_json({"error": "Kamera tidak tersedia"})
        await websocket.close(code=1013)
        return

    try:
        while True:
            t_start = time.perf_counter()

            frame = camera.read()

            if frame is None:
                await websocket.send_json({"error": "Frame tidak tersedia", "frame": None})
                await asyncio.sleep(0.1)
                continue

            frame_small = cv2.resize(frame, (640, 360))
            _, buffer   = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64   = base64.b64encode(buffer).decode("utf-8")

            # Classifier hanya dipanggil jika session aktif
            if session_active:
                result = classify_frame(frame_small)
                session_history.append({
                    "timestamp":        time.time(),
                    "detected":         result["detected"],
                    "engagement_index": result["engagement_index"],
                    "activities":       result["activities"],
                })
            else:
                result = {
                    "detected": 0,
                    "engagement_index": 0,
                    "activities": {k: 0 for k in
                        ["memperhatikan","menulis","membaca","angkat_tangan","tidur","bermain_hp"]}
                }

            await websocket.send_json({
                "frame":            frame_b64,
                "session_active":   session_active,
                "detected":         result["detected"],
                "engagement_index": result["engagement_index"],
                "activities":       result["activities"],
            })

            elapsed = time.perf_counter() - t_start
            await asyncio.sleep(max(0.0, 0.04 - elapsed))

    except WebSocketDisconnect:
        log.info(f"WebSocket terputus: {websocket.client}")
    except Exception as e:
        log.error(f"Error WebSocket: {e}", exc_info=True)


# ── Session control ───────────────────────────────────
@app.post("/api/session/start")
async def start_session():
    global session_active, session_start, session_history

    if session_active:
        return JSONResponse({"status": "already_active"}, status_code=400)

    session_history.clear()
    session_start  = time.time()
    session_active = True
    log.info("Session dimulai")

    return {"status": "started", "started_at": datetime.now().isoformat()}


@app.post("/api/session/stop")
async def stop_session():
    global session_active, session_start, session_history

    if not session_active:
        return JSONResponse({"status": "no_active_session"}, status_code=400)

    session_active = False
    filename       = save_session(session_history, session_start)
    total_frames   = len(session_history)

    session_history.clear()
    session_start = None
    log.info("Session dihentikan dan direset")

    return {
        "status":       "stopped",
        "saved_as":     filename,
        "total_frames": total_frames,
    }


@app.get("/api/session/status")
async def session_status():
    duration = round(time.time() - session_start) if session_start else 0
    return {
        "active":           session_active,
        "duration_seconds": duration,
        "total_frames":     len(session_history),
    }


@app.get("/api/summary")
async def get_summary():
    if not session_history:
        return JSONResponse(
            {"total_frames": 0, "message": "Belum ada data sesi"},
            status_code=200
        )

    keys = ["memperhatikan", "menulis", "membaca", "angkat_tangan", "tidur", "bermain_hp"]
    avg_activities = {
        k: round(sum(h["activities"][k] for h in session_history) / len(session_history))
        for k in keys
    }
    avg_engagement = round(
        sum(h["engagement_index"] for h in session_history) / len(session_history)
    )
    learning_pct = sum(avg_activities[k] for k in ["memperhatikan","menulis","membaca","angkat_tangan"])
    duration_sec = round(time.time() - session_start) if session_start else 0

    return {
        "total_frames":      len(session_history),
        "duration_seconds":  duration_sec,
        "avg_engagement":    avg_engagement,
        "avg_activities":    avg_activities,
        "learning_time_pct": learning_pct,
        "peak_engagement":   max(session_history, key=lambda h: h["engagement_index"]),
    }


@app.get("/api/health")
async def health_check():
    camera: VideoStream = app.state.camera
    return {
        "status":         "ok",
        "camera_active":  camera.ret if camera else False,
        "session_active": session_active,
        "session_frames": len(session_history),
    }