import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["OMP_NUM_THREADS"] = "2"

try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api import api_router
from backend.websocket import routes as websocket_routes

app = FastAPI(
    title="AI Border Surveillance System",
    description="Intelligent Video Analytics Platform for Border Surveillance",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app.include_router(api_router.router, prefix="/api/v1")
app.include_router(websocket_routes.router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")
EVIDENCE_DIR = "/tmp/evidence" if os.getenv("VERCEL") else os.path.abspath("evidence")

try:
    os.makedirs(STATIC_DIR, exist_ok=True)
except Exception:
    pass

try:
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
except Exception:
    pass

if os.path.exists(STATIC_DIR):
    try:
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    except Exception:
        pass

if os.path.exists(EVIDENCE_DIR):
    try:
        app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")
    except Exception:
        pass

@app.get("/")
@app.get("/dashboard")
@app.get("/monitor")
@app.get("/analytics")
@app.get("/assistant")
@app.get("/perimeter")
@app.get("/reports")
@app.get("/index.html")
async def serve_dashboard():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return {"message": "AI Border Surveillance System API Live", "version": "1.0.0"}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Border Surveillance System...")
    # Initialize database connection here
    from backend.core.database import SessionLocal, Base, engine
    from backend.models.schema import Event, Alert, Vehicle, Track, ANPRRecord, TrackPosition
    from backend.services.entity_registry import entity_registry
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.query(Alert).delete()
        db.query(Event).delete()
        db.query(ANPRRecord).delete()
        db.query(TrackPosition).delete()
        db.query(Vehicle).delete()
        db.query(Track).delete()
        db.commit()
        entity_registry.entities.clear()
        logger.info("Completely reset SQLite database: 0 old vehicles, tracks, alerts, or events for a fresh live session.")
    except Exception as e:
        logger.error(f"Failed to clear database on startup: {e}")
    finally:
        db.close()

    # Skip starting background threads on Vercel Serverless environment
    if not os.getenv("VERCEL"):
        from backend.services.db_worker import db_worker
        db_worker.start()

        # Auto-initialize CAM-01 with Example Vid if available
        try:
            from video.stream_manager import stream_manager
            from ai.pipeline import CameraPipeline
            from backend.api.api_router import active_pipelines, _ensure_camera_in_db

            example_path = os.path.abspath("uploads/example_vid.mp4")
            if os.path.exists(example_path):
                _ensure_camera_in_db("CAM-01", name="Example Vid", location="Highway Traffic")
                stream_manager.add_stream("CAM-01", example_path, "MP4")
                pipeline = CameraPipeline("CAM-01")
                pipeline.start()
                active_pipelines["CAM-01"] = pipeline
                logger.info("CAM-01 initialized with Example Vid.")
        except Exception as e:
            logger.error(f"Failed to auto-start CAM-01 with example video: {e}")
async def shutdown_event():
    logger.info("Shutting down Border Surveillance System...")
    # Clean up resources here

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
