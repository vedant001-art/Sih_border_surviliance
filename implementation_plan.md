# AI-Based Intelligent Video Analytics Platform for Border Surveillance (SIH26187)

This document outlines the implementation plan for the professional, fully functional prototype of an AI-powered border surveillance and video analytics platform based on the SIH Problem Statement SIH26187. 

The system will ingest video from various sources (MP4, Webcam, RTSP), process it using a computer vision pipeline (YOLO, ByteTrack, OSNet/FastReID, SCRFD, ArcFace, PaddleOCR) to track people and vehicles, detect events (intrusion, loitering, abandoned objects, etc.), compute risk scores, generate evidence clips, and provide real-time alerts to a professional command-and-control React dashboard. No mocking of AI results will be performed.

## User Review Required

> [!WARNING]
> This is a large-scale architecture encompassing multiple AI models and a real-time event-driven backend. To ensure stability and adherence to requirements, the project will be built strictly in the defined phases. Please review the phases below.

> [!CAUTION]
> The AI pipeline requires significant computational resources. We will implement configurable FPS and inference batching to manage load, but running multiple models (Detection, Tracking, ReID, ANPR, Face Recognition) concurrently on multiple camera streams may be demanding for typical laptop hardware without a dedicated GPU.

## Open Questions

> [!IMPORTANT]
> 1. **Database:** Should we use Docker for PostgreSQL and Redis, or assume they will be installed natively on your system? (Docker is highly recommended for reproducibility).
> 2. **AI Models:** Do you have a dedicated NVIDIA GPU (CUDA) on this machine, or should we default to CPU execution (which will be significantly slower for YOLO/ReID/Face)?
> 3. **Demo Footage:** Do you have specific MP4 files you'd like to use for testing, or should I download some standard sample footage for testing person/vehicle tracking and intrusion?
> 4. **Mistral API Key:** You provided 5 API keys in the prompt. Which one should be used for the Mistral Event Summarization?

## Proposed Changes

We will construct the project using the required directory structure. 

### Phase 1: Foundation & Video Ingestion
- Set up the project structure (`backend/`, `frontend/`, `ai/`, `video/`, `database/`, etc.).
- Set up PostgreSQL schema (SQLAlchemy models) for cameras, streams, and system logs.
- Implement `video/` module to handle MP4, Webcam, and RTSP stream ingestion using OpenCV/FFmpeg.
- Build the FastAPI foundation and WebSocket skeleton.

### Phase 2: Object Detection (YOLO)
- Integrate Ultralytics YOLO.
- Configure inference to detect person, car, truck, bus, motorcycle, bicycle.
- Expose bounding boxes, class, and confidence to the pipeline.

### Phase 3: Object Tracking (ByteTrack)
- Integrate ByteTrack for persistent local tracking.
- Assign stable Track IDs to detected objects.
- Store trajectory history (centroids) for downstream event logic.

### Phase 4: Virtual Fence & Real-time Alerts
- Add database models for `zones`, `events`, and `alerts`.
- Implement spatial logic: check if tracked centroids intersect/enter drawn polygon zones.
- Emit `INTRUSION ALERT` over WebSockets to the frontend.

### Phase 5: Behavior & Advanced Detection
- Implement loitering detection (time-in-zone).
- Implement direction tracking and speed estimation (using pixel displacement).
- Implement vehicle stopping detection.
- Implement abandoned object detection (state machine: DETECTED -> MONITORED -> ABANDONED).

### Phase 6: ANPR (Number Plate Recognition)
- Use YOLO for plate crop.
- Integrate PaddleOCR for plate text extraction.
- Store `anpr_records` linked to vehicle Track IDs.

### Phase 7: Face Detection & Recognition
- Integrate SCRFD (insightface) for face detection.
- Integrate ArcFace for face embeddings and recognition.
- Create local enrollment database in PostgreSQL/Qdrant.

### Phase 8: Multi-camera & Cross-Camera Tracking
- Integrate OSNet/FastReID.
- Extract appearance embeddings and assign Global Track IDs.
- Correlate transitions between cameras using time/topology constraints.

### Phase 9: Risk Engine & Predictive Intrusion
- Implement a predictive Kalman filter to extrapolate trajectories.
- Generate `PREDICTED INTRUSION` if the extrapolated path crosses a restricted zone.
- Implement deterministic Risk Scoring (0-100) based on events.
- Implement incident correlation (linking multiple alerts to one incident).

### Phase 10: Evidence Capture
- Implement a rolling video buffer.
- On critical alerts, use FFmpeg to trim 10s before and 10s after the event into an evidence clip.
- Save to disk and link to the event in PostgreSQL.

### Phase 11: Mistral AI Integration
- Integrate Mistral API for generating natural language summaries of structured incident data.
- Add conversational capability for querying database events.

### Phase 12: Professional Dashboard (Frontend)
- Set up Vite + React + TypeScript + Tailwind CSS.
- Build command-center dark theme UI.
- Integrate Leaflet/MapLibre for camera/zone mapping.
- Implement live camera grid, active alerts list, and historical search.
- Implement RBAC (Admin, Supervisor, Operator, Analyst) using JWT.

### Phase 13: Performance Optimization & Demo Mode
- Implement frame skipping, multi-threading, and queue backpressure.
- Set up the Demo Control interface.

## Verification Plan

### Automated Tests
- Unit tests for the behavior logic (loitering, intersection, direction).
- Unit tests for database models and relationships.

### Manual Verification
- We will use sample MP4 videos and a local webcam to verify the pipeline end-to-end.
- Validate that the dashboard receives WebSocket events instantly without page refreshes.
- Validate that evidence clips are properly captured and playback correctly.
