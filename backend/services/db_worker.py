import queue
import threading
from loguru import logger
from backend.core.database import SessionLocal
from backend.models.schema import Track, TrackPosition, Vehicle, ANPRRecord, Event, Alert, AlertSeverity, TrackStatus, Camera
from backend.services.mistral_service import mistral_service
from datetime import datetime
import json

class DBWorker:
    def __init__(self):
        self.task_queue = queue.Queue(maxsize=1000)
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info("Database Worker Queue started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3.0)

    def enqueue_task(self, task_type: str, data: dict):
        try:
            self.task_queue.put_nowait({"type": task_type, "data": data})
        except queue.Full:
            logger.warning(f"DB Worker queue is full! Dropping task {task_type}")

    def _worker_loop(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                self._process_task(task)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"DB Worker exception: {e}")

    def _process_task(self, task: dict):
        task_type = task["type"]
        data = task["data"]
        db = SessionLocal()
        try:
            if task_type == "UPDATE_TRACK":
                self._handle_update_track(db, data)
            elif task_type == "SAVE_POSITION":
                self._handle_save_position(db, data)
            elif task_type == "SAVE_ANPR":
                self._handle_save_anpr(db, data)
            elif task_type in ["CREATE_EVENT", "CREATE_ALERT"]:
                self._handle_create_event(db, data)
            db.commit()
        except Exception as e:
            logger.error(f"DB task '{task_type}' failed: {e}")
            db.rollback()
        finally:
            db.close()

    def _ensure_camera(self, db, camera_id: str):
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if not cam:
            db.add(Camera(id=camera_id, name=f"Camera {camera_id}"))
            db.flush()

    def _get_or_create_track(self, db, camera_id: str, local_track_id: int, object_type: str, class_name: str, confidence: float):
        self._ensure_camera(db, camera_id)
        track = db.query(Track).filter(
            Track.camera_id == camera_id,
            Track.local_track_id == local_track_id
        ).first()
        
        if not track:
            track = Track(
                local_track_id=local_track_id,
                camera_id=camera_id,
                object_type=object_type,
                class_name=class_name,
                initial_confidence=confidence,
                last_confidence=confidence,
                status=TrackStatus.ACTIVE
            )
            db.add(track)
            db.flush()
        return track

    def _handle_update_track(self, db, data: dict):
        track = self._get_or_create_track(
            db, data["camera_id"], data["local_track_id"], 
            data["object_type"], data["class_name"], data["confidence"]
        )
        track.last_seen = datetime.now()
        track.last_confidence = data["confidence"]
        track.duration_seconds = (track.last_seen - track.first_seen).total_seconds()
        
        if "status" in data:
            track.status = data["status"]
            
        if data.get("object_type") == "vehicle":
            veh = db.query(Vehicle).filter(Vehicle.track_id == track.id).first()
            new_type = data.get("class_name", "Vehicle").capitalize()
            if not veh:
                veh = Vehicle(
                    track_id=track.id,
                    camera_id=data["camera_id"],
                    vehicle_type=new_type,
                    status="ACTIVE",
                    first_seen=track.first_seen,
                    last_seen=track.last_seen
                )
                db.add(veh)
            else:
                veh.last_seen = track.last_seen
                veh.status = "ACTIVE"
                # Update vehicle_type if updated to a four-wheeler class
                if new_type in ["Car", "Truck", "Bus"] and veh.vehicle_type not in ["Car", "Truck", "Bus"]:
                    veh.vehicle_type = new_type

    def _handle_save_position(self, db, data: dict):
        track = self._get_or_create_track(
            db, data["camera_id"], data["local_track_id"], 
            data["object_type"], data["class_name"], data["confidence"]
        )
        pos = TrackPosition(
            track_id=track.id,
            frame_number=data.get("frame_number"),
            center_x=data["center_x"],
            center_y=data["center_y"],
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
            confidence=data["confidence"]
        )
        db.add(pos)

    def _handle_save_anpr(self, db, data: dict):
        track = self._get_or_create_track(
            db, data["camera_id"], data["local_track_id"], 
            "vehicle", data.get("vehicle_type", "vehicle"), 0.0
        )
        
        veh = db.query(Vehicle).filter(Vehicle.track_id == track.id).first()
        if not veh:
            veh = Vehicle(
                track_id=track.id,
                camera_id=data["camera_id"],
                vehicle_type=data.get("vehicle_type", "vehicle"),
                plate_number=data["plate_text"],
                plate_confidence=data["ocr_confidence"]
            )
            db.add(veh)
            db.flush()
        else:
            # Update best observation
            if data["ocr_confidence"] > (veh.plate_confidence or 0):
                veh.plate_number = data["plate_text"]
                veh.plate_confidence = data["ocr_confidence"]
                
        anpr = ANPRRecord(
            vehicle_id=veh.id,
            camera_id=data["camera_id"],
            plate_text=data["plate_text"],
            raw_ocr_text=data.get("raw_ocr_text"),
            ocr_confidence=data["ocr_confidence"]
        )
        db.add(anpr)
        db.flush()
        
        # Broadcast ANPR confirmation over WebSocket
        self._broadcast({
            "type": "ANPR",
            "data": {
                "id": anpr.id,
                "camera_id": data["camera_id"],
                "plate_text": data["plate_text"],
                "ocr_confidence": data["ocr_confidence"],
                "vehicle_type": data.get("vehicle_type", "vehicle"),
                "timestamp": datetime.now().isoformat(),
                "track_id": data.get("local_track_id")
            }
        })

    def _broadcast(self, msg_dict: dict):
        try:
            import asyncio
            from backend.websocket.manager import manager as ws_manager
            
            # Run in a new or running event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws_manager.broadcast(msg_dict), loop)
                    return
            except Exception:
                pass
                
            new_loop = asyncio.new_event_loop()
            new_loop.run_until_complete(ws_manager.broadcast(msg_dict))
            new_loop.close()
        except Exception as e:
            logger.debug(f"WS Broadcast error: {e}")

    def _handle_create_event(self, db, data: dict):
        self._ensure_camera(db, data["camera_id"])
        
        # Optionally generate Mistral summary
        summary = None
        if mistral_service.enabled:
            # We don't want to block the queue completely, but making a quick HTTP call 
            # inside the DB worker is acceptable since it decouples from the video AI loop.
            summary = mistral_service.generate_event_summary(data)
        else:
            summary = mistral_service._deterministic_fallback(data)
            
        track_id = None
        if "local_track_id" in data:
            track = db.query(Track).filter(
                Track.camera_id == data["camera_id"],
                Track.local_track_id == data["local_track_id"]
            ).first()
            if track:
                track_id = track.id

        evt = Event(
            event_type=data["type"],
            camera_id=data["camera_id"],
            track_id=track_id,
            object_type=data.get("object_type"),
            severity=data.get("severity", AlertSeverity.LOW),
            title=data.get("title", data["type"]),
            description=json.dumps(data),
            summary=summary
        )
        db.add(evt)
        db.flush()
        
        alt_message = data.get("message") or summary or evt.title
        alt = Alert(
            event_id=evt.id,
            severity=evt.severity,
            title=evt.title,
            message=alt_message,
            camera_id=evt.camera_id,
            track_id=evt.track_id
        )
        db.add(alt)
        db.flush()

        # Prepare alert payload
        severity_str = evt.severity.value if hasattr(evt.severity, 'value') else str(evt.severity)
        alert_payload = {
            "id": alt.id,
            "event_id": evt.id,
            "severity": severity_str,
            "title": evt.title,
            "message": alt.message,
            "camera_id": evt.camera_id,
            "track_id": data.get("local_track_id"),
            "object_type": evt.object_type,
            "class_name": data.get("class_name"),
            "details": data.get("details", {}),
            "person_info": data.get("details") if evt.object_type == "person" else None,
            "vehicle_info": data.get("details") if evt.object_type == "vehicle" else None,
            "timestamp": datetime.now().isoformat(),
            "status": "NEW"
        }

        # Store-and-Forward Queue Data Structure:
        # If data connection is off, alerts are buffered in the queue and never lost.
        from backend.services.offline_alert_queue import offline_alert_queue
        if not offline_alert_queue.is_data_connected:
            offline_alert_queue.enqueue(alert_payload)
            logger.info(f"[OfflineBuffer] Data connection is OFF. Video recording continued; alert #{alt.id} saved in queue data structure ({offline_alert_queue.count()} buffered).")
        else:
            # Broadcast Event & Alert over WebSocket if data connection is active
            self._broadcast({
                "type": "ALERT",
                "data": alert_payload
            })

db_worker = DBWorker()
