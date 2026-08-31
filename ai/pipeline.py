from video.stream_manager import stream_manager
from ai.detection.yolo_detector import YOLODetector
from ai.behavior.virtual_fence import VirtualFence
from ai.behavior.loitering import LoiteringDetector
from ai.behavior.speed_direction import SpeedDirectionAnalyzer
from ai.behavior.abandoned_object import AbandonedObjectDetector
from ai.face.face_recognizer import FaceRecognizer
from ai.anpr.plate_reader import ANPRSystem
from backend.websocket.manager import manager as ws_manager
from backend.services.db_worker import db_worker
from backend.models.schema import TrackStatus
from backend.services.entity_registry import entity_registry
from backend.services.tracker import TrackMemory
from backend.services.face_reid_engine import reid_engine
from backend.services.anpr_engine import vehicle_attribute_engine
from backend.services.authorized_plates import authorized_plates
from loguru import logger
import threading
import time
import asyncio
import cv2
import numpy as np
from datetime import datetime

class CameraPipeline:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.detector = YOLODetector()
        self.motion_tracker = TrackMemory()
        
        self.virtual_fence = VirtualFence(zones=[])
        self.loitering_detector = LoiteringDetector(threshold_seconds=10.0)
        self.abandoned_detector = AbandonedObjectDetector(stationary_threshold_seconds=15.0)
        
        self.face_recognizer = FaceRecognizer()
        self.anpr_system = ANPRSystem()
        
        self.running = False
        self.ai_thread = None
        self.render_thread = None
        
        # Shared state between threads
        self.latest_detections = []
        self.latest_faces = []
        self.rendered_frame = None
        
        # Metrics
        self.ai_fps = 0.0
        self.display_fps = 0.0
        self.inference_latency_ms = 0
        
        self._plate_cache = {}
        self._plate_cache_ttl = 15.0
        self.track_classes = {}
        self.track_class_votes = {}
        self.authorized_tracks = set()
        self.zone_entry_times = {}
        self.virtual_fence_has_custom = False
        self._pending_fence_update = None
        self.virtual_fence_enabled = True
        self.approaching_alert_cache = {}

    @property
    def actual_fps(self):
        return self.display_fps if self.display_fps > 0 else self.ai_fps

    def start(self):
        self.running = True
        self.ai_thread = threading.Thread(target=self._ai_worker_thread, daemon=True)
        self.render_thread = threading.Thread(target=self._render_worker_thread, daemon=True)
        self.ai_thread.start()
        self.render_thread.start()
        logger.info(f"AI & Render Pipelines started for camera {self.camera_id}")

    def update_virtual_fence(self, zone_name: str, normalized_coords: list):
        self._pending_fence_update = {
            "zone_name": zone_name,
            "normalized_coords": normalized_coords
        }

    def _broadcast_sync(self, message: dict):
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(ws_manager.broadcast(message))
            loop.close()
        except Exception as e:
            logger.warning(f"WS broadcast error: {e}")

    def _ai_worker_thread(self):
        stream = stream_manager.get_stream(self.camera_id)
        if not stream:
            logger.error(f"Stream for {self.camera_id} not found.")
            return

        frame_count = 0
        start_time = time.time()
        frames_processed = 0

        while self.running:
            # Drain queue to ALWAYS fetch the latest frame and drop stale buffered frames (zero lag)
            frame = None
            while not stream.frame_queue.empty():
                try:
                    frame = stream.frame_queue.get_nowait()
                except Exception:
                    break
            if frame is None:
                try:
                    frame = stream.frame_queue.get(timeout=0.05)
                except Exception:
                    continue

            now = time.time()
            t0 = time.time()
            frame_count += 1
            
            if self._pending_fence_update:
                from ai.behavior.virtual_fence import VirtualFence
                zone = self._pending_fence_update
                h, w = frame.shape[:2]
                abs_coords = [(int(pt[0] * w), int(pt[1] * h)) for pt in zone["normalized_coords"]]
                self.virtual_fence = VirtualFence(zones=[{
                    "id": 1, "name": zone["zone_name"], "type": "POLYGON", "coords": abs_coords
                }])
                logger.info(f"Updated Virtual Fence '{zone['zone_name']}'")
                self._pending_fence_update = None

            # 1. AI Inference
            detections = self.detector.detect_and_track(frame)
            
            enhanced_detections = []
            current_track_ids = set()
            
            for det in detections:
                tid = det["track_id"]
                current_track_ids.add(tid)
                cls_name = det.get("class_name", "unknown")
                is_vehicle = det.get("is_vehicle", False)
                
                # Maintain temporal track class stability
                if tid not in self.track_class_votes:
                    self.track_class_votes[tid] = []
                self.track_class_votes[tid].append(cls_name)
                if len(self.track_class_votes[tid]) > 25:
                    self.track_class_votes[tid].pop(0)
                    
                # If this track was ever confirmed as a car (or has plate), do not degrade to motorcycle
                if "car" in self.track_class_votes[tid] and cls_name in ["motorcycle", "bicycle"]:
                    cls_name = "car"
                    det["class_name"] = "car"
                    det["class_id"] = 2
                    det["is_vehicle"] = True
                    is_vehicle = True

                self.track_classes[tid] = cls_name
                
                # Get or Create Global Entity (Zero-Drift)
                obj_type = "vehicle" if is_vehicle else "person"
                g_entity = entity_registry.get_or_create(self.camera_id, tid, obj_type)
                
                # Update motion tracker
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                self.motion_tracker.update(tid, frame_count, cx, cy)
                
                speed = self.motion_tracker.get_speed_kmh(tid)
                heading = self.motion_tracker.get_heading(tid)
                
                # Filter stationary phantom ground/road detections (zero-motion fixed artifacts)
                if obj_type == "person":
                    hist = self.motion_tracker.get_history(tid)
                    if len(hist) >= 12:
                        dx = abs(hist[-1][0] - hist[0][0])
                        dy = abs(hist[-1][1] - hist[0][1])
                        if dx < 6.0 and dy < 6.0 and speed < 0.5:
                            continue
                
                det["global_id"] = g_entity.entity_id
                det["speed_kmh"] = speed
                det["heading"] = heading
                
                # Assign attributes
                if obj_type == "person":
                    profile = reid_engine.get_person_profile(tid)
                    det.update(profile)
                    g_entity.update(profile)
                else:
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    profile = vehicle_attribute_engine.get_vehicle_profile(tid, cls_name, crop)
                    det.update(profile)
                    g_entity.update(profile)

                # Send track update to DB occasionally (fast real-time refresh)
                if entity_registry.can_write_db("track_update", g_entity.entity_id, cooldown_duration=2.0):
                    db_worker.enqueue_task("UPDATE_TRACK", {
                        "camera_id": self.camera_id,
                        "local_track_id": tid,
                        "object_type": obj_type,
                        "class_name": cls_name,
                        "confidence": det.get("confidence", 0.0),
                        "status": TrackStatus.ACTIVE
                    })

                # Position sampling (every 30 frames approx 1 sec at 30fps)
                if frame_count % 30 == 0:
                    db_worker.enqueue_task("SAVE_POSITION", {
                        "camera_id": self.camera_id,
                        "local_track_id": tid,
                        "object_type": obj_type,
                        "class_name": cls_name,
                        "confidence": det.get("confidence", 0.0),
                        "frame_number": frame_count,
                        "center_x": cx,
                        "center_y": cy,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
                    
                enhanced_detections.append(det)

            entity_registry.cleanup()
            self.motion_tracker.cleanup(frame_count)
            self._evaluate_rules(enhanced_detections, frame)

            # 2. ANPR Processing
            from ai.detection.plate_detector import PlateDetector
            from backend.services.anpr_fusion import anpr_fusion
            import os
            import concurrent.futures
            
            if not hasattr(self, 'plate_detector'):
                self.plate_detector = PlateDetector()
                self.anpr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                os.makedirs("evidence/plates", exist_ok=True)
                os.makedirs("evidence/vehicles", exist_ok=True)
                
            def process_anpr_async(det, frame, now):
                tid = det["track_id"]
                g_id = det.get("global_id", f"{self.camera_id}_{tid}")
                x1, y1, x2, y2 = map(int, det["bbox"])
                h, w = frame.shape[:2]
                veh_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if veh_crop is None or veh_crop.size == 0:
                    return

                # 1. Try YOLO Plate Detector first
                plate_det = self.plate_detector.detect_in_crop(veh_crop, global_offset_x=max(0, x1), global_offset_y=max(0, y1))
                plate_crop = None
                plate_bbox = None
                plate_conf = 0.5

                if plate_det:
                    plate_crop = plate_det["crop"]
                    plate_bbox = plate_det["bbox"]
                    plate_conf = plate_det["confidence"]
                else:
                    # 2. Fallback: Lower half of vehicle where plates are situated
                    vh, vw = veh_crop.shape[:2]
                    if vh >= 25 and vw >= 35:
                        plate_crop = veh_crop[int(vh * 0.45):, :]
                        plate_bbox = [x1, y1 + int(vh * 0.45), x2, y2]

                if plate_crop is not None and plate_crop.size > 0:
                    # 3. Plate Reading (EasyOCR)
                    plate_res = self.anpr_system.read_plate(plate_crop)
                    if plate_res:
                        # 4. Temporal Fusion
                        fused_plate = anpr_fusion.add_observation(tid, plate_res, plate_crop)
                        if fused_plate and fused_plate.get("normalized_text"):
                            final_text = fused_plate["normalized_text"]
                            # A vehicle with a detected license plate is confirmed as a Car/Four-wheeler
                            det["class_name"] = "car"
                            self.track_classes[tid] = "car"
                            if tid in self.track_class_votes:
                                self.track_class_votes[tid].append("car")
                                
                            self._plate_cache[tid] = {
                                "plate": final_text,
                                "conf": fused_plate["confidence"],
                                "last_seen": now,
                                "bbox": plate_bbox or [x1, y1, x2, y2]
                            }
                            g_ent = entity_registry.entities.get(g_id)
                            if g_ent:
                                g_ent.update({"plate": final_text, "class_name": "car", "type": "Car"})

                            raw_ocr = fused_plate.get("raw_text", "")
                            if authorized_plates.is_authorized(final_text) or authorized_plates.is_authorized(raw_ocr):
                                self.authorized_tracks.add(tid)
                                logger.info(f"[{self.camera_id}] Track #{tid} confirmed AUTHORIZED with plate '{final_text}'. Alerts will be suppressed.")
                                
                            # Save evidence image
                            plate_img_path = ""
                            if fused_plate["confidence"] > 0.35:
                                plate_img_path = f"evidence/plates/plate_{self.camera_id}_T{tid}_{int(now)}.jpg"
                                cv2.imwrite(plate_img_path, fused_plate["best_crop"])
                                veh_img_path = f"evidence/vehicles/veh_{self.camera_id}_T{tid}_{int(now)}.jpg"
                                cv2.imwrite(veh_img_path, veh_crop)
                                
                            # DB Write with cooldown per plate (Record keeping only)
                            if entity_registry.can_write_db("anpr", f"{self.camera_id}_{tid}_{final_text}", cooldown_duration=3.0):
                                db_worker.enqueue_task("SAVE_ANPR", {
                                    "camera_id": self.camera_id,
                                    "local_track_id": tid,
                                    "vehicle_type": "Car",
                                    "plate_text": final_text,
                                    "raw_ocr_text": fused_plate["raw_text"],
                                    "ocr_confidence": fused_plate["confidence"],
                                    "plate_detection_confidence": plate_conf,
                                    "plate_image_path": plate_img_path
                                })
            
            for det in enhanced_detections:
                if det.get("is_vehicle", False):
                    tid = det["track_id"]
                    if anpr_fusion.can_run_ocr(tid):
                        # Mark timestamp immediately to prevent spamming executor queue
                        anpr_fusion.last_ocr_time[tid] = now
                        self.anpr_executor.submit(process_anpr_async, det, frame.copy(), now)
            
            active_tids = {d["track_id"] for d in enhanced_detections}
            anpr_fusion.cleanup(active_tids)
            
            expired = [tid for tid, info in self._plate_cache.items() if now - info["last_seen"] > self._plate_cache_ttl]
            for tid in expired:
                del self._plate_cache[tid]

            t1 = time.time()
            self.inference_latency_ms = int((t1 - t0) * 1000)
            self.latest_detections = enhanced_detections

            # Metrics
            frames_processed += 1
            if time.time() - start_time > 1.0:
                self.ai_fps = frames_processed / (time.time() - start_time)
                start_time = time.time()
                frames_processed = 0

            # Broadcast
            if frame_count % 3 == 0:
                det_summary = []
                for d in enhanced_detections:
                    entry = {
                        "track_id": d["track_id"],
                        "global_id": d["global_id"],
                        "class_name": d["class_name"],
                        "speed_kmh": d.get("speed_kmh", 0),
                        "heading": d.get("heading", ""),
                        "confidence": round(d["confidence"], 2),
                    }
                    if d.get("is_vehicle"):
                        entry["type"] = d.get("type")
                        entry["color"] = d.get("color")
                        entry["make"] = d.get("make")
                    else:
                        entry["name"] = d.get("name")
                        entry["clearance"] = d.get("clearance")
                        entry["clothing"] = d.get("clothing")
                        
                    if d["track_id"] in self._plate_cache:
                        entry["plate"] = self._plate_cache[d["track_id"]]["plate"]
                    det_summary.append(entry)
                    
                payload = {
                    "type": "detections",
                    "camera_id": self.camera_id,
                    "data": det_summary
                }
                threading.Thread(target=self._broadcast_sync, args=(payload,), daemon=True).start()
            
            # Yield GIL
            time.sleep(0.01)

    def _render_worker_thread(self):
        stream = stream_manager.get_stream(self.camera_id)
        if not stream:
            return

        start_time = time.time()
        frames_rendered = 0
        _last_rendered_id = id(None)
        _target_fps = 30.0

        while self.running:
            frame = stream.latest_frame
            if frame is None:
                time.sleep(0.033)
                continue

            # Only render when the stream has a new frame (avoids duplicate renders = less lag)
            frame_id = id(frame)
            if frame_id == _last_rendered_id:
                time.sleep(0.008)
                continue
            _last_rendered_id = frame_id

            # Adapt render rate to stream FPS to prevent CPU-induced lag
            _target_fps = max(10.0, min(stream.fps if stream.fps > 1 else 30.0, 30.0))

            annotated = frame.copy()
            overlay = annotated.copy()

            h_img, w_img = annotated.shape[:2]
            scale = max(1.0, w_img / 640.0)
            
            font_scale = 0.55 * scale
            font_thick = max(1, int(1.8 * scale))
            box_thick = max(2, int(2.5 * scale))
            corner_len = max(15, int(22 * scale))
            trail_thick_bg = max(4, int(8 * scale))
            trail_thick_fg = max(2, int(3 * scale))

            # Draw Virtual Fence zones on live video
            if getattr(self, 'virtual_fence_enabled', True) and hasattr(self, 'virtual_fence') and self.virtual_fence.polygons:
                for zone_name, poly in self.virtual_fence.polygons.items():
                    try:
                        ext_coords = list(poly.exterior.coords)
                        if len(ext_coords) >= 3:
                            pts = np.array(ext_coords, np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(overlay, [pts], (0, 0, 160))
                            cv2.polylines(annotated, [pts], True, (0, 0, 255), max(2, int(2.5 * scale)))
                            cx = int(poly.centroid.x)
                            cy = int(poly.centroid.y)
                            cv2.putText(annotated, f"VIRTUAL FENCE: {zone_name}", (max(10, cx - int(100*scale)), max(int(30*scale), cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), max(2, int(2*scale)))
                    except Exception:
                        pass
            
            # Draw trajectories and motion paths for tracked targets
            for det in self.latest_detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                # Clamp coordinates strictly inside image bounds
                x1 = max(0, min(w_img - 1, x1))
                y1 = max(0, min(h_img - 1, y1))
                x2 = max(0, min(w_img - 1, x2))
                y2 = max(0, min(h_img - 1, y2))
                
                bw = x2 - x1
                bh = y2 - y1
                if bw < 10 or bh < 10:
                    continue

                tid = det["track_id"]
                hist = self.motion_tracker.get_history(tid)
                is_veh = det.get("is_vehicle", False)
                trail_color = (255, 220, 0) if is_veh else (0, 255, 100)
                
                if len(hist) >= 1:
                    rev_hist = list(reversed(hist))
                    valid_trail = []
                    for p in rev_hist:
                        px = int(max(2, min(w_img - 2, p[0])))
                        py = int(max(2, min(h_img - 2, p[1])))
                        if not valid_trail or np.hypot(px - valid_trail[-1][0], py - valid_trail[-1][1]) < (150 * scale):
                            valid_trail.append((px, py))
                        else:
                            break
                    
                    if len(valid_trail) >= 2:
                        pts = np.array(valid_trail, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(overlay, [pts], False, trail_color, trail_thick_bg, cv2.LINE_AA)
                        cv2.polylines(annotated, [pts], False, (255, 255, 255), trail_thick_fg, cv2.LINE_AA)
                    
                    if len(valid_trail) >= 1:
                        cv2.circle(overlay, valid_trail[-1], max(4, int(5 * scale)), trail_color, -1)
                        cv2.circle(annotated, valid_trail[0], max(4, int(5 * scale)), (255, 255, 255), -1)
                    
                is_vehicle = det.get("is_vehicle", False)
                color = (255, 180, 0) if is_vehicle else (0, 255, 0)
                
                # Full bounding box outline + corner brackets
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, max(1, int(box_thick / 2)))
                cv2.line(annotated, (x1, y1), (x1+corner_len, y1), color, box_thick)
                cv2.line(annotated, (x1, y1), (x1, y1+corner_len), color, box_thick)
                cv2.line(annotated, (x2, y1), (x2-corner_len, y1), color, box_thick)
                cv2.line(annotated, (x2, y1), (x2, y1+corner_len), color, box_thick)
                cv2.line(annotated, (x1, y2), (x1+corner_len, y2), color, box_thick)
                cv2.line(annotated, (x1, y2), (x1, y2-corner_len), color, box_thick)
                cv2.line(annotated, (x2, y2), (x2-corner_len, y2), color, box_thick)
                cv2.line(annotated, (x2, y2), (x2, y2-corner_len), color, box_thick)
                
                # Position Info Panel: If near top of screen (y1 < 75), render BELOW y2 to prevent box overflow off-screen!
                draw_above = y1 >= int(75 * scale)
                y_offset = y1 - int(10 * scale) if draw_above else y2 + int(15 * scale)
                
                if is_vehicle:
                    plate = self._plate_cache.get(tid, {}).get("plate", "SCANNING...")
                    v_type = det.get('type') or det.get('class_name', 'Vehicle').capitalize()
                    info_lines = [
                        f"TRACK #{tid} | {v_type}",
                        f"PLATE: {plate}",
                        f"SPD: {det.get('speed_kmh', 0):.0f}km/h {det.get('heading', '')}"
                    ]
                else:
                    info_lines = [
                        f"TRACK #{tid} | {det.get('name', 'Subject')}",
                        f"{det.get('clearance', 'UNVERIFIED')}",
                        f"{det.get('clothing', '')}"
                    ]
                    
                line_order = reversed(info_lines) if draw_above else info_lines
                for text in line_order:
                    if not text.strip(): continue
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                    box_x1 = max(5, min(w_img - tw - int(12 * scale), x1))
                    box_y1 = max(5, min(h_img - th - int(10 * scale), y_offset - th - int(6 * scale))) if draw_above else max(5, min(h_img - th - int(10 * scale), y_offset))
                    box_x2 = min(w_img - 2, box_x1 + tw + int(8 * scale))
                    box_y2 = min(h_img - 2, box_y1 + th + int(10 * scale))
                    
                    cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), (15, 18, 24), -1)
                    cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), color, 1)
                    cv2.putText(annotated, text, (box_x1 + int(4 * scale), box_y1 + th + int(2 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)
                    if draw_above:
                        y_offset -= (th + int(10 * scale))
                    else:
                        y_offset += (th + int(10 * scale))
                    
                # Draw Plate Box if available
                if is_vehicle and tid in self._plate_cache:
                    p_info = self._plate_cache[tid]
                    if "bbox" in p_info:
                        px1, py1, px2, py2 = p_info["bbox"]
                        px1 = max(0, min(w_img - 1, px1))
                        py1 = max(0, min(h_img - 1, py1))
                        px2 = max(0, min(w_img - 1, px2))
                        py2 = max(0, min(h_img - 1, py2))
                        if (px2 - px1) > 10 and (py2 - py1) > 5:
                            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 165, 255), max(2, int(2*scale)))
                            cv2.putText(annotated, "PLATE", (px1, max(15, py1 - int(5*scale))), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 165, 255), max(1, int(1.5*scale)))

            # Apply overlay blend for transparent fence & glowing trajectory ribbons
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

            # Draw Performance Metrics
            metrics = [
                f"DISPLAY FPS: {self.display_fps:.1f}",
                f"AI FPS: {self.ai_fps:.1f}",
                f"LATENCY: {self.inference_latency_ms} ms",
                f"QUEUE: {stream.frame_queue.qsize()}"
            ]
            for i, line in enumerate(metrics):
                cv2.putText(annotated, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                self.rendered_jpeg_bytes = buf.tobytes()
            self.rendered_frame = annotated
            
            frames_rendered += 1
            if time.time() - start_time > 1.0:
                self.display_fps = frames_rendered / (time.time() - start_time)
                start_time = time.time()
                frames_rendered = 0

            time.sleep(1.0 / _target_fps)

    def _evaluate_rules(self, detections, frame):
        # Only produce alerts when Virtual Fence feature is enabled
        if not getattr(self, 'virtual_fence_enabled', True):
            return

        # Auto-initialize restricted perimeter zone if none configured
        if not self.virtual_fence.polygons and not getattr(self, 'virtual_fence_has_custom', False) and frame is not None:
            h, w = frame.shape[:2]
            self.virtual_fence = VirtualFence(zones=[{
                "id": 1,
                "name": f"Restricted Sector {self.camera_id}",
                "type": "POLYGON",
                "coords": [(0, int(h * 0.35)), (w, int(h * 0.35)), (w, h), (0, h)]
            }])

        # Alerts & Predictive Approaching Trajectory Analysis
        intrusion_events = self.virtual_fence.check_intrusion(detections)
        approaching_events = self.virtual_fence.check_approaching(detections, motion_tracker=self.motion_tracker)
        all_events = intrusion_events + approaching_events
        now = time.time()
        
        for event in all_events:
            try:
                tid = event.get("track_id", "")
                evt_type = event.get("type", "INTRUSION")
                cls_name = event.get("class_name", self.track_classes.get(tid, "target")).lower()
                is_veh = cls_name in ["car", "truck", "bus", "vehicle", "motorcycle"]
                zone = event.get('zone_name', f'Restricted Sector {self.camera_id}')
                det = next((d for d in detections if d.get("track_id") == tid), {})

                if evt_type == "APPROACHING":
                    # Check 5-second cooldown per target to prevent duplicate warnings
                    if now - self.approaching_alert_cache.get(tid, 0) < 5.0:
                        continue
                    self.approaching_alert_cache[tid] = now

                    # Suppress approaching alert if track is pre-authorized or whitelisted
                    if tid in self.authorized_tracks:
                        continue
                    plate = self._plate_cache.get(tid, {}).get("plate")
                    if plate and authorized_plates.is_authorized(plate):
                        self.authorized_tracks.add(tid)
                        continue

                    plate_disp = plate if plate else f"IND-P{tid:04d}"
                    event_title = f"APPROACHING RESTRICTED ZONE: {cls_name.upper()} #{tid}"
                    event_msg = f"WARNING: Trajectory of {cls_name.upper()} #{tid} (Plate: {plate_disp}) is approaching Restricted Sector {zone}."
                    logger.warning(f"[{self.camera_id}] APPROACHING ALERT: {event_msg}")

                    db_worker.enqueue_task("CREATE_ALERT", {
                        "type": "APPROACHING",
                        "camera_id": self.camera_id,
                        "local_track_id": tid,
                        "object_type": "vehicle" if is_veh else "person",
                        "class_name": cls_name,
                        "title": event_title,
                        "message": event_msg,
                        "severity": "HIGH",
                        "timestamp": datetime.now().isoformat(),
                        "details": {
                            "object_type": "vehicle" if is_veh else "person",
                            "class_name": cls_name,
                            "plate": plate_disp if is_veh else None,
                            "speed_kmh": round(det.get("speed_kmh", 0), 1),
                            "zone_name": zone,
                            "event_type": "APPROACHING"
                        }
                    })
                    continue
                
                if is_veh:
                    # 1. If this track is already confirmed authorized, SUPPRESS!
                    if tid in self.authorized_tracks:
                        logger.info(f"[{self.camera_id}] Track #{tid} is pre-authorized. Alert SUPPRESSED.")
                        continue

                    # 2. Check plate cache or entity registry
                    plate = self._plate_cache.get(tid, {}).get("plate")
                    if not plate:
                        g_id = f"{self.camera_id}_{tid}"
                        g_ent = entity_registry.entities.get(g_id)
                        if g_ent and getattr(g_ent, 'plate', None):
                            plate = g_ent.plate

                    # 3. Check if plate is in the authorized whitelist
                    if plate and authorized_plates.is_authorized(plate):
                        self.authorized_tracks.add(tid)
                        logger.info(f"[{self.camera_id}] Authorized vehicle {cls_name.upper()} #{tid} (Plate: {plate}) entered {zone} - Alert SUPPRESSED.")
                        continue

                    # 4. Trigger alert for unauthorized or unregistered vehicle breach
                    plate_disp = plate if plate else f"IND-P{tid:04d}"
                    event_title = f"UNAUTHORIZED VEHICLE: {cls_name.upper()} #{tid}"
                    event_msg = f"UNAUTHORIZED {cls_name.upper()} #{tid} (Plate: {plate_disp}) crossed Virtual Fence into {zone}."
                    logger.warning(f"[{self.camera_id}] SECURITY ALERT: {event_msg}")
                    
                    details = {
                        "object_type": "vehicle",
                        "class_name": cls_name,
                        "plate": plate_disp,
                        "type": det.get("type") or cls_name.capitalize(),
                        "make": det.get("make", "Vehicle"),
                        "color": det.get("color", "Unknown"),
                        "speed_kmh": round(det.get("speed_kmh", 0), 1),
                        "zone_name": zone
                    }
                    target_obj_type = "vehicle"
                else:
                    target_obj_type = "person"
                    p_name = det.get("name", f"Subject #{tid}")
                    p_clearance = det.get("clearance", "UNVERIFIED")
                    p_clothing = det.get("clothing", "Standard Wear")
                    p_activity = det.get("activity", "Walking")
                    p_speed = round(det.get("speed_kmh", 0), 1)
                    
                    event_title = f"PERIMETER BREACH: PERSON #{tid}"
                    event_msg = f"PERSON #{tid} ({p_name}, {p_clearance}, Clothing: {p_clothing}) crossed Virtual Fence into {zone}."
                    logger.warning(f"[{self.camera_id}] SECURITY ALERT: {event_msg}")
                    
                    details = {
                        "object_type": "person",
                        "class_name": "person",
                        "name": p_name,
                        "clearance": p_clearance,
                        "clothing": p_clothing,
                        "activity": p_activity,
                        "speed_kmh": p_speed,
                        "zone_name": zone
                    }
            
                if entity_registry.can_write_db("alert", f"{self.camera_id}_{tid}_{evt_type}", cooldown_duration=6.0):
                    severity = "CRITICAL" if ("INTRUSION" in evt_type or is_veh) else "HIGH"
                    
                    db_worker.enqueue_task("CREATE_EVENT", {
                        "type": evt_type,
                        "camera_id": self.camera_id,
                        "local_track_id": tid,
                        "object_type": target_obj_type,
                        "class_name": cls_name,
                        "title": event_title,
                        "message": event_msg,
                        "details": details,
                        "severity": severity
                    })
                    
                    alert_payload = {
                        "type": "ALERT",
                        "camera_id": self.camera_id,
                        "data": {
                            "severity": severity,
                            "title": event_title,
                            "message": event_msg,
                            "camera_id": self.camera_id,
                            "track_id": tid,
                            "object_type": target_obj_type,
                            "class_name": cls_name,
                            "plate": details.get("plate", "N/A"),
                            "details": details,
                            "person_info": details if target_obj_type == "person" else None,
                            "vehicle_info": details if target_obj_type == "vehicle" else None,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    threading.Thread(target=self._broadcast_sync, args=(alert_payload,), daemon=True).start()
            except Exception as e:
                logger.error(f"[{self.camera_id}] Error evaluating intrusion rule for event: {e}")

    def stop(self):
        self.running = False
        if self.ai_thread: self.ai_thread.join(timeout=3)
        if self.render_thread: self.render_thread.join(timeout=3)
        logger.info(f"AI Pipeline stopped for camera {self.camera_id}")
