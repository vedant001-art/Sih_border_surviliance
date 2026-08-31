import cv2
import threading
import time
from loguru import logger
import queue
import collections

class VideoStream:
    def __init__(self, camera_id: str, source: str, stream_type: str = "MP4"):
        self.camera_id = camera_id
        self.source = source
        self.stream_type = stream_type
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=1)  # Strictly maxsize=1 for zero-latency latest-frame delivery
        self.thread = None
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.fps = 0
        self.frame_counter = 0
        self.rolling_buffer = collections.deque(maxlen=150) # Keep last ~5s at 30fps

    def start(self):
        if self.stream_type == "WEBCAM":
            source = int(self.source) if self.source.isdigit() else self.source
        else:
            source = self.source
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source {self.source} for camera {self.camera_id}")
            return False
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or self.fps != self.fps:
            self.fps = 30 # fallback
            
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info(f"Started video stream for camera {self.camera_id}")
        return True

    def _update(self):
        start_time = time.time()
        frame_count = 0
        while self.running:
            if not self.cap.isOpened():
                break
                
            ret, frame = self.cap.read()
            if not ret:
                if self.stream_type != "WEBCAM":
                    # Smoothly seek to frame 0 without destroying VideoCapture handle
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.cap.release()
                        time.sleep(0.2)
                        self.cap = cv2.VideoCapture(self.source)
                        continue
                else:
                    break
            
            frame_count += 1
            if time.time() - start_time > 1.0:
                self.fps = frame_count / (time.time() - start_time)
                start_time = time.time()
                frame_count = 0
                
            self.frame_counter += 1
            self.latest_frame = frame.copy()
            self.rolling_buffer.append(self.latest_frame)
            
            # Non-blocking put with latest-frame strategy
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    # Drop the oldest frame to make room for the fresh one
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass
                except queue.Full:
                    pass
                
            # Smooth pacing for file streams to prevent freezing and CPU spikes
            if self.stream_type != "WEBCAM":
                time.sleep(0.025)
            
        self.cap.release()

    def get_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        logger.info(f"Stopped video stream for camera {self.camera_id}")

class StreamManager:
    def __init__(self):
        self.streams = {}
        
    def add_stream(self, camera_id: str, source: str, stream_type: str = "MP4"):
        if camera_id in self.streams:
            logger.warning(f"Stream {camera_id} already exists.")
            return False
            
        stream = VideoStream(camera_id, source, stream_type)
        if stream.start():
            self.streams[camera_id] = stream
            return True
        return False
        
    def remove_stream(self, camera_id: str):
        if camera_id in self.streams:
            self.streams[camera_id].stop()
            del self.streams[camera_id]
            return True
        return False
        
    def get_stream(self, camera_id: str) -> VideoStream:
        return self.streams.get(camera_id)

stream_manager = StreamManager()
