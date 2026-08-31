import time
import threading
from collections import deque
from loguru import logger

class OfflineAlertQueue:
    """
    Thread-safe FIFO Queue Data Structure for Store-and-Forward Alert Resiliency.
    When client/data connection is OFF, edge video pipeline does NOT stop
    analyzing or recording alerts. Every breach is safely stored in this queue.
    When reconnected, all buffered alerts are flushed and delivered in chronological order.
    """
    def __init__(self, maxsize: int = 1000):
        self._queue = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self.is_data_connected = True
        self.total_buffered_historical = 0

    def set_connection_status(self, connected: bool):
        with self._lock:
            self.is_data_connected = bool(connected)
            logger.info(f"[OfflineAlertQueue] Data Connection status changed: {'ONLINE' if self.is_data_connected else 'OFFLINE (BUFFERING ALERTS IN QUEUE)'}")

    def enqueue(self, alert_item: dict):
        """Enqueues an alert into the FIFO queue data structure."""
        with self._lock:
            item = dict(alert_item)
            item["queued_at"] = time.time()
            item["buffered_offline"] = not self.is_data_connected
            self._queue.append(item)
            self.total_buffered_historical += 1
            logger.info(f"[OfflineAlertQueue] Alert #{item.get('id', '')} stored in queue. Queue depth: {len(self._queue)} (Data Connected: {self.is_data_connected})")

    def flush(self) -> list:
        """Drains and returns all queued alerts when connection is restored."""
        with self._lock:
            alerts = []
            while self._queue:
                item = self._queue.popleft()
                item["buffered_offline"] = True
                item["is_buffered"] = True
                alerts.append(item)
            logger.info(f"[OfflineAlertQueue] Flushed {len(alerts)} buffered alerts upon reconnection.")
            return alerts

    def peek_queue(self) -> list:
        with self._lock:
            return list(self._queue)

    def count(self) -> int:
        with self._lock:
            return len(self._queue)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "queue_length": len(self._queue),
                "is_data_connected": self.is_data_connected,
                "total_buffered_historical": self.total_buffered_historical
            }

offline_alert_queue = OfflineAlertQueue()
