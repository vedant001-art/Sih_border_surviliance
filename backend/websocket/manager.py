from typing import Dict, List
from fastapi import WebSocket
import json
from loguru import logger

class ConnectionManager:
    def __init__(self):
        # Maps client_id to WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket Client {client_id} connected")
        
        # When client reconnects, restore data connection and flush buffered queue
        from backend.services.offline_alert_queue import offline_alert_queue
        offline_alert_queue.set_connection_status(True)
        buffered = offline_alert_queue.flush()
        if buffered:
            logger.info(f"Delivering {len(buffered)} queued offline alerts to newly connected client {client_id}")
            for b_alert in buffered:
                try:
                    await websocket.send_text(json.dumps({
                        "type": "ALERT",
                        "data": b_alert,
                        "buffered_sync": True
                    }))
                except Exception:
                    pass

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket Client {client_id} disconnected")
            # If all clients disconnected (e.g. WiFi off / offline), switch queue to buffering mode
            if len(self.active_connections) == 0:
                from backend.services.offline_alert_queue import offline_alert_queue
                offline_alert_queue.set_connection_status(False)

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def broadcast(self, message: dict):
        # We send JSON stringified messages
        payload = json.dumps(message)
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(payload)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")

manager = ConnectionManager()
