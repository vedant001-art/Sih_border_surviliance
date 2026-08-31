from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from .manager import manager
import uuid

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # For a real system we would authenticate here
    client_id = str(uuid.uuid4())
    await manager.connect(client_id, websocket)
    try:
        while True:
            # We don't necessarily expect messages from the dashboard in this basic implementation,
            # but we can listen for pings or control commands.
            data = await websocket.receive_text()
            if data == "ping":
                await manager.send_personal_message("pong", client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
