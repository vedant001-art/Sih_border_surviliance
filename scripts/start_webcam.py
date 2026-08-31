import requests

url = "http://localhost:8000/api/v1/cameras/start"
payload = {
    "camera_id": "CAM-01",
    "source": "0",  # Webcam index 0
    "stream_type": "WEBCAM"
}

try:
    response = requests.post(url, json=payload)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
