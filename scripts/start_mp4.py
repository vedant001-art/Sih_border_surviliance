import requests
import sys

if len(sys.argv) < 2:
    print("Usage: python start_mp4.py <path_to_mp4_file>")
    sys.exit(1)

video_path = sys.argv[1]

url = "http://localhost:8000/api/v1/cameras/start"
payload = {
    "camera_id": "CAM-02",
    "source": video_path,
    "stream_type": "MP4"
}

try:
    response = requests.post(url, json=payload)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
