import urllib.request
import os

url = "http://127.0.0.1:8000/api/v1/cameras/upload"
video_path = os.path.abspath("uploads/example_vid.mp4")

if not os.path.exists(video_path):
    print("Test video file not found")
    exit(1)

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
with open(video_path, "rb") as f:
    video_bytes = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test_upload.mp4"\r\n'
    f"Content-Type: video/mp4\r\n\r\n"
).encode("utf-8") + video_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    url,
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        print("Status:", resp.status)
        print("Response:", resp.read().decode("utf-8"))
except Exception as e:
    print("Upload failed:", e)
