import requests
import sys

if len(sys.argv) < 3:
    print("Usage: python enroll_face.py <name> <path_to_image>")
    sys.exit(1)

name = sys.argv[1]
image_path = sys.argv[2]

url = "http://localhost:8000/api/v1/faces/enroll"

try:
    with open(image_path, 'rb') as f:
        files = {'file': (image_path, f, 'image/jpeg')}
        data = {'name': name}
        response = requests.post(url, files=files, data=data)
        print("Response:", response.json())
except Exception as e:
    print("Error:", e)
