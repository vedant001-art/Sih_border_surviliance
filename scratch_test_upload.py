import requests

url = 'http://localhost:8000/api/v1/cameras/upload'
files = {'file': ('test.txt', 'dummy content')}
try:
    response = requests.post(url, files=files)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.text)
except Exception as e:
    print("Error:", e)
