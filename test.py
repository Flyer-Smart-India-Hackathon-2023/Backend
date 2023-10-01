import requests

api_url = 'http://127.0.0.1:5000/classify' 

# Image file to be classified
image_file = 'clear-light-bulb.jpg'  

# Send a POST request with the image data
with open(image_file, 'rb') as f:
    files = {'image': (image_file, f)}
    response = requests.post(api_url, files=files)

# Check the response
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print('Error:', response.status_code)