import requests
from dotenv import load_dotenv
import os

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
load_dotenv()
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	if response.status_code != 200:
		raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
	return response.content

try:
	image_bytes = query({
		"inputs": "Make me a viking background. Make it realistic, in portrait mode. The sky should be overcast, with dark clouds and a hint of sunlight breaking through. The ocean should be rough, with white-capped waves crashing against the rocks below. The cliff should be rugged and rocky, with sparse vegetation.",
	})
except Exception as e:
	print(e)
	image_bytes = None

import io
from PIL import Image
from datetime import datetime

# Try opening the image
if image_bytes:
	try:
		image = Image.open(io.BytesIO(image_bytes))
		image.show()
	except Exception as e:
		print(f"Failed to open image: {e}")
else:
	print("No image bytes to display")
# Save the image bytes to a file for debugging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"/image/{timestamp}.png"

with open(filename, "wb") as f:
	f.write(image_bytes)

# Try opening the image
image = Image.open(io.BytesIO(image_bytes))
image.show()