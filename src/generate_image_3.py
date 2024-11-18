import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline
from datetime import datetime

# torch.set_default_device("mps")

device = "mps"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

url = "https://cache.magicmaman.com/data/photo/w1000_ci/6w/rihanna-video-fils-riot-rose.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "Make the person in the picture a viking warrior with a sword and a shield, really realistic, in portrait mode. The background should have sky should be overcast, with dark clouds and a hint of sunlight breaking through. The ocean should be rough, with white-capped waves crashing against the rocks below. The cliff should be rugged and rocky, with sparse vegetation."

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"/image/{timestamp}.png"

images[0].save("fantasy_landscape.png")