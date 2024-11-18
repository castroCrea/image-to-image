📁 stable-diffusion-project/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📁 src/
    └── 📄 generate_image.py

# First, create the above folder structure
mkdir stable-diffusion-project
cd stable-diffusion-project
mkdir src

# Contents of README.md:
# Stable Diffusion 3 Image Generator

This project uses Stability AI's Stable Diffusion 3.5 model to generate images from text descriptions.

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU with at least 8GB VRAM
- CUDA toolkit installed

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

Run the image generation script:
```bash
python src/generate_image.py
```

The generated image will be saved as "capybara.png" in the project root directory.

# Contents of requirements.txt:
torch>=2.0.0
diffusers>=0.26.0
transformers>=4.36.0
accelerate>=0.27.0

# Contents of .gitignore:
venv/
__pycache__/
*.png
*.pyc
.env

# Contents of src/generate_image.py:
import torch
from diffusers import StableDiffusion3Pipeline

def generate_image(prompt: str, output_path: str = "capybara.png") -> None:
    """
    Generate an image using Stable Diffusion 3.5
    
    Args:
        prompt (str): Text description of the image to generate
        output_path (str): Path where to save the generated image
    """
    # Initialize the pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", 
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    # Generate the image
    image = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    prompt = "A capybara holding a sign that reads Hello World"
    generate_image(prompt)


# Configure Pytorch for Mac M1 chips
Step 1: Install Xcode Install the Command Line Tools:

xcode-select --install
Step 2: Setup a new conda environment

conda create -n torch-gpu python=3.8
conda activate torch-gpu
Step 2: Install PyTorch packages

conda install pytorch torchvision torchaudio -c pytorch-nightly
Step 3: Install Jupyter notebook for validating installation

conda install -c conda-forge jupyter jupyterlab
jupter-notebook
Create new notebook file and execute this code

dtype = torch.float
device = torch.device("mps")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

# Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
If you don’t see any error, everything works as expected!

Ref: https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c