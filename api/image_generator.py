import requests
from PIL import Image
import io

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}


def generate_image(prompt):
    payload = {"inputs": prompt}
    print(f"Prompt: {prompt}")
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.content.decode('utf-8')}")
        return None

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        print("Error: Response is not an image.")
        print(f"Response content: {response.content.decode('utf-8')}")
        return None

    return response.content


def save_image(image_bytes, filename):
    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.save(filename, format="JPEG")
            return True
        except Exception as e:
            print(f"Error opening or saving image: {e}")
    return False
