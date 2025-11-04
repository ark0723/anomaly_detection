import base64
import io
from pathlib import Path
from PIL import Image
import requests


def encode_image_to_base64(image_path: Path) -> str:
    """
    encode the image file path to a base64 string
    if the file exists, read the original bytes
    if the file does not exist, create a dummy black image
    """
    if image_path.exists():
        # if the file exists, read the original bytes
        print(f"Encoding real image from: {image_path}")
        with image_path.open("rb") as image_file:
            image_bytes = image_file.read()
    else:
        # if the file does not exist, create a dummy black image
        print(f"Warning: '{image_path}' not found. Creating a dummy black image.")
        img = Image.new("RGB", (224, 224), color="black")

        # convert the dummy image to in-memory bytes
        with io.BytesIO() as buf:
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

    # encode the image bytes to base64 string
    return base64.b64encode(image_bytes).decode("utf-8")


api_url = "http://0.0.0.0:8000/predict"
base_folder = Path("frames")

img_paths = list(base_folder.glob("*.jpg"))

print(f"Encoding {len(img_paths)} images for batch request...")
try:
    base64_images = [encode_image_to_base64(img_path) for img_path in img_paths]
except Exception as e:
    print(f"Error encoding images: {e}")
    raise e

# The JSON payload must match the Pydantic model in api.py
payload = {"images": base64_images}

# send the request to the API
print(f"Sending batch request to {api_url}...")
try:
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    print("Request successful!")
    print(f"Response: {response.json()}")

except requests.exceptions.ConnectionError:
    print("\n--- ERROR ---")
    print("Could not connect to the API server.")

except Exception as e:
    print(f"An error occurred: {e}")
