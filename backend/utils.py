import json
import torch
import clip
import requests
from PIL import Image
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_text(text):
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token)
    return text_features / text_features.norm(dim=-1, keepdim=True)

def embed_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features / image_features.norm(dim=-1, keepdim=True)

def embed_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        return image_features[0].cpu().numpy().tolist()
    except Exception as e:
        print(f"Failed to process image {url}: {e}")
        return None