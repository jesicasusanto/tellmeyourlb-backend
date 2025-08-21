import json
from utils import embed_text, embed_image_from_url

# Load your JSON
with open("brands_raw.json") as f:
    brands = json.load(f)

# Embed each brand’s style description
brands = brands["data"]
for brand in brands:
    embedding = embed_text(brand["style_desc"])
    brand["embedding"] = embedding.cpu().numpy().tolist()[0]
    #brand["image_embedding"] = embed_image_from_url(brand["image"])


# Save it to a new JSON
with open("brands_embedded.json", "w") as f:
    json.dump(brands, f, indent=2)
