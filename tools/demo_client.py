import base64, json
from pathlib import Path
import requests
from PIL import Image, ImageDraw

API = "http://127.0.0.1:8000/api/v1/frontal/crop/submit"

IMAGE_PATH = "/home/amir/Work/United-Kingdom/test-task/QOVES_INC/data/original_image.png"      # your provided image
LANDMARKS_TXT = "/home/amir/Work/United-Kingdom/test-task/QOVES_INC/data/landmarks.txt"  # your uploaded file content
MASK_PATH = "mask_demo.png"
OUT_SVG = "result.svg"

def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def create_demo_mask(img_path: str, mask_path: str):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    # very simple mask with a few labeled regions for testing (1..5)
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    d.ellipse((w*0.23, h*0.22, w*0.80, h*0.50), fill=1)   # forehead
    d.ellipse((w*0.35, h*0.40, w*0.48, h*0.50), fill=2)   # left eye region
    d.ellipse((w*0.55, h*0.40, w*0.68, h*0.50), fill=2)   # right eye region
    d.ellipse((w*0.46, h*0.52, w*0.56, h*0.68), fill=3)   # nose
    d.rectangle((w*0.30, h*0.62, w*0.70, h*0.78), fill=4) # cheeks
    d.rectangle((w*0.40, h*0.72, w*0.60, h*0.82), fill=5) # mouth
    m.save(mask_path)

def load_landmarks(path: str):
    # your file is JSON with {'landmarks': [[{x,y},...]], 'dimensions': [W,H], ...}
    print(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data["landmarks"][0]

if __name__ == "__main__":
    assert Path(IMAGE_PATH).exists(), "Put original_image.png next to this script"
    create_demo_mask(IMAGE_PATH, MASK_PATH)

    payload = {
        "image": b64(IMAGE_PATH),
        "segmentation_map": b64(MASK_PATH),
        "landmarks": load_landmarks(LANDMARKS_TXT),
        # "upright_svg": False  # set True to un-tilt using eye angle
    }

    r = requests.post(API, json=payload, timeout=60)
    print("status:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        raise SystemExit(1)
    data = r.json()
    svg_bytes = base64.b64decode(data["svg"])
    Path(OUT_SVG).write_bytes(svg_bytes)
    print("saved:", OUT_SVG)
