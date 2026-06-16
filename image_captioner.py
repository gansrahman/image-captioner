import argparse, json
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="captions.json")
    p.add_argument("--limit", type=int, default=1000)
    a = p.parse_args()
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(sorted(Path(a.input).glob(ext)))
    imgs = imgs[:a.limit]
    print(f"Captioning {len(imgs)} images...")
    results = []
    for i, p in enumerate(imgs):
        inp = proc(Image.open(p).convert("RGB"), return_tensors="pt").to("cuda")
        out = model.generate(**inp, max_length=75)
        cap = proc.decode(out[0], skip_special_tokens=True)
        results.append({"file": p.name, "caption": cap})
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(imgs)}")
    with open(a.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} captions")

if __name__ == "__main__":
    main()
