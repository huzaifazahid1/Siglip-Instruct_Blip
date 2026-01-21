import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration
)

# ==========================================
# ⚙️ PATH SETUP (AUTO)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

if not os.path.exists(IMAGE_DIR):
    print("❌ 'images' folder nahi mila.")
    print("👉 main.py ke saath 'images' folder create karo aur images daalo.")
    exit()

# ==========================================
# 🧠 TAXONOMY
# ==========================================
TAXONOMY = {
    "Backgrounds": ["Abstract Pattern", "Texture", "Gradient Colors", "Bokeh Blur", "Geometric Shapes"],
    "Business": ["Office Meeting", "Financial Charts", "Handshake", "Working on Laptop", "Corporate Team"],
    "People": ["Portrait", "Crowd", "Family", "Fitness & Sports", "Happy Emotion"],
    "Technology": ["Artificial Intelligence", "Coding & Programming", "Virtual Reality", "Circuit Board", "Smart Devices"],
    "Travel": ["Mountains & Hiking", "Beach & Ocean", "Cityscape & Architecture", "Airport & Luggage", "Historical Monuments"],
    "Nature": ["Forest & Trees", "Sky & Clouds", "Flowers & Garden", "Desert & Sand", "Waterfalls & Rivers"],
    "Medical": ["Doctor & Nurse", "Hospital Bed", "Pills & Medicine", "Microscope & Lab", "Surgery"],
    "Food": ["Fresh Fruits", "Fast Food", "Desserts & Cakes", "Coffee & Drinks", "Healthy Salad"],
    "Sci-Fi & Fantasy": ["Space & Galaxy", "Cyberpunk & Futuristic", "Surrealism", "3D Render", "Aliens & Robots"],
    "Animals & Wildlife": ["Pets", "Wild Animals", "Birds", "Marine Life", "Insects", "Farm Animals"]
}

# ==========================================
# 🚀 MODEL LOADING
# ==========================================
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    print("🔹 Loading SigLIP...")
    siglip_proc = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    siglip_model = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384"
    ).to(device)

    print("🔹 Loading InstructBLIP...")
    blip_proc = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl"
    )
    blip_model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✅ Models loaded\n")
    return device, siglip_proc, siglip_model, blip_proc, blip_model

# ==========================================
# 🧠 IMAGE PROCESSING
# ==========================================
def process_image(img_path, device, s_proc, s_model, b_proc, b_model):
    try:
        image = Image.open(img_path).convert("RGB")
    except:
        return None

    # ---- CATEGORY (SigLIP) ----
    main_keys = list(TAXONOMY.keys())
    inputs = s_proc(text=main_keys, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out = s_model(**inputs)

    main_idx = torch.sigmoid(out.logits_per_image).argmax().item()
    main_cat = main_keys[main_idx]

    sub_keys = TAXONOMY[main_cat]
    inputs_sub = s_proc(text=sub_keys, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out_sub = s_model(**inputs_sub)

    sub_idx = torch.sigmoid(out_sub.logits_per_image).argmax().item()
    sub_cat = sub_keys[sub_idx]

    # ---- DESCRIPTION (InstructBLIP) ----
    prompt = "Describe the image in detail including objects, style, lighting and mood."
    inputs_blip = b_proc(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        gen = b_model.generate(**inputs_blip, max_new_tokens=90)

    caption = b_proc.batch_decode(gen, skip_special_tokens=True)[0]

    keywords = list(set([
        w.strip(".,").lower()
        for w in caption.split()
        if len(w) > 4
    ]))

    return {
        "Filename": os.path.basename(img_path),
        "Main Category": main_cat,
        "Sub Category": sub_cat,
        "Description": caption,
        "Keywords": ", ".join(keywords)
    }

# ==========================================
# ▶️ MAIN
# ==========================================
if __name__ == "__main__":
    device, s_proc, s_model, b_proc, b_model = load_models()

    images = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"📸 Found {len(images)} images\n")

    results = []
    for img in tqdm(images, desc="Processing Images", unit="img"):
        data = process_image(img, device, s_proc, s_model, b_proc, b_model)
        if data:
            results.append(data)

    if results:
        out_csv = os.path.join(BASE_DIR, "metadata_results.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n✅ DONE! CSV saved at:\n{out_csv}")
    else:
        print("⚠️ No data generated.")
