import os
import torch
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoModel,
    AutoProcessor,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration
)

# ==========================================
# 🛠️ SETUP & GUI INPUT
# ==========================================
def select_folder():
    print("📂 Opening Folder Selector...")
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Image Folder for Processing")

    if not folder_path:
        print("❌ Operation Cancelled")
        exit()

    print(f"✅ Folder Selected: {folder_path}")
    return folder_path


# ==========================================
# 🚀 MODEL LOADING
# ==========================================
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ Device: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    # --- SigLIP ---
    print("[1/2] Loading SigLIP (Categorization)...")
    siglip_proc = AutoProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )
    siglip_model = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384"
    ).to(device)

    # --- InstructBLIP ---
    print("[2/2] Loading InstructBLIP (Metadata)...")
    blip_proc = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl"
    )
    blip_model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✅ Models Loaded Successfully\n")
    return device, siglip_proc, siglip_model, blip_proc, blip_model


# ==========================================
# 📂 TAXONOMY
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
    "Sci-Fi & Fantasy": ["Space & Galaxy", "Cyberpunk & Futuristic", "Surrealism & Dreamlike", "3D Render & Hyper-realism", "Aliens & Robots"],
    "Animals & Wildlife": ["Pets", "Wild Animals", "Birds & Aerial", "Marine Life", "Insects", "Farm Animals"]
}


# ==========================================
# 🧠 IMAGE PROCESSING
# ==========================================
def process_single_image(img_path, device, s_proc, s_model, b_proc, b_model):
    try:
        image = Image.open(img_path).convert("RGB")
    except:
        return None

    # -------- SigLIP Category --------
    main_keys = list(TAXONOMY.keys())
    inputs = s_proc(text=main_keys, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = s_model(**inputs)

    main_idx = outputs.logits_per_image.argmax().item()
    main_cat = main_keys[main_idx]

    sub_keys = TAXONOMY[main_cat]
    inputs_sub = s_proc(text=sub_keys, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs_sub = s_model(**inputs_sub)

    sub_idx = outputs_sub.logits_per_image.argmax().item()
    sub_cat = sub_keys[sub_idx]

    # -------- InstructBLIP Metadata --------
    prompt = (
        "Describe this image clearly. "
        "Mention style, concept, main elements and mood. "
        "Do not guess objects that are not visible."
    )

    inputs_blip = b_proc(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        out = b_model.generate(**inputs_blip, max_new_tokens=120)

    description = b_proc.batch_decode(out, skip_special_tokens=True)[0].strip()

    # Simple keyword extraction
    ignore = {"image", "photo", "picture", "background", "scene"}
    tags = sorted(
        set(
            w.strip(".,").lower()
            for w in description.split()
            if len(w) > 3 and w.lower() not in ignore
        )
    )

    return {
        "Filename": os.path.basename(img_path),
        "Main Category": main_cat,
        "Sub Category": sub_cat,
        "Description": description,
        "Keywords": ", ".join(tags)
    }


# ==========================================
# ▶️ MAIN
# ==========================================
if __name__ == "__main__":
    folder = select_folder()
    device, s_proc, s_model, b_proc, b_model = load_models()

    images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"📸 {len(images)} images found\n")

    results = []
    for img in tqdm(images, desc="Processing", unit="img"):
        data = process_single_image(img, device, s_proc, s_model, b_proc, b_model)
        if data:
            results.append(data)

    if results:
        out_csv = os.path.join(folder, "metadata_results.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n✅ DONE → {out_csv}")
    else:
        print("\n⚠️ No results generated")
