"""
Simple data preparation script:
- Crops images to 16:9 aspect ratio
- Splits into Training/Validation/Testing (70/15/15)
- Saves to Data/Processed/

NOTE: Run data_balancing.py first to create balanced dataset!
"""
import random
from pathlib import Path
from PIL import Image, ImageOps

# Config
INPUT_DIR = Path("Data/Balanced")  # Uses balanced data (run data_balancing.py first)
OUTPUT_DIR = Path("Data/Processed")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
IMG_SIZE = (1280, 720)  # 16:9
SEED = 42

CLASSES = ["Indonesia", "Laos", "Malaysia", "Philippines", "Singapore", "Thailand"]


def crop_to_16_9(img):
    """Center-crop image to 16:9 aspect ratio."""
    w, h = img.size
    target_ratio = 16 / 9
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Too wide: crop width
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # Too tall: crop height
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    
    return img


def process_image(src_path, out_path):
    """Load, crop to 16:9, resize, and save."""
    try:
        img = Image.open(src_path)
        img = ImageOps.exif_transpose(img)  # Fix rotation
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = crop_to_16_9(img)
        img = img.resize(IMG_SIZE, Image.BICUBIC)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"Skipping {src_path.name}: {e}")
        return False


def prepare_data():
    random.seed(SEED)
    
    for cls in CLASSES:
        class_dir = INPUT_DIR / cls
        if not class_dir.exists():
            print(f"Skipping {cls} (not found)")
            continue
        
        # Get all images
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not files:
            print(f"No images in {cls}")
            continue
        
        # Shuffle and split
        files.sort()
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Process each split
        for split_name, split_files in [("Training", train_files), ("Validation", val_files), ("Testing", test_files)]:
            for f in split_files:
                out_path = OUTPUT_DIR / split_name / cls / f.with_suffix(".jpg").name
                process_image(f, out_path)
        
        print(f"{cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    print("Images have completed processing")


if __name__ == "__main__":
    prepare_data()
