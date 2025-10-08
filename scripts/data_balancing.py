"""
Balance dataset to ensure each country has similar number of images (200-300).
Uses oversampling (duplicate minority classes) and undersampling (reduce majority classes).
"""
import random
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm


# Config
INPUT_DIR = Path("Data/Raw")
OUTPUT_DIR = Path("Data/Balanced")
TARGET_MIN = 500  # Minimum images per class
TARGET_MAX = 600  # Maximum images per class
SEED = 42

CLASSES = ["Indonesia", "Laos", "Malaysia", "Philippines", "Singapore", "Thailand"]


def count_images(data_dir):
    """Count images per class."""
    counts = {}
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            images = [f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            counts[cls] = len(images)
        else:
            counts[cls] = 0
    return counts


def get_all_images(data_dir, cls):
    """Get all image paths for a class."""
    cls_dir = data_dir / cls
    return [f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]


def oversample(images, target_count):
    """
    Oversample by duplicating images until target count is reached.
    Duplicates are renamed with suffix _dup1, _dup2, etc.
    """
    if len(images) >= target_count:
        return images[:target_count]
    
    result = images.copy()
    needed = target_count - len(images)
    
    # Repeat images cyclically
    dup_count = 0
    while len(result) < target_count:
        for img in images:
            if len(result) >= target_count:
                break
            dup_count += 1
            # Create a duplicate entry with modified name
            new_name = f"{img.stem}_dup{dup_count}{img.suffix}"
            result.append((img, new_name))  # (source, target_name)
    
    # Convert back to standard format
    final = []
    for item in result:
        if isinstance(item, tuple):
            final.append(item)
        else:
            final.append((item, item.name))
    
    return final


def undersample(images, target_count):
    """Undersample by randomly selecting target_count images."""
    if len(images) <= target_count:
        return [(img, img.name) for img in images]
    
    sampled = random.sample(images, target_count)
    return [(img, img.name) for img in sampled]


def balance_dataset():
    """Balance dataset by over/undersampling each class."""
    random.seed(SEED)
    
    # Count current distribution
    print("Current distribution:")
    counts = count_images(INPUT_DIR)
    for cls, count in counts.items():
        print(f"  {cls}: {count}")
    
    if all(count == 0 for count in counts.values()):
        print("\nError: No images found in Data/Raw/")
        return
    
    # Determine target count (middle of range)
    total = sum(counts.values())
    avg = total // len([c for c in counts.values() if c > 0])
    target = max(TARGET_MIN, min(TARGET_MAX, avg))
    
    print(f"\nTarget images per class: {target}")
    print(f"\nBalancing strategy:")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    balanced_counts = {}
    
    for cls in CLASSES:
        images = get_all_images(INPUT_DIR, cls)
        original_count = len(images)
        
        if original_count == 0:
            print(f"  {cls}: SKIPPED (no images found)")
            continue
        
        # Determine strategy
        if original_count < target:
            strategy = f"OVERSAMPLE (duplicate {target - original_count} images)"
            balanced_images = oversample(images, target)
        elif original_count > target:
            strategy = f"UNDERSAMPLE (remove {original_count - target} images)"
            balanced_images = undersample(images, target)
        else:
            strategy = "NO CHANGE"
            balanced_images = [(img, img.name) for img in images]
        
        print(f"  {cls}: {original_count} → {len(balanced_images)} ({strategy})")
        
        # Copy images to balanced folder
        out_cls_dir = OUTPUT_DIR / cls
        out_cls_dir.mkdir(exist_ok=True)
        
        for src_path, target_name in tqdm(balanced_images, desc=f"Processing {cls}", leave=False):
            dest_path = out_cls_dir / target_name
            shutil.copy2(src_path, dest_path)
        
        balanced_counts[cls] = len(balanced_images)
    
    # Summary
    print(f"\n✓ Balanced dataset saved to {OUTPUT_DIR.resolve()}")
    print("\nFinal distribution:")
    for cls, count in balanced_counts.items():
        print(f"  {cls}: {count}")
    
    total_balanced = sum(balanced_counts.values())
    print(f"\nTotal images: {sum(counts.values())} → {total_balanced}")


if __name__ == "__main__":
    balance_dataset()
