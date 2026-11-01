import os
import shutil
import random
import yaml
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# Paths
# Verified paths based on directory listing
DRONE_DATASET_PATH = r"D:\inav3\drone-detection-system-main\drone-detection-system-main\drone-dataset\drone-detection-new.v5-new-train.yolov8"
BIRDS_DATASET_PATH = r"D:\inav3\drone-detection-system-main\drone-detection-system-main\birds-dataset\Birds" 
OUTPUT_DIR = r"D:\inav3\drone-detection-system-main\drone-detection-system-main\dataset_combined"

# Classes
# Drone dataset: 0: AirPlane, 1: Drone, 2: Helicopter
# We will add 3: Bird
CLASSES = ['AirPlane', 'Drone', 'Helicopter', 'Bird']

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Cleaned {OUTPUT_DIR}")
        except Exception as e:
            print(f"Error cleaning directory: {e}")
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

def process_drone_dataset():
    print("Processing Drone Dataset...")
    # Drone dataset has train, valid, test folders
    splits_mapping = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }
    
    for original_split, new_split in splits_mapping.items():
        src_images = os.path.join(DRONE_DATASET_PATH, original_split, 'images')
        src_labels = os.path.join(DRONE_DATASET_PATH, original_split, 'labels')
        
        if not os.path.exists(src_images):
            print(f"Warning: {src_images} not found, skipping {original_split}")
            continue

        image_files = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Copying {original_split}"):
            # Copy Image
            src_img_path = os.path.join(src_images, img_file)
            dst_img_path = os.path.join(OUTPUT_DIR, new_split, 'images', f"drone_{img_file}")
            shutil.copy(src_img_path, dst_img_path)
            
            # Copy Label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label_path = os.path.join(src_labels, label_file)
            if os.path.exists(src_label_path):
                 dst_label_path = os.path.join(OUTPUT_DIR, new_split, 'labels', f"drone_{label_file}")
                 shutil.copy(src_label_path, dst_label_path)

def process_bird_dataset():
    print("Processing Bird Dataset (Auto-labeling)...")
    model = YOLO('yolov8n.pt')  # Pretrained model to detect birds
    
    # Check 'train' and 'test' folders in birds dataset
    # We'll combine them and re-split to ensure balanced distribution
    all_images = []
    for subdir in ['train', 'test']:
        d = os.path.join(BIRDS_DATASET_PATH, subdir)
        if os.path.exists(d):
            imgs = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            all_images.extend(imgs)
    
    print(f"Found {len(all_images)} bird images total.")
    
    # Shuffle and Split manually: 80% train, 10% val, 10% test
    random.shuffle(all_images)
    num_imgs = len(all_images)
    train_end = int(num_imgs * 0.8)
    val_end = int(num_imgs * 0.9)
    
    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }
    
    for split, images in splits.items():
        # Process in batches or individually if simple
        for img_path in tqdm(images, desc=f"Auto-labeling birds for {split}"):
            try:
                # Run inference with low confidence threshold to catch most candidates
                results = model(img_path, verbose=False, conf=0.25)
                
                basename = os.path.basename(img_path)
                new_img_name = f"bird_{basename}"
                new_label_name = os.path.splitext(new_img_name)[0] + ".txt"
                
                save_img_path = os.path.join(OUTPUT_DIR, split, 'images', new_img_name)
                save_label_path = os.path.join(OUTPUT_DIR, split, 'labels', new_label_name)
                
                # Filter for bird class (COCO class 14 is bird)
                detected_boxes = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id == 14: # COCO bird class
                             # Format: class x_center y_center width height
                             coords = box.xywhn[0].tolist()
                             # Remap class 14 (Bird) to class 3 (Bird in our new dataset)
                             new_line = f"3 {' '.join(map(str, coords))}"
                             detected_boxes.append(new_line)
                
                if detected_boxes:
                    # Copy image only if bird detected
                    shutil.copy(img_path, save_img_path)
                    with open(save_label_path, 'w') as f:
                        f.write('\n'.join(detected_boxes))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def create_yaml():
    yaml_content = {
        'path': OUTPUT_DIR,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Created dataset.yaml at {yaml_path}")

if __name__ == "__main__":
    setup_directories()
    # Ensure ultralytics is installed? It is imported.
    process_drone_dataset()
    process_bird_dataset()
    create_yaml()
