from pathlib import Path
import shutil

base_path = Path(r"C:/Users/fabia/Documents/PGAI/Internship/FINE/yolo11-fine/_data")
image_root = base_path / "images/iSense2/AI/Liam/FINE/Originele beelden/Zorgband2025/Images"
label_root = base_path / "labels/iSense2/AI/Liam/FINE/Originele beelden/Zorgband2025/Images"

train_image_dst = base_path / "images/train"
val_image_dst = base_path / "images/val"
train_label_dst = base_path / "labels/train"
val_label_dst = base_path / "labels/val"

train_image_dst.mkdir(parents=True, exist_ok=True)
val_image_dst.mkdir(parents=True, exist_ok=True)
train_label_dst.mkdir(parents=True, exist_ok=True)
val_label_dst.mkdir(parents=True, exist_ok=True)

# Sort folders by name (dates), assuming all are valid
folders = sorted([f.name for f in image_root.iterdir() if f.is_dir()])
if len(folders) < 10:
    raise Exception("Expected at least 10 date folders.")

train_dates = folders[:8]
val_dates = folders[-2:]

def move_with_labels(dates, image_src_root, label_src_root, image_dst, label_dst):
    for date in dates:
        img_rgb_folder = image_src_root / date / "RGB"
        lbl_rgb_folder = label_src_root / date / "RGB"

        if not img_rgb_folder.exists():
            continue

        for img_file in img_rgb_folder.iterdir():
            img_dst_path = image_dst / img_file.name
            shutil.move(str(img_file), str(img_dst_path))

            base_name = img_file.stem
            label_file = base_name + ".txt"
            label_src = lbl_rgb_folder / label_file
            label_dst_path = label_dst / label_file

            if label_src.exists():
                shutil.move(str(label_src), str(label_dst_path))
            else:
                # Create empty label file
                label_dst_path.write_text("")

# Move images and ensure labels exist
move_with_labels(train_dates, image_root, label_root, train_image_dst, train_label_dst)
move_with_labels(val_dates, image_root, label_root, val_image_dst, val_label_dst)
