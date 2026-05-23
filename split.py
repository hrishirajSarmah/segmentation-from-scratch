import os
import shutil
from sklearn.model_selection import train_test_split


def create_val_split(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, val_split=0.2):
    # Create validation directories
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    # Get all image filenames
    images = os.listdir(train_img_dir)

    # Split into train and validation
    _, val_images_list = train_test_split(images, test_size=val_split, random_state=42)

    # Move validation files
    for img_name in val_images_list:
        # Move image
        src_img = os.path.join(train_img_dir, img_name)
        dst_img = os.path.join(val_img_dir, img_name)
        shutil.move(src_img, dst_img)

        # Move corresponding mask
        mask_name = img_name.replace('.jpg', '_mask.gif')
        src_mask = os.path.join(train_mask_dir, mask_name)
        dst_mask = os.path.join(val_mask_dir, mask_name)
        shutil.move(src_mask, dst_mask)

    print(f"Moved {len(val_images_list)} images to validation set")


if __name__ == "__main__":
    create_val_split(
        train_img_dir="data/train_images/",
        train_mask_dir="data/train_masks/",
        val_img_dir="data/val_images/",
        val_mask_dir="data/val_masks/",
        val_split=0.2
    )
