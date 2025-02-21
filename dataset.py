import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(root_path):
    root_path = Path(root_path)
    img_path = root_path / 'images'
    label_path = root_path / 'labels'
    
    all_images = list(img_path.glob('*.tif'))
    
    train_imgs, temp_imgs = train_test_split(all_images, train_size=0.8, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, train_size=0.5, random_state=42)
    
    for folder in ['train', 'val', 'test']:
        (Path('data') / folder / 'images').mkdir(parents=True, exist_ok=True)
        (Path('data') / folder / 'labels').mkdir(parents=True, exist_ok=True)
    
    for img in train_imgs:
        shutil.copy(img, f'data/train/images/{img.name}')
        label = label_path / f'{img.stem}.txt'
        if label.exists():
            shutil.copy(label, f'data/train/labels/{label.name}')
    
    for img in val_imgs:
        shutil.copy(img, f'data/val/images/{img.name}')
        label = label_path / f'{img.stem}.txt'
        if label.exists():
            shutil.copy(label, f'data/val/labels/{label.name}')
    
    for img in test_imgs:
        shutil.copy(img, f'data/test/images/{img.name}')
        label = label_path / f'{img.stem}.txt'
        if label.exists():
            shutil.copy(label, f'data/test/labels/{label.name}')
    
    print(f"Train: {len(train_imgs)} images")
    print(f"Val: {len(val_imgs)} images")
    print(f"Test: {len(test_imgs)} images")

if __name__ == "__main__":
    prepare_dataset("dataset")