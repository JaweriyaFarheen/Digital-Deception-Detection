from pathlib import Path
from PIL import Image
import random
import shutil

# Dataset locations
DATASET_DIR = Path("dataset")
AUTHENTIC_DIR = DATASET_DIR / "authentic"
MANIPULATED_DIR = DATASET_DIR / "manipulated"

# Output directories
OUTPUT_DIR = Path("processed_dataset")

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

IMAGE_SIZE = (224, 224)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def get_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]


def process_class(source_dir, class_name):
    images = get_images(source_dir)

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\n{class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    for split_name, split_images in splits.items():

        output_folder = OUTPUT_DIR / split_name / class_name
        output_folder.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:

            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize(IMAGE_SIZE)

                output_path = output_folder / image_path.name
                image.save(output_path, "JPEG", quality=95)

            except Exception as error:
                print(f"Could not process {image_path.name}: {error}")


def main():

    print("Starting dataset preprocessing...")

    # Remove previous processed dataset
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    process_class(AUTHENTIC_DIR, "authentic")
    process_class(MANIPULATED_DIR, "manipulated")

    print("\nPreprocessing completed successfully.")
    print(f"Processed dataset location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()