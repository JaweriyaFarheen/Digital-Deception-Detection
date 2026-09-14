import shutil
import random
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# 1. SETTINGS
# ============================================================

# CHANGE THIS to the location of your extracted CiFAKE folder
SOURCE_DIR = Path(
    r"C:\Users\Shaik Sadaf Patel\Downloads\archive (3)"
)

# Where the new processed dataset will be created
OUTPUT_DIR = Path(
    "cifake_processed"
)

# Required dataset size
TOTAL_IMAGES = 12_614
TRAIN_IMAGES = 8_829
VAL_IMAGES = 1_891
TEST_IMAGES = 1_894

SEED = 42

random.seed(SEED)


# ============================================================
# 2. CHECK SOURCE DATASET
# ============================================================

source_train = SOURCE_DIR / "train"
source_test = SOURCE_DIR / "test"

if not source_train.exists():
    raise FileNotFoundError(
        f"CiFAKE train folder not found:\n{source_train}"
    )

if not source_test.exists():
    raise FileNotFoundError(
        f"CiFAKE test folder not found:\n{source_test}"
    )


# ============================================================
# 3. COLLECT IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def collect_images(folder, class_name):
    images = []

    class_folder = folder / class_name

    if not class_folder.exists():
        print(
            f"WARNING: {class_folder} does not exist."
        )
        return images

    for path in class_folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in extensions
        ):
            images.append(path)

    return images


print("\nCollecting CiFAKE images...")


real_images = (
    collect_images(source_train, "REAL")
    +
    collect_images(source_test, "REAL")
)

fake_images = (
    collect_images(source_train, "FAKE")
    +
    collect_images(source_test, "FAKE")
)


print(
    f"REAL images found: {len(real_images)}"
)

print(
    f"FAKE images found: {len(fake_images)}"
)

print(
    f"Total images found: "
    f"{len(real_images) + len(fake_images)}"
)


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

def file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def remove_duplicates(images):

    unique_images = []
    hashes = set()

    for image in images:

        try:
            image_hash = file_hash(image)

        except Exception as error:

            print(
                f"Could not read {image}: {error}"
            )

            continue

        if image_hash not in hashes:

            hashes.add(image_hash)
            unique_images.append(image)

    return unique_images


print("\nRemoving duplicate images...")

real_images = remove_duplicates(real_images)
fake_images = remove_duplicates(fake_images)

print(
    f"Unique REAL images: {len(real_images)}"
)

print(
    f"Unique FAKE images: {len(fake_images)}"
)


# ============================================================
# 5. CHECK WHETHER DATASET IS LARGE ENOUGH
# ============================================================

required_per_class = TOTAL_IMAGES // 2

if len(real_images) < required_per_class:

    raise RuntimeError(
        f"Not enough REAL images.\n"
        f"Required: {required_per_class}\n"
        f"Available: {len(real_images)}"
    )

if len(fake_images) < required_per_class:

    raise RuntimeError(
        f"Not enough FAKE images.\n"
        f"Required: {required_per_class}\n"
        f"Available: {len(fake_images)}"
    )


# ============================================================
# 6. SELECT 12,614 IMAGES
# ============================================================

print(
    f"\nSelecting exactly {TOTAL_IMAGES} images..."
)

random.shuffle(real_images)
random.shuffle(fake_images)

# Balanced dataset:
# 6,307 REAL
# 6,307 FAKE

real_selected = real_images[:6307]
fake_selected = fake_images[:6307]

print(
    f"Selected REAL: {len(real_selected)}"
)

print(
    f"Selected FAKE: {len(fake_selected)}"
)

print(
    f"Selected total: "
    f"{len(real_selected) + len(fake_selected)}"
)


# ============================================================
# 7. COMBINE SELECTED DATA
# ============================================================

all_images = []

for image in real_selected:
    all_images.append(
        (image, "REAL")
    )

for image in fake_selected:
    all_images.append(
        (image, "FAKE")
    )


random.shuffle(all_images)


# ============================================================
# 8. CREATE EXACT TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First create TEST = 1,894
train_val, test_data = train_test_split(
    all_images,
    test_size=TEST_IMAGES,
    random_state=SEED,
    stratify=[
        label
        for _, label in all_images
    ]
)


# Then create VALIDATION = 1,891
train_data, val_data = train_test_split(
    train_val,
    test_size=VAL_IMAGES,
    random_state=SEED,
    stratify=[
        label
        for _, label in train_val
    ]
)


print("\nFinal split:")
print(
    f"Train:       {len(train_data)}"
)

print(
    f"Validation:  {len(val_data)}"
)

print(
    f"Test:        {len(test_data)}"
)

print(
    f"Total:       "
    f"{len(train_data) + len(val_data) + len(test_data)}"
)


# ============================================================
# 9. CREATE OUTPUT DIRECTORIES
# ============================================================

for split in [
    "train",
    "validation",
    "test"
]:

    for class_name in [
        "REAL",
        "FAKE"
    ]:

        folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 10. COPY IMAGES
# ============================================================

def copy_images(data, split):

    print(
        f"\nCreating {split} dataset..."
    )

    counters = {
        "REAL": 0,
        "FAKE": 0
    }

    for index, (source, class_name) in enumerate(data):

        destination_folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        # Give every image a unique filename
        destination = (
            destination_folder
            / f"{class_name.lower()}_{index:06d}"
            f"{source.suffix.lower()}"
        )

        shutil.copy2(
            source,
            destination
        )

        counters[class_name] += 1

    print(
        f"REAL: {counters['REAL']}"
    )

    print(
        f"FAKE: {counters['FAKE']}"
    )


copy_images(
    train_data,
    "train"
)

copy_images(
    val_data,
    "validation"
)

copy_images(
    test_data,
    "test"
)


# ============================================================
# 11. FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("CiFAKE DATASET PREPARATION COMPLETED")
print("=" * 60)

print(
    f"Dataset location: {OUTPUT_DIR}"
)

print(
    f"Train:       {len(train_data)}"
)

print(
    f"Validation:  {len(val_data)}"
)

print(
    f"Test:        {len(test_data)}"
)

print(
    f"Total:       "
    f"{len(train_data) + len(val_data) + len(test_data)}"
)

print("\nDataset structure:")

print(
    """
cifake_processed/
├── train/
│   ├── REAL/
│   └── FAKE/
│
├── validation/
│   ├── REAL/
│   └── FAKE/
│
└── test/
    ├── REAL/
    └── FAKE/
"""
)

print("Ready for CiFAKE ResNet-50 training.")
