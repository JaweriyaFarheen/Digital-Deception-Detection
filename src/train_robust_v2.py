import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
import numpy as np


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 2. PATHS
# ============================================================

# IMPORTANT:
# Only the processed CASIA dataset is used.
# No external images are used.

DATASET_DIR = Path("processed_dataset")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = (
    MODEL_DIR /
    "resnet50_robust_v2_best.pth"
)


# ============================================================
# 3. CHECK DATASET PATHS
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation folder not found:\n{VAL_DIR}"
    )


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        224,
        scale=(0.80, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 5. LOAD TRAINING DATA
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)


# ============================================================
# 6. LOAD VALIDATION DATA
# ============================================================

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


print("\nDataset classes:")
print(train_dataset.classes)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ============================================================
# 7. CHECK CLASSES
# ============================================================

if train_dataset.classes != val_dataset.classes:

    raise ValueError(
        "\nClass mismatch detected!\n"
        f"Training classes: {train_dataset.classes}\n"
        f"Validation classes: {val_dataset.classes}"
    )


# ============================================================
# 8. CLASS BALANCING
# ============================================================

targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    targets,
    minlength=2
)

print("\nTraining class counts:")

for class_name, count in zip(
    train_dataset.classes,
    class_counts
):

    print(
        f"{class_name}: {count}"
    )


# Inverse-frequency class weights

class_weights = (
    1.0 /
    np.maximum(
        class_counts,
        1
    )
)


sample_weights = (
    class_weights[
        targets
    ]
)


sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# 9. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 10. LOAD PRETRAINED RESNET-50
# ============================================================

print(
    "\nLoading ResNet-50..."
)

weights = (
    models.ResNet50_Weights.DEFAULT
)

model = models.resnet50(
    weights=weights
)


# ============================================================
# 11. REPLACE CLASSIFIER
# ============================================================

num_features = (
    model.fc.in_features
)

model.fc = nn.Sequential(

    nn.Dropout(
        p=0.3
    ),

    nn.Linear(
        num_features,
        2
    )
)


model = model.to(device)


print(
    "ResNet-50 created successfully."
)


# ============================================================
# 12. CLASS-WEIGHTED LOSS
# ============================================================

total_samples = class_counts.sum()


loss_weights = torch.tensor(
    [
        total_samples /
        (
            2 *
            max(class_counts[0], 1)
        ),

        total_samples /
        (
            2 *
            max(class_counts[1], 1)
        )
    ],
    dtype=torch.float32
).to(device)


criterion = nn.CrossEntropyLoss(
    weight=loss_weights
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)


# ============================================================
# 14. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 15. TRAINING SETTINGS
# ============================================================

EPOCHS = 30

EARLY_STOPPING_PATIENCE = 6

best_val_loss = float("inf")

patience_counter = 0


# ============================================================
# 16. TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        total += labels.size(0)


        correct += (
            predictions == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_accuracy = (
        100 *
        correct /
        total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item()
            )


            predictions = (
                outputs.argmax(
                    dim=1
                )
            )


            val_total += (
                labels.size(0)
            )


            val_correct += (
                predictions == labels
            ).sum().item()


    val_loss = (
        val_loss /
        len(val_loader)
    )


    val_accuracy = (
        100 *
        val_correct /
        val_total
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step(
        val_loss
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )


    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    print(
        f"Learning Rate: "
        f"{current_lr:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        patience_counter = 0


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            "✓ Best ResNet-50 "
            "v2 model saved."
        )


    else:

        patience_counter += 1


        print(
            f"No improvement "
            f"({patience_counter}/"
            f"{EARLY_STOPPING_PATIENCE})"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        patience_counter >=
        EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# 17. COMPLETED
# ============================================================

print(
    "\nResNet-50 v2 training completed."
)

print(
    "Best model saved at:"
)

print(
    MODEL_PATH
)