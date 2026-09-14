import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path


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

DATASET_DIR = Path(
    "cifake_processed"
)

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    "resnet50_cifake_best.pth"
)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_DIR}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation dataset not found:\n{VAL_DIR}"
    )


# ============================================================
# 4. TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
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

    transforms.Resize((224, 224)),

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
# 5. LOAD DATASETS
# ============================================================

print(
    "\nLoading CiFAKE training dataset..."
)

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print(
    "Classes:",
    train_dataset.classes
)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)


# ============================================================
# 6. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 7. LOAD RESNET-50
# ============================================================

print(
    "\nLoading ResNet-50..."
)

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Sequential(

    nn.Dropout(
        p=0.4
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
# 8. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 9. TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 30

PATIENCE = 6

best_val_loss = float("inf")

epochs_without_improvement = 0


# ============================================================
# 10. TRAINING LOOP
# ============================================================

for epoch in range(
    NUM_EPOCHS
):

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        train_total += labels.size(0)

    train_loss /= train_total

    train_accuracy = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            val_total += labels.size(0)

    val_loss /= val_total

    val_accuracy = (
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "✓ Best CiFAKE ResNet-50 model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# 11. COMPLETED
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CiFAKE ResNet-50 training completed."
)

print(
    "=" * 60
)

print(
    "Best model saved at:"
)

print(
    MODEL_PATH
)