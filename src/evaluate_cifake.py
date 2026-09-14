import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

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

TEST_DIR = Path(
    "cifake_processed/test"
)

MODEL_PATH = Path(
    "models/resnet50_cifake_best.pth"
)


# ============================================================
# 3. TEST TRANSFORMATION
# ============================================================

test_transform = transforms.Compose([

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
# 4. CHECK PATHS
# ============================================================

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"""
CiFAKE test dataset not found:

{TEST_DIR}

Make sure cifake_processed/test exists.
"""
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"""
CiFAKE model not found:

{MODEL_PATH}

Run train_cifake.py first.
"""
    )


# ============================================================
# 5. LOAD TEST DATASET
# ============================================================

print(
    "\nLoading CiFAKE test dataset..."
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print(
    "Classes:",
    test_dataset.classes
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# 6. DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 7. CREATE RESNET-50
# ============================================================

print(
    "\nLoading CiFAKE ResNet-50..."
)

model = models.resnet50(
    weights=None
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


# ============================================================
# 8. LOAD TRAINED MODEL
# ============================================================

print(
    "Model path:",
    MODEL_PATH
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print(
    "CiFAKE model loaded successfully."
)


# ============================================================
# 9. PREDICTIONS
# ============================================================

all_labels = []

all_predictions = []


print(
    "\nRunning final CiFAKE test evaluation..."
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(
            images
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# 10. METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# 11. FINAL RESULTS
# ============================================================

print("\n")

print(
    "=" * 60
)

print(
    "FINAL CiFAKE TEST RESULTS"
)

print(
    "=" * 60
)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        zero_division=0
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(
    "Confusion Matrix:"
)

print(cm)


# ============================================================
# 14. COMPLETED
# ============================================================

print(
    "\nCiFAKE final evaluation completed."
)

print(
    "Dataset evaluated:",
    TEST_DIR
)

print(
    "Model evaluated:",
    MODEL_PATH
)