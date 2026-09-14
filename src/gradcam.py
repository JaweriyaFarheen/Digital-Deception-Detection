import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = Path(
    "models/resnet50_robust_v2_best.pth"
)

IMAGE_PATH = Path(
    "processed_dataset/test/authentic"
)

OUTPUT_DIR = Path(
    "gradcam_output"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 3. LOAD ROBUST V2 RESNET-50
# ============================================================

print(
    "\nLoading Robust V2 ResNet-50..."
)

model = models.resnet50(
    weights=None
)

num_features = model.fc.in_features


# IMPORTANT:
# This must match train_robust_v2.py

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
# 4. CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"""
Model not found:

{MODEL_PATH}

Make sure train_robust_v2.py
has completed successfully.
"""
    )


# ============================================================
# 5. LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print(
    "Robust V2 model loaded successfully."
)


# ============================================================
# 6. IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

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
# 7. TARGET LAYER
# ============================================================

# Last convolutional block of ResNet-50

target_layer = model.layer4[-1]


# ============================================================
# 8. GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )


    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output


    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def generate(
        self,
        image_tensor
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            image_tensor
        )

        prediction = output.argmax(
            dim=1
        ).item()

        score = output[
            0,
            prediction
        ]

        score.backward()

        if (
            self.gradients is None
            or self.activations is None
        ):

            raise RuntimeError(
                "Grad-CAM hooks did not "
                "capture activations/gradients."
            )

        gradients = self.gradients[0]

        activations = self.activations[0]

        # Global average pooling
        # over spatial dimensions

        weights = gradients.mean(
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i in range(
            activations.shape[0]
        ):

            cam += (
                weights[i]
                * activations[i]
            )

        # Keep only positive influence

        cam = torch.relu(
            cam
        )

        cam = cam.detach().cpu().numpy()

        # Robust normalization

        cam_min = cam.min()

        cam_max = cam.max()

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )

        return (
            cam,
            prediction,
            output.detach()
        )


# ============================================================
# 9. FIND IMAGE
# ============================================================

image_files = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]:

    image_files.extend(
        IMAGE_PATH.glob(extension)
    )


if not image_files:

    raise FileNotFoundError(
        f"""
No images found in:

{IMAGE_PATH}

Change IMAGE_PATH to a folder
containing test images.
"""
    )


image_path = image_files[0]

print(
    "\nUsing image:",
    image_path
)


# ============================================================
# 10. LOAD ORIGINAL IMAGE
# ============================================================

original = Image.open(
    image_path
).convert("RGB")


original_width, original_height = (
    original.size
)

print(
    "Original image size:",
    original_width,
    "x",
    original_height
)


# ============================================================
# 11. PREPARE INPUT
# ============================================================

input_tensor = transform(
    original
).unsqueeze(0).to(device)


# ============================================================
# 12. GENERATE GRAD-CAM
# ============================================================

gradcam = GradCAM(
    model,
    target_layer
)

cam, prediction, output = gradcam.generate(
    input_tensor
)


# ============================================================
# 13. PREDICTION + CONFIDENCE
# ============================================================

probabilities = torch.softmax(
    output,
    dim=1
)

confidence = (
    probabilities[
        0,
        prediction
    ].item()
    * 100
)


classes = [
    "authentic",
    "manipulated"
]

predicted_class = classes[
    prediction
]


print(
    "\nPrediction:",
    predicted_class
)

print(
    f"Confidence: {confidence:.2f}%"
)

print(
    "Authentic probability:"
    f" {probabilities[0, 0].item() * 100:.2f}%"
)

print(
    "Manipulated probability:"
    f" {probabilities[0, 1].item() * 100:.2f}%"
)


# ============================================================
# 14. CONVERT ORIGINAL IMAGE
# ============================================================

original_cv = np.array(
    original
)

original_cv = cv2.cvtColor(
    original_cv,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# 15. RESIZE CAM TO ORIGINAL IMAGE
# ============================================================

cam = cv2.resize(
    cam,
    (
        original_width,
        original_height
    ),
    interpolation=cv2.INTER_LINEAR
)


# ============================================================
# 16. CREATE HEATMAP
# ============================================================

heatmap = np.uint8(
    cam * 255
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


# ============================================================
# 17. CREATE OVERLAY
# ============================================================

overlay = cv2.addWeighted(
    original_cv,
    0.60,
    heatmap,
    0.40,
    0
)


# ============================================================
# 18. SAVE ORIGINAL COPY
# ============================================================

original_output = (
    OUTPUT_DIR /
    f"{image_path.stem}_original.jpg"
)

cv2.imwrite(
    str(original_output),
    original_cv
)


# ============================================================
# 19. SAVE HEATMAP
# ============================================================

heatmap_output = (
    OUTPUT_DIR /
    f"{image_path.stem}_heatmap.jpg"
)

cv2.imwrite(
    str(heatmap_output),
    heatmap
)


# ============================================================
# 20. SAVE OVERLAY
# ============================================================

overlay_output = (
    OUTPUT_DIR /
    f"{image_path.stem}_gradcam.jpg"
)

cv2.imwrite(
    str(overlay_output),
    overlay
)


# ============================================================
# 21. RESULTS
# ============================================================

print(
    "\nOriginal image saved to:"
)

print(
    original_output
)

print(
    "\nHeatmap saved to:"
)

print(
    heatmap_output
)

print(
    "\nGrad-CAM overlay saved to:"
)

print(
    overlay_output
)

print(
    "\nGrad-CAM completed successfully."
)