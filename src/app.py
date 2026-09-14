import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Digital Deception Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# 2. MODERN APPLE-INSPIRED UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 50% -15%,
                rgba(99, 102, 241, 0.16),
                transparent 34%
            ),
            radial-gradient(
                circle at 100% 45%,
                rgba(56, 189, 248, 0.07),
                transparent 28%
            ),
            #090b10;
        color: #f5f7fb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.7rem;
        padding-bottom: 4rem;
    }

    /* ---------- Typography ---------- */

    h1 {
        font-size: 3rem !important;
        line-height: 1.05 !important;
        font-weight: 760 !important;
        letter-spacing: -0.055em !important;
        margin: 0.25rem 0 0.55rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 720 !important;
        letter-spacing: -0.025em !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 680 !important;
    }

    p, label {
        color: #c5cad5;
    }

    .hero-subtitle {
        color: #8f98aa;
        font-size: 1.02rem;
        letter-spacing: -0.01em;
        margin-bottom: 1.55rem;
    }

    /* ---------- Top badge ---------- */

    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        background: rgba(129, 140, 248, 0.09);
        border: 1px solid rgba(129, 140, 248, 0.20);
        color: #b9c0ff;
        font-size: 0.72rem;
        font-weight: 720;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* ---------- Cards ---------- */

    .ui-card {
        background: linear-gradient(
            180deg,
            rgba(25, 29, 39, 0.86),
            rgba(17, 20, 28, 0.86)
        );
        border: 1px solid rgba(255, 255, 255, 0.075);
        border-radius: 22px;
        padding: 1.35rem 1.45rem;
        box-shadow:
            0 18px 50px rgba(0, 0, 0, 0.22),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
        margin: 0.55rem 0 1.2rem;
    }

    .mode-card {
        background: rgba(18, 21, 29, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.075);
        border-radius: 18px;
        padding: 0.8rem 1rem;
        margin: 0.7rem 0 1.3rem;
    }

    .muted {
        color: #919bad;
        font-size: 0.90rem;
        line-height: 1.65;
    }

    .section-kicker {
        color: #8993a7;
        font-size: 0.68rem;
        font-weight: 760;
        letter-spacing: 0.12em;
        margin: 0.35rem 0 0.55rem;
    }

    /* ---------- Result cards ---------- */

    .result-card {
        background: linear-gradient(
            145deg,
            rgba(30, 34, 46, 0.95),
            rgba(17, 20, 28, 0.92)
        );
        border: 1px solid rgba(255, 255, 255, 0.075);
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        min-height: 112px;
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.18);
    }

    .result-label {
        color: #858fa3;
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-weight: 720;
        margin-bottom: 0.55rem;
    }

    .result-value {
        color: #f7f8fb;
        font-size: 1.55rem;
        line-height: 1.15;
        font-weight: 760;
        letter-spacing: -0.025em;
    }

    /* ---------- Image presentation ---------- */

    .image-frame {
        background: #0d1016;
        border: 1px solid rgba(255,255,255,0.075);
        border-radius: 20px;
        padding: 0.45rem;
        box-shadow: 0 18px 45px rgba(0,0,0,0.20);
    }

    [data-testid="stImage"] img {
        border-radius: 16px;
    }

    /* ---------- File uploader ---------- */

    [data-testid="stFileUploader"] {
        background: rgba(18, 21, 29, 0.72);
        border: 1px dashed rgba(148, 163, 184, 0.28);
        border-radius: 20px;
        padding: 0.65rem;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(129, 140, 248, 0.65);
    }

    [data-testid="stFileUploader"] section {
        border: none !important;
    }

    /* ---------- Radio / Selectbox ---------- */

    div[role="radiogroup"] {
        gap: 0.55rem;
    }

    div[role="radiogroup"] label {
        background: rgba(24, 28, 38, 0.78);
        border: 1px solid rgba(255,255,255,0.075);
        border-radius: 12px;
        padding: 0.55rem 0.85rem;
    }

    div[data-baseweb="select"] > div {
        background: #171b24;
        border-color: rgba(255,255,255,0.09);
        border-radius: 12px;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        font-weight: 680;
        min-height: 2.7rem;
    }

    /* ---------- Progress ---------- */

    [data-testid="stProgress"] {
        margin-top: 0.35rem;
    }

    [data-testid="stProgress"] > div {
        background: #202532;
        border-radius: 999px;
    }

    [data-testid="stProgress"] > div > div {
        border-radius: 999px;
    }

    /* ---------- Alerts ---------- */

    [data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.075);
    }

    /* ---------- Expander ---------- */

    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.075);
        border-radius: 16px;
        background: rgba(18, 21, 29, 0.62);
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #0b0e14;
        border-right: 1px solid rgba(255,255,255,0.055);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.25rem !important;
        letter-spacing: -0.025em !important;
    }

    /* ---------- Divider ---------- */

    hr {
        border-color: rgba(255,255,255,0.075);
        margin: 2rem 0;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #667083;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1.3rem;
        border-top: 1px solid rgba(255,255,255,0.065);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. PATHS + DEVICE
# ============================================================

CASIA_MODEL_PATH = Path(
    "models/resnet50_robust_v2_best.pth"
)

CIFAKE_MODEL_PATH = Path(
    "models/resnet50_cifake_best.pth"
)

CASIA_TEST_DIR = Path(
    "processed_dataset/test"
)

CIFAKE_TEST_DIR = Path(
    "cifake_processed/test"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. MODEL CREATION
# ============================================================

def create_resnet50(classifier_type="dropout"):
    """
    Build the ResNet-50 architecture.

    classifier_type:
        - dropout: ResNet-50 + Dropout + Linear
        - linear:  ResNet-50 + Linear

    The loader detects which classifier was used from the
    saved checkpoint, so the CiFAKE model does not have to
    be assumed to use the same classifier as the CASIA model.
    """

    model = models.resnet50(
        weights=None
    )

    num_features = model.fc.in_features

    if classifier_type == "dropout":

        model.fc = nn.Sequential(
            nn.Dropout(
                p=0.4
            ),
            nn.Linear(
                num_features,
                2
            )
        )

    else:

        model.fc = nn.Linear(
            num_features,
            2
        )

    return model


def load_checkpoint(model_path):
    """
    Load a PyTorch checkpoint safely and return its state dict.
    """

    try:

        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )

    except TypeError:

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

    return checkpoint


@st.cache_resource
def load_model(model_path):

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    state_dict = load_checkpoint(
        model_path
    )

    # Detect the classifier structure from the
    # actual checkpoint instead of assuming it.
    if "fc.1.weight" in state_dict:

        classifier_type = "dropout"

    elif "fc.weight" in state_dict:

        classifier_type = "linear"

    else:

        raise RuntimeError(
            "Unsupported ResNet-50 checkpoint structure. "
            "The classifier layer could not be detected."
        )

    model = create_resnet50(
        classifier_type
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    model.eval()

    return model, classifier_type


# ============================================================
# 5. IMAGE TRANSFORMATION
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
# 6. GRAD-CAM
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

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
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

        # Do NOT use torch.no_grad() here.
        # Grad-CAM requires gradients.
        output = self.model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
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
                "Grad-CAM could not obtain "
                "activations or gradients."
            )

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(
            dim=(1, 2)
        )

        cam = torch.sum(
            weights[:, None, None]
            * activations,
            dim=0
        )

        cam = torch.relu(
            cam
        )

        cam = cam.detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

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
            probabilities.detach().cpu().numpy()[0]
        )

    def close(self):

        """
        Remove hooks after analysis.

        This prevents hooks from accumulating when Streamlit
        analyses several images during the same session.
        """

        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    """
    <div style="margin-bottom:0.55rem;">
        <span class="brand-badge">✦ AI Image Forensics</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Digital Deception Detection")

st.markdown(
    '<div class="hero-subtitle">'
    'Deep Learning Based Image Authenticity Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="ui-card" style="padding:1.15rem 1.35rem; margin-bottom:1.15rem;">
        <div style="font-size:1rem; line-height:1.55; color:#eef1f7;">
            Choose a detection mode and upload an image for analysis.
        </div>
        <div class="muted" style="margin-top:0.45rem;">
            Select <b style="color:#e6e9f0;">Tampered Image</b> for
            manipulation detection or <b style="color:#e6e9f0;">AI-Generated Image</b>
            for synthetic-image detection.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 8. DETECTION MODE
# ============================================================

st.markdown(
    '<div class="section-kicker">DETECTION MODE</div>',
    unsafe_allow_html=True
)

detection_mode = st.radio(
    "Detection mode",
    [
        "Tampered Image",
        "AI-Generated Image"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


if detection_mode == "Tampered Image":

    MODEL_PATH = CASIA_MODEL_PATH

    TEST_DIR = CASIA_TEST_DIR

    classes = [
        "Authentic",
        "Manipulated"
    ]

    model_name = (
        "Robust ResNet-50 V2"
    )

    dataset_name = (
        "CASIA 2.0"
    )

    description = (
        "Detects image manipulation using the robust "
        "ResNet-50 V2 model trained on CASIA 2.0."
    )

else:

    MODEL_PATH = CIFAKE_MODEL_PATH

    TEST_DIR = CIFAKE_TEST_DIR

    # CiFAKE training uses ImageFolder ordering:
    # FAKE = 0
    # REAL = 1
    classes = [
        "FAKE",
        "REAL"
    ]

    model_name = (
        "CiFAKE ResNet-50"
    )

    dataset_name = (
        "CiFAKE"
    )

    description = (
        "Distinguishes real images from AI-generated "
        "images using the CiFAKE ResNet-50 model."
    )


# ============================================================
# 8. LOAD SELECTED MODEL
# ============================================================

try:

    model, classifier_type = load_model(
        MODEL_PATH
    )

except Exception as error:

    st.error(
        "Unable to load the selected model."
    )

    st.exception(error)

    st.stop()


target_layer = model.layer4[-1]


# ============================================================
# 9. MODEL SUMMARY
# ============================================================

# Keep the selected model information concise and safe.  In particular,
# description is plain text here, so no HTML can accidentally be displayed
# as literal text inside the page.
st.markdown(
    f"""
    <div class="mode-card">
        <div style="font-size:0.96rem; color:#eef1f7; font-weight:650;">
            {detection_mode}
            <span style="color:#687286; padding:0 0.45rem;">·</span>
            {model_name}
        </div>
        <div class="muted" style="margin-top:0.35rem;">
            {description} Grad-CAM provides visual model explainability.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 10. SIDEBAR
# ============================================================

st.sidebar.title(
    "System Information"
)

st.sidebar.write(
    f"**Detection Mode:** {detection_mode}"
)

st.sidebar.write(
    f"**Model:** {model_name}"
)

st.sidebar.write(
    f"**Dataset:** {dataset_name}"
)

st.sidebar.write(
    "**Architecture:** ResNet-50"
)

st.sidebar.write(
    "**Task:** Binary Image Classification"
)

if detection_mode == "Tampered Image":

    st.sidebar.write(
        "**Classes:** Authentic / Manipulated"
    )

else:

    st.sidebar.write(
        "**Classes:** Real / AI-Generated"
    )

st.sidebar.write(
    "**Explainability:** Grad-CAM"
)

st.sidebar.write(
    "**Input Size:** 224 × 224"
)

st.sidebar.write(
    "**Device:** " + str(device)
)

if torch.cuda.is_available():

    st.sidebar.write(
        "**GPU:** "
        + torch.cuda.get_device_name(0)
    )

st.sidebar.success(
    f"{model_name} loaded"
)

st.sidebar.caption(
    f"Model: {MODEL_PATH}"
)


# ============================================================
# 11. IMAGE SOURCE
# ============================================================

st.markdown(
    '<div class="muted" style="margin-top:1.5rem; margin-bottom:0.25rem; '
    'font-weight:700; color:#aeb7c7;">DETECTION MODE</div>',
    unsafe_allow_html=True
)

# The detection mode selector is rendered near the top of the page.
# The existing selector remains the source of truth for the selected model.

st.subheader(
    "Choose Image Source"
)

image_source = st.radio(
    "Select how you want to provide the image",
    [
        "Upload Image",
        "Test Dataset"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# 12. GET IMAGE
# ============================================================

image = None
image_name = None
dataset_category = None


if image_source == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            image_name = uploaded_file.name

        except Exception:

            st.error(
                "Unable to read this image."
            )

            st.stop()


else:

    if not TEST_DIR.exists():

        st.error(
            f"Test dataset not found:\n{TEST_DIR}"
        )

        st.stop()

    if detection_mode == "Tampered Image":

        categories = [
            "Authentic",
            "Manipulated"
        ]

        dataset_category = st.selectbox(
            "Choose test dataset category",
            categories
        )

        image_dir = (
            TEST_DIR / "authentic"
            if dataset_category == "Authentic"
            else TEST_DIR / "manipulated"
        )

    else:

        categories = [
            "Real",
            "AI-Generated"
        ]

        dataset_category = st.selectbox(
            "Choose test dataset category",
            categories
        )

        image_dir = (
            TEST_DIR / "REAL"
            if dataset_category == "Real"
            else TEST_DIR / "FAKE"
        )

    if not image_dir.exists():

        st.error(
            f"Dataset category folder not found:\n{image_dir}"
        )

        st.stop()

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
        "*.WEBP"
    ]:

        image_files.extend(
            image_dir.glob(extension)
        )

    image_files = sorted(
        set(image_files)
    )

    if not image_files:

        st.warning(
            f"No images found in:\n{image_dir}"
        )

        st.stop()

    selected_file = st.selectbox(
        "Choose a test image",
        image_files,
        format_func=lambda path: path.name
    )

    try:

        image = Image.open(
            selected_file
        ).convert("RGB")

        image_name = selected_file.name

    except Exception:

        st.error(
            "Unable to read the selected test image."
        )

        st.stop()


# ============================================================
# 13. PROCESS IMAGE
# ============================================================

if image is not None:

    st.subheader(
        "Selected Image"
    )

    st.markdown(
        f'<div class="muted" style="margin-bottom:0.65rem;">'
        f'Selected file · <b style="color:#d7dce6;">{image_name}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        width=520
    )

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # ========================================================
    # 14. PREDICTION + GRAD-CAM
    # ========================================================

    with st.spinner(
        "Analysing image..."
    ):

        gradcam = GradCAM(
            model,
            target_layer
        )

        try:

            cam, prediction, probabilities = (
                gradcam.generate(
                    image_tensor
                )
            )

        except Exception as error:

            st.error(
                "Grad-CAM analysis failed."
            )

            st.exception(error)

            st.stop()

        finally:

            gradcam.close()

    # ========================================================
    # 15. PREDICTION VALUES
    # ========================================================

    predicted_class = classes[
        prediction
    ]

    confidence = (
        probabilities[prediction]
        * 100
    )

    if detection_mode == "Tampered Image":

        authentic_probability = (
            probabilities[0] * 100
        )

        manipulated_probability = (
            probabilities[1] * 100
        )

        display_prediction = (
            predicted_class
        )

    else:

        fake_probability = (
            probabilities[0] * 100
        )

        real_probability = (
            probabilities[1] * 100
        )

        display_prediction = (
            "AI-Generated"
            if predicted_class == "FAKE"
            else "Real"
        )

    # ========================================================
    # 16. DETECTION RESULT
    # ========================================================

    st.subheader(
        "Detection Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Prediction</div>
                <div class="result-value">
                    {display_prediction}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Confidence</div>
                <div class="result-value">
                    {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # 17. PREDICTION PROBABILITIES
    # ========================================================

    st.subheader(
        "Prediction Probabilities"
    )

    col1, col2 = st.columns(2)

    if detection_mode == "Tampered Image":

        with col1:

            st.write(
                f"**Authentic:** "
                f"{authentic_probability:.2f}%"
            )

            st.progress(
                int(
                    min(
                        100,
                        max(
                            0,
                            authentic_probability
                        )
                    )
                )
            )

        with col2:

            st.write(
                f"**Manipulated:** "
                f"{manipulated_probability:.2f}%"
            )

            st.progress(
                int(
                    min(
                        100,
                        max(
                            0,
                            manipulated_probability
                        )
                    )
                )
            )

    else:

        with col1:

            st.write(
                f"**Real:** "
                f"{real_probability:.2f}%"
            )

            st.progress(
                int(
                    min(
                        100,
                        max(
                            0,
                            real_probability
                        )
                    )
                )
            )

        with col2:

            st.write(
                f"**AI-Generated:** "
                f"{fake_probability:.2f}%"
            )

            st.progress(
                int(
                    min(
                        100,
                        max(
                            0,
                            fake_probability
                        )
                    )
                )
            )

    # ========================================================
    # 18. GRAD-CAM
    # ========================================================

    st.subheader(
        "Model Attention"
    )

    st.write(
        """
        The Grad-CAM visualization shows the image regions
        that contributed most strongly to the model's prediction.
        """
    )

    original = np.array(
        image
    )

    height, width = (
        original.shape[:2]
    )

    cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap,
        0.40,
        0
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            overlay,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    # ========================================================
    # 19. INTERPRETATION
    # ========================================================

    st.subheader(
        "Interpretation"
    )

    if detection_mode == "Tampered Image":

        if predicted_class == "Manipulated":

            st.warning(
                """
                The model predicts that this image may contain
                patterns associated with manipulation.

                The Grad-CAM visualization highlights regions that
                contributed to the Manipulated prediction.
                """
            )

        else:

            st.success(
                """
                The model predicts that this image is likely
                Authentic based on patterns learned during training.

                The prediction does not guarantee that the image
                is completely unmodified.
                """
            )

    else:

        if predicted_class == "FAKE":

            st.warning(
                """
                The model predicts that this image is likely
                AI-generated based on patterns learned during
                CiFAKE training.

                The Grad-CAM visualization highlights regions that
                contributed to the AI-Generated prediction.
                """
            )

        else:

            st.success(
                """
                The model predicts that this image is likely
                Real based on patterns learned during CiFAKE training.

                The prediction does not guarantee that the image
                has never been edited or processed.
                """
            )

    # ========================================================
    # 20. MODEL LIMITATION
    # ========================================================

    st.info(
        """
        **Important:** This system is a research/deep-learning
        classifier, not a definitive forensic authentication tool.

        The confidence value represents the model's prediction
        probability. Images from sources, cameras, editing methods,
        or generative models that differ from the training data
        may produce incorrect predictions.
        """
    )

    # ========================================================
    # 21. TECHNICAL INFORMATION
    # ========================================================

    with st.expander(
        "Technical Information"
    ):

        st.write(
            f"**Detection Mode:** {detection_mode}"
        )

        st.write(
            f"**Model:** {model_name}"
        )

        st.write(
            f"**Dataset:** {dataset_name}"
        )

        st.write(
            "**Architecture:** ResNet-50"
        )

        st.write(
            f"**Classifier:** {classifier_type}"
        )

        st.write(
            "**Input:** 224 × 224 RGB image"
        )

        st.write(
            "**Target Layer:** ResNet-50 layer4[-1]"
        )

        st.write(
            "**Explainability:** Grad-CAM"
        )

        if detection_mode == "Tampered Image":

            st.write(
                "**Classes:** Authentic / Manipulated"
            )

        else:

            st.write(
                "**Classes:** Real / AI-Generated"
            )

        st.write(
            f"**Model File:** {MODEL_PATH}"
        )


# ============================================================
# 22. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Digital Deception Detection • ResNet-50 • Grad-CAM Explainability
    </div>
    """,
    unsafe_allow_html=True
)
