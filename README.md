<div align="center">

# 🔍 Digital Deception Detection For Images

### AI-Powered Image Forensics for Tampered & AI-Generated Images

<p>
  <strong>Detect. Analyse. Explain.</strong>
</p>

<p>
  A deep-learning based image forensics system designed to detect
  <strong>traditional image manipulation</strong> and
  <strong>AI-generated imagery</strong>, with visual explanations using Grad-CAM.
</p>

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Large%20Files-222222?style=for-the-badge&logo=git&logoColor=white)



---

## 🧠 About the Project

**Digital Deception Detection For Images** is an AI-powered image forensics application that investigates whether a digital image is potentially deceptive.

The system focuses on **two different forms of digital deception**:

| Detection Mode | Dataset | Model | Classes |
|---|---|---|---|
| 🖼️ **Tampered Image Detection** | CASIA 2.0 | Robust ResNet-50 V2 | Authentic / Manipulated |
| 🤖 **AI-Generated Image Detection** | CiFAKE | ResNet-50 | Real / AI-Generated |

The application combines deep-learning classification with **Grad-CAM explainability**, allowing users to see not only the model's prediction but also the image regions that contributed to that prediction.

> **Important:** The system provides a model-based forensic assessment. A prediction should not be interpreted as definitive proof of authenticity or manipulation.

---

# ✨ Key Features

- 🔍 **Two independent detection modes**
- 🖼️ **Traditional image tampering detection**
- 🤖 **AI-generated image detection**
- 🧠 **ResNet-50 deep-learning architectures**
- 📊 **Prediction probabilities**
- 🎯 **Confidence-based classification**
- 🔥 **Grad-CAM visual explanations**
- 📤 **Custom image upload**
- 🧪 **Built-in test dataset images**
- 🌐 **Interactive Streamlit interface**
- ⚡ **Automatic GPU/CPU detection**
- ☁️ **Streamlit Community Cloud deployment**
- 📦 **Git LFS support for large datasets and model weights**

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │  DIGITAL DECEPTION       │
                         │       DETECTION          │
                         └────────────┬─────────────┘
                                      │
                       ┌──────────────┴──────────────┐
                       │                             │
                       ▼                             ▼
              ┌─────────────────┐          ┌─────────────────┐
              │ Tampered Image  │          │ AI-Generated    │
              │    Detection    │          │     Detection   │
              └────────┬────────┘          └────────┬────────┘
                       │                            │
                       ▼                            ▼
                ┌─────────────┐              ┌─────────────┐
                │ CASIA 2.0   │              │   CiFAKE     │
                └──────┬──────┘              └──────┬──────┘
                       │                            │
                       ▼                            ▼
                ┌─────────────┐              ┌─────────────┐
                │ ResNet-50    │              │ ResNet-50    │
                │ Robust V2    │              │              │
                └──────┬──────┘              └──────┬──────┘
                       │                            │
                       ▼                            ▼
                ┌─────────────┐              ┌─────────────┐
                │ Authentic /  │              │ Real /      │
                │ Manipulated  │              │ AI-Generated│
                └──────┬──────┘              └──────┬──────┘
                       │                            │
                       └──────────────┬─────────────┘
                                      │
                                      ▼
                           ┌────────────────────┐
                           │ Prediction Results │
                           └─────────┬──────────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                         ▼                       ▼
                  📊 Probabilities        🔥 Grad-CAM
                         │                       │
                         └───────────┬───────────┘
                                     ▼
                              Final Analysis
```

---

# 🛠️ Technology Stack

<div align="center">

### Programming & Deep Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Image Processing

![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-3776AB?style=for-the-badge)

### Web Application

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Explainable AI

![GradCAM](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-8A2BE2?style=for-the-badge)

### Version Control & Deployment

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-222222?style=for-the-badge&logo=git&logoColor=white)
![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

# 🗂️ Datasets

## 1. CASIA 2.0

The **CASIA 2.0** dataset is used for traditional image tampering detection.

The model distinguishes between:

- ✅ **Authentic**
- ⚠️ **Manipulated**

The processed dataset is organised as:

```text
processed_dataset/
│
├── train/
│   ├── authentic/
│   └── manipulated/
│
├── val/
│   ├── authentic/
│   └── manipulated/
│
└── test/
    ├── authentic/
    └── manipulated/
```

### Classes

| Class | Description |
|---|---|
| ✅ Authentic | Original / non-manipulated image |
| ⚠️ Manipulated | Image containing manipulation or tampering |

---

# 2. CiFAKE

The **CiFAKE** dataset is used for AI-generated image detection.

Instead of using the entire available dataset, this project uses a selected balanced subset containing:

```text
12,614 images
```

### Dataset Composition

```text
REAL  = 6,307
FAKE  = 6,307
```

The processed dataset is organised as:

```text
cifake_processed/
│
├── train/
│   ├── real/
│   └── fake/
│
├── val/
│   ├── real/
│   └── fake/
│
└── test/
    ├── real/
    └── fake/
```

### Dataset Distribution

| Split | Images |
|---|---:|
| 🏋️ Training | 8,829 |
| 🔎 Validation | 1,891 |
| 🧪 Testing | 1,894 |
| **Total** | **12,614** |

---

# ⚙️ Image Preprocessing

Images are prepared using a standard preprocessing pipeline compatible with ResNet-based models.

### Processing Steps

```text
Input Image
     │
     ▼
Resize to 224 × 224
     │
     ▼
Convert to Tensor
     │
     ▼
ImageNet Normalisation
     │
     ▼
Model Input
```

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

During inference:

```python
image_tensor = transform(image).unsqueeze(0).to(device)
```

---

# 🧠 Model 01 — Tampered Image Detection

### Architecture

**Robust ResNet-50 V2**

### Dataset

**CASIA 2.0**

### Classification

```text
Authentic
Manipulated
```

### Model Weights

```text
models/resnet50_robust_v2_best.pth
```

---

## 📊 CASIA Evaluation Results

The final model was evaluated on **1,894 test images**.

| Metric | Result |
|---|---:|
| 🎯 Accuracy | **74.39%** |
| Precision | **74.45%** |
| Recall | **74.39%** |
| F1 Score | **74.42%** |

### Confusion Matrix

```text
                         Predicted
                    Authentic  Manipulated

Actual Authentic       877         248

Actual Manipulated     237         532
```

### Test Set

```text
Authentic   : 1,125
Manipulated :   769
Total       : 1,894
```

---

# 🤖 Model 02 — AI-Generated Image Detection

### Architecture

**ResNet-50**

### Dataset

**CiFAKE**

### Classification

```text
Real
AI-Generated
```

### Model Weights

```text
models/resnet50_cifake_best.pth
```

---

## 📊 CiFAKE Evaluation Results

The model was evaluated on **1,894 test images**.

| Metric | Result |
|---|---:|
| 🎯 Accuracy | **96.20%** |
| Precision | **96.20%** |
| Recall | **96.20%** |
| F1 Score | **96.20%** |

### Confusion Matrix

```text
                    Predicted
                     Real    Fake

Actual Real           909     38

Actual Fake            34    913
```

### Test Set

```text
Real : 947
Fake : 947
Total: 1,894
```

---

# 🔥 Explainable AI — Grad-CAM

A major component of the system is **Grad-CAM
(Gradient-weighted Class Activation Mapping)**.

Deep-learning models can provide highly accurate predictions, but the reasoning behind those predictions can be difficult to interpret.

Grad-CAM addresses this by generating a **visual heatmap** highlighting regions that contributed strongly to the model's prediction.

---

## Grad-CAM Pipeline

```text
              Input Image
                   │
                   ▼
             Preprocessing
                   │
                   ▼
                ResNet-50
                   │
                   ▼
               Prediction
                   │
                   ▼
              Target Layer
                   │
                   ▼
        Gradients + Activations
                   │
                   ▼
            Grad-CAM Heatmap
                   │
                   ▼
          Heatmap + Original
                   │
                   ▼
          Visual Explanation
```

The target layer used for the ResNet-based models is:

```python
model.layer4[-1]
```

The resulting heatmap is resized to the original image dimensions and overlaid on the input image.

---

# 🖥️ Application Workflow

```text
                     Start Application
                            │
                            ▼
                 Select Detection Mode
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Tampered Image              AI-Generated Image
          Detection                    Detection
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                   Select Input Method
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
             Upload Image       Test Dataset
                  │                   │
                  └─────────┬─────────┘
                            │
                            ▼
                     Preprocessing
                            │
                            ▼
                    Model Inference
                            │
                            ▼
                     Classification
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          Probability Scores        Grad-CAM
                 │                     │
                 └──────────┬──────────┘
                            │
                            ▼
                     Final Results
```

---

# 📤 Input Methods

The application supports two input methods.

## Upload Image

Users can upload their own images.

Supported formats:

```text
.jpg
.jpeg
.png
.webp
```

---

## 🧪 Test Dataset

The application also provides images from the prepared test datasets.

### CASIA

```text
Authentic
Manipulated
```

### CiFAKE

```text
Real
AI-Generated
```

This provides a controlled environment for testing the trained models.

---

# 📊 Prediction Output

After processing an image, the application provides:

### Classification

For CASIA:

```text
Authentic
       OR
Manipulated
```

For CiFAKE:

```text
Real
       OR
AI-Generated
```

### Probability Scores

Example:

```text
Authentic       72.4%
Manipulated     27.6%
```

The probability scores provide additional insight into the model's prediction.

---

# 🧩 Project Structure

```text
Digital-Deception-Detection/
│
├── 📁 dataset/
│   ├── authentic/
│   └── manipulated/
│
├── 📁 processed_dataset/
│   ├── train/
│   │   ├── authentic/
│   │   └── manipulated/
│   │
│   ├── val/
│   │   ├── authentic/
│   │   └── manipulated/
│   │
│   └── test/
│       ├── authentic/
│       └── manipulated/
│
├── 📁 cifake_processed/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   │
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   │
│   └── test/
│       ├── real/
│       └── fake/
│
├── 📁 models/
│   ├── resnet50_robust_v2_best.pth
│   └── resnet50_cifake_best.pth
│
├── 📁 src/
│   ├── app.py
│   ├── evaluate.py
│   ├── evaluate_cifake.py
│   ├── gradcam.py
│   ├── organize_dataset.py
│   ├── prepare_cifake.py
│   ├── preprocessing.py
│   ├── train_cifake.py
│   └── train_robust_v2.py
│
├── requirements.txt
└── README.md
```

---

# 📄 Source Code Overview

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `train_robust_v2.py` | Train CASIA Robust ResNet-50 V2 |
| `train_cifake.py` | Train CiFAKE ResNet-50 |
| `evaluate.py` | Evaluate CASIA model |
| `evaluate_cifake.py` | Evaluate CiFAKE model |
| `gradcam.py` | Grad-CAM implementation |
| `preprocessing.py` | Image preprocessing |
| `organize_dataset.py` | Organise CASIA dataset |
| `prepare_cifake.py` | Prepare CiFAKE dataset |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/JaweriyaFarheen/Digital-Deception-Detection.git
```

```bash
cd Digital-Deception-Detection
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses packages including:

```text
PyTorch
Torchvision
Streamlit
OpenCV
Pillow
NumPy
```

---

# ▶️ Run the Application

From the project root:

```bash
streamlit run src/app.py
```

The Streamlit application will then be available through the local browser interface.

---

# 💻 Hardware Acceleration

The application automatically determines whether CUDA is available.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Therefore:

```text
NVIDIA GPU + CUDA
        │
        ▼
      GPU
```

or:

```text
No CUDA available
        │
        ▼
      CPU
```

This allows the application to run locally as well as in cloud environments.

---

# ☁️ Deployment

The application is designed to be deployed using:

**Streamlit Community Cloud**

Application entry point:

```text
src/app.py
```

Dependency file:

```text
requirements.txt
```

Large datasets and trained model weights are managed using **Git LFS**.

---

# 📦 Git LFS

Because this project contains a large number of image files and trained PyTorch model weights, Git LFS is used for large files.

Tracked file types include:

```text
*.pth
*.jpg
*.jpeg
*.png
*.tif
*.bmp
```

Install Git LFS:

```bash
git lfs install
```

Check tracked files:

```bash
git lfs ls-files
```

---

# 📈 Results at a Glance

<div align="center">

| Detection Task | Dataset | Architecture | Accuracy |
|:---:|:---:|:---:|:---:|
| 🖼️ Image Tampering | CASIA 2.0 | Robust ResNet-50 V2 | **74.39%** |
| 🤖 AI-Generated | CiFAKE | ResNet-50 | **96.20%** |

</div>

---

# 🔬 Research Contribution

The project combines two complementary image-forensics tasks within a single application:

```text
       Traditional Image Manipulation
                    +
            AI-Generated Images
                    │
                    ▼
        ┌─────────────────────────┐
        │ Digital Deception       │
        │       Detection         │
        └────────────┬────────────┘
                     │
                     ▼
              Explainable AI
                 Grad-CAM
```

Rather than treating image classification as a simple binary prediction problem, the application also provides visual explanations through Grad-CAM.

This creates a more interpretable workflow:

```text
Prediction
    +
Probability
    +
Visual Explanation
    =
Forensic Assessment
```

---

# ⚠️ Limitations

The system is a research prototype and should not be considered a universal image-authentication solution.

### Dataset Dependence

Model performance depends on the characteristics of the datasets used during training.

### Generalisation

Images from unseen manipulation methods, cameras, compression pipelines, or newer generative AI systems may produce different results.

### Image Compression

Resizing, compression, screenshots, and social-media processing can affect image artifacts and therefore influence predictions.

### Probability ≠ Proof

A high prediction probability does not establish that an image is definitively authentic or manipulated.

---

# 🔮 Future Improvements

Potential extensions include:

- 🌐 Cross-dataset evaluation
- 🧠 Vision Transformer architectures
- 🔗 Ensemble models
- 📸 More diverse image sources
- 🧪 Additional manipulation techniques
- 🤖 Detection of newer generative AI models
- 🗜️ Robustness against image compression
- 📊 Confidence calibration
- 🔍 Additional explainability methods
- 📁 Batch image analysis
- 🎥 Video/deepfake detection
- 🧾 Metadata and EXIF analysis
- 📈 More extensive error analysis

---

# ⚖️ Ethical Considerations

Image-forensics technology can support:

- Media verification
- Digital investigations
- Academic research
- Misinformation analysis
- Content moderation
- Forensic image analysis

However, automated predictions should not be used as the sole evidence in high-stakes decisions.

Human review and additional forensic evidence may be required.

---

# 📌 Disclaimer

This project has been developed for **academic and research purposes** 

The system's predictions represent a machine-learning-based assessment and should not be interpreted as definitive proof of image authenticity, manipulation, or AI generation.

For legal, forensic, journalistic, or other high-stakes applications, model predictions should be combined with expert analysis and independent evidence.

---

# 👨‍💻 Author

<div align="center">

### Jaweriya

**Digital Deception Detection**

Built with:

`Python` · `PyTorch` · `ResNet-50` · `OpenCV` · `Streamlit` · `Grad-CAM`

</div>

---

# 📜 License

This repository is intended primarily for **academic and research purposes**.

Please ensure that any use or redistribution of the datasets complies with their respective licenses and terms of use.

---

<div align="center">

### 🔍 Detect Digital Deception with Deep Learning

**Digital Deception Detection**

</div>
