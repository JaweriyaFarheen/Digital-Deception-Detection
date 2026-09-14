from pathlib import Path
import shutil

# CASIA2 location
casia_path = Path.home() / "Downloads" / "archive (2)" / "CASIA2"

# Our project dataset folders
project_path = Path.cwd() / "dataset"
authentic_path = project_path / "authentic"
manipulated_path = project_path / "manipulated"

# Create destination folders
authentic_path.mkdir(parents=True, exist_ok=True)
manipulated_path.mkdir(parents=True, exist_ok=True)

# Copy authentic images
for image in (casia_path / "Au").iterdir():
    if image.is_file():
        shutil.copy2(image, authentic_path / image.name)

# Copy manipulated images
for image in (casia_path / "Tp").iterdir():
    if image.is_file():
        shutil.copy2(image, manipulated_path / image.name)

print("Dataset organisation completed.")
print("Authentic images:", len(list(authentic_path.iterdir())))
print("Manipulated images:", len(list(manipulated_path.iterdir())))