# 🌿 CropGuard AI — Complete Build Guide
## YOLOv8 Crop Disease Detection System

---

## Project Overview

| Component | Technology |
|-----------|-----------|
| AI Model  | YOLOv8 (Ultralytics) |
| Backend   | FastAPI + Python |
| Frontend  | HTML + TailwindCSS |
| Dataset   | PlantVillage (54,306 images) |
| Training  | Google Colab T4 GPU (free) |
| Deploy    | Docker + Railway/Render |

---

## PHASE 1: Dataset Setup

### Option A — Kaggle (Recommended, Free)

1. Create Kaggle account at kaggle.com
2. Go to: Account → API → Create New Token → download `kaggle.json`
3. Place in `~/.kaggle/kaggle.json`

```bash
pip install kaggle
kaggle datasets download -d emmarex/plantdisease --unzip -p raw_dataset/
```

4. Prepare dataset:
```bash
python training/train.py --prepare \
  --source raw_dataset/PlantVillage \
  --output dataset/
```

### Option B — Roboflow (Labeled, Ready-to-Train)

1. Go to roboflow.com → Create account
2. Search: "PlantVillage" or "Plant Disease Detection"
3. Fork the dataset → Export as "YOLOv8" format
4. Copy the download code snippet they give you

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("your-workspace").project("plant-disease")
dataset = project.version(1).download("yolov8")
```

### Dataset Structure After Preparation

```
dataset/
├── train/
│   ├── Apple___Apple_scab/        [2,016 images]
│   ├── Apple___Black_rot/         [620 images]
│   ├── Tomato___Early_blight/     [800 images]
│   ├── Tomato___Late_blight/      [1,851 images]
│   └── ...                        [38 classes total]
└── val/
    ├── Apple___Apple_scab/        [356 images]
    └── ...
```

### PlantVillage Dataset Stats

| Stat | Value |
|------|-------|
| Total Images | 54,306 |
| Crops | 14 |
| Disease Classes | 38 |
| Healthy Classes | 14 |
| Image Format | JPG (256×256) |
| License | CC0 Public Domain |

---

## PHASE 2: Training on Google Colab

### Step 1 — Open Colab

Go to: colab.research.google.com
Runtime → Change runtime type → **T4 GPU**

### Step 2 — Colab Training Notebook

```python
# Cell 1: Install dependencies
!pip install ultralytics roboflow

# Cell 2: Mount Google Drive (to save model)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Download dataset from Kaggle
!pip install kaggle
from google.colab import files
files.upload()  # upload your kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d emmarex/plantdisease --unzip -p raw_dataset/

# Cell 4: Prepare dataset
import shutil, random
from pathlib import Path

def prepare(source, output, val_split=0.15):
    for class_dir in Path(source).iterdir():
        if not class_dir.is_dir(): continue
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
        random.shuffle(images)
        split = int(len(images) * (1 - val_split))
        for split_name, imgs in [('train', images[:split]), ('val', images[split:])]:
            dest = Path(output) / split_name / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

prepare('raw_dataset/PlantVillage', 'dataset/')
print("Dataset prepared!")

# Cell 5: Train YOLOv8 classification model
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # nano = fast; try yolov8s-cls.pt for better accuracy

results = model.train(
    data='dataset/',
    epochs=50,
    imgsz=224,
    batch=64,
    lr0=0.01,
    patience=10,
    optimizer='AdamW',
    cos_lr=True,
    augment=True,
    device='0',   # GPU
    project='runs/train',
    name='crop_disease_v1',
    save=True,
    plots=True,
)

# Cell 6: Evaluate
metrics = model.val(data='dataset/')
print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")

# Cell 7: Save best model to Drive
!cp runs/train/crop_disease_v1/weights/best.pt /content/drive/MyDrive/crop_disease_best.pt
print("Model saved to Google Drive!")
```

### Expected Training Results

| Model | Top-1 Accuracy | Top-5 Accuracy | Train Time (T4) |
|-------|---------------|---------------|----------------|
| YOLOv8n-cls | ~93–95% | ~99% | ~45 min |
| YOLOv8s-cls | ~95–97% | ~99% | ~90 min |
| YOLOv8m-cls | ~97–98% | ~99.5% | ~3 hours |

---

## PHASE 3: Backend Setup

### Prerequisites
- Python 3.10+
- Download `best.pt` from your Colab training

### Installation

```bash
cd backend
pip install -r requirements.txt
```

### Place Your Model

```bash
mkdir -p models
cp /path/to/best.pt models/best.pt
```

### Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Health check |
| GET  | `/health` | API status |
| POST | `/predict` | Upload image file |
| POST | `/predict/base64` | Send base64 image |
| GET  | `/docs` | Swagger UI |

### Test the API

```bash
# Using curl
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/tomato_leaf.jpg"

# Using Python
import requests
with open('tomato.jpg', 'rb') as f:
    resp = requests.post('http://localhost:8000/predict', files={'file': f})
print(resp.json())
```

---

## PHASE 4: Frontend Setup

The frontend is a single HTML file (no Node.js required).

```bash
# Serve locally
cd frontend
python -m http.server 3000
# Open: http://localhost:3000
```

OR serve with any web server:
- VS Code Live Server extension
- Nginx
- Netlify (drag and drop the HTML file)

---

## PHASE 5: Docker Deployment

### Run Full Stack with Docker

```bash
# Build and start everything
docker-compose up --build

# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Production Deployment Options

#### Option A: Railway (Easiest)
1. Push code to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Add environment variable: `MODEL_PATH=models/best.pt`
4. Upload `best.pt` via Railway volume

#### Option B: Render
1. Connect GitHub repo at render.com
2. Create Web Service → Docker
3. Add `MODEL_PATH` env var

#### Option C: AWS EC2 (Full Control)
```bash
# On EC2 instance
sudo apt update && sudo apt install docker.io docker-compose -y
git clone your-repo && cd your-repo
# Upload best.pt to backend/models/
docker-compose up -d
```

---

## PHASE 6: Model Improvement Tips

### Data Augmentation (Already in train.py)
- Horizontal/vertical flips
- Rotation ±15°
- HSV color jitter
- Random crop/scale

### Fine-tune for Your Region
If your crops have different disease strains:

1. Collect your own images (min 100 per class)
2. Label with Roboflow (free tier available)
3. Fine-tune the pre-trained model:

```python
model = YOLO('runs/train/crop_disease_v1/weights/best.pt')  # load your trained model
model.train(
    data='my_local_dataset/',
    epochs=20,       # fewer epochs for fine-tuning
    lr0=0.001,       # lower LR for fine-tuning
    freeze=10,       # freeze first 10 layers
)
```

### Improve Accuracy
- Use YOLOv8s-cls or YOLOv8m-cls instead of nano
- Train more epochs (100+)
- Add more augmentations
- Collect real-field images for your local crops

---

## File Structure

```
crop-disease-detection/
├── backend/
│   ├── app/
│   │   ├── main.py          ← FastAPI app entry point
│   │   ├── inference.py     ← YOLOv8 inference engine
│   │   ├── models.py        ← Pydantic request/response models
│   │   └── disease_info.py  ← Disease treatment database (38 classes)
│   ├── models/
│   │   └── best.pt          ← YOUR TRAINED MODEL (add this!)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html           ← Complete single-page app
├── training/
│   └── train.py             ← Full training + evaluation script
├── docker-compose.yml
└── docs/
    └── COMPLETE_GUIDE.md    ← This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Place `best.pt` in `backend/models/` |
| `CORS error` | Add your frontend URL to `allow_origins` in `main.py` |
| `Low accuracy` | Use `yolov8s-cls.pt` instead of nano, train more epochs |
| `Slow inference` | Enable GPU, or use ONNX exported model |
| `Memory error in Colab` | Reduce batch size to 32 or 16 |
| `FileNotFoundError` | Check dataset path is correct |

---

## Resources

- **YOLOv8 Docs**: docs.ultralytics.com
- **PlantVillage Kaggle**: kaggle.com/datasets/emmarex/plantdisease
- **Roboflow**: roboflow.com
- **FastAPI Docs**: fastapi.tiangolo.com
- **Original Paper**: "Using Deep Learning for Image-Based Plant Disease Detection" — Mohanty et al. (2016)

---

*Built with ❤️ for farmers and agricultural researchers*
