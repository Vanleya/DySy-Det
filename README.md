# DySy-Det: A Synergistic Framework with Dynamic Reconstruction-Path Consistency for AI-Generated Image Detection [AAAI2026]

## ⚙️ Environment Setup
```bash
git clone https://github.com/Vanleya/DySy-Det
cd DySy-Det
conda create -n dysydet python=3.8 -y
conda activate dysydet
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 📦 Model Download
source: https://drive.google.com/drive/folders/11SBgHT5r2FjlWl6vSew-AuuU10Gy8kx-?usp=sharing

---

## 🗃️ Dataset Download
### GenImage Dataset
source: https://github.com/GenImage-Dataset/GenImage
### UniversalFakeDetect Dataset
source: https://github.com/WisconsinAIVision/UniversalFakeDetect

For this project, **create one JSON file per generation method**, and each JSON must include:

- `image_path` — path to the image  
- `label` — 0 or 1 (0 = real, 1 = fake)

Example:
```json
{
    "image_path": "path/to/image.jpg",
    "label": 1
}
```

---

## 💻 Testing Code
```bash
python test.py \
    --dataset_name UniversalFakeDetect
    --test_dataset ./UniFD/json \
    --model_ckpt ./checkpoints/progan.pth \
    --batch_size 32 \
    --ensemble_size 4 \
    --t 200 250 300 \
    --prompt "a photo" \
    --save_path ./results
```
