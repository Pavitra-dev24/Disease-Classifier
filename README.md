# Disease-Classifier

A dual‑modality deep‑learning pipeline for disease diagnosis that combines an image‑based and a text‑based model, then fuses their predictions by user‑specified weights.

---

## 📖 Overview

**Disease-Classifier** consists of three main components:

1. **Image Classifier**: A ResNet‑50 model fine‑tuned on skin disease images from a publicly‑available Kaggle dataset.
2. **Text Classifier**: A BERT‑based model trained on a self‑curated symptom↔disease CSV.
3. **Ensembler**: A wrapper that loads both models, takes an image file and a symptom description as input, and outputs the weighted ensemble prediction.

Users choose an image‑text weight (between 0.0 and 1.0) to control how much each modality contributes to the final decision.

---

## 🔍 Repository Structure

```
Disease-Classifier/
├── data/
│   ├── image/                  # (not tracked) Full Kaggle SkinDisease images
│   │   ├── Train/              # 22 classes (Acne, Psoriasis, …)
│   │   └── Test/
│   └── text/
│       └── skin_disease_symptoms_dataset.csv  # Symptom→Disease pairs
├── notebooks/                  # (optional) EDA or experiments
├── models/                     # Saved checkpoints
│   ├── image_model.pth
│   └── text_model.pth
├── src/
│   ├── image_classifier.py     # Training script for image model
│   ├── text_classifier.py      # Training script for text model
│   └── ensemble.py             # Script to run prediction on both
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

* Python 3.8+
* PyTorch 1.10+
* torchvision 0.11+
* transformers 4.5+
* scikit-learn 0.24+
* matplotlib 3.4+

Install via:

```bash
pip install -r requirements.txt
```

---

## 📥 Data Preparation

### 1. Image Dataset (Kaggle)

This project uses the **SkinDisease** dataset from Kaggle (22 skin conditions):

> [https://www.kaggle.com/datasets/pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)

To download:

```bash
# 1. Install & configure Kaggle API
pip install kaggle
export KAGGLE_USERNAME="<your_username>"
export KAGGLE_KEY="<your_api_key>"

# 2. Download & unzip into `data/image`
kaggle datasets download -d pacificrm/skindiseasedataset -p data/image --unzip
```

The directory should mirror:

```
data/image/
├── Train/
│   ├── Acne/
│   ├── Actinic_Keratosis/
│   └── …
└── Test/
    ├── Acne/
    ├── Actinic_Keratosis/
    └── …
```

### 2. Text Dataset

A self‑curated CSV of symptom lists and disease labels is provided at:

```
data/text/skin_disease_symptoms_dataset.csv
```

No external download is needed—this file is tracked in the repository.

---

## 🚀 Training

### 1. Train Image Model

```bash
python src/image_classifier.py \
  --train-dir data/image/Train \
  --test-dir  data/image/Test \
  --checkpoint-path models/image_model.pth \
  --epochs 15 --batch-size 32 --lr 1e-4
```

After training, the script prints epoch‑level loss/accuracy, evaluation metrics (precision, recall, F1, Cohen’s Kappa), displays a confusion matrix, and saves the checkpoint.

### 2. Train Text Model

```bash
python src/text_classifier.py \
  --csv-path data/text/skin_disease_symptoms_dataset.csv \
  --checkpoint-path models/text_model.pth \
  --epochs 10 --batch-size 16 --lr 2e-5
```

This will fine‑tune BERT, report epoch‑level metrics and a confusion matrix, then save `text_model.pth`.

---

## 🎯 Inference & Ensembling

Run the color-coded ensemble script to classify a new sample:

```bash
python src/ensemble.py
```

* **Inputs**: Path to an image + symptom description text.
* **Prompt**: Specify an image weight `w_img` between 0.0 and 1.0 (text weight is `1 - w_img`).
* **Output**: Top‑class prediction and top‑3 probabilities.

Example:

```
Enter IMAGE FILE path: data/image/Test/Acne/img_001.jpg
Enter TEXT DESCRIPTION of symptoms: "pimples on cheeks, redness"
Enter IMAGE weight [0.0–1.0]: 0.7

Ensembled ➔ Acne
  Acne                     0.8432
  Rosacea                  0.0721
  Eczema                   0.0378
```

---

## 📜 License & Citation

* **Image Dataset**: CC‑BY‑4.0 ([https://www.kaggle.com/datasets/pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset))
* **Text Dataset**: Proprietary/curated; for research use.

Please cite this repository if you use it.
