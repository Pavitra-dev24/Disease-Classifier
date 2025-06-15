import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from transformers import BertForSequenceClassification, BertTokenizerFast

CLASS_NAMES = [
    "Acne","Actinic Keratosis","Benign Tumors","Bullous",
    "Candidiasis","Drug Eruption","Eczema","Infestations/Bites",
    "Lichen","Lupus","Moles","Psoriasis","Rosacea",
    "Seborrheic Keratoses","Skin Cancer","Sun/Sunlight Damage",
    "Tinea","Unknown/Normal","Vascular Tumors","Vasculitis",
    "Vitiligo","Warts"
]

class ImageClassifierWrapper(nn.Module):
    def __init__(self, ckpt_path, num_classes=len(CLASS_NAMES)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        x = self.tfms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            return F.softmax(logits, dim=1).cpu().squeeze(0)

class TextClassifierWrapper:
    def __init__(self, ckpt_path, num_classes=len(CLASS_NAMES)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_classes
        )
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict(self, text):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(
                input_ids=enc.input_ids.to(self.device),
                attention_mask=enc.attention_mask.to(self.device)
            ).logits
            return F.softmax(logits, dim=1).cpu().squeeze(0)

def main():
    
    img_ckpt = r"image_model.pth"
    txt_ckpt = r"text_model.pth"

    img_clf = ImageClassifierWrapper(img_ckpt)
    txt_clf = TextClassifierWrapper(txt_ckpt)

    
    img_path = input("Enter IMAGE FILE path: ").strip()
    txt_desc = input("Enter TEXT DESCRIPTION of symptoms: ").strip()

    
    while True:
        try:
            w_img = float(input("Enter IMAGE weight [0.0–1.0]: ").strip())
            if 0.0 <= w_img <= 1.0:
                break
        except ValueError:
            pass
        print("Please enter a decimal between 0 and 1.")
    w_txt = 1.0 - w_img

    
    p_img = img_clf.predict(img_path)
    p_txt = txt_clf.predict(txt_desc)
    p_final = w_img * p_img + w_txt * p_txt

    
    idx = torch.argmax(p_final).item()
    print(f"\nEnsembled ➔ {CLASS_NAMES[idx]}")
    top3 = torch.topk(p_final, 3)
    for score, i in zip(top3.values.tolist(), top3.indices.tolist()):
        print(f"  {CLASS_NAMES[i]:25s} {score:.4f}")

if __name__ == "__main__":
    main()
