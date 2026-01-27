import torch
from torchvision import transforms
from PIL import Image

from model import get_model

DEVICE = "cpu"
MODEL_PATH = "output/deepfake_model.pth"

# Load model
model = get_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    # Probabilities
    p_real = probs[0][0].item()
    p_fake = probs[0][1].item()

    # Trust score (0–100)
    trust_score = int(p_real * 100)

    # Risk level
    if trust_score >= 70:
        risk = "LOW"
    elif trust_score >= 40:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # Final label
    label = "FAKE" if p_fake > p_real else "REAL"

    # Output
    print(f"Prediction: {label}")
    print(f"Trust Score: {trust_score}/100")
    print(f"Risk Level: {risk}")

# --------- Test ---------
if __name__ == "__main__":
    predict("output/face_crop.jpg")   # change image path if needed
