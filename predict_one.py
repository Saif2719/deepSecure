import torch
from torchvision import transforms
from PIL import Image

from model import get_model
from suspicious_factors import (
    detect_facial_boundary_issue,
    detect_abnormal_eye_texture,
    detect_skin_tone_irregularity,
    detect_gan_artifacts
)


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
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

    p_real = probs[0][0].item()
    p_fake = probs[0][1].item()

    trust_score = int(p_real * 100)

    # Final label
    label = "FAKE" if p_fake > p_real else "REAL"

    # Risk level (based on Trust Score)
    if trust_score >= 80:
        risk = "LOW"
    elif trust_score >= 40:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    print(f"Prediction: {label}")
    print(f"Trust Score: {trust_score}/100")
    print(f"Risk Level: {risk}")

    # Suspicious factors ONLY for FAKE
    if label == "FAKE":
        suspicious = []

        if detect_facial_boundary_issue(image_path):
            suspicious.append("Facial boundary inconsistency")

        if detect_abnormal_eye_texture(image_path):
            suspicious.append("Abnormal eye texture")

        if detect_skin_tone_irregularity(image_path):
            suspicious.append("Skin tone irregularity")

        if detect_gan_artifacts(image_path):
            suspicious.append("GAN artifact patterns detected")

        if suspicious:
            print("\nSuspicious Factors:")
            for s in suspicious:
                print("•", s)
        else:
            print("\nSuspicious Factors: none")
    else:
        print("\nSuspicious Factors: none")

# --------- Test ---------
if __name__ == "__main__":
    predict("profile-2.jpg")   # change image path if needed
