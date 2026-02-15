from predict_one import IMAGE_PATH
from suspicious_factors import detect_skin_tone_irregularity

print("Skin tone irregularity:",
      detect_skin_tone_irregularity("IMAGE_PATH"))
