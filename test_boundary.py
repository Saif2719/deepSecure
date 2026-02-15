from predict_one import IMAGE_PATH
from suspicious_factors import detect_facial_boundary_issue

result = detect_facial_boundary_issue("IMAGE_PATH")

print("Facial boundary inconsistency:", result)
