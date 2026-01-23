import os

def count_images(path):
    return len([
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

base = "dataset/train"

real_count = count_images(os.path.join(base, "real"))
fake_count = count_images(os.path.join(base, "fake"))

print("Real images:", real_count)
print("Fake images:", fake_count)
