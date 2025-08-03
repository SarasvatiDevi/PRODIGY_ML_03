import cv2
import joblib
from skimage.feature import hog
import numpy as np
model = joblib.load("model.pkl")
IMG_SIZE = 64
test_image_path = "C:/Users/Sanjina/Downloads/dogs_vs_cats/test/dogs/dog.1533.jpg"
img = cv2.imread(test_image_path)
if img is None:
    print("Could not read the image. Check the file path.")
else:
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=True)
    prediction = model.predict([features])[0]
    label = "Dog" if prediction == 1 else "Cat"
    print(f"Prediction: {label}")
