import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import hog
import joblib

DATA_DIR = "C:/Users/Sanjina/Downloads/dogs_vs_cats/train"
IMG_SIZE = 64

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
    return features

def load_data(limit=1000):
    X, y = [], []
    classes = {'cats': 0, 'dogs': 1}

    for class_name, label in classes.items():
        folder = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        files = os.listdir(folder)
        count = 0
        for f in files:
            path = os.path.join(folder, f)
            if count >= limit:
                break
            if os.path.isfile(path) and f.lower().endswith(('.jpg', '.png')):
                features = extract_features(path)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    count += 1

    return np.array(X), np.array(y)

def main():
    print("Loading data...")
    X, y = load_data(limit=1000)

    if len(X) == 0:
        print("No data loaded! Check dataset path.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training SVM...")
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

    joblib.dump(clf, "model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    main()
