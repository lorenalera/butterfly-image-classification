import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the SVM model
svm_model = joblib.load('svm_model_linear.pkl')

def preprocess_images(images):
    # Resize images to a fixed size (e.g., 224x224)
    resized_images = [cv2.resize(img, (224, 224)) for img in images]
    # Flatten images
    flattened_images = [img.flatten() for img in resized_images]
    return np.array(flattened_images)

def load_images_from_csv(csv_file, folder):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = os.path.join(folder, row['filename'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        if img is not None:
            images.append(img)
            labels.append(row['label'])
    return images, labels

# CSV file containing file paths and labels
train_csv_file = 'Dataset/Training_set.csv'
test_csv_file = 'Dataset/Testing_set.csv'
# Folder paths for the images
train_folder = 'Dataset/train'
test_folder = 'Dataset/test'

# Load images and labels from CSV files for training and test sets
train_images, train_labels = load_images_from_csv(train_csv_file, train_folder)
test_images, test_labels = load_images_from_csv(test_csv_file, test_folder)

X_train = preprocess_images(train_images)
y_train = np.array(train_labels)
X_test = preprocess_images(test_images)
y_test = np.array(test_labels)

# Preprocess images (assuming you have X_test and y_test already loaded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)


# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision, recall, and F1-score for each class
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

# Extract scores for the first 10 classes
class_names = np.unique(y_test)[:10]
precision_10 = precision[:10]
recall_10 = recall[:10]
f1_10 = f1[:10]

# Plot histograms for precision, recall, and F1-score
plt.figure(figsize=(15, 10))

# Precision histogram
plt.subplot(2, 2, 1)
plt.bar(class_names, precision_10)
plt.title('Precision for First 10 Classes')
plt.xlabel('Class')
plt.ylabel('Precision')

# Recall histogram
plt.subplot(2, 2, 2)
plt.bar(class_names, recall_10)
plt.title('Recall for First 10 Classes')
plt.xlabel('Class')
plt.ylabel('Recall')

# F1-score histogram
plt.subplot(2, 2, 3)
plt.bar(class_names, f1_10)
plt.title('F1-Score for First 10 Classes')
plt.xlabel('Class')
plt.ylabel('F1-Score')

plt.tight_layout()
plt.show()
