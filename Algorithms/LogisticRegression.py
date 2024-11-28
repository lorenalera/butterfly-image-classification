import cv2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_image(img):
    # Resize image to a fixed size
    img = cv2.resize(img, (320, 320))
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Apply histogram equalization
    img_eq = cv2.equalizeHist(img_blur)
    # Apply edge detection (Canny)
    img_edges = cv2.Canny(img_eq, 100, 200)
    return img_edges

def load_dataset(csv_file, image_dir, max_classes=10):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    class_counter = {}
    for index, row in df.iterrows():
        if row['label'] in class_counter:
            class_counter[row['label']] += 1
        else:
            class_counter[row['label']] = 1
        if class_counter[row['label']] > max_classes:
            continue
        img_path = os.path.join(image_dir, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img_processed = preprocess_image(img)
            images.append(img_processed.flatten())
            labels.append(row['label'])
    return np.array(images), np.array(labels)

# Load training dataset with only the first 10 classes
X_train, Y_train = load_dataset('Dataset/training_set.csv', 'Dataset/train')

# Load testing dataset with only the first 10 classes
X_test, Y_test = load_dataset('Dataset/testing_set.csv', 'Dataset/test')

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, Y_train)

# Save the trained model
joblib.dump(model, 'logistic_regression_model.pkl')

# Make predictions
Y_pred = model.predict(X_test_scaled)

# Calculate the performance metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average=None)
recall = recall_score(Y_test, Y_pred, average=None)
f1 = f1_score(Y_test, Y_pred, average=None)

# Print classification report
print(classification_report(Y_test, Y_pred))

# Print the performance metrics
print("Accuracy:", accuracy)

# Plot histograms for precision, recall, and F1-score for each class
class_names = sorted(set(Y_test))
plt.figure(figsize=(15, 10))

# Precision histogram
plt.subplot(2, 2, 1)
plt.bar(class_names, precision)
plt.title('Precision for Each Class')
plt.xlabel('Class')
plt.ylabel('Precision')
plt.xticks(rotation=45)

# Recall histogram
plt.subplot(2, 2, 2)
plt.bar(class_names, recall)
plt.title('Recall for Each Class')
plt.xlabel('Class')
plt.ylabel('Recall')
plt.xticks(rotation=45)

# F1-score histogram
plt.subplot(2, 2, 3)
plt.bar(class_names, f1)
plt.title('F1-Score for Each Class')
plt.xlabel('Class')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Plot confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
