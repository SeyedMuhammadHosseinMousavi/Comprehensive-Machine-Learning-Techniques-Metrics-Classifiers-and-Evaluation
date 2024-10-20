%reset -f
#----------------------------------------------------
# Chapter Four
#----------------------------------------------------
# 4. Metrics
# TP, FN, FP, FN
# Classification Report
# Confusion Matrix
# ROC-AUC
# Precision-Recall AUC
# Log Loss / Cross-Entropy Loss
# Cohen's Kappa
# MCC
# Precision
# Recall (sensitivity)
# Unweighted Average Recall
# Specificity
# F1-Score
# Accuracy
# Balanced Accuracy
# Hamming Loss
# Jaccard Index
#----------------------------------------------------

import numpy as np
import pandas as pd
import os
import warnings
import time
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, log_loss, cohen_kappa_score, 
    matthews_corrcoef, balanced_accuracy_score, 
    precision_score, recall_score, f1_score, accuracy_score,
    hamming_loss, jaccard_score
)
from xgboost import XGBClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, chi2

#----------------------------------------------------
# Record the start time
start_time = time.time()

# Suppress warnings
warnings.filterwarnings("ignore")

# Parse BVH files
def parse_bvh(file_path):
    """Parse the BVH file to extract the header and motion data."""
    print(f"Parsing file: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header, motion_data = [], []
    capture_data = False
    for line in lines:
        if "MOTION" in line:
            capture_data = True
        elif capture_data:
            if line.strip().startswith("Frames") or line.strip().startswith("Frame Time"):
                continue
            motion_data.append(np.fromstring(line, sep=' '))
        else:
            header.append(line)
    print(f"Finished parsing: {file_path}")
    return header, np.array(motion_data)

def read_bvh(filename):
    """Reads a BVH file and returns motion data parsed."""
    header, motion_data = parse_bvh(filename)
    return motion_data

def interpolate_frames(motion_data, target_frame_count):
    """Interpolates BVH frames to match a target frame count."""
    print(f"Interpolating frames to match target frame count: {target_frame_count}")
    original_frame_count = len(motion_data)
    original_time = np.linspace(0, 1, original_frame_count)
    target_time = np.linspace(0, 1, target_frame_count)
    interpolated_frames = []
    for frame in np.array(motion_data).T:
        interpolator = interp1d(original_time, frame.astype(float), kind='linear')
        interpolated_frame = interpolator(target_time)
        interpolated_frames.append(interpolated_frame)
    print(f"Finished interpolation for frames: Original frame count = {original_frame_count}, Target frame count = {target_frame_count}")
    return np.array(interpolated_frames).T

def find_max_frames(folder_path):
    """Finds the maximum number of frames among all BVH files in a folder."""
    print(f"Finding maximum number of frames in folder: {folder_path}")
    max_frames = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            motion_data = read_bvh(os.path.join(folder_path, filename))
            max_frames = max(max_frames, len(motion_data))
    print(f"Maximum frames found: {max_frames}")
    return max_frames

def process_bvh_files(folder_path, max_frames):
    """Processes each BVH file in the folder after interpolating to the same number of frames."""
    print(f"Processing BVH files in folder: {folder_path}")
    all_features = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            print(f"Processing file: {filename}")
            full_path = os.path.join(folder_path, filename)
            motion_data = read_bvh(full_path)
            interpolated_frames = interpolate_frames(motion_data, max_frames)
            all_features.append(interpolated_frames)  # Keep the 2D array before flattening for feature extraction
            print(f"Finished processing file: {filename}")
    print(f"Finished processing all BVH files in folder: {folder_path}")
    return np.array(all_features)

#------------------------------------------------------------------
# Feature Extraction
#------------------------------------------------------------------

def extract_motion_features(motion_data):
    """Extracts key motion features such as angular velocity, acceleration, jerk, and range of motion."""
    print("Extracting motion features...")
    channels_per_joint = 3  # 3 channels per joint (e.g., X, Y, Z rotations)
    num_joints = motion_data.shape[1] // channels_per_joint

    features = []

    for i in range(num_joints):
        joint_rotations = motion_data[:, i * channels_per_joint:(i + 1) * channels_per_joint]

        # Angular velocity: derivative of joint rotation
        angular_velocity = np.diff(joint_rotations, axis=0)

        # Acceleration: derivative of angular velocity
        acceleration = np.diff(angular_velocity, axis=0)

        # Jerk: derivative of acceleration
        jerk = np.diff(acceleration, axis=0)

        # Range of Motion: max rotation - min rotation
        range_of_motion = np.max(joint_rotations, axis=0) - np.min(joint_rotations, axis=0)

        # Flatten all features for this joint and append
        joint_features = np.concatenate([angular_velocity.flatten(), acceleration.flatten(), jerk.flatten(), range_of_motion])
        features.append(joint_features)

    # Flatten the entire features array for all joints
    flattened_features = np.concatenate(features)
    print("Feature extraction complete.")
    return flattened_features

# -------------------------------
# Train Folder
train_folder_path = 'Small Dataset/'
# -------------------------------

# Find maximum frame size in the training data
print("Finding maximum frames in the training dataset...")
max_frames_train = find_max_frames(train_folder_path)

# Process BVH files using the maximum frame size
print("Processing training data...")
all_features_train = process_bvh_files(train_folder_path, max_frames_train)

# Extract motion features for each processed BVH file
print("Extracting features from processed data...")
X_features = np.array([extract_motion_features(interpolate_frames(motion_data, max_frames_train)) for motion_data in all_features_train])

# Train Labels
print("Assigning labels to the dataset...")
C_Angry = [0] * 42
C_Depressed = [1] * 42
C_Neutral = [2] * 42
C_Proud = [3] * 32
Labels = np.array(C_Angry + C_Depressed + C_Neutral + C_Proud, dtype=np.int32)
print("Labels assigned.")

# Standardize the data
print("Standardizing the data...")
scaler = StandardScaler()
X_standardized = scaler.fit_transform(all_features_train.reshape(len(all_features_train), -1))  # Flatten for standardization
print("Data standardized.")

# Normalize the data (min-max scaling)
print("Normalizing the data...")
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X_standardized)
print("Data normalized.")

# Handle NaN values
print("Handling NaN values...")
imputer = SimpleImputer(strategy='mean')
X_cleaned = imputer.fit_transform(X_normalized)
print("NaN values handled.")

# Final feature set and labels
X = X_cleaned
Y = Labels


# -------------------------------------------
# Perform feature selection-------------------------------------------
k = 1000  # Number of top features 
selector = SelectKBest(chi2, k=k)
X = selector.fit_transform(X, Y)
# Display the indices of the selected features
selected_indices = selector.get_support(indices=True)
# -------------------------------------------


# X_features contains the flattened extracted features
print(f"Feature extraction completed. X_features shape: {X_features.shape}")

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# XGBoost Classifier
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
print("Training the XGBoost classifier...")
xgb_classifier.fit(X_train, Y_train)
print("Training completed.")

# Predict probabilities and labels
print("Making predictions...")
y_proba_XGB = xgb_classifier.predict_proba(X_test)
y_pred_XGB = xgb_classifier.predict(X_test)

# Confusion Matrix and Basic Elements: TP, FP, FN, TN
# Confusion Matrix and metrics for multiclass problem
cm = confusion_matrix(Y_test, y_pred_XGB)
print(f"Confusion Matrix:\n{cm}")

# For multiclass, we'll sum the diagonal for True Positives (TP), and calculate FP, FN, and TN
TP = np.diag(cm)  # True Positives are the diagonal elements of the confusion matrix
FP = cm.sum(axis=0) - TP  # False Positives are column-wise sums minus the True Positives
FN = cm.sum(axis=1) - TP  # False Negatives are row-wise sums minus the True Positives
TN = cm.sum() - (FP + FN + TP)  # True Negatives are the total sum minus FP, FN, and TP
tn=TN
fp=FP
# Sum them up to get overall values if needed
TP_sum = TP.sum()
FP_sum = FP.sum()
FN_sum = FN.sum()
TN_sum = TN.sum()

print(f"True Positives (TP): {TP_sum}, False Positives (FP): {FP_sum}, False Negatives (FN): {FN_sum}, True Negatives (TN): {TN_sum}")

# Metrics Calculation
print("Calculating metrics...")

# 1. ROC-AUC (for Multiclass)
roc_auc_XGB = roc_auc_score(Y_test, y_proba_XGB, multi_class='ovr')
print(f"XGBoost ROC-AUC: {roc_auc_XGB:.4f}")

# 2. Precision-Recall AUC
pr_auc_XGB = average_precision_score(Y_test, y_proba_XGB, average='macro')
print(f"XGBoost Precision-Recall AUC: {pr_auc_XGB:.4f}")

# 3. Log Loss (Cross-Entropy Loss)
log_loss_XGB = log_loss(Y_test, y_proba_XGB)
print(f"XGBoost Log Loss: {log_loss_XGB:.4f}")

# 4. Cohen's Kappa
cohen_kappa_XGB = cohen_kappa_score(Y_test, y_pred_XGB)
print(f"XGBoost Cohen's Kappa: {cohen_kappa_XGB:.4f}")

# 5. Matthews Correlation Coefficient (MCC)
mcc_XGB = matthews_corrcoef(Y_test, y_pred_XGB)
print(f"XGBoost MCC: {mcc_XGB:.4f}")

# 6. Precision
precision_XGB = precision_score(Y_test, y_pred_XGB, average='macro')
print(f"XGBoost Precision: {precision_XGB:.4f}")

# 7. Recall (Sensitivity)
recall_XGB = recall_score(Y_test, y_pred_XGB, average='macro')
print(f"XGBoost Recall (Sensitivity): {recall_XGB:.4f}")

# 8. Unweighted Average Recall (UAR)
uar_XGB = recall_score(Y_test, y_pred_XGB, average='macro')
print(f"XGBoost Unweighted Average Recall (UAR): {uar_XGB:.4f}")

# 9. Specificity (True Negative Rate)
specificity_XGB = tn / (tn + fp)
# Calculate and print average specificity
avg_specificity_XGB = np.mean(specificity_XGB)
print(f"XGBoost Average Specificity (True Negative Rate): {avg_specificity_XGB:.4f}")

# 10. F1-Score
f1_XGB = f1_score(Y_test, y_pred_XGB, average='macro')
print(f"XGBoost F1-Score: {f1_XGB:.4f}")

# 11. Accuracy
accuracy_XGB = accuracy_score(Y_test, y_pred_XGB)
print(f"XGBoost Accuracy: {accuracy_XGB:.4f}")

# 12. Balanced Accuracy
balanced_accuracy_XGB = balanced_accuracy_score(Y_test, y_pred_XGB)
print(f"XGBoost Balanced Accuracy: {balanced_accuracy_XGB:.4f}")

# 13. Hamming Loss
hamming_loss_XGB = hamming_loss(Y_test, y_pred_XGB)
print(f"XGBoost Hamming Loss: {hamming_loss_XGB:.4f}")

# 14. Jaccard Index (Intersection over Union)
jaccard_XGB = jaccard_score(Y_test, y_pred_XGB, average='macro')
print(f"XGBoost Jaccard Index: {jaccard_XGB:.4f}")

# 15. Classification Report
class_report_XGB = classification_report(Y_test, y_pred_XGB)
print(f"Classification Report:\n{class_report_XGB}")

# Print completion message
end_time = time.time()
print(f"Pre-processing, training, and metrics calculation completed in {end_time - start_time:.2f} seconds.")
