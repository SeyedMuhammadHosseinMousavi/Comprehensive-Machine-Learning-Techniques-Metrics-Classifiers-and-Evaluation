%reset -f
#----------------------------------------------------
# Chapter Two
#----------------------------------------------------
# 2.Mid-Level Processing
# Feature Extraction
# Data Augmentation
# Feature Selection
# Dimensionality Reduction
# Multi-Modal Fusion
#----------------------------------------------------

# Standard libraries
import numpy as np
import pandas as pd
import os
import warnings
import time

# Scipy for interpolation
from scipy.interpolate import interp1d

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, VarianceThreshold
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# XGBoost
from xgboost import XGBClassifier

# TensorFlow/Keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim


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
    channels_per_joint = 3  # Assuming 3 channels per joint (e.g., X, Y, Z rotations)
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

# X_features contains the flattened extracted features
print(f"Feature extraction completed. X_features shape: {X_features.shape}")

# Print completion message
end_time = time.time()
print(f"Pre-processing and feature extraction completed in {end_time - start_time:.2f} seconds.")


#------------------------------------------------------------------
# Data Augmentation
#------------------------------------------------------------------

# Augmentation function: Applies basic operations such as noise, scaling, and flipping
def augment_data(X_data, num_augmentations=3):
    augmented_data = []
    for i in range(num_augmentations):
        for sample in X_data:
            # Apply augmentation operations
            noise = np.random.normal(0, 0.01, sample.shape)  # Adding random noise
            scaled_sample = sample * np.random.uniform(0.9, 1.1)  # Scaling the data
            flipped_sample = np.flip(sample)  # Flipping the sample (reverse order)
            
            # Choose one augmentation per sample
            augmented_sample = sample + noise  # Example: adding noise
            augmented_data.append(augmented_sample)

    return np.array(augmented_data)

# Amount of desired augmented samples (multiplication)
num_augmentations = 2  # Adjust this value for more or fewer augmentations

# Augment extracted features (X_features)
X_features_augmented = augment_data(X_features, num_augmentations)

# Augment final cleaned data (X)
X_augmented = augment_data(X, num_augmentations)

# Append augmented samples to the original data
X_features = np.vstack([X_features, X_features_augmented])
X = np.vstack([X, X_augmented])

# Adjust labels for augmented data (duplicating the original labels)
Y_augmented = np.tile(Y, num_augmentations)
Y = np.concatenate([Y, Y_augmented])

print(f"Augmentation complete. New dataset size: {X.shape[0]} samples, {X_features.shape[0]} augmented features.")


#------------------------------------------------------------------
# Feature Selection
#------------------------------------------------------------------

#------------------------------------------------------------------
# Set the number of features to select (this can be adjusted as needed)
num_features_to_select = 10  # Example number, adjust based on dataset
#------------------------------------------------------------------

# Min-Max scaling to ensure non-negative values (range [0, 1])
print("Scaling data with Min-Max Scaler for Chi-squared test...")
min_max_scaler = MinMaxScaler()
X_features_scaled = min_max_scaler.fit_transform(X_features)
# Perform Chi-squared feature selection
print("Performing Chi-squared feature selection...")
chi_selector = SelectKBest(chi2, k=num_features_to_select)
X_features_chi_selected = chi_selector.fit_transform(X_features_scaled, Y)
print(f"Selected {num_features_to_select} features using Chi-squared test.")

# # Mutual Information
# print("Performing Mutual Information feature selection...")
# mi_selector = SelectKBest(mutual_info_classif, k=num_features_to_select)
# X_features_mi_selected = mi_selector.fit_transform(X_features, Y)
# print(f"Selected {num_features_to_select} features using Mutual Information.")

# # ANOVA F-value
# print("Performing ANOVA F-value feature selection...")
# anova_selector = SelectKBest(f_classif, k=num_features_to_select)
# X_features_anova_selected = anova_selector.fit_transform(X_features, Y)
# print(f"Selected {num_features_to_select} features using ANOVA F-value.")

# Lasso-based feature selection (L1 regularization)
print("Performing Lasso-based feature selection...")
lasso = Lasso(alpha=0.01)  # Alpha controls regularization strength, can be tuned
lasso.fit(X_features, Y)
X_features_lasso_selected = lasso.coef_ != 0  # Select features with non-zero coefficients
print(f"Selected features using Lasso: {sum(X_features_lasso_selected)} features selected.")

# # Recursive Feature Elimination (RFE) with SVM as estimator
# print("Performing RFE feature selection...")
# svc = SVC(kernel="linear")
# rfe = RFE(estimator=svc, n_features_to_select=num_features_to_select)
# X_features_rfe_selected = rfe.fit_transform(X_features, Y)
# print(f"Selected {num_features_to_select} features using RFE.")


# Standardize the data before applying NCA
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)
# Perform Neighborhood Components Analysis (NCA)
print("Performing NCA feature selection...")
nca = NeighborhoodComponentsAnalysis(n_components=num_features_to_select, random_state=42)
X_features_nca_selected = nca.fit_transform(X_features_scaled, Y)
print(f"Selected {num_features_to_select} features using NCA.")


# Apply variance thresholding (e.g., remove features with variance below 0.1)
print("Applying Variance Thresholding...")
variance_selector = VarianceThreshold(threshold=0.1)
X_features_variance_selected = variance_selector.fit_transform(X_features)
print(f"Selected {X_features_variance_selected.shape[1]} features using Variance Thresholding.")


# Train a Random Forest classifier
print("Training Random Forest for feature importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_features, Y)
# Get feature importances and select top features
importances = rf.feature_importances_
indices = np.argsort(importances)[-num_features_to_select:]
X_features_rf_selected = X_features[:, indices]
print(f"Selected {num_features_to_select} features using Random Forest.")


# from sklearn.ensemble import GradientBoostingClassifier
# # Train a Gradient Boosting classifier
# print("Training Gradient Boosting for feature importance...")
# gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
# gbc.fit(X_features, Y)
# # Get feature importances and select top features
# importances = gbc.feature_importances_
# indices = np.argsort(importances)[-num_features_to_select:]
# X_features_gbc_selected = X_features[:, indices]
# print(f"Selected {num_features_to_select} features using Gradient Boosting.")


#------------------------------------------------------------------
# Dimensionality Reduction
#------------------------------------------------------------------

# Apply PCA for dimensionality reduction
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=num_features_to_select)  # Choose number of components to keep
X_features_pca = pca.fit_transform(X_features)
print(f"Reduced to {num_features_to_select} dimensions using PCA.")


# Apply t-SNE for dimensionality reduction (typically to 2 or 3 dimensions)
print("Applying t-SNE for dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42)
X_features_tsne = tsne.fit_transform(X_features)
print("Reduced to 2 dimensions using t-SNE.")


# Apply Truncated SVD for dimensionality reduction
print("Applying Truncated SVD for dimensionality reduction...")
svd = TruncatedSVD(n_components=num_features_to_select)
X_features_svd = svd.fit_transform(X_features)
print(f"Reduced to {num_features_to_select} dimensions using Truncated SVD.")


# Apply ICA for dimensionality reduction
print("Applying ICA for dimensionality reduction...")
ica = FastICA(n_components=num_features_to_select, random_state=42)
X_features_ica = ica.fit_transform(X_features)
print(f"Reduced to {num_features_to_select} dimensions using ICA.")


# Apply Isomap for dimensionality reduction
print("Applying Isomap for dimensionality reduction...")
isomap = Isomap(n_components=num_features_to_select)
X_features_isomap = isomap.fit_transform(X_features)
print(f"Reduced to {num_features_to_select} dimensions using Isomap.")


# Number of classes in the dataset
num_classes = len(np.unique(Y))
# Set n_components to be the minimum of num_classes - 1
n_components_lda = min(num_features_to_select, num_classes - 1)
# Apply LDA for dimensionality reduction
print("Applying LDA for dimensionality reduction...")
lda = LDA(n_components=n_components_lda)
X_features_lda = lda.fit_transform(X_features, Y)
print(f"Reduced to {n_components_lda} dimensions using LDA.")

# Autoencoder dimensionality reduction
# Step 1: Normalize the features before feeding into the Autoencoder
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)
# Step 2: Define the dimensions for the Autoencoder
input_dim = X_features_scaled.shape[1]  # Number of original features
encoding_dim = num_features_to_select  # Desired dimensionality (the size of the bottleneck)
# Step 3: Build the Autoencoder model
input_layer = Input(shape=(input_dim,))
# Encoder layers (compressing)
encoded = Dense(encoding_dim, activation='relu')(input_layer)
# Decoder layers (reconstructing)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
# Step 4: Construct the Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)
# Step 5: Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
# Step 6: Train the Autoencoder
autoencoder.fit(X_features_scaled, X_features_scaled, 
                epochs=10, 
                batch_size=32, 
                shuffle=True, 
                validation_split=0.2, 
                verbose=1)
# Step 7: Extract the encoder part to get reduced representation
encoder = Model(inputs=input_layer, outputs=encoded)
X_features_autoencoder = encoder.predict(X_features_scaled)
print(f"Reduced to {encoding_dim} dimensions using Autoencoder.")


#------------------------------------------------------------------
# Multi-Modal Fusion
#------------------------------------------------------------------

# Concatenation Early Stage KNN Imputation Fusion

# Step 1: Load the GSR data from the CSV file
gsr_data = pd.read_csv('GSR.csv')
# Step 2: Separate labels (Quad_Cat) and GSR features
gsr_labels = gsr_data['Quad_Cat'].values  # Extract the label column
gsr_features = gsr_data.drop(columns=['Quad_Cat']).values  # Extract the GSR features (drop the label column)
# Step 3: Check the shape of both datasets
print(f"GSR features shape: {gsr_features.shape}")
print(f"Chi-squared extracted features shape: {X_features_chi_selected.shape}")
# Step 4: Determine the maximum number of rows and columns between the two datasets
max_rows = max(len(X_features_chi_selected), len(gsr_features))
max_cols = max(X_features_chi_selected.shape[1], gsr_features.shape[1])
# Step 5: Pad both datasets to ensure they have the same number of rows and columns by adding NaNs
gsr_features_padded = np.pad(gsr_features, 
                             ((0, max_rows - len(gsr_features)), (0, max_cols - gsr_features.shape[1])),
                             mode='constant', constant_values=np.nan)

X_features_chi_padded = np.pad(X_features_chi_selected, 
                               ((0, max_rows - len(X_features_chi_selected)), (0, max_cols - X_features_chi_selected.shape[1])),
                               mode='constant', constant_values=np.nan)
# Step 6: Concatenate the padded GSR and Chi-squared features
combined_features = np.hstack([X_features_chi_padded, gsr_features_padded])
# Step 7: Apply KNN Imputation to handle the NaN values in the combined dataset
print("Applying KNN Imputation...")
knn_imputer = KNNImputer(n_neighbors=5)  # Adjust n_neighbors as needed
combined_features_imputed = pd.DataFrame(knn_imputer.fit_transform(combined_features))
# Step 8: Standardize the imputed features
scaler = StandardScaler()
combined_features_standardized = pd.DataFrame(scaler.fit_transform(combined_features_imputed))
# Step 9: Ensure that labels are aligned (GSR labels apply across both modalities)
gsr_labels = np.pad(gsr_labels, (0, max_rows - len(gsr_labels)), 'edge')  # Forward fill missing labels if needed
# Now the fused features are stored in `combined_features_standardized` and the labels in `gsr_labels`
print(f"Fusion completed. Fused dataset shape: {combined_features_standardized.shape}, Labels shape: {gsr_labels.shape}")




# PCA Early Stage KNN Imputation Fusion------------------------------------------

# Step 1: Load the GSR data from the CSV file
gsr_data = pd.read_csv('GSR.csv')
# Step 2: Separate labels (Quad_Cat) and GSR features
gsr_labels = gsr_data['Quad_Cat'].values  # Extract the label column
gsr_features = gsr_data.drop(columns=['Quad_Cat']).values  # Extract the GSR features (drop the label column)
# Step 3: Check the shape of both datasets
print(f"GSR features shape: {gsr_features.shape}")
print(f"Chi-squared extracted features shape: {X_features_chi_selected.shape}")
# Step 4: Determine the maximum number of rows and columns between the two datasets
max_rows = max(len(X_features_chi_selected), len(gsr_features))
max_cols = max(X_features_chi_selected.shape[1], gsr_features.shape[1])
# Step 5: Pad both datasets to ensure they have the same number of rows and columns by adding NaNs
gsr_features_padded = np.pad(gsr_features, 
                             ((0, max_rows - len(gsr_features)), (0, max_cols - gsr_features.shape[1])),
                             mode='constant', constant_values=np.nan)
X_features_chi_padded = np.pad(X_features_chi_selected, 
                               ((0, max_rows - len(X_features_chi_selected)), (0, max_cols - X_features_chi_selected.shape[1])),
                               mode='constant', constant_values=np.nan)
# Step 6: Concatenate the padded GSR and Chi-squared features
combined_features = np.hstack([X_features_chi_padded, gsr_features_padded])
# Step 7: Apply KNN Imputation to handle the NaN values in the combined dataset
print("Applying KNN Imputation...")
knn_imputer = KNNImputer(n_neighbors=5)  # Adjust n_neighbors as needed
combined_features_imputed = pd.DataFrame(knn_imputer.fit_transform(combined_features))
# Step 8: Standardize the imputed features
scaler = StandardScaler()
combined_features_standardized = pd.DataFrame(scaler.fit_transform(combined_features_imputed))
# Step 9: Apply PCA for dimensionality reduction
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)  # Retain 95% of variance (adjust as needed)
X_fused_pca = pca.fit_transform(combined_features_standardized)
# Step 10: Ensure that labels are aligned (assuming GSR labels apply across both modalities)
gsr_labels = np.pad(gsr_labels, (0, max_rows - len(gsr_labels)), 'edge')  # Forward fill missing labels if needed
# Now the PCA-reduced fused features are stored in `X_fused_pca` and the labels in `gsr_labels`
print(f"PCA-based fusion completed. Fused dataset shape: {X_fused_pca.shape}, Labels shape: {gsr_labels.shape}")


# Weighted Averaging Late Stage Fusion------------------------------------------

# Step 1: Load GSR data and Chi-squared extracted features (X_features_chi_selected)
print("Loading GSR data...")
gsr_data = pd.read_csv('GSR.csv')
gsr_labels = gsr_data['Quad_Cat'].values  # Assuming 'Quad_Cat' column has the labels
gsr_features = gsr_data.drop(columns=['Quad_Cat']).values  # Drop the label column
# Step 2: Align the number of rows between GSR and Chi-squared features
max_rows = max(len(gsr_features), len(X_features_chi_selected))
# Pad both datasets to have the same number of rows by adding NaNs
gsr_features_padded = np.pad(gsr_features, ((0, max_rows - len(gsr_features)), (0, 0)), mode='constant', constant_values=np.nan)
chi_features_padded = np.pad(X_features_chi_selected, ((0, max_rows - len(X_features_chi_selected)), (0, 0)), mode='constant', constant_values=np.nan)
# Step 3: Use KNN Imputation to fill missing values in both modalities
print("Applying KNN Imputation...")
imputer = KNNImputer(n_neighbors=5)
gsr_features_imputed = imputer.fit_transform(gsr_features_padded)
chi_features_imputed = imputer.fit_transform(chi_features_padded)
# Step 4: Standardize both datasets
scaler = StandardScaler()
gsr_features_standardized = scaler.fit_transform(gsr_features_imputed)
chi_features_standardized = scaler.fit_transform(chi_features_imputed)
# Step 5: Ensure labels are correctly aligned with the imputed data
gsr_labels = np.pad(gsr_labels, (0, max_rows - len(gsr_labels)), mode='edge')
# Step 6: Train-test split
X_train_gsr, X_test_gsr, y_train, y_test = train_test_split(gsr_features_standardized, gsr_labels, test_size=0.2, random_state=42)
X_train_chi, X_test_chi, _, _ = train_test_split(chi_features_standardized, gsr_labels, test_size=0.2, random_state=42)
# Step 7: Train separate models for each modality
# GSR model using Random Forest
model_gsr = RandomForestClassifier(n_estimators=100, random_state=42)
model_gsr.fit(X_train_gsr, y_train)
# Chi-squared model using Logistic Regression
model_chi = LogisticRegression(max_iter=1000, random_state=42)
model_chi.fit(X_train_chi, y_train)
# Step 8: Get the predictions (probabilities) for both modalities
gsr_pred_train = model_gsr.predict_proba(X_train_gsr)
gsr_pred_test = model_gsr.predict_proba(X_test_gsr)
chi_pred_train = model_chi.predict_proba(X_train_chi)
chi_pred_test = model_chi.predict_proba(X_test_chi)
# Step 9: Apply weighted averaging to the predictions (adjust weights as needed)
weight_gsr = 0.6
weight_chi = 0.4
train_weighted_avg = (weight_gsr * gsr_pred_train) + (weight_chi * chi_pred_train)
test_weighted_avg = (weight_gsr * gsr_pred_test) + (weight_chi * chi_pred_test)
# Step 10: Convert weighted average predictions to class labels
train_pred_classes = np.argmax(train_weighted_avg, axis=1)
test_pred_classes = np.argmax(test_weighted_avg, axis=1)



# Majority Voting Late Stage Fusion------------------------------------------

# Step 1: Load GSR data and Chi-squared extracted features (X_features_chi_selected)
print("Loading GSR data...")
gsr_data = pd.read_csv('GSR.csv')
gsr_labels = gsr_data['Quad_Cat'].values  # Assuming 'Quad_Cat' column has the labels
gsr_features = gsr_data.drop(columns=['Quad_Cat']).values  # Drop the label column
# Step 2: Align the number of rows between GSR and Chi-squared features
max_rows = max(len(gsr_features), len(X_features_chi_selected))
# Pad both datasets to have the same number of rows by adding NaNs
gsr_features_padded = np.pad(gsr_features, ((0, max_rows - len(gsr_features)), (0, 0)), mode='constant', constant_values=np.nan)
chi_features_padded = np.pad(X_features_chi_selected, ((0, max_rows - len(X_features_chi_selected)), (0, 0)), mode='constant', constant_values=np.nan)
# Step 3: Use KNN Imputation to fill missing values in both modalities
print("Applying KNN Imputation...")
imputer = KNNImputer(n_neighbors=5)
gsr_features_imputed = imputer.fit_transform(gsr_features_padded)
chi_features_imputed = imputer.fit_transform(chi_features_padded)
# Step 4: Standardize both datasets
scaler = StandardScaler()
gsr_features_standardized = scaler.fit_transform(gsr_features_imputed)
chi_features_standardized = scaler.fit_transform(chi_features_imputed)
# Step 5: Ensure labels are correctly aligned with the imputed data
gsr_labels = np.pad(gsr_labels, (0, max_rows - len(gsr_labels)), mode='edge')
# Step 6: Train-test split
X_train_gsr, X_test_gsr, y_train, y_test = train_test_split(gsr_features_standardized, gsr_labels, test_size=0.2, random_state=42)
X_train_chi, X_test_chi, _, _ = train_test_split(chi_features_standardized, gsr_labels, test_size=0.2, random_state=42)
# Step 7: Train separate models for each modality
# GSR model using Random Forest
model_gsr = RandomForestClassifier(n_estimators=100, random_state=42)
model_gsr.fit(X_train_gsr, y_train)
# Chi-squared model using Logistic Regression
model_chi = LogisticRegression(max_iter=1000, random_state=42)
model_chi.fit(X_train_chi, y_train)
# Step 8: Create a Voting Classifier for Majority Voting
# Majority voting combines predictions from multiple models
voting_clf = VotingClassifier(
    estimators=[('gsr', model_gsr), ('chi', model_chi)],
    voting='hard'  # Use 'hard' for majority voting
)
# Train the Voting Classifier on the concatenated features
voting_clf.fit(np.concatenate([X_train_gsr, X_train_chi], axis=1), y_train)
# Step 9: Make predictions using the Voting Classifier
y_pred = voting_clf.predict(np.concatenate([X_test_gsr, X_test_chi], axis=1))



# PCA and LDA Hybrid Fusion------------------------------------------

# Load CSVs into DataFrames
print("Loading data...")
gsr_df = pd.read_csv('GSR.csv')
# 'Quad_Cat' is the label column for GSR and X_features_chi_selected is chi-squared features
labels_gsr = gsr_df['Quad_Cat'].fillna(method='ffill')  # Ensure there are no missing labels
features_gsr = gsr_df.drop(columns=['Quad_Cat'])
# Align the number of rows across modalities by padding with NaN
print("Aligning the number of rows across modalities...")
max_rows = max(len(features_gsr), len(X_features_chi_selected))
# Reindex to make sure both features and labels have the same number of rows
features_gsr = features_gsr.reindex(range(max_rows), fill_value=np.nan)
X_features_chi_selected = pd.DataFrame(X_features_chi_selected).reindex(range(max_rows), fill_value=np.nan)
# Reindex labels to ensure consistency with features
labels_gsr = labels_gsr.reindex(range(max_rows), fill_value=labels_gsr.mode()[0])
# Impute missing values for each modality
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
features_gsr_imputed = pd.DataFrame(imputer.fit_transform(features_gsr), columns=features_gsr.columns)
features_chi_imputed = pd.DataFrame(imputer.fit_transform(X_features_chi_selected), columns=X_features_chi_selected.columns)
# Standardize the features for each modality
print("Standardizing features...")
scaler = StandardScaler()
features_gsr_standardized = pd.DataFrame(scaler.fit_transform(features_gsr_imputed), columns=features_gsr_imputed.columns)
features_chi_standardized = pd.DataFrame(scaler.fit_transform(features_chi_imputed), columns=features_chi_imputed.columns)
# Apply PCA separately to each modality
print("Applying PCA to each modality...")
pca_gsr = PCA(n_components=5)
pca_chi = PCA(n_components=5)
reduced_gsr = pca_gsr.fit_transform(features_gsr_standardized)
reduced_chi = pca_chi.fit_transform(features_chi_standardized)
# Combine the reduced features from all modalities
combined_reduced_features = np.concatenate([reduced_gsr, reduced_chi], axis=1)
# Perform LDA for class separation on the combined PCA-reduced features
print("Applying LDA...")
lda = LDA(n_components=2)
lda_features = lda.fit_transform(combined_reduced_features, labels_gsr)
# Final fused features stored in a clear variable
final_fused_featuresPCALDA = lda_features



# Attention-Weighted XGBoost Hybrid Fusion------------------------------------------

# Load the CSV files into pandas DataFrames
print("Loading data...")
gsr_df = pd.read_csv('GSR.csv')
# 'Quad_Cat' is the label column for GSR and X_features_chi_selected is chi-squared features
labels_gsr = gsr_df['Quad_Cat'].fillna(method='ffill')
features_gsr = gsr_df.drop(columns=['Quad_Cat'])
# Align the number of rows across modalities by padding with NaN
max_rows = max(len(features_gsr), len(X_features_chi_selected))
# Reindex both GSR and Chi-squared features to have the same number of rows
features_gsr = features_gsr.reindex(range(max_rows), fill_value=np.nan)
X_features_chi_selected = pd.DataFrame(X_features_chi_selected).reindex(range(max_rows), fill_value=np.nan)
# Reindex labels to ensure consistency with features
labels_gsr = labels_gsr.reindex(range(max_rows), fill_value=labels_gsr.mode()[0])
# Impute missing values for each modality
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
features_gsr_imputed = pd.DataFrame(imputer.fit_transform(features_gsr), columns=features_gsr.columns)
features_chi_imputed = pd.DataFrame(imputer.fit_transform(X_features_chi_selected), columns=X_features_chi_selected.columns)
# Standardize the features
scaler = StandardScaler()
features_gsr_standardized = pd.DataFrame(scaler.fit_transform(features_gsr_imputed), columns=features_gsr_imputed.columns)
features_chi_standardized = pd.DataFrame(scaler.fit_transform(features_chi_imputed), columns=features_chi_imputed.columns)
# Attention mechanism for each modality
def attention_layer(inputs):
    attention_probs = Dense(inputs.shape[1], activation='softmax')(inputs)
    attention_mul = Multiply()([inputs, attention_probs])
    return attention_mul
# Apply attention on each modality
input_gsr = Input(shape=(features_gsr_standardized.shape[1],))
input_chi = Input(shape=(features_chi_standardized.shape[1],))
attention_gsr = attention_layer(input_gsr)
attention_chi = attention_layer(input_chi)
# Build and compile models to extract attention-weighted features
model_gsr = Model(inputs=input_gsr, outputs=attention_gsr)
model_chi = Model(inputs=input_chi, outputs=attention_chi)
# Extract attention-weighted features
attention_features_gsr = model_gsr.predict(features_gsr_standardized)
attention_features_chi = model_chi.predict(features_chi_standardized)
# Concatenate the attention-weighted features from both GSR and Chi-squared features
combined_attention_features = np.concatenate([attention_features_gsr, attention_features_chi], axis=1)
# Train-test split for the combined attention-weighted features
X_train, X_test, y_train, y_test = train_test_split(combined_attention_features, labels_gsr, test_size=0.2, random_state=42)
# Train an XGBoost classifier on the attention-weighted features
print("Training XGBoost classifier...")
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier.fit(X_train, y_train)
# Final fused features stored in a clear variable
final_fused_featuresAttention = combined_attention_features



# CNN and LSTM Hybrid Fusion------------------------------------------

# Load the CSV files into pandas DataFrames
print("Loading data...")
gsr_df = pd.read_csv('GSR.csv')
# 'Quad_Cat' is the label column for GSR and X_features_chi_selected is chi-squared features
labels_gsr = gsr_df['Quad_Cat'].fillna(method='ffill')
features_gsr = gsr_df.drop(columns=['Quad_Cat'])
# Align the number of rows across modalities by padding with NaN
max_rows = max(len(features_gsr), len(X_features_chi_selected))
# Reindex to make sure both features and labels have the same number of rows
features_gsr = features_gsr.reindex(range(max_rows), fill_value=np.nan)
X_features_chi_selected = pd.DataFrame(X_features_chi_selected).reindex(range(max_rows), fill_value=np.nan)
# Reindex labels to ensure consistency with features
labels_gsr = labels_gsr.reindex(range(max_rows), fill_value=labels_gsr.mode()[0])
# Impute missing values
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
features_gsr_imputed = pd.DataFrame(imputer.fit_transform(features_gsr), columns=features_gsr.columns)
features_chi_imputed = pd.DataFrame(imputer.fit_transform(X_features_chi_selected), columns=X_features_chi_selected.columns)
# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
features_gsr_standardized = pd.DataFrame(scaler.fit_transform(features_gsr_imputed), columns=features_gsr_imputed.columns)
features_chi_standardized = pd.DataFrame(scaler.fit_transform(features_chi_imputed), columns=features_chi_imputed.columns)
# Convert the data to numpy arrays
X_gsr = features_gsr_standardized.values
X_chi = features_chi_standardized.values
y = labels_gsr.values
# Reshape data for CNN (batch_size, channels, num_features)
X_gsr = X_gsr.reshape(X_gsr.shape[0], 1, X_gsr.shape[1])  # 1 channel for each modality
X_chi = X_chi.reshape(X_chi.shape[0], 1, X_chi.shape[1])
# Train-test split
X_gsr_train, X_gsr_test, X_chi_train, X_chi_test, y_train, y_test = train_test_split(
    X_gsr, X_chi, y, test_size=0.2, random_state=42
)
# Convert data to PyTorch tensors
X_gsr_train = torch.tensor(X_gsr_train, dtype=torch.float32)
X_gsr_test = torch.tensor(X_gsr_test, dtype=torch.float32)
X_chi_train = torch.tensor(X_chi_train, dtype=torch.float32)
X_chi_test = torch.tensor(X_chi_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
# Define the CNN + LSTM model
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        
        # CNN for each modality (1D convolution)
        self.cnn_gsr = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.cnn_chi = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # LSTM for each modality
        self.lstm_gsr = nn.LSTM(32, 64, batch_first=True)  # 32 input size from CNN
        self.lstm_chi = nn.LSTM(32, 64, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(64 * 2, 4)  # 2 modalities, 4 classes
        
    def forward(self, X_gsr, X_chi):
        # CNN for each modality
        X_gsr = self.cnn_gsr(X_gsr)
        X_chi = self.cnn_chi(X_chi)
        
        # LSTM for each modality
        _, (X_gsr, _) = self.lstm_gsr(X_gsr)
        _, (X_chi, _) = self.lstm_chi(X_chi)
        
        # Concatenate LSTM outputs from all modalities
        X = torch.cat((X_gsr[-1], X_chi[-1]), dim=1)
        
        # Fully connected layer for final classification
        out = self.fc(X)
        return out
# the model, loss function, and optimizer
model = CNNLSTM()
criterion = nn.CrossEntropyLoss()  # No need for class weights here
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_gsr_train, X_chi_train)
    loss = criterion(outputs, y_train)
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing the model
model.eval()
with torch.no_grad():
    y_pred = model(X_gsr_test, X_chi_test)
    _, predicted = torch.max(y_pred, 1)

# Save the final concatenated features (fused) into a clear variable
CNNLSTMHybrid = np.concatenate([X_gsr.reshape(X_gsr.shape[0], -1), 
                                X_chi.reshape(X_chi.shape[0], -1)], axis=1)
