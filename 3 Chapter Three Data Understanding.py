%reset -f
#----------------------------------------------------
# Chapter Three
#----------------------------------------------------
# 3. Data Understanding
# Feature Distribution Over Classes
# Feature Correlation
# Feature Correlation by Regression
# Feature Importance
# SHAP
# LIME
# Class Distribution Ratio
# PCA and t-SNE Feature Plots
#----------------------------------------------------
import numpy as np
import pandas as pd
import os
import warnings
import time
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import lime
import lime.lime_tabular
import shap

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

# X_features contains the flattened extracted features
print(f"Feature extraction completed. X_features shape: {X_features.shape}")

Xtemp = X

#------------------------------------------------------------------
# Set the number of features to select (this can be adjusted as needed)
num_features_to_select = 10  # Example number, adjust based on dataset
#------------------------------------------------------------------

# Standardize the data before applying NCA
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X)
# Perform Neighborhood Components Analysis (NCA)
print("Performing NCA feature selection...")
nca = NeighborhoodComponentsAnalysis(n_components=num_features_to_select, random_state=42)
X = nca.fit_transform(X_features_scaled, Y)
print(f"Selected {num_features_to_select} features using NCA.")


#-----------------------------------------------------------------------------------
# Data Understanding and Features
#-----------------------------------------------------------------------------------
# Feature distribution over classes
# Define the feature indices to visualize (for example, feature indices 1, 5, and 10)
feature_indices_to_plot = [5, 7, 2]
# Select the desired features for visualization using their indices
X_selected_for_plot = X[:, feature_indices_to_plot]
# Number of features to plot
num_features_to_plot = len(feature_indices_to_plot)
# Create subplots for each feature
fig, axes = plt.subplots(num_features_to_plot, 1, figsize=(10, 20), sharex=True)
# Loop through the selected features and plot distribution for each class
for idx, feature_index in enumerate(feature_indices_to_plot):
    feature_values = X_selected_for_plot[:, idx]
    for class_label in np.unique(Y):
        sns.kdeplot(feature_values[Y == class_label], ax=axes[idx], label=f'Class {class_label}', fill=True)
    axes[idx].set_title(f'Feature {feature_index} Distribution Over Classes')
    axes[idx].set_xlabel('Feature Value')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()

plt.tight_layout()
plt.show()

#--------------------------------
# Feature Correlation
# Define the number of features to plot (select a subset of features)
desired_features = 8  # Change this to the number of features to visualize
num_features = min(desired_features, X.shape[1])
# Select the subset of the correlation matrix for the desired number of features
correlation_matrix_subset = np.corrcoef(X[:, :num_features], rowvar=False)
# Plot the correlation matrix for the selected number of features
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_subset, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
plt.title(f"Feature Correlation Matrix for {num_features} Features")
plt.show()

#Feature Correlation by Regression
# Convert X to a DataFrame for easier manipulation
Xtemp_df = pd.DataFrame(X)
# Define the specific features to plot a regression between (feature 0 and feature 4 in this case)
feature_1_index = 7
feature_2_index = 8
# Extract the data for the two features
x = Xtemp_df.iloc[:, feature_1_index].values.reshape(-1, 1)  # Reshape for the model
y = Xtemp_df.iloc[:, feature_2_index].values
# Compute correlation coefficient
correlation_value = np.corrcoef(x[:, 0], y)[0, 1]
# Set up the plot
plt.figure(figsize=(15, 5))
# Linear regression plot
plt.subplot(1, 2, 1)
sns.regplot(x=x[:, 0], y=y, scatter_kws={"s": 10}, line_kws={"color": "red"})
plt.title(f"Linear Regression (Corr: {correlation_value:.2f})")
plt.xlabel(f"Feature {feature_1_index + 1}")
plt.ylabel(f"Feature {feature_2_index + 1}")
# Polynomial (non-linear) regression plot
plt.subplot(1, 2, 2)
# Fit a 2nd degree polynomial
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)
poly_reg_model = LinearRegression()
poly_reg_model.fit(x_poly, y)
# Predict the polynomial regression line
y_poly_pred = poly_reg_model.predict(x_poly)
# Plot the non-linear regression with a smooth curve
plt.scatter(x, y, s=10, label="Data Points")
plt.plot(np.sort(x[:, 0]), y_poly_pred[np.argsort(x[:, 0])], color='red', label="Polynomial Fit (Degree 3)")
plt.title(f"Polynomial Regression (Corr: {correlation_value:.2f})")
plt.xlabel(f"Feature {feature_1_index + 1}")
plt.ylabel(f"Feature {feature_2_index + 1}")
plt.legend()
plt.tight_layout()
plt.show()


# RandomForest model (use RandomForestClassifier for classification tasks)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the data
model.fit(X, Y)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for easier handling and sorting
feature_importance_df = pd.DataFrame({
    'Feature': np.arange(X.shape[1]),
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importances
print("Feature Importance:\n", feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='b', align='center')
plt.gca().invert_yaxis()  # To display the most important at the top
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


#-----------------------------------------------
#SHAP 

# RandomForest model (use RandomForestClassifier for classification tasks)
model = RandomForestRegressor(n_estimators=100, random_state=42)
# Fit the model on the data
model.fit(X, Y)
# SHAP explainer
explainer = shap.TreeExplainer(model)
# Calculate SHAP values (Shapley values) for all data points
shap_values = explainer.shap_values(X)
# Print SHAP values for the first instance (for interpretation)
print("SHAP values for the first instance:\n", shap_values[0])
# Plot SHAP summary plot for feature importance
shap.summary_plot(shap_values, X, plot_type="bar")
# the detailed SHAP summary plot (with feature distribution)
shap.summary_plot(shap_values, X)

#-------------------------------------------
# LIME

# RandomForest model (use RandomForestClassifier for classification tasks)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)
# LIME explainer for tabular data
explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=[f"Feature {i}" for i in range(X.shape[1])], 
                                                   verbose=True, mode='regression')  # Use mode='classification' if using a classifier
# Select a specific instance to explain
instance_to_explain = X[1]  # Replace 0 with the index of the instance to explain
# Generate LIME explanation for the selected instance
exp = explainer.explain_instance(instance_to_explain, model.predict)
# Print the feature importance for this instance
print("LIME explanation for the selected instance:\n")
exp.show_in_notebook(show_table=True)
# Plot the LIME explanation (bar chart)
exp.as_pyplot_figure()
plt.show()

#--------------------------------------
# Class Distribution Ratio

# Count the number of samples per class
class_counts = Counter(Y)
# Print the class distribution
print("Class Distribution (Number of samples per class):")
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} samples")
# Plot the class distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette='viridis')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.show()


# PCA Plot ---------------------------------------------

# Step 1: Augment the data to exactly 2000 samples
target_samples = 2000  # Total desired samples
classes, counts = np.unique(Y, return_counts=True)
# Adjust the target for each class depending on the current distribution
target_per_class = target_samples // len(classes)  # Divide 2000 equally across all classes
sampling_strategy = {cls: target_per_class for cls in classes}  # Adjust this if classes are not balanced
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_augmented, Y_augmented = ros.fit_resample(X, Y)
# Step 2: Standardize the features before PCA to ensure proper scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_augmented)
# Step 3: Apply PCA to reduce the dimensionality to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Check the number of samples after augmentation
print(f"Original number of samples: {Xtemp.shape[0]}")
print(f"Augmented number of samples: {X_augmented.shape[0]}")
# Step 4: Plot the augmented data with PCA in 2D, and make sure the points are visibly scattered
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=Y_augmented, palette='viridis', s=100, alpha=0.7)
# Add labels and title
plt.title('PCA of Augmented Features (2D Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Class")
plt.show()


# #t-SNE Plot ----------------------------------
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# # Step 1: Augment the data to exactly 2000 samples
# target_samples = 2000  # Total desired samples
# classes, counts = np.unique(Y, return_counts=True)
# # Adjust the target for each class depending on the current distribution
# target_per_class = target_samples // len(classes)  # Divide 2000 equally across all classes
# sampling_strategy = {cls: target_per_class for cls in classes}  # Adjust this if classes are not balanced
# ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
# X_augmented, Y_augmented = ros.fit_resample(X, Y)
# # Step 2: Standardize the features before t-SNE to ensure proper scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_augmented)
# # Step 3: Apply t-SNE to reduce the dimensionality to 2 components
# tsne = TSNE(n_components=2, random_state=42, perplexity=100, n_iter=1000)  # Adjust parameters as needed
# X_tsne = tsne.fit_transform(X_scaled)
# # Check the number of samples after augmentation
# print(f"Original number of samples: {Xtemp.shape[0]}")
# print(f"Augmented number of samples: {X_augmented.shape[0]}")
# # Step 4: Plot the augmented data with t-SNE in 2D, and make sure the points are visibly scattered
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=Y_augmented, palette='viridis', s=50, alpha=0.7)
# # Add labels and title
# plt.title('t-SNE of Augmented Features (2D Projection)')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.legend(title="Class")
# plt.show()