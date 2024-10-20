%reset -f
#----------------------------------------------------
# Chapter Six
#----------------------------------------------------
# 6. Classifiers
# Naïve Bayes (NB)
# K-Nearest Neighborhood (KNN)
# Support Vector Machine (SVM)
# Logistic Regression (LR)
# Decision Tree (DT)
# Gradient Boosting
# eXtreme Gradient Boosting (XGBoost)
# Random Forest (RF)
# AdaBoost
#----------------------------------------------------

import numpy as np
import os
from bvh import Bvh
from scipy.interpolate import interp1d
import warnings
import time
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.tree as tr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# Record the start time
start_time = time.time()

# Suppress warnings
warnings.filterwarnings("ignore")

def read_bvh(filename):
    """Reads a BVH file and returns a Bvh object."""
    with open(filename) as f:
        mocap = Bvh(f.read())
    return mocap

def interpolate_frames(mocap, target_frame_count):
    """Interpolates BVH frames to match a target frame count."""
    original_frame_count = len(mocap.frames)
    original_time = np.linspace(0, 1, original_frame_count)
    target_time = np.linspace(0, 1, target_frame_count)
    interpolated_frames = []
    for frame in np.array(mocap.frames).T:
        interpolator = interp1d(original_time, frame.astype(float), kind='linear')
        interpolated_frame = interpolator(target_time)
        interpolated_frames.append(interpolated_frame)
    return np.array(interpolated_frames).T

def find_max_frames(folder_path):
    """Finds the maximum number of frames among all BVH files in a folder."""
    max_frames = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            mocap = read_bvh(os.path.join(folder_path, filename))
            max_frames = max(max_frames, len(mocap.frames))
    return max_frames


def extract_motion_features(mocap):
    motion_data = np.array(mocap.frames)
    channels_per_joint = 3
    num_joints = motion_data.shape[1] // channels_per_joint

    motion_features = {
        f'joint_{i}': {
            'rotations': [],
            # 'average_rotation': [],
            # 'angular_velocity': [],
            # 'acceleration': [],
            # 'speed': [],
            # 'jerk': [],
            # 'range_of_motion': [],
            # 'spatial_path': [],
            # 'harmonics': [],
            # 'frequency_analysis': []
        } for i in range(num_joints)
    }

    # Process each joint
    for i in range(num_joints):
        
        # Rotation
        joint_rotations = motion_data[:, i*channels_per_joint:(i+1)*channels_per_joint]
        motion_features[f'joint_{i}']['rotations'] = joint_rotations.tolist()

        # # Average rotation
        # average_rotation = np.mean(joint_rotations, axis=0)
        # motion_features[f'joint_{i}']['average_rotation'] = average_rotation.tolist()
        
        # # Angular velocity  
        # angular_velocity = np.diff(joint_rotations, axis=0, prepend=joint_rotations[:1])
        # motion_features[f'joint_{i}']['angular_velocity'] = angular_velocity.tolist()

        # # Acceleration
        # acceleration = np.diff(angular_velocity, axis=0, prepend=angular_velocity[:1])
        # motion_features[f'joint_{i}']['acceleration'] = acceleration.tolist()
        
        # # Speed (magnitude of angular velocity)
        # speed = np.linalg.norm(angular_velocity, axis=1)
        # motion_features[f'joint_{i}']['speed'] = speed.tolist()

        # # Jerk: derivative of speed
        # jerk = np.diff(speed, prepend=[speed[0]])  # Ensuring same length by prepending the first element
        # motion_features[f'joint_{i}']['jerk'] = jerk.tolist()

        # # Range of Motion
        # range_of_motion = np.max(joint_rotations, axis=0) - np.min(joint_rotations, axis=0)
        # motion_features[f'joint_{i}']['range_of_motion'] = range_of_motion.tolist()

        # # Spatial path: sum of absolute angular changes
        # angular_changes = np.abs(np.diff(joint_rotations, axis=0))
        # spatial_path = np.sum(angular_changes)
        # motion_features[f'joint_{i}']['spatial_path'] = spatial_path

        # # Harmonics using real FFT for reduced spectral leakage and efficiency
        # window = np.hanning(len(joint_rotations))
        # windowed_rotations = joint_rotations * window[:, np.newaxis]  # Apply window along time axis
        # harmonics = np.fft.rfft(windowed_rotations, axis=0)
        # motion_features[f'joint_{i}']['harmonics'] = np.abs(harmonics).tolist()

        # # Frequency Analysis: FFT magnitude
        # fs = 100.0  # Example: 100 Hz
        # # Process each joint
        # for i in range(num_joints):
        #     joint_rotations = motion_data[:, i*channels_per_joint:(i+1)*channels_per_joint]
        #     motion_features[f'joint_{i}']['rotations'] = joint_rotations.tolist()
        #     # Apply a window function to reduce spectral leakage
        #     window = np.hanning(len(joint_rotations))
        #     windowed_rotations = joint_rotations * window[:, np.newaxis]
        #     # Frequency analysis with real FFT and frequency resolution
        #     harmonics = np.fft.rfft(windowed_rotations, axis=0)
        #     magnitude_spectrum = np.abs(harmonics)
        #     motion_features[f'joint_{i}']['harmonics'] = magnitude_spectrum.tolist()
        #     # Frequency bins calculation
        #     freqs = np.fft.rfftfreq(n=len(joint_rotations), d=1/fs)
        #     motion_features[f'joint_{i}']['frequency_analysis'] = magnitude_spectrum.tolist()
        #     # Identify dominant frequencies
        #     dominant_indices = np.argmax(magnitude_spectrum, axis=0)
        #     dominant_frequencies = freqs[dominant_indices]
        #     motion_features[f'joint_{i}']['dominant_frequencies'] = dominant_frequencies.tolist()

    return motion_features


def process_bvh_files(folder_path, max_frames):
    """Processes each BVH file in the folder after interpolating to the same number of frames."""
    all_features = {}
    processed_files_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            print(f"Processing file: {filename}")
            full_path = os.path.join(folder_path, filename)
            mocap = read_bvh(full_path)
            interpolated_frames = interpolate_frames(mocap, max_frames)
            mocap.frames = interpolated_frames
            # extract_motion_features function is defined elsewhere
            motion_features = extract_motion_features(mocap)
            all_features[filename] = motion_features
            processed_files_count += 1
            print(f"Processed {processed_files_count} files.")
    return all_features

def print_class_metrics(y_true, y_pred, class_labels):
    print("Accuracy per class:")
    class_correct = np.zeros(len(class_labels), dtype=int)
    class_incorrect = np.zeros(len(class_labels), dtype=int)
    for i, label in enumerate(class_labels):
        correct = np.sum((y_true == label) & (y_pred == label))
        incorrect = np.sum((y_true == label) & (y_pred != label))
        class_correct[i] = correct
        class_incorrect[i] = incorrect
        print(f"Class {label}: Correctly Classified = {correct}, Incorrectly Classified = {incorrect}, Accuracy = {correct / (correct + incorrect) if correct + incorrect > 0 else 0:.2f}")
    total_correct = np.sum(class_correct)
    total_incorrect = np.sum(class_incorrect)
    print(f"Overall: Correctly Classified = {total_correct}, Incorrectly Classified = {total_incorrect}, Total Accuracy = {total_correct / (total_correct + total_incorrect):.2f}")


# Train Folder-----------------------------------------------------------------
train_folder_path = 'Small Dataset/'
# -----------------------------------------------------------------------------

# Find maximum frame size in the training data
max_frames_train = find_max_frames(train_folder_path)

# Process training and test BVH files using the training max frame size
all_features_train = process_bvh_files(train_folder_path, max_frames_train)

def recursive_flatten(input_item):
    """Recursively flattens nested lists or lists of lists into a flat list."""
    if isinstance(input_item, dict):
        return [sub_item for value in input_item.values() for sub_item in recursive_flatten(value)]
    elif isinstance(input_item, list):
        return [element for item in input_item for element in recursive_flatten(item)]
    else:
        return [input_item]

# an empty list to hold all flattened features from all samples
all_samples_flattened_features = []

# Iterate over all_features to flatten each sample's features and stack them
for filename, features in all_features_train.items():
    flattened_features = recursive_flatten(features)
    all_samples_flattened_features.append(flattened_features)

# Now, all_samples_flattened_features is a list where each item is the flattened feature list of a sample
# Convert the list of lists into a NumPy array for numerical processing
all_samples_flattened_features_array = np.array(all_samples_flattened_features, dtype=object)
array_of_lists = [list(row) for row in all_samples_flattened_features_array]
array_of_float64 = np.array(array_of_lists, dtype='float64')
flattened_data = array_of_float64

# Train Labels
C_Angry = [0] * 42
C_Depressed = [1] * 42
C_Neutral = [2] * 42
C_Proud = [3] * 32

# Concatenate the two lists
Labels = C_Angry + C_Depressed + C_Neutral + C_Proud
Labels_int32= np.array(Labels, dtype=np.int32)

X=flattened_data
Y=Labels_int32


# Standardize the data --------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
stacked_data = np.array(X)
scaler = StandardScaler()
# Fit the scaler to the data and transform it
X = scaler.fit_transform(stacked_data)
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Feature selection by SelectKBest with  ANOVA F-value----------------------
from sklearn.feature_selection import SelectKBest, f_classif
# Define the number of desired features
desired_features = 500   
# SelectKBest with ANOVA F-test (for classification tasks)
selector = SelectKBest(score_func=f_classif, k=desired_features)
# Fit the selector to the data and transform it
X = selector.fit_transform(X, Y)
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Classification
# XGBoost--------------------------------------------------------------------
# Number of runs
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []
# Run XGBoost multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    # Train the classifier on the selected features
    xgb_clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = xgb_clf.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]
# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs
# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")
# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)
# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)
# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_xgb=average_accuracy


# Gradient Boosting--------------------------------------------------------------------
# Number of runs  
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run Gradient Boosting Classifier multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the Gradient Boosting classifier
    gb_clf = GradientBoostingClassifier()
    
    # Train the classifier on the selected features
    gb_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = gb_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_gb=average_accuracy


# Naive Bayes--------------------------------------------------------------------
# Number of runs  
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run Naive Bayes Classifier multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the Naive Bayes classifier
    nb_clf = GaussianNB()
    
    # Train the classifier on the selected features
    nb_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = nb_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_nb=average_accuracy


# KNN--------------------------------------------------------------------
# Number of runs  
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run KNN Classifier multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the KNN classifier (default k=5, adjust as needed)
    knn_clf = KNeighborsClassifier()
    
    # Train the classifier on the selected features
    knn_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_knn=average_accuracy


# SVM--------------------------------------------------------------------
# Number of runs
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run SVM Classifier multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the SVM classifier
    svm_clf = SVC()  # Default kernel is 'rbf', adjust to 'linear', 'poly', etc.
    
    # Train the classifier on the selected features
    svm_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = svm_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_svm=average_accuracy

# LR--------------------------------------------------------------------
# Number of runs  
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run Multi-class Logistic Regression multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the Logistic Regression classifier with multi-class setting
    lr_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)  # adjust solver and max_iter
    
    # Train the classifier on the selected features
    lr_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = lr_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_lr=average_accuracy

# DT--------------------------------------------------------------------
# Number of runs
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run Decision Tree Classifier multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the Decision Tree classifier
    dt_clf = DecisionTreeClassifier()
    
    # Train the classifier on the selected features
    dt_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = dt_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_dt=average_accuracy

# RF--------------------------------------------------------------------
# Number of runs  
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run Random Forest Classifier multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the Random Forest classifier
    rf_clf = RandomForestClassifier()
    
    # Train the classifier on the selected features
    rf_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_rf=average_accuracy

# AdaBoost--------------------------------------------------------------------
# Number of runs
n_runs = 5
# arrays to store aggregated confusion matrix and classification metrics
confusion_matrices_sum = np.zeros((4, 4))  # 4 classes (adjust based on class count)
# variables to aggregate classification metrics
classification_reports_sum = {
    "precision": np.zeros(4),
    "recall": np.zeros(4),
    "f1-score": np.zeros(4),
    "support": np.zeros(4)
}
# For storing accuracies over multiple runs
accuracy_list = []

# Run AdaBoost Classifier multiple times
for i in range(n_runs):
    print(f"Run {i+1}/{n_runs}")
    
    # Split the data into training and testing sets (without specifying random_state to shuffle each time)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    # the AdaBoost classifier
    ab_clf = AdaBoostClassifier()
    
    # Train the classifier on the selected features
    ab_clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = ab_clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # Print the accuracy for this run
    print(f"Test Accuracy (Run {i+1}): {accuracy:.2f}")
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += conf_matrix  # Summing confusion matrices for later averaging
    
    # Generate classification report (output_dict=True to extract values)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Aggregate precision, recall, f1-score, and support for each class
    for class_idx in range(4):  # Adjust the range based on the number of classes
        class_label = str(class_idx)
        classification_reports_sum["precision"][class_idx] += report[class_label]["precision"]
        classification_reports_sum["recall"][class_idx] += report[class_label]["recall"]
        classification_reports_sum["f1-score"][class_idx] += report[class_label]["f1-score"]
        classification_reports_sum["support"][class_idx] += report[class_label]["support"]

# Compute average precision, recall, f1-score for each class
precision_avg = classification_reports_sum["precision"] / n_runs
recall_avg = classification_reports_sum["recall"] / n_runs
f1_avg = classification_reports_sum["f1-score"] / n_runs
support_avg = classification_reports_sum["support"]

# Compute the average confusion matrix
confusion_matrix_avg = confusion_matrices_sum / n_runs

# Print the final aggregated classification report in its original format
print("\nFinal Averaged Classification Report After Multiple Runs:")
for class_idx in range(4):  # Adjust the range based on the number of classes
    print(f"Class {class_idx}:")
    print(f"  Precision: {precision_avg[class_idx]:.2f}")
    print(f"  Recall: {recall_avg[class_idx]:.2f}")
    print(f"  F1-score: {f1_avg[class_idx]:.2f}")
    print(f"  Support: {support_avg[class_idx]:.0f}")

# Print the average confusion matrix
print("\nAverage Confusion Matrix After Multiple Runs:")
print(confusion_matrix_avg)

# Print the list of all test accuracies over the runs
print("\nTest Accuracies Over Multiple Runs:")
print(accuracy_list)

# Calculate and print the final average accuracy over all runs
average_accuracy = np.mean(accuracy_list)
print(f"\nFinal Average Accuracy Over All Runs: {average_accuracy:.2f}")

average_accuracy_ab=average_accuracy

# ----------------------------------------------------------------------
# Store the final average accuracy for each classifier in a dictionary
final_accuracies = {
    "XGBoost": average_accuracy_xgb,  # Average accuracy for XGBoost
    "Gradient Boosting": average_accuracy_gb,  # Average accuracy for Gradient Boosting
    "Naive Bayes": average_accuracy_nb,  # Average accuracy for Naive Bayes
    "KNN": average_accuracy_knn,  # Average accuracy for KNN
    "SVM": average_accuracy_svm,  # Average accuracy for SVM
    "Logistic Regression": average_accuracy_lr,  # Average accuracy for Logistic Regression
    "Decision Tree": average_accuracy_dt,  # Average accuracy for Decision Tree
    "Random Forest": average_accuracy_rf,  # Average accuracy for Random Forest
    "AdaBoost": average_accuracy_ab  # Average accuracy for AdaBoost
}

# Print the final average accuracy for each classifier
print("\nFinal Average Accuracy for All Classifiers After Multiple Runs:")
for clf_name, avg_accuracy in final_accuracies.items():
    print(f"{clf_name}: {avg_accuracy:.2f}")
