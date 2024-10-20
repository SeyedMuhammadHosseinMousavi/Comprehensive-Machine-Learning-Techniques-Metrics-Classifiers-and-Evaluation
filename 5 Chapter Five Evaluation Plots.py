%reset -f
#----------------------------------------------------
# Chapter Five
#----------------------------------------------------
# 5. Evaluation Plots
# Violin Plot for Prediction Probabilities
# Confusion Matrix Heatmap
# Test Accuracy Over Multiple Runs
# Residual Plot
# Precision-Recall Curve
# ROC Curve
# Cumulative Gain Chart
# Histogram of Prediction Probabilities
# Class Distribution Plot
# Box Plot for Prediction Probabilities
#----------------------------------------------------

import time
import numpy as np
import os
import warnings
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, 
    average_precision_score, 
    balanced_accuracy_score, 
    precision_score, recall_score, f1_score, accuracy_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------------
# Record the start time
start_time = time.time()

# Suppress warnings
warnings.filterwarnings("ignore")

# Parse BVH files
def parse_bvh(file_path):
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
    header, motion_data = parse_bvh(filename)
    return motion_data

def interpolate_frames(motion_data, target_frame_count):
    print(f"Interpolating frames to match target frame count: {target_frame_count}")
    original_frame_count = len(motion_data)
    original_time = np.linspace(0, 1, original_frame_count)
    target_time = np.linspace(0, 1, target_frame_count)
    interpolated_frames = []
    for frame in np.array(motion_data).T:
        interpolator = interp1d(original_time, frame.astype(float), kind='linear')
        interpolated_frame = interpolator(target_time)
        interpolated_frames.append(interpolated_frame)
    return np.array(interpolated_frames).T

def find_max_frames(folder_path):
    print(f"Finding maximum number of frames in folder: {folder_path}")
    max_frames = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            motion_data = read_bvh(os.path.join(folder_path, filename))
            max_frames = max(max_frames, len(motion_data))
    return max_frames

def process_bvh_files(folder_path, max_frames):
    print(f"Processing BVH files in folder: {folder_path}")
    all_features = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            full_path = os.path.join(folder_path, filename)
            motion_data = read_bvh(full_path)
            interpolated_frames = interpolate_frames(motion_data, max_frames)
            all_features.append(interpolated_frames)
    return np.array(all_features)

# Feature Extraction
def extract_motion_features(motion_data):
    print("Extracting motion features...")
    channels_per_joint = 3
    num_joints = motion_data.shape[1] // channels_per_joint

    features = []
    for i in range(num_joints):
        joint_rotations = motion_data[:, i * channels_per_joint:(i + 1) * channels_per_joint]
        angular_velocity = np.diff(joint_rotations, axis=0)
        acceleration = np.diff(angular_velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        range_of_motion = np.max(joint_rotations, axis=0) - np.min(joint_rotations, axis=0)
        joint_features = np.concatenate([angular_velocity.flatten(), acceleration.flatten(), jerk.flatten(), range_of_motion])
        features.append(joint_features)

    flattened_features = np.concatenate(features)
    return flattened_features

# -------------------------------
# Load and preprocess data
train_folder_path = 'Small Dataset/'
# -------------------------------

max_frames_train = find_max_frames(train_folder_path)
all_features_train = process_bvh_files(train_folder_path, max_frames_train)
X_features = np.array([extract_motion_features(interpolate_frames(motion_data, max_frames_train)) for motion_data in all_features_train])

# Labels
C_Angry = [0] * 42
C_Depressed = [1] * 42
C_Neutral = [2] * 42
C_Proud = [3] * 32
Labels = np.array(C_Angry + C_Depressed + C_Neutral + C_Proud, dtype=np.int32)

# Standardize and normalize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(all_features_train.reshape(len(all_features_train), -1))
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X_standardized)

# Handle NaN values
imputer = SimpleImputer(strategy='mean')
X_cleaned = imputer.fit_transform(X_normalized)

# Final feature set and labels
X = X_cleaned
Y = Labels

# Feature selection
k = 500
selector = SelectKBest(chi2, k=k)
X_selected = selector.fit_transform(X, Y)
selected_indices = selector.get_support(indices=True)
print(f"Selected feature indices: {selected_indices}")
print(f"Shape of X before selection: {X.shape}")
print(f"Shape of X after selection: {X_selected.shape}")

# Train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.3, random_state=42)

# Train XGBoost Classifier
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_classifier.fit(X_train, Y_train)

# Predictions
y_proba_XGB = xgb_classifier.predict_proba(X_test)
y_pred_XGB = xgb_classifier.predict(X_test)

# Plots------------------------------------------------------
# 1. Violin Plot for Prediction Probabilities
def plot_violin_with_medians(y_proba, y_true, classifier_name, class_labels, accuracies):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=y_proba, inner="quartile", palette="Set2")
    for i in range(y_proba.shape[1]):
        plt.scatter([i] * len(y_proba[:, i]), y_proba[:, i], color='black', alpha=0.6)
    for i in range(y_proba.shape[1]):
        median = np.median(y_proba[:, i])
        plt.scatter(i, median, color='white', edgecolor='black', s=100, zorder=3)
    for i, acc in enumerate(accuracies):
        plt.text(i, 1.05, f'Acc: {acc:.2f}', horizontalalignment='center', fontsize=10, color='black')
    plt.title(f'Violin Plot of Predicted Probabilities - {classifier_name}')
    plt.xlabel("Class")
    plt.ylabel("Predicted Probability")
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels)
    plt.show()

# 2. Confusion Matrix Heatmap
def plot_confusion_matrix_heatmap(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 3. Test Accuracy Over Multiple Runs
def plot_test_accuracy(accuracies):
    runs = range(1, len(accuracies) + 1)
    plt.figure(figsize=(8, 6))
    plt.bar(runs, accuracies)
    plt.title("Test Accuracy Over Multiple Runs")
    plt.xlabel("Run")
    plt.ylabel("Test Accuracy")
    plt.xticks(runs)
    plt.ylim([0, 1])
    plt.show()

# 4. Residual Plot (for classification)
def plot_residuals(y_true, y_pred, classifier_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=20)
    plt.title(f'Residuals Distribution - {classifier_name}')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

# 5. Precision-Recall Curve (for multiclass)
def plot_precision_recall_curve(y_true, y_proba, n_classes, classifier_name):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.title(f'Precision-Recall Curve - {classifier_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.show()

# 6. ROC Curve (for multiclass)
def plot_multiclass_roc_curve(y_true, y_proba, n_classes, classifier_name):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {classifier_name}')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# 7. Cumulative Gain Chart
def plot_cumulative_gain_chart(y_true, y_proba, classifier_name):
    sorted_indices = np.argsort(-y_proba[:, 1])
    sorted_y_true = y_true[sorted_indices]
    cumulative_true = np.cumsum(sorted_y_true) / np.sum(sorted_y_true)
    cumulative_total = np.cumsum(np.ones_like(sorted_y_true)) / len(sorted_y_true)
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_total, cumulative_true, label="Cumulative Gain")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Model")
    plt.xlabel('Fraction of Samples')
    plt.ylabel('Cumulative Gain')
    plt.title(f'Cumulative Gain Chart - {classifier_name}')
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# 8. Histogram of Prediction Probabilities
def plot_prediction_probabilities_histogram(y_proba, classifier_name):
    plt.figure(figsize=(8, 6))
    plt.hist(y_proba, bins=10, edgecolor="k")
    plt.title(f'Histogram of Prediction Probabilities - {classifier_name}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.show()

# 9. Class Distribution Plot
def plot_class_distribution(Y, class_labels):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=Y)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(ticks=range(len(class_labels)), labels=class_labels)
    plt.show()

# Call the function to plot class distribution
plot_class_distribution(Y, np.array(["Angry", "Depressed", "Neutral", "Proud"]))

# 10. Box Plot for Prediction Probabilities
def plot_boxplot_proba_with_enhancements(y_proba, classifier_name, class_labels):
    plt.figure(figsize=(10, 6))

    # Create the box plot
    sns.boxplot(data=y_proba, showmeans=True, meanline=True)

    # Add additional statistical annotations
    for i in range(y_proba.shape[1]):
        # Calculate summary statistics
        median = np.median(y_proba[:, i])
        mean = np.mean(y_proba[:, i])
        q1 = np.percentile(y_proba[:, i], 25)
        q3 = np.percentile(y_proba[:, i], 75)
        iqr = q3 - q1
        lower_whisker = max(min(y_proba[:, i]), q1 - 1.5 * iqr)
        upper_whisker = min(max(y_proba[:, i]), q3 + 1.5 * iqr)
        outliers = y_proba[(y_proba[:, i] < lower_whisker) | (y_proba[:, i] > upper_whisker), i]

        # Annotate the plot with median, mean, and whiskers
        plt.text(i, median, f'Median: {median:.2f}', horizontalalignment='center', fontsize=9, color='black', weight='bold')
        plt.text(i, mean, f'Mean: {mean:.2f}', horizontalalignment='center', fontsize=9, color='blue', weight='bold')
        plt.text(i, lower_whisker, f'Low: {lower_whisker:.2f}', horizontalalignment='center', fontsize=8, color='green')
        plt.text(i, upper_whisker, f'High: {upper_whisker:.2f}', horizontalalignment='center', fontsize=8, color='green')

        # Annotate outliers
        if len(outliers) > 0:
            plt.text(i, upper_whisker + 0.05, f'{len(outliers)} outliers', horizontalalignment='center', fontsize=8, color='red')

    # Add titles and labels
    plt.title(f'Enhanced Box Plot of Predicted Probabilities - {classifier_name}')
    plt.xlabel("Class")
    plt.ylabel("Predicted Probability")
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels)

    plt.show()

# Call the enhanced box plot function
plot_boxplot_proba_with_enhancements(y_proba_XGB, "XGBoost", np.array(["Angry", "Depressed", "Neutral", "Proud"]))

# Call the function for the Violin Plot
plot_violin_with_medians(y_proba_XGB, Y_test, "XGBoost", np.array(["Angry", "Depressed", "Neutral", "Proud"]), 
                         [accuracy_score(Y_test[Y_test == i], y_pred_XGB[Y_test == i]) for i in range(4)])

# Call the function to plot the Confusion Matrix Heatmap
plot_confusion_matrix_heatmap(Y_test, y_pred_XGB, np.array(["Angry", "Depressed", "Neutral", "Proud"]))

# Call the function to plot Test Accuracy Over Multiple Runs
# accuracy scores over multiple runs
XGB_test_accuracies = [0.9, 0.85, 0.87, 0.88]  # Dummy accuracy values
plot_test_accuracy(XGB_test_accuracies)

# Call the function to plot Residuals
plot_residuals(Y_test, y_pred_XGB, "XGBoost")

# Call the function to plot the Precision-Recall Curve (for multiclass)
plot_precision_recall_curve(Y_test, y_proba_XGB, 4, "XGBoost")

# Call the function to plot the ROC Curve (for multiclass)
plot_multiclass_roc_curve(Y_test, y_proba_XGB, 4, "XGBoost")

# Call the function for the Cumulative Gain Chart
plot_cumulative_gain_chart(Y_test, y_proba_XGB, "XGBoost")

# Call the function for the Histogram of Prediction Probabilities
plot_prediction_probabilities_histogram(y_proba_XGB, "XGBoost")

# Call the function to plot Class Distribution
plot_class_distribution(Y_test, np.array(["Angry", "Depressed", "Neutral", "Proud"]))
