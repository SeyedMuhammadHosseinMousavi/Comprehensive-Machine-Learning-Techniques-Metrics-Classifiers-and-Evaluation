%reset -f
#----------------------------------------------------
# Chapter Seven
#----------------------------------------------------
# 7. Advanced Techniques and Analysis
# Leave One Participant Out CV (LOPO)
# Leave-One-Out Cross-Validation (LOO)
# K-Fold Cross-Validation
# Stratified K-Fold Cross-Validation
# Repeated K-Fold Cross-Validation
# Nested Cross-Validation
# Bootstrapping
#----------------------------------------------------

import numpy as np
import pandas as pd
from scipy import stats
from bvh import Bvh
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
import os
import warnings
import pywt
import time
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.tree as tr
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils import resample

#---------------------------------------------------------------------------
# Suppressing All Warnings
warnings.filterwarnings("ignore")
# Start time for the overall process
start_time = time.time()
# storage for final test accuracies of each classifier for all runs
final_accuracies = {
    "XGB": []
}

# storage for accumulating predictions and true labels
accumulated_predictions = {
    "XGB": {"true": [], "pred": []}
}

def read_bvh(filename):
    with open(filename) as f:
        mocap = Bvh(f.read())
    return mocap
label_encoder = LabelEncoder()

# Folder Path ------------------------------------------
folder_path = 'Small Dataset/'
# Number of Runs ---------------------------------------

num_runs = 5

# ------------------------------------------------------
bvh_data = {}
run=0
file_count = 0  # To count the files processed
for file in os.listdir(folder_path):
    if file.endswith('.bvh'):
        file_path = os.path.join(folder_path, file)
        mocap = read_bvh(file_path)
        bvh_data[file] = mocap
        file_count += 1
        print(f"Loaded file {file_count}: {file}")
print(f"Total files loaded in run {run + 1}: {file_count}")

for run in range(num_runs):
    print(f"\nStarting run {run + 1} of {num_runs}")
    ListofDicVal = list(bvh_data.values())
    stacked_data = []
    for iloop, bvh in enumerate(ListofDicVal, start=1):
        frames = bvh.frames
        flat_vector = [value for frame in frames for value in frame]
        float_list = [float(item) for item in flat_vector]

        # Feature extraction
        mean = np.mean(float_list)
        median = np.median(float_list)
        std_dev = np.std(float_list)
        variance = np.var(float_list)
        skewness = stats.skew(float_list)
        kurtosis = stats.kurtosis(float_list)
        minval = np.min(float_list)
        maxval = np.max(float_list)
        sumval = np.sum(float_list)

        # Correcting the autocorrelation calculation:
        # Directly use `float_list` with `acf` since it's already a 1D array
        autocorrelation = acf(float_list, nlags=5, fft=True)  # Note: No need to flatten

        peaks, _ = find_peaks(np.array(float_list))
        num_peaks = len(peaks)
        sum_peaks = np.sum(peaks)
        percentiles = np.percentile(float_list, [20, 40, 60, 80, 99])

        signal = np.array([float_list])
        coeffs = pywt.wavedec(signal, 'db1', mode='symmetric', level=None)
        coeff_0_mean = np.mean(coeffs[0])
        coeff_0_variance = np.var(coeffs[0])
        coeff_0_std_dev = np.std(coeffs[0])
        coeff_0_max = np.max(coeffs[0])
        coeff_0_min = np.min(coeffs[0])

        AllStatics = np.concatenate(([mean, median, std_dev, variance, skewness, kurtosis, minval, maxval, sumval], 
                                     autocorrelation, [num_peaks, sum_peaks], percentiles, 
                                     [coeff_0_mean, coeff_0_variance, coeff_0_std_dev, coeff_0_max, coeff_0_min]))

        stacked_data.append(AllStatics)

        print(f"Processed {iloop}/{len(ListofDicVal)} files for feature extraction in run {run + 1}")

    print(f"Feature extraction completed for all files in run {run + 1}.")

# Standardize the data --------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    stacked_data = np.array(stacked_data)
    scaler = StandardScaler()
    # Fit the scaler to the data and transform it
    scaled_data = scaler.fit_transform(stacked_data)
    
# Labels ------------------------------------------------------------------------------------
    FinalFeatures = np.array(scaled_data)
    Labels = [0] * 42 + [1] * 42 + [2] * 42 + [3] * 32
    
# ------------------------------------------------------------------------------------
   
    Labels_int32 = np.array(Labels, dtype=np.int32)
    X_train, X_test, Y_train, Y_test = ms.train_test_split(FinalFeatures, Labels_int32, train_size=0.8, stratify=Labels_int32)
    Y_train_encoded = label_encoder.fit_transform(Y_train)
    Y_test_encoded = label_encoder.transform(Y_test)

    # XGBoost
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,  # Number of gradient boosted trees
        learning_rate=0.1,  # Step size shrinkage used to prevent overfitting
        max_depth=6,  # Maximum tree depth for base learners
        subsample=0.3,  # Subsample ratio of the training instance
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        objective='multi:softmax',  # Multi-class classification using the softmax objective
        num_class=4,  # Specify the number of classes in dataset
        use_label_encoder=False,  # Avoids a future warning for label encoding
        eval_metric='mlogloss'  # Evaluation metric for multi-class classification
        )
    xgb_classifier.fit(X_train, Y_train_encoded)  
    y_pred = xgb_classifier.predict(X_test)  # Prediction
    # Append the accuracy score to final_accuracies dictionary under "XGB"
    final_accuracies["XGB"].append(accuracy_score(Y_test_encoded, y_pred))
    # Extend accumulated_predictions dictionary for "XGB" with true labels and predictions
    accumulated_predictions["XGB"]["true"].extend(Y_test_encoded)
    accumulated_predictions["XGB"]["pred"].extend(y_pred)

#----------------------------------------------------------------------------
# Classification report and confusion matrix
# Generate the classification report and confusion matrix
classification_rep = classification_report(accumulated_predictions["XGB"]["true"], accumulated_predictions["XGB"]["pred"])
confusion_mat = confusion_matrix(accumulated_predictions["XGB"]["true"], accumulated_predictions["XGB"]["pred"])
# Print classification report and confusion matrix
print("\nClassification Report for XGBoost:\n", classification_rep)
print("\nConfusion Matrix for XGBoost:\n", confusion_mat)
# Calculate and print the classification report, confusion matrix, and final test accuracy
print("XGBoost Final Test Accuracy:", np.mean(final_accuracies["XGB"]))
#-----------------------------------------------------------------

#------------------------------------------
#Leave-One-Participant-Out (LOPO)
#------------------------------------------
# here as we dont have participant manually splited data into half for two participants.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Split data for each class into two "participants"
half_size = len(FinalFeatures) // 2
participant_1_indices = []
participant_2_indices = []

# Labels_int32 is the list of class labels and classes are balanced
for class_label in np.unique(Labels_int32):
    class_indices = np.where(Labels_int32 == class_label)[0]
    np.random.shuffle(class_indices)  # Shuffle to randomize participant splits
    mid_point = len(class_indices) // 2
    
    # Assign half of the class samples to participant 1 and half to participant 2
    participant_1_indices.extend(class_indices[:mid_point])
    participant_2_indices.extend(class_indices[mid_point:])

# Define participant splits
participant_splits = [
    participant_1_indices,  # Leave Participant 1 out
    participant_2_indices   # Leave Participant 2 out
]

# storage for results
lopo_true_labels = []
lopo_pred_labels = []

# Perform Leave-One-Participant-Out Cross-Validation
for participant_idx, test_indices in enumerate(participant_splits, start=1):
    print(f"\nStarting LOPO iteration {participant_idx} - Leave Participant {participant_idx} out")
    
    # Create training and testing data
    train_indices = [i for i in range(len(FinalFeatures)) if i not in test_indices]
    X_train, X_test = FinalFeatures[train_indices], FinalFeatures[test_indices]
    Y_train, Y_test = Labels_int32[train_indices], Labels_int32[test_indices]
    
    # XGBoost Classifier
    xgb_classifier_lopo = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        subsample=0.3, 
        colsample_bytree=0.8, 
        objective='multi:softmax', 
        num_class=4,  # Adjust according to the number of classes
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    
    # Train the classifier
    xgb_classifier_lopo.fit(X_train, Y_train)
    
    # Predict on the test set
    y_pred_lopo = xgb_classifier_lopo.predict(X_test)
    
    # Store the true and predicted labels
    lopo_true_labels.extend(Y_test)
    lopo_pred_labels.extend(y_pred_lopo)
    
    # Print progress and results for this iteration
    accuracy = accuracy_score(Y_test, y_pred_lopo)
    print(f"LOPO iteration {participant_idx} completed. Accuracy: {accuracy:.4f}")

# Generate classification report and confusion matrix
lopo_classification_rep = classification_report(lopo_true_labels, lopo_pred_labels)
lopo_confusion_mat = confusion_matrix(lopo_true_labels, lopo_pred_labels)

# Calculate overall accuracy
final_lopo_accuracy = accuracy_score(lopo_true_labels, lopo_pred_labels)

# Print final results
print("\nLeave-One-Participant-Out (LOPO) Results:")
print(f"Final LOPO Accuracy: {final_lopo_accuracy:.4f}")
print("\nLOPO Classification Report:\n", lopo_classification_rep)
print("\nLOPO Confusion Matrix:\n", lopo_confusion_mat)
print(f"  ")
print(f"Final LOPO Accuracy: {final_lopo_accuracy:.4f}")
print(f"  ")
print(f"  ")
print(f"----------------")

#-----------------------------------------------------------------
# Leave-One-Out Cross-Validation ---------------------------------------------
#------------------------------------------

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# storage for true labels, predicted labels, and correctness tracking
loo_true_labels = []
loo_pred_labels = []
loo_accuracies = []

# dictionary to track correct and incorrect predictions per class
class_correct_incorrect = {0: {"correct": 0, "incorrect": 0},
                           1: {"correct": 0, "incorrect": 0},
                           2: {"correct": 0, "incorrect": 0},
                           3: {"correct": 0, "incorrect": 0}}  # Adjust if have more classes

# Get the total number of iterations (samples) for progress tracking
total_loops = len(FinalFeatures)

# Perform LOO Cross-Validation with progress updates
for idx, (train_index, test_index) in enumerate(loo.split(FinalFeatures), start=1):
    X_train, X_test = FinalFeatures[train_index], FinalFeatures[test_index]
    Y_train, Y_test = Labels_int32[train_index], Labels_int32[test_index]
    
    # XGBoost Classifier
    xgb_classifier_loo = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        subsample=0.3, 
        colsample_bytree=0.8, 
        objective='multi:softmax', 
        num_class=4,  # Change if have more or fewer classes
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    
    # Train the classifier
    xgb_classifier_loo.fit(X_train, Y_train)
    
    # Predict the test sample
    y_pred_loo = xgb_classifier_loo.predict(X_test)
    
    # Store the true and predicted labels
    loo_true_labels.append(Y_test[0])
    loo_pred_labels.append(y_pred_loo[0])
    
    # Check if the prediction is correct (1 if correct, 0 if incorrect)
    correct = int(Y_test[0] == y_pred_loo[0])
    
    # Update the class tracking dictionary
    if correct == 1:
        class_correct_incorrect[Y_test[0]]["correct"] += 1
    else:
        class_correct_incorrect[Y_test[0]]["incorrect"] += 1
    
    # Print iteration progress with correctness
    print(f"LOO iteration {idx}/{total_loops} - Prediction: {'Correct' if correct == 1 else 'Incorrect'}")

# Generate classification report and confusion matrix
loo_classification_rep = classification_report(loo_true_labels, loo_pred_labels)
loo_confusion_mat = confusion_matrix(loo_true_labels, loo_pred_labels)

# Calculate overall accuracy from LOO
final_loo_accuracy = np.mean(loo_accuracies)

# Print final results
print("\nLeave-One-Out Cross-Validation (LOO) Results:")
# print("Final LOO Accuracy:", final_loo_accuracy)
print("\nLOO Classification Report:\n", loo_classification_rep)
print("\nLOO Confusion Matrix:\n", loo_confusion_mat)

# Print the total number of correct and incorrect predictions for each class
print("\nCorrect/Incorrect Predictions per Class:")
for class_label, results in class_correct_incorrect.items():
    print(f"Class {class_label} - Correct: {results['correct']}, Incorrect: {results['incorrect']}")
# Calculate overall accuracy from LOO
final_loo_accuracy = np.mean([1 if true == pred else 0 for true, pred in zip(loo_true_labels, loo_pred_labels)])

# Print final accuracy
print("\nFinal Leave-One-Out Cross-Validation (LOO) Accuracy: {:.4f}".format(final_loo_accuracy))

#-----------------------------------------------------------------
# K-Fold Cross-Validation
#-----------------------------------------------------------------
# K-Fold Cross-Validation
kf = KFold(n_splits=7, shuffle=True, random_state=42)  # Adjust n_splits for the number of folds

# storage for results
kfold_true_labels = []
kfold_pred_labels = []
kfold_accuracies = []

# Perform K-Fold Cross-Validation
for fold_idx, (train_index, test_index) in enumerate(kf.split(FinalFeatures), start=1):
    print(f"\nStarting K-Fold iteration {fold_idx} - Training on {len(train_index)} samples, Testing on {len(test_index)} samples")
    
    # Split the data into training and testing sets
    X_train, X_test = FinalFeatures[train_index], FinalFeatures[test_index]
    Y_train, Y_test = Labels_int32[train_index], Labels_int32[test_index]
    
    # XGBoost Classifier
    xgb_classifier_kfold = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        subsample=0.3, 
        colsample_bytree=0.8, 
        objective='multi:softmax', 
        num_class=4,  # Adjust based on the number of classes
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    
    # Train the classifier
    xgb_classifier_kfold.fit(X_train, Y_train)
    
    # Predict on the test set
    y_pred_kfold = xgb_classifier_kfold.predict(X_test)
    
    # Store the true and predicted labels
    kfold_true_labels.extend(Y_test)
    kfold_pred_labels.extend(y_pred_kfold)
    
    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(Y_test, y_pred_kfold)
    kfold_accuracies.append(fold_accuracy)
    
    # Print progress and results for this iteration
    print(f"K-Fold iteration {fold_idx} completed. Accuracy: {fold_accuracy:.4f}")

# Generate classification report and confusion matrix
kfold_classification_rep = classification_report(kfold_true_labels, kfold_pred_labels)
kfold_confusion_mat = confusion_matrix(kfold_true_labels, kfold_pred_labels)

# Calculate overall accuracy across all folds
final_kfold_accuracy = np.mean(kfold_accuracies)

# Print final results
print("\nK-Fold Cross-Validation Results:")
print(f"Final K-Fold Accuracy: {final_kfold_accuracy:.4f}")
print("\nK-Fold Classification Report:\n", kfold_classification_rep)
print("\nK-Fold Confusion Matrix:\n", kfold_confusion_mat)
print(f"  ")
print(f"Final K-Fold Accuracy: {final_kfold_accuracy:.4f}")
print(f"  ")
print(f"  ")
print(f"----------------")

#-----------------------------------------------------------------
# Stratified K-Fold Cross-Validation
#-----------------------------------------------------------------
# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)  # Adjust n_splits for the number of folds

# storage for results
stratified_true_labels = []
stratified_pred_labels = []
stratified_accuracies = []

# Perform Stratified K-Fold Cross-Validation
for fold_idx, (train_index, test_index) in enumerate(skf.split(FinalFeatures, Labels_int32), start=1):
    print(f"\nStarting Stratified K-Fold iteration {fold_idx} - Training on {len(train_index)} samples, Testing on {len(test_index)} samples")
    
    # Split the data into training and testing sets
    X_train, X_test = FinalFeatures[train_index], FinalFeatures[test_index]
    Y_train, Y_test = Labels_int32[train_index], Labels_int32[test_index]
    
    # XGBoost Classifier
    xgb_classifier_kfold = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        subsample=0.3, 
        colsample_bytree=0.8, 
        objective='multi:softmax', 
        num_class=4,  # Adjust based on the number of classes
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    
    # Train the classifier
    xgb_classifier_kfold.fit(X_train, Y_train)
    
    # Predict on the test set
    y_pred_kfold = xgb_classifier_kfold.predict(X_test)
    
    # Store the true and predicted labels
    stratified_true_labels.extend(Y_test)
    stratified_pred_labels.extend(y_pred_kfold)
    
    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(Y_test, y_pred_kfold)
    stratified_accuracies.append(fold_accuracy)
    
    # Print progress and results for this iteration
    print(f"Stratified K-Fold iteration {fold_idx} completed. Accuracy: {fold_accuracy:.4f}")

# Generate classification report and confusion matrix
stratified_classification_rep = classification_report(stratified_true_labels, stratified_pred_labels)
stratified_confusion_mat = confusion_matrix(stratified_true_labels, stratified_pred_labels)

# Calculate overall accuracy across all folds
final_stratified_accuracy = np.mean(stratified_accuracies)

# Print final results
print("\nStratified K-Fold Cross-Validation Results:")
print(f"Final Stratified K-Fold Accuracy: {final_stratified_accuracy:.4f}")
print("\nStratified K-Fold Classification Report:\n", stratified_classification_rep)
print("\nStratified K-Fold Confusion Matrix:\n", stratified_confusion_mat)
print(f"Final Stratified K-Fold Accuracy: {final_stratified_accuracy:.4f}")
print(f"  ")
print(f"  ")
print(f"----------------")

#-----------------------------------------------------------------
# Repeated K-Fold Cross-Validation
#-----------------------------------------------------------------
# Repeated K-Fold Cross-Validation
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)  # Adjust n_splits and n_repeats as needed

# storage for results
repeated_true_labels = []
repeated_pred_labels = []
repeated_accuracies = []

# Perform Repeated K-Fold Cross-Validation
for fold_idx, (train_index, test_index) in enumerate(rkf.split(FinalFeatures), start=1):
    print(f"\nStarting Repeated K-Fold iteration {fold_idx} - Training on {len(train_index)} samples, Testing on {len(test_index)} samples")
    
    # Split the data into training and testing sets
    X_train, X_test = FinalFeatures[train_index], FinalFeatures[test_index]
    Y_train, Y_test = Labels_int32[train_index], Labels_int32[test_index]
    
    # XGBoost Classifier
    xgb_classifier_rkf = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        subsample=0.3, 
        colsample_bytree=0.8, 
        objective='multi:softmax', 
        num_class=4,  # Adjust based on the number of classes
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    
    # Train the classifier
    xgb_classifier_rkf.fit(X_train, Y_train)
    
    # Predict on the test set
    y_pred_rkf = xgb_classifier_rkf.predict(X_test)
    
    # Store the true and predicted labels
    repeated_true_labels.extend(Y_test)
    repeated_pred_labels.extend(y_pred_rkf)
    
    # Calculate accuracy for this fold
    fold_accuracy = accuracy_score(Y_test, y_pred_rkf)
    repeated_accuracies.append(fold_accuracy)
    
    # Print progress and results for this iteration
    print(f"Repeated K-Fold iteration {fold_idx} completed. Accuracy: {fold_accuracy:.4f}")

# Generate classification report and confusion matrix
repeated_classification_rep = classification_report(repeated_true_labels, repeated_pred_labels)
repeated_confusion_mat = confusion_matrix(repeated_true_labels, repeated_pred_labels)

# Calculate overall accuracy across all folds
final_repeated_accuracy = np.mean(repeated_accuracies)

# Print final results
print("\nRepeated K-Fold Cross-Validation Results:")
print(f"Final Repeated K-Fold Accuracy: {final_repeated_accuracy:.4f}")
print("\nRepeated K-Fold Classification Report:\n", repeated_classification_rep)
print("\nRepeated K-Fold Confusion Matrix:\n", repeated_confusion_mat)
print(f"Final Repeated K-Fold Accuracy: {final_repeated_accuracy:.4f}")
print(f"  ")
print(f"  ")
print(f"----------------")

#-----------------------------------------------------------------
# Nested Cross-Validation 
#-----------------------------------------------------------------
# Define the parameter grid for hyperparameter tuning (inner loop)
param_grid = {
    'n_estimators': [50, 100],  # Balanced range for number of trees
    'learning_rate': [0.05, 0.1],  # Reasonable learning rates
    'max_depth': [4, 6],  # Limited depth values
    'subsample': [0.8, 0.9],  # Two subsample values
    'colsample_bytree': [0.8, 0.9],  # Two feature fractions
}

# the XGBoost classifier with early stopping
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,  # Adjust based on the number of classes
    use_label_encoder=False,
    eval_metric='mlogloss',
    early_stopping_rounds=10  # Early stopping to speed up training
)

# the outer KFold cross-validation with moderate splits
outer_cv = KFold(n_splits=4, shuffle=True, random_state=42)  # 4 outer folds

# storage for outer loop results
outer_accuracies = []
all_true_labels = []
all_pred_labels = []

# Perform Nested Cross-Validation
for outer_fold_idx, (train_index, test_index) in enumerate(outer_cv.split(FinalFeatures), start=1):
    print(f"\nOuter Fold {outer_fold_idx} - Training on {len(train_index)} samples, Testing on {len(test_index)} samples")

    # Split the data into training and testing sets for this outer fold
    X_train, X_test = FinalFeatures[train_index], FinalFeatures[test_index]
    Y_train, Y_test = Labels_int32[train_index], Labels_int32[test_index]
    
    # Use GridSearchCV for the inner loop (hyperparameter tuning)
    print("  Performing hyperparameter tuning (inner loop) with GridSearchCV...")
    inner_cv = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, n_jobs=-1)  # Moderate inner loop with 3 CV splits
    inner_cv.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)  # Using eval_set for early stopping
    
    # Best hyperparameters from the inner loop
    print(f"  Best hyperparameters for Outer Fold {outer_fold_idx}: {inner_cv.best_params_}")
    
    # Evaluate on the outer test set using the best model from inner loop
    y_pred = inner_cv.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    outer_accuracies.append(accuracy)
    
    # Append true and predicted labels for confusion matrix and classification report
    all_true_labels.extend(Y_test)
    all_pred_labels.extend(y_pred)
    
    print(f"  Outer Fold {outer_fold_idx} completed with Accuracy: {accuracy:.4f}")

# Calculate and print overall accuracy from outer loop
final_nested_accuracy = np.mean(outer_accuracies)
print("\nNested Cross-Validation Results:")
print(f"Final Nested CV Accuracy: {final_nested_accuracy:.4f}")

# Generate and print classification report and confusion matrix for all folds
classification_rep = classification_report(all_true_labels, all_pred_labels)
confusion_mat = confusion_matrix(all_true_labels, all_pred_labels)

print("\nAggregated Classification Report for All Outer Folds:\n", classification_rep)
print("\nAggregated Confusion Matrix for All Outer Folds:\n", confusion_mat)

print(f"Final Nested CV Accuracy: {final_nested_accuracy:.4f}")
print(f"  ")
print(f"  ")
print(f"----------------")

#-----------------------------------------------------------------
# Number of bootstrap iterations
n_iterations = 100

# storage for results
bootstrap_true_labels = []
bootstrap_pred_labels = []
bootstrap_accuracies = []

# Perform Bootstrapping
for i in range(n_iterations):
    print(f"\nStarting Bootstrap iteration {i + 1}/{n_iterations}")
    
    # Create a bootstrap sample with replacement
    X_resampled, Y_resampled = resample(FinalFeatures, Labels_int32, replace=True, random_state=i)
    
    # Split into training and testing sets
    split_ratio = 0.8  # Use 80% for training, 20% for testing
    n_train = int(split_ratio * len(X_resampled))
    X_train, X_test = X_resampled[:n_train], X_resampled[n_train:]
    Y_train, Y_test = Y_resampled[:n_train], Y_resampled[n_train:]
    
    # the XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        objective='multi:softmax', 
        num_class=4,  # Adjust based on the number of classes
        use_label_encoder=False, 
        eval_metric='mlogloss'
    )
    
    # Train the classifier on the resampled data
    xgb_classifier.fit(X_train, Y_train)
    
    # Predict on the test set
    y_pred = xgb_classifier.predict(X_test)
    
    # Calculate and store the accuracy for this iteration
    accuracy = accuracy_score(Y_test, y_pred)
    bootstrap_accuracies.append(accuracy)
    
    # Append true and predicted labels for confusion matrix and classification report
    bootstrap_true_labels.extend(Y_test)
    bootstrap_pred_labels.extend(y_pred)
    
    # Print progress and results for this iteration
    print(f"Bootstrap iteration {i + 1} completed with Accuracy: {accuracy:.4f}")

# Calculate and print the final accuracy across all bootstrap iterations
final_bootstrap_accuracy = np.mean(bootstrap_accuracies)
print("\nBootstrapping Results:")
print(f"Final Bootstrap Accuracy (averaged over {n_iterations} iterations): {final_bootstrap_accuracy:.4f}")

# Generate and print the classification report and confusion matrix for all bootstrap iterations
classification_rep = classification_report(bootstrap_true_labels, bootstrap_pred_labels)
confusion_mat = confusion_matrix(bootstrap_true_labels, bootstrap_pred_labels)

print("\nAggregated Classification Report for All Bootstrap Iterations:\n", classification_rep)
print("\nAggregated Confusion Matrix for All Bootstrap Iterations:\n", confusion_mat)
print(f"  ")
print(f"Final Bootstrap Accuracy (averaged over {n_iterations} iterations): {final_bootstrap_accuracy:.4f}")
print(f"  ")
print(f"  ")
print(f"----------------")

#-----------------------------------------------------------------
