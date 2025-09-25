from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
import numpy as np

class AUCEvaluator:
    def __init__(self):
        self.name = "multi_task_auc"

    def eval(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]

        num_tasks = y_true.shape[1]
        auc_list = []

        for task in range(num_tasks):
            y_true_task = y_true[:, task]
            y_pred_task = y_pred[:, task]

            # Skip tasks with only one class
            if len(np.unique(y_true_task)) < 2:
                continue

            try:
                auc = roc_auc_score(y_true_task, y_pred_task)
                auc_list.append(auc)
            except ValueError:
                continue

        if len(auc_list) == 0:
            print("⚠️ No valid tasks to calculate AUC")
            return {"auc": float("nan")}

        return {"auc": float(np.mean(auc_list))}


# class AUCEvaluator:
#     def __init__(self):
#         self.name = "custom_auc"
#
#     def eval(self, input_dict):
#         """
#         Evaluate the predictions using the AUC metric.
#
#         Parameters:
#             input_dict (dict): Contains 'y_true' and 'y_pred'.
#
#         Returns:
#             dict: A dictionary with the AUC score.
#         """
#         y_true = input_dict['y_true']
#         y_pred = input_dict['y_pred']
#
#         try:
#             # Compute AUC
#             auc = roc_auc_score(y_true, y_pred)
#             return {"auc": auc}
#         except ValueError as e:
#             # Handle cases where AUC can't be calculated (e.g., only one class in y_true)
#             print(f"Error calculating AUC: {e}")
#             return {"auc": float("nan")}

def min_max_scaling(data, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to the input tensor for each time step and feature.

    Parameters:
    - data: Input tensor of shape [batch_size, time_steps, features]
    - feature_range: Tuple specifying the desired range (min, max) for scaling.

    Returns:
    - Scaled tensor of the same shape as the input.
    """
    min_val, max_val = feature_range
    batch_min = data.min(dim=1, keepdim=True)[0]  # Min across time steps for each sample
    batch_max = data.max(dim=1, keepdim=True)[0]  # Max across time steps for each sample

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    scaled_data = (data - batch_min) / (batch_max - batch_min + epsilon)
    scaled_data = scaled_data * (max_val - min_val) + min_val

    return scaled_data
def validate_test_set(test_set):

    valid = True

    # Iterate over each column

    for col in range(test_set.size(1)):

        # Extract non -1 entries for the current column

        non_neg_ones = test_set[test_set[:, col] != -1, col]

        # Skip validation if the column is entirely -1

        if len(non_neg_ones) == 0:

            valid = False

            print(f"Column {col} is entirely filled with -1 entries!")

            break

        # Check if there's both 0 and 1 in non -1 entries

        if torch.any(non_neg_ones == 0) and torch.any(non_neg_ones == 1):

            valid = True

            continue

        elif not torch.any(non_neg_ones == 0) and not torch.any(non_neg_ones == 1):

            valid = True

            continue

        else:

            valid = False

            print(f"Column {col} does not have both 0 and 1 in non -1 entries!")

            break

    return valid

def min_max_scaling(data, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to the input tensor for each time step and feature.

    Parameters:
    - data: Input tensor of shape [batch_size, time_steps, features]
    - feature_range: Tuple specifying the desired range (min, max) for scaling.

    Returns:
    - Scaled tensor of the same shape as the input.
    """
    min_val, max_val = feature_range
    batch_min = data.min(dim=1, keepdim=True)[0]  # Min across time steps for each sample
    batch_max = data.max(dim=1, keepdim=True)[0]  # Max across time steps for each sample

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    scaled_data = (data - batch_min) / (batch_max - batch_min + epsilon)
    scaled_data = scaled_data * (max_val - min_val) + min_val

    return scaled_data

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

