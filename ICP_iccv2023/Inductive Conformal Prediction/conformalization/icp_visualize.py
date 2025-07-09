import numpy as np
import matplotlib.pyplot as plt

def visualize_conformal_prediction_sets(X_test, test_smx, qhat, test_labels, model_test_pred, label_strings):
    """
    Visualize and print conformal prediction sets for test images.
    For each test image, display the image, print the prediction set, true label, model prediction, and softmax scores.
    X_test: test images (N, H, W, C) or (N, C, H, W)
    test_smx: test softmax outputs (N, num_classes)
    qhat: quantile threshold
    test_labels: true labels (N,)
    model_test_pred: model predictions (N,)
    label_strings: array of class label strings (num_classes,)
    """
    for i in range(test_labels.shape[0]):
        normalized_image = X_test[i]
        # If image is (C, H, W), transpose to (H, W, C)
        if normalized_image.shape[0] == 3 and normalized_image.ndim == 3:
            normalized_image = np.transpose(normalized_image, (1, 2, 0))
        clipped_image = np.clip(normalized_image, 0, 1)
        prediction_set = test_smx[i] > 1 - qhat
        plt.figure()
        plt.imshow(clipped_image)
        plt.axis('off')
        plt.show()
        if len(prediction_set) == 0:
            print(f'{150 *"_"}')
            print(f"The prediction set for image {i} is: [], and the true label is {test_labels[i]}, and model's prediction is {model_test_pred[i]}")
            print(f"The Softmax score for class 0: {test_smx[i, 0]}, and Softmax score for class 1: {test_smx[i, 1]}")
            print(f'{150 *"_"}')
        else:
            print(f'{150 *"_"}')
            print(f"The prediction set for image {i} is: {list(label_strings[prediction_set])}, the true label is {test_labels[i]}, and model's prediction is {model_test_pred[i]}")
            print(f"The Softmax score for class 0: {test_smx[i, 0]}, and Softmax score for class 1: {test_smx[i, 1]}")
            print(f'{150 *"_"}')

def analyze_conformal_prediction_sets(X_final, test_final_smx, qhat, test_final_labels, model_final_test_pred, label_strings):
    """
    Visualize and analyze conformal prediction sets for the test set during inference.
    For each test image, display the image, print the prediction set, true label, model prediction, and softmax scores.
    Collect and return lists for single-label and multi-label prediction sets.
    Returns: y_cp_true, y_cp_pred, output_set, y_cp_double_class_true, y_cp_double_class_pred, y_cp_double
    """
    pred_set = []
    output_set = []
    y_cp_true = []
    y_cp_pred = []
    y_cp_double = []
    y_cp_double_class_true = []
    y_cp_double_class_pred = []
    for i in range(test_final_labels.shape[0]):
        normalized_image = X_final[i]
        if normalized_image.shape[0] == 3 and normalized_image.ndim == 3:
            normalized_image = np.transpose(normalized_image, (1, 2, 0))
        clipped_image = np.clip(normalized_image, 0, 1)
        prediction_set = test_final_smx[i] > 1 - qhat
        plt.figure()
        plt.imshow(clipped_image)
        plt.axis('off')
        plt.show()
        if sum(prediction_set) == 0:
            print(f'{150 *"_"}')
            print(f"The prediction set for image {i} is: [], and the true label is {test_final_labels[i]}, and model's prediction is {test_final_labels[i]}")
            print(f"The Softmax score for class 0: {test_final_smx[i, 0]}, and Softmax score for class 1: {test_final_smx[i, 1]}")
            print(f'{150 *"_"}')
            pred_set.append(prediction_set)
        else:
            print(f'{150 *"_"}')
            print(f"The prediction set for image {i} is: {list(label_strings[prediction_set])}, the true label is {test_final_labels[i]}, and model's prediction is {model_final_test_pred[i]}")
            print(f"The Softmax score for class 0: {test_final_smx[i, 0]}, and Softmax score for class 1: {test_final_smx[i, 1]}")
            print(f'{150 *"_"}')
            pred_set.append(prediction_set)
            if sum(prediction_set) == 1:
                y_cp_true.append(test_final_labels[i])
                y_cp_pred.append(model_final_test_pred[i])
                output_set.append(list(label_strings[prediction_set]))
            elif sum(prediction_set) > 1:
                y_cp_double_class_true.append(test_final_labels[i])
                y_cp_double_class_pred.append(model_final_test_pred[i])
                y_cp_double.append(prediction_set)
    return y_cp_true, y_cp_pred, output_set, y_cp_double_class_true, y_cp_double_class_pred, y_cp_double

def plot_alpha_ablation(alpha_test, single_pred_sets, double_pred_sets, empty_pred_sets, save_path=None):
    """
    Plot the results of the alpha ablation study: counts of single, double, and empty sets vs. alpha.
    alpha_test: array of alpha values
    single_pred_sets, double_pred_sets, empty_pred_sets: arrays of counts for each set type
    save_path: if provided, saves the figure to this path
    """
    fig, ax = plt.subplots()
    ax.plot(alpha_test, double_pred_sets, label='Double Sets')
    ax.fill_between(alpha_test, double_pred_sets, alpha=0.3)
    ax.plot(alpha_test, single_pred_sets, label='Single Sets')
    ax.fill_between(alpha_test, single_pred_sets, alpha=0.3)
    ax.plot(alpha_test, empty_pred_sets, label='Empty sets')
    ax.fill_between(alpha_test, empty_pred_sets, alpha=0.3)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid()
    ax.set_xlim((0.1, 1))
    ax.set_ylim((0, 200))
    ax.vlines(x=0.25, ymin=0, ymax=190, colors='black', linestyles='dashed', alpha=0.5)
    if save_path:
        plt.savefig(save_path, format='png', dpi=600)
    plt.show()

# Example usage (commented):
# label_strings = np.array(['no-harvest', 'harvest'])
# y_cp_true, y_cp_pred, output_set, y_cp_double_class_true, y_cp_double_class_pred, y_cp_double = analyze_conformal_prediction_sets(X_final, test_final_smx, qhat, test_final_labels, model_final_test_pred, label_strings) 
# plot_alpha_ablation(alpha_test, single_pred_sets, double_pred_sets, empty_pred_sets, save_path='ICP_sens_analysis.png') 