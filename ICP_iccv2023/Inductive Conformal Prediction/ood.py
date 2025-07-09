import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def add_black_box(image, box_size):
    """
    Add a black box of size box_size x box_size to the center of the image.
    image: numpy array (H, W, C)
    Returns a new numpy array with the black box applied.
    """
    h, w = image.shape[:2]
    center_h, center_w = h // 2, w // 2
    half_box_size = box_size // 2
    modified_image = np.copy(image)
    top = center_h - half_box_size
    bottom = center_h + half_box_size
    left = center_w - half_box_size
    right = center_w + half_box_size
    modified_image[top:bottom, left:right, :] = 0
    return modified_image

def mc_dropout_single_sample(model, image_tensor, device, num_samples=1000):
    """
    Perform MC Dropout sampling for a single image tensor (C, H, W).
    Returns a numpy array of shape (num_samples, num_classes) with softmax probabilities.
    """
    import torch.nn as nn
    model.eval()
    mc_smx_one_sample = []
    for i in range(num_samples):
        # Unfreeze dropout layers
        for module in model.children():
            for child in module.children():
                if isinstance(child, nn.Dropout2d):
                    child.train(True)
        if hasattr(model, 'drop'):
            model.drop.train(True)
        with torch.no_grad():
            y_hat = model(torch.unsqueeze(image_tensor, 0).to(device))
            y_hat = torch.exp(y_hat)
            mc_smx_one_sample.append(y_hat.cpu())
    mc_smx_one = torch.cat(mc_smx_one_sample, dim=0)
    return mc_smx_one.numpy()

def plot_mc_histograms(mc_probs, mc_probs_mod, class_idx=1):
    """
    Plot probability distributions before and after occlusion for a given class index.
    """
    sns.set(style="ticks")
    plt.figure(dpi=600)
    bins = np.linspace(min(mc_probs[:, class_idx]), max(mc_probs[:, class_idx]), 25)
    sns.histplot(np.array(mc_probs[:, class_idx]), kde=True, bins=bins)
    plt.axvline(x=np.array(mc_probs[:, class_idx]).mean(), color='red', linestyle='dashed')
    bins_mod = np.linspace(min(mc_probs_mod[:, class_idx]), max(mc_probs_mod[:, class_idx]), 25)
    sns.histplot(np.array(mc_probs_mod[:, class_idx]), kde=True, bins=bins_mod)
    plt.axvline(x=np.array(mc_probs_mod[:, class_idx]).mean(), color='black', linestyle='dashed')
    plt.xlabel(r'Probability $p$')
    plt.ylabel('Count')
    plt.legend([
        r'$P(\mathrm{y}=1| {X}_{test})$',
        r'Mean ($\mu$)',
        r'$P_{modified}(\mathrm{y}=1|X_{test})$',
        r'Mean ($\mu_{modified}$)'
    ])
    plt.grid()
    plt.show()

# Example usage (commented):
# img = X_test[172].view(256, 256, 3).numpy()
# box_size = 100
# modified_image = add_black_box(img, box_size)
# clipped_img = np.clip(modified_image, 0, 1)
# plt.imshow(clipped_img)
# modified_image_tensor = torch.Tensor(modified_image).view(3, 256, 256)
# orig_image_tensor = X_test[172]
# mc_probs = mc_dropout_single_sample(model_mc, orig_image_tensor, device, num_samples=1000)
# mc_probs_mod = mc_dropout_single_sample(model_mc, modified_image_tensor, device, num_samples=1000)
# plot_mc_histograms(mc_probs, mc_probs_mod, class_idx=1) 