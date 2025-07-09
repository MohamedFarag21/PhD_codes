import os
import pandas as pd
import numpy as np
import glob
import re
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader

# Set random seed for reproducibility
SEED = 78
np.random.seed(SEED)
torch.manual_seed(SEED)
import random
random.seed(SEED)

# Data directories and parameters
main_dr = r"D:\GrowliFlower Data"  # Update this path as needed
file_no = [str(i) for i in range(1,6)]
sub_dir = ['2021_08_23', '2021_08_25', '2021_08_30', '2021_09_03']
train_pattern = r"\d{4}_\d{2}_\d{2}_Ref_Plot[1-5]_[A-Za-z]\d{1,2}_\d{1,2}_\d{1,2}$"

# 1. Load label files
df_all = pd.DataFrame()
for root, dirs, files in os.walk(main_dr):
    for i in file_no:
        for file in files:
            if file == f"2021_Ref_Plot{i}_UTM_Coordinates.txt":
                file_path = os.path.join(root, file)
                temp = pd.read_csv(file_path, header=None)
                df_all = pd.concat([df_all, temp ], axis=0, ignore_index=True)
                print(f"File {file} is found")

# 2. Split into train/val/test files
train_files = []
remain_files = []
val_files = []
test_files = []

val_df = df_all[df_all[:][4] == 'val']
test_df = df_all[df_all[:][4] == 'test']

for subdr in sub_dir:
    base_file_name = main_dr + "/" + subdr + '/'
    file_pattern = f"{base_file_name}*.tif"
    matching_files = glob.glob(file_pattern)
    for file_path in matching_files:
        if re.search(train_pattern, file_path[32:-4]) is not None:
            train_files.append(file_path)
        else:
            remain_files.append(file_path)
    for i in val_df[:][0]:
        i = subdr + i[4:]
        for j in remain_files:
            if i == j[32:-6]:
                val_files.append(j)
    for i in test_df[:][0]:
        i = subdr + i[4:]
        for j in remain_files:
            if i == j[32:-6]:
                test_files.append(j)

# 3. Load and resize images, extract labels
def load_images(files, img_height=256, img_width=256, label_type=int):
    images = np.empty((len(files), img_height, img_width, 3), dtype=np.uint8)
    labels = np.empty((len(files),), dtype=np.int64)
    for i, filepath in enumerate(files):
        img = Image.open(filepath)
        resized_image = img.resize((img_height, img_width))
        arr = np.array(resized_image)
        images[i, :, :, :] = arr
        labels[i] = label_type(filepath[-5])
    return images, labels

X_train, y_train = load_images(train_files)
X_val, y_val = load_images(val_files, label_type=str)
y_val = np.array([int(lbl) for lbl in y_val])
X_test, y_test = load_images(test_files, label_type=str)
y_test = np.array([int(lbl) for lbl in y_test])

# 4. Convert to tensors and normalize
X_train = torch.Tensor(X_train / 255.0)
X_val = torch.Tensor(X_val / 255.0)
X_test = torch.Tensor(X_test / 255.0)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)

# 5. Reshape for PyTorch (N, C, H, W)
X_train = X_train.view(-1, 3, 256, 256)
X_val = X_val.view(-1, 3, 256, 256)
X_test = X_test.view(-1, 3, 256, 256)

# 6. Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
tr_dloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
vl_dloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
ts_dloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 7. Print dataset sizes
print(f"Train set size: {len(tr_dloader.dataset)}")
print(f"Validation set size: {len(vl_dloader.dataset)}")
print(f"Test set size: {len(ts_dloader.dataset)}")

# 8. Calculate class weights for weighted loss function
label, counts = np.unique(y_train.cpu(), return_counts=True)
W = (1/counts)*(len(y_train)/2)
print(f"Class weights for loss function: {W}") 