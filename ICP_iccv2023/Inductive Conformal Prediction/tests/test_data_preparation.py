import numpy as np
import os
from PIL import Image
from data_preparation import load_images

def test_load_images(tmp_path):
    # Create dummy images
    img_height, img_width = 32, 32
    n = 3
    filepaths = []
    for i in range(n):
        arr = np.random.randint(0, 255, (img_height, img_width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        fname = tmp_path / f"img{i}0.png"  # label is i
        img.save(fname)
        filepaths.append(str(fname))
    images, labels = load_images(filepaths, img_height=img_height, img_width=img_width, label_type=int)
    assert images.shape == (n, img_height, img_width, 3)
    assert labels.shape == (n,)
    assert set(labels) == {0, 1, 2} 