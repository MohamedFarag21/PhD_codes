# Help: How to Use This Codebase

This project provides a modular pipeline for deep learning, uncertainty quantification, and conformal prediction on the GrowliFlower dataset. Below are step-by-step instructions and example command lines for each major workflow.

---

## 1. **Setup**
- Ensure you have all dependencies installed (see `requirements.txt` or the README).
- Prepare your data as described in the README.

---

## 2. **Training a Model**
Train a LeNet model using the provided script:

```bash
python scripts/train_model.py \
    --epochs 20 \
    --device cuda \
    --lr 0.001 \
    --output trained_model.pt
```
- `--epochs`: Number of training epochs
- `--device`: 'cpu' or 'cuda'
- `--lr`: Learning rate
- `--output`: Path to save the trained model

---

## 3. **Conformalization (Calibration & Test)**
Apply conformal prediction to a trained model:

```bash
python scripts/conformalize_model.py \
    --model_weights trained_model.pt \
    --alpha 0.1 \
    --device cuda \
    --output conformal_results.npz
```
- `--model_weights`: Path to trained model weights
- `--alpha`: Miscoverage level (1 - desired coverage)
- `--output`: Path to save results

---

## 4. **Monte Carlo Dropout Inference**
Generate MC Dropout outputs for uncertainty estimation:

```bash
python scripts/mc_inference.py \
    --model_weights my_mc_model.pt \
    --num_samples 100 \
    --device cuda \
    --output mc_outputs.npz
```
- `--num_samples`: Number of MC samples

---

## 5. **Ablation Study on Alpha (Conformal Prediction)**
Run an ablation study over different alpha values:

```bash
python scripts/ablation_alpha_conformal.py \
    --model_weights trained_model.pt \
    --alpha_points 30 \
    --alpha_min 0.01 \
    --alpha_max 0.5 \
    --device cuda \
    --output ablation_results.npz
```
- `--alpha_points`: Number of alpha values to test
- `--alpha_min`, `--alpha_max`: Range of alpha values

---

## 6. **Conformal Prediction Inference**
### **Single Image:**
```bash
python scripts/conformal_inference.py \
    --model_weights trained_model.pt \
    --qhat 0.15 \
    --image path/to/image.jpg \
    --output single_image_conformal.npz
```

### **Batch (Test Set):**
```bash
python scripts/conformal_inference.py \
    --model_weights trained_model.pt \
    --qhat 0.15 \
    --batch \
    --output batch_conformal.npz
```

### **Compute qhat from Calibration Set:**
```bash
python scripts/conformal_inference.py \
    --model_weights trained_model.pt \
    --calib_probs cal_probs.npy \
    --calib_labels cal_labels.npy \
    --alpha 0.1 \
    --image path/to/image.jpg
```

---

## 7. **General Notes**
- All scripts support `--device cpu` or `--device cuda`.
- Make sure your data is preprocessed and available as expected by the scripts.
- For custom models or dataloaders, you may need to adapt the scripts.

---

## 8. **Troubleshooting**
- If you encounter missing module errors, check your Python environment and PYTHONPATH.
- For CUDA errors, ensure your GPU drivers and CUDA toolkit are properly installed.
- If you have merge conflicts or git issues, see the README or ask for help.

---

For more details, see the README or the docstrings in each script/module. 