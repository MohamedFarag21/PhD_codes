# Verification Checklist

This file contains a step-by-step checklist to verify that the codebase is working as expected. For each task, check off completion and record any notes or results.

---

## 1. Load the Data
- [ ] Load the dataset using the provided data preparation scripts or modules.
- **Notes/Results:**

---

## 2. Create Dataloaders
- [ ] Create PyTorch dataloaders for training, validation, and test sets.
- **Notes/Results:**

---

## 3. Build the Model
- [ ] Instantiate the model (e.g., LeNet or MC Dropout variant) using the models module.
- **Notes/Results:**

---

## 4. Train the Model and Save the Weights
- [ ] Train the model using the training script/module and save the resulting weights.
- **Notes/Results:**

---

## 5. Load Model Weights and Evaluate Performance
- [ ] Load the saved model weights and evaluate the model on the test set.
- **Notes/Results:**

---

## 6. Conformalization and Its Evaluation
- [ ] Apply conformal prediction to the trained model and evaluate empirical coverage and set size.
- **Notes/Results:**

---

## 7. MC Model and Sampling
- [ ] Instantiate and evaluate the MC Dropout model, perform MC sampling, and analyze uncertainty.
- **Notes/Results:**

---

## 8. Alpha Ablation Study
- [ ] Run the ablation study over alpha (miscoverage) values for conformal prediction using the provided script/module.
- **Notes/Results:**

---

*Add more tasks as needed to further verify the codebase or to test new features.* 