# Research-
I include here the paper's code and the results.

## How to Use This Codebase

For a complete guide to training, conformalization, MC Dropout, ablation, and inference, see [`help_me.md`](./help_me.md).

### Main Scripts
- `scripts/train_model.py`: Train a LeNet model from the command line.
- `scripts/conformalize_model.py`: Apply conformal prediction to a trained model.
- `scripts/mc_inference.py`: Run Monte Carlo Dropout inference for uncertainty estimation.
- `scripts/ablation_alpha_conformal.py`: Perform an ablation study over alpha (miscoverage) for conformal prediction.
- `scripts/conformal_inference.py`: Run conformal prediction inference on a single image or a batch.

### Example Usage
See `help_me.md` for example command lines and argument details for each script.

---

**Related Paper:** This code is associated with the following publication: [Inductive Conformal Prediction for Harvest-Readiness Classification of Cauliflower Plants: A Comparative Study of Uncertainty Quantification Methods](https://ieeexplore.ieee.org/document/10350846)
