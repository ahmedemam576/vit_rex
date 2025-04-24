# T-REX-Wild: Transformer Relevance & Explainability with Uncertainty

**T-REX (Transformer Relevance & EXplainability)** is a framework for interpreting Vision Transformer (ViT) predictions on satellite imagery while quantifying uncertainty.

---

## 🧠 Overview

The `attentionLRP.ipynb` notebook demonstrates how to integrate pixel-wise explanations with uncertainty quantification to create a comprehensive understanding of model predictions.

This approach helps identify:
- **What** regions influenced a model's prediction
- **How confident** the model is about different parts of the image

---

## 🔧 Key Components

### 1. ViT with Layer-wise Relevance Propagation (LRP)
- Modified ViT architecture with support for LRP-based visualizations
- Implements attention-based relevance propagation for pixel-level explanation maps
- Fine-tunes a pretrained ViT model on satellite imagery classification tasks

### 2. Custom Dataset for Satellite Imagery
- Loads satellite images from TIFF files with multiple spectral channels
- Normalizes and preprocesses images for ViT input
- Supports training/testing splits via CSV configuration

### 3. Uncertainty Quantification
- **Reconstruction-based Uncertainty**: Uses a decoder head to reconstruct input images
- **Monte Carlo Dropout**: Enables dropout during inference for stochastic predictions
- **Entropy-based Uncertainty**: Computes pixel-wise entropy over MC samples

### 4. Visualization Tools
- Generates attention and relevance heatmaps
- Creates uncertainty maps highlighting regions of uncertainty
- Provides side-by-side visual comparison of different uncertainty estimation methods

---

## 🖥️ Working Environment

### 🧩 Dependencies
- `torch`, `torchvision`
- `numpy`, `matplotlib`, `opencv-python`
- `tifffile` (for TIFF satellite imagery)
- `scikit-learn` (for evaluation metrics)
- `tqdm` (for progress bars)

### 📦 External Dependencies
- Requires the [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) repository
- Set the `path_to_transformer_explainability` in the notebook

---

## 📁 Data Requirements

- Satellite imagery in **TIFF** format
- A **CSV** file with:
  - File paths to images
  - Corresponding class labels
- Default setup assumes 3-channel spectral imagery

---

## 🚀 Usage

1. Clone this repository and the [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) dependency
2. Prepare your dataset as described above
3. Run the notebook cells sequentially to:
   - Train the ViT-LRP model
   - Generate LRP explanation maps
   - Train the reconstruction decoder
   - Compute uncertainty maps
   - Visualize and compare uncertainty estimation methods

---

## 📊 Results

The notebook demonstrates how to combine:
- **LRP attribution maps**
- **Uncertainty estimates**

...to provide a complete, interpretable view of ViT predictions on satellite imagery. This is especially valuable for remote sensing tasks, where **model reliability and transparency** are crucial.

---
