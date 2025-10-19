# NaT-ReX: Naturalness Assessment with Transformer-Based Reliable Explainability

**NaT-ReX (Naturalness Assessment with Transformer-Based Reliable Explainability)** is a Vision Transformer–based framework that integrates **Layer-wise Relevance Propagation (LRP)** with **uncertainty quantification** to generate **fine-grained, uncertainty-aware relevance maps** for naturalness assessment in satellite imagery.

---

## 🧠 Overview

The `nat_rex.ipynb` notebook demonstrates how **explainability** and **uncertainty estimation** can be combined to assess the reliability of model attributions for naturalness prediction. NaT-ReX identifies:
- **Where** the model focuses when assessing naturalness  
- **How confidently** it attributes specific image regions to natural or non-natural areas

This addresses limitations of earlier work (limited interpretability, lack of data-driven attribution, missing uncertainty), improving trust and applicability in real-world settings.

---

## 🧩 Architecture

<p align="center">
  <img src="rex-inference (1).png" alt="NaT-ReX Architecture" width="700"/>
</p>

**Figure:** A ViT encoder feeds two heads. The **classification head** produces relevance maps via LRP attention rollout (naturalness vs. non-naturalness). The **reconstruction head** uses MC-Dropout to estimate pixel-wise epistemic uncertainty. Joint multitask training enables uncertainty-aware, interpretable analysis.

---

## 🔧 Key Components

### 1) Vision Transformer with LRP Attention Rollout
- ViT encoder adapted for satellite imagery  
- Attention-based LRP for **pixel-level** relevance to the naturalness class  
- Preserves spatial structure for fine-grained attributions

### 2) Dual-Head Multitask Architecture
- **Classification Head:** class prediction + LRP relevance  
- **Reconstruction Head:** **Monte Carlo Dropout** reconstructions for uncertainty  
- Joint optimization balances semantic interpretability and UQ

### 3) Uncertainty Quantification
- Stochastic forward passes with **MC-Dropout**  
- **Pixel-wise uncertainty maps** from reconstruction variance  
- Produces **uncertainty-weighted relevance (ReX) maps**

### 4) ReX Score
- Combines normalized relevance and uncertainty  
- Quantifies **confidence-weighted contribution** of each pixel to naturalness  
- Supports aggregation to **class-level** (land-cover) analysis

### 5) Visualization Tools
- Relevance, uncertainty, and ReX maps  
- Side-by-side qualitative views and class-level summaries

---

## 🖥️ Working Environment

### Dependencies
`torch`, `torchvision`, `numpy`, `matplotlib`, `opencv-python`, `tifffile`, `scikit-learn`, `tqdm`

### External Dependency
- [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) (Chefer et al., 2021)  
- Set `path_to_transformer_explainability` in the notebook

---

## 📁 Data Requirements

- Satellite imagery in **TIFF** format  
- **CSV** with:
  - Image paths  
  - Binary labels (*natural* / *non-natural*)  
  - Optional land-cover segmentation masks for class-level evaluation

Recommended datasets: **AnthroProtect** and **MapInWild**.

---

## 🚀 Usage

1. Clone this repository and the Transformer-Explainability dependency  
2. Prepare your dataset as above  
3. Run the notebook sequentially to:
   - Train NaT-ReX  
   - Generate LRP attention maps  
   - Estimate pixel-wise uncertainty via MC-Dropout  
   - Compute and visualize **ReX** scores

---

## 📊 Results

NaT-ReX outputs:
- **Uncertainty-aware relevance maps** highlighting confidently natural regions  
- **ReX scores** quantifying contribution and confidence per pixel and per land-cover class

