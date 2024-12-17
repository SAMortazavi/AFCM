# Pipeline Safety Models and Feature Extraction

This repository implements:

1. **[DOI: 10.1109/JSEN.2021.3087537](https://doi.org/10.1109/JSEN.2021.3087537):** A model for pipeline safety early warning (PSEW) systems based on distributed optical fiber sensing and semi-supervised learning.
2. **[DOI: 10.1609/aaai.v35i17.17759](https://doi.org/10.1609/aaai.v35i17.17759):** A feature extraction framework for distributed optical fiber systems using a machine learning approach tailored for robust action recognition.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)

---

## **Introduction**

This repository provides implementations for:
1. A semi-supervised learning-based model designed to enhance pipeline safety monitoring by leveraging distributed optical fiber sensing technology.
2. A complementary feature extraction pipeline utilizing high-dimensional signal processing techniques for robust action recognition and localization in noisy environments.

The implementations utilize **PyTorch** and **PyTorch Lightning** for streamlined and scalable deep learning workflows.

---

## **Features**

1. **Model (JSEN, 2021):**
   - Employs sparse stacked autoencoders (SSAE) for unsupervised feature learning.
   - Utilizes Bi-LSTM and self-attention mechanisms to capture spatiotemporal dependencies in pipeline signals.
   - Optimized for real-time deployment with reduced latency and model size.

2. **Feature Extractor (AAAI, 2021):**
   - Extracts complementary features: high-frequency peaks and low-frequency energy signatures.
   - Leverages CNNs and Bi-LSTM layers for action recognition in distributed optical fiber signals.

3. **Performance Highlights:**
   - **Model Accuracy:** Up to 99.26% (500 Hz) and 97.20% (100 Hz) on real-world pipelines.
   - **Feature Adaptability:** Robust against environmental noise and varying hardware setups.

---

## **Requirements**

- Python 3.8+
- PyTorch 2.x
- PyTorch Lightning 2.x
- NumPy
- Matplotlib (for visualizations)
- GPU (recommended for training models)

