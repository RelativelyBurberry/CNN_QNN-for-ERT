# Comparative Analysis of CNN and QNN-Hybrid Architectures for Electrical Resistivity Prediction

This repository presents a **comparative study between classical Convolutional Neural Networks (CNNs) and Hybrid Quantum Neural Networks (QNNs)** for predicting **electrical resistivity** of materials using **synthetically generated Electrical Resistivity Tomography (ERT) data**.

The project explores whether **quantum-enhanced machine learning** can match or outperform classical deep learning models in solving **inverse problems** commonly encountered in geophysics and material science.

---

## 📌 Overview

Electrical resistivity prediction is a challenging inverse problem due to:
- High-dimensional dependencies  
- Noise sensitivity  
- Non-linear spatial relationships  

Traditional numerical inversion techniques are computationally expensive and poorly scalable. This project investigates:
- A **spatial-attention-based CNN**
- A **hybrid Quantum–Classical Neural Network (QNN)**

Both models are trained and evaluated on the **same synthetically generated ERT dataset**, enabling a fair and controlled comparison.

---

## 🧠 Problem Statement

> Can hybrid quantum neural networks provide comparable or improved performance over classical CNNs for resistivity prediction — even when executed on quantum simulators?

---

## 🏗️ System Architecture

### 1️⃣ Data Generation Pipeline

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/028e2a0f-50cb-4be7-8b27-947206b20bf4" />

### 2️⃣ CNN Architecture (Classical)

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/eedd8ce6-aef5-4d06-8d7e-2e9f8ba456a6" />

### 3️⃣ Quantum Neural Network Architecture
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/884f4400-daeb-461a-96ec-ca22493a4143" />

### 4️⃣ Hybrid QNN Architecture

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/3cd9df9a-6f94-4e62-8f05-64a0a089ca50" />


---

## ✨ Key Features
🔹 Physics-inspired synthetic ERT data generation  <br>
🔹 Polynomial feature engineering for spatial relations   <br>
🔹 Log-scaled resistivity modeling (geophysical best practice)   <br>
🔹 Spatial attention mechanism in CNN   <br>
🔹 Hybrid quantum-classical learning pipeline   <br>
🔹 Ensemble learning for robustness   <br>
🔹 Detailed evaluation using regression metrics   <br>

---

## ⚙️ Functionalities

- Generate realistic multi-anomaly ERT datasets
- Train CNN and QNN models on identical data
- Perform fair metric-based comparison
- Visualize:
  - Training vs validation loss
  - Predicted vs actual resistivity
  - Residual distributions
- Export reproducible resul ts

---

## 🧰 Tech Stack

### Programming & ML
- Python 3.10+
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

### Quantum Computing
- PennyLane
- Qiskit (hardware-compatible)

### Geophysical Simulation
- PyGIMLi (ERT forward modeling)

---

## 📂 Repository Structure
├── data_gen.py # Synthetic ERT data generation <br>
├── cnn.py # Spatial Attention CNN model <br>
├── qnn.py # Hybrid Quantum Neural Network <br>
└── README.md <br>


---

## 🔬 How We Built It

1. **Synthetic Data Creation**
   - Designed layered subsurface with embedded anomalies
   - Simulated dipole–dipole ERT surveys
   - Added realistic noise and filtering <br>
    <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/5f3921ca-078d-46c8-9fb6-5e5ca078e16a" />


2. **Feature Engineering**
   - Electrode positions & spacings
   - Pseudo-depth & geometric factors
   - Polynomial interaction features

3. **Model Design**
   - CNN: Attention + dilation + residual learning
   - Hybrid-QNN: CNN feature extractor + quantum circuit

4. **Training Strategy**
   - Log-transformed targets
   - Huber loss for robustness
   - One-cycle learning rate scheduling
   - Early stopping

5. **Evaluation**
   - RMSE, MAE, MSE, R²
   - Visual diagnostics

---

## 📊 Results & Comparison

| Metric | CNN | QNN |
|------|-----|-----|
| R² Score | **0.8910** | 0.8862 |
| RMSE | **7.5094** | 7.6745 |
| MSE | **56.39** | 58.89 |
| MAE | 3.8923 | **3.8785** |

### Observations
- CNN shows slightly better **global accuracy**
- QNN achieves **lower MAE**, indicating better local anomaly handling
- Performance gap is minimal despite quantum simulation overhead

> **Note:** QNNs were executed on quantum simulators due to limited access to real hardware.

---

## 📈 Result Visualizations

- Training vs Validation Loss
- Predicted vs Actual Resistivity
- Residual Error Analysis

###  CNN  
<img width="1700" height="560" alt="image" src="https://github.com/user-attachments/assets/d2e584d5-5067-4e2b-a553-61e75f7800fd" />

### Hybrid QNN 
<img width="1700" height="560" alt="image" src="https://github.com/user-attachments/assets/2df14e00-53c9-423f-838e-ef71e88a49b1" />


These confirm:
- Stable convergence
- Minimal bias
- Strong generalization

---

## 🧠 Key Learnings

- CNNs remain strong baselines for spatial inverse problems
- Quantum layers can integrate meaningfully with classical models
- Hybrid QNNs are **already competitive** despite hardware limitations
- Data preprocessing is as critical as model architecture
- Scientific ML benefits from physics-informed data generation

---

## 📚 Literature & References

- Liu et al., *ERS-InvNet*, IEEE TGRS (2020)
- Vu & Jardani, *CNN-3D-ERT*, GJI (2021)
- Li et al., *VD-Net*, IEEE TIM (2021)
- Aleardi et al., *CNN for ERT*, Politecnico di Milano (2024)
- Schuld & Petruccione, *Machine Learning with Quantum Computers*

(Full reference list available in project report)

---

## 🚀 Future Work

- Deploy QNNs on real quantum hardware
- Expand to 3D resistivity inversion
- Add uncertainty quantification
- Explore deeper quantum circuits
- Apply framework to other material properties

---

## 👥 Team

Developed collaboratively by an 8-member interdisciplinary team as part of an academic physics research project.
- **Team Leader**: <a href = "https://www.linkedin.com/in/purushotham-reddy-812b33312/"> **Yalluru Purushotham Reddy** </a> - Hybrid CNN–QNN architecture design, integration and team co-ordination
-  <a href = "https://www.linkedin.com/in/nipun-anand-saxena/">**Nipun Saxena** </a> — CNN model design, training, and comparative analysis  
- <a href = "https://www.linkedin.com/in/saswata-bastia/">**Saswata Bastia** </a> — Data preprocessing and feature engineering  
- <a href = "https://www.linkedin.com/in/caleb-kurian-george/"> **Caleb Kurian George** </a> — ERT physics modeling and synthetic data simulation  
- Abhinav Saikumar — Implementation support and debugging  
- Pranav — General assistance  
- Atharva — General assistance

Under the guidance of <a href = "https://www.linkedin.com/in/divya-bharathi-943b09120/"> Dr. Korlepara Divya Bharathi </a> VIT Chennai Assistant Professor Grade II

---

## 📜 License

This project is intended for academic and research use.

