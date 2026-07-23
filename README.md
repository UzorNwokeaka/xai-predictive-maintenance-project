

# Explainable AI for Human-Centred Predictive Maintenance Using Industrial IoT Sensor Data

**A Case Study of Remaining Useful Life Prediction in NASA Turbofan Engines**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![XAI](https://img.shields.io/badge/Explainable%20AI-TimeSHAP-success)
![Industry 5.0](https://img.shields.io/badge/Industry-5.0-green)

---

## MSc Dissertation Project

**Programme:** MSc Data Science and Artificial Intelligence  
**Institution:** University of Suffolk, United Kingdom  
**Year:** 2026

---

## Project Overview

Predictive maintenance has become a key enabler of Industry 4.0 and Industry 5.0 by allowing organisations to predict equipment failures before they occur. However, many state-of-the-art deep learning models operate as "black boxes", making it difficult for maintenance engineers to understand why predictions are made.

This dissertation addresses this challenge by developing an **Explainable Artificial Intelligence (XAI)** framework for Remaining Useful Life (RUL) prediction using the NASA C-MAPSS FD001 turbofan engine dataset.

The project combines an **Improved Long Short-Term Memory (LSTM)** model with **TimeSHAP** and **Integrated Gradients** to produce accurate, transparent and human-centred predictive maintenance decisions.

The resulting Streamlit application transforms AI predictions into actionable maintenance recommendations through an interactive engineering dashboard.

---

# Research Objectives

The objectives of this research were to:

- Develop an accurate deep learning model for Remaining Useful Life prediction.
- Compare machine learning and deep learning approaches.
- Investigate Explainable Artificial Intelligence techniques for predictive maintenance.
- Improve transparency and trust in AI-assisted maintenance decisions.
- Develop an interactive decision-support dashboard for maintenance engineers.
- Align predictive maintenance with the principles of Industry 5.0.

---

# Dashboard Features

The application consists of six interactive pages:

### Home

- Project overview
- Research motivation
- Dashboard navigation
- Performance summary

---

### Prediction

- Engine sensor input
- Remaining Useful Life prediction
- Health status classification
- Maintenance recommendation
- Risk assessment

---

### Explainability

- TimeSHAP Feature Attribution
- Event Attribution
- Coalition Pruning
- Integrated Gradients Validation
- Critical Degradation Window Analysis
- Sensor-Time Attribution
- Feature Deletion Faithfulness

---

### Fleet Health Dashboard

- Fleet overview
- Health distribution
- Remaining Useful Life distribution
- Engines requiring immediate attention
- Fleet interpretation
- Decision support

---

### Model Comparison

Performance comparison between:

- Improved LSTM
- Improved CNN
- Improved CNN-LSTM
- Random Forest
- XGBoost
- Linear Regression

---

### About

- Project summary
- Methodology
- Research contribution
- Technology stack
- Industry 5.0 alignment
- Future work

---

# System Workflow

```
NASA C-MAPSS Dataset
            │
            ▼
     Data Pre-processing
            │
            ▼
      Feature Engineering
            │
            ▼
     Improved LSTM Model
            │
            ▼
 Remaining Useful Life Prediction
            │
            ▼
         TimeSHAP
            │
            ▼
 Integrated Gradients Validation
            │
            ▼
 Health Status Classification
            │
            ▼
 Maintenance Recommendation
            │
            ▼
 Interactive Decision Support Dashboard
```

---

# Dataset

This project uses the **NASA C-MAPSS FD001** turbofan engine degradation dataset.

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Dataset | NASA C-MAPSS FD001 |
| Engines | 100 |
| Observations | 20,631 |
| Sensors | 21 |
| Operating Conditions | 1 |
| Fault Modes | 1 |

---

# Model Performance

The final experimental evaluation produced the following results:

| Model | RMSE | MAE | R² |
|------|------:|------:|------:|
| **Improved LSTM** | **14.5466** | **10.8172** | **0.8682** |
| Random Forest Tuned | 16.7867 | 11.9994 | 0.8245 |
| Improved CNN | 16.9846 | 13.0673 | 0.8204 |
| XGBoost Tuned | 17.0384 | 11.9609 | 0.8192 |
| Random Forest Baseline | 17.1555 | 12.0183 | 0.8167 |
| XGBoost Baseline | 17.4161 | 12.3387 | 0.8111 |
| Linear Regression | 20.6005 | 16.3723 | 0.7357 |
| Improved CNN-LSTM | 31.5626 | 26.8227 | 0.3797 |

The **Improved LSTM** achieved the best predictive performance and was selected as the deployment model for the decision-support application.

---

# Explainable AI Framework

Unlike conventional predictive maintenance systems, this project integrates multiple Explainable AI techniques.

## Primary Explainability

- TimeSHAP

## Supplementary Validation

- Integrated Gradients

### Key Explainability Findings

- Critical degradation window identified between operational cycles **112–121**
- Strong agreement between explanation methods
- Feature Rank Correlation: **0.9779**
- Top-5 Feature Overlap: **80%**
- Transparent sensor-level and temporal explanations supporting engineering interpretation

---

# Technology Stack

## Machine Learning

- TensorFlow / Keras
- Scikit-learn
- XGBoost

## Explainable AI

- TimeSHAP
- Integrated Gradients
- SHAP

## Data Processing

- Pandas
- NumPy

## Visualisation

- Plotly
- Matplotlib

## Deployment

- Streamlit

---

# Repository Structure

```
.
├── assets/
├── data/
├── models/
├── pages/
├── utils/
├── Home.py
├── config.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-github-username>/<repository-name>.git
```

Navigate to the project directory:

```bash
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the Streamlit application:

```bash
streamlit run Home.py
```

---

# Research Contributions

This dissertation contributes by:

- Developing an Improved LSTM architecture for Remaining Useful Life prediction.
- Integrating TimeSHAP for direct explanation of deep learning predictions.
- Validating explanations using Integrated Gradients.
- Translating AI predictions into practical maintenance recommendations.
- Designing an interactive Explainable AI dashboard for maintenance decision support.
- Aligning predictive maintenance with Industry 5.0 principles through human-centred AI.

---

# Industry 5.0 Alignment

This research aligns with Industry 5.0 by placing maintenance engineers at the centre of the decision-making process.

Rather than replacing human expertise, the application provides transparent AI predictions and interpretable explanations that support:

- Trustworthy Artificial Intelligence
- Human-AI collaboration
- Transparent decision making
- Responsible AI adoption
- Maintenance planning
- Operational resilience

---

# Future Work

Potential extensions include:

- Live Industrial IoT sensor streaming
- Digital Twin integration
- Edge AI deployment
- Federated Learning
- Online model updating
- Multi-failure mode prediction
- Industrial-scale deployment

---

# Dashboard Screenshots

Add screenshots of the application in the `assets/` directory.

Example:

```
assets/
├── home.png
├── prediction.png
├── explainability.png
├── fleet_health.png
├── model_comparison.png
└── about.png
```

---

# Citation

If you use or reference this work, please cite:

> Nwokeaka, U. (2026). *Explainable AI for Human-Centred Predictive Maintenance Using Industrial IoT Sensor Data: A Case Study of Remaining Useful Life Prediction in NASA Turbofan Engines.* MSc Dissertation, University of Suffolk.

---

# License

This project is released under the **MIT License**.

---

# Acknowledgements

- University of Suffolk
- NASA Prognostics Center of Excellence
- TensorFlow
- Streamlit
- The open-source machine learning and Explainable AI communities

---
