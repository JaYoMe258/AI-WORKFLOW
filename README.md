<<<<<<< HEAD
# AI-WORKFLOW
🎓 Week 5 PLP Assignment on AI Development Workflow. Includes student dropout and hospital readmission prediction models, preprocessing pipeline, and evaluation reports.
=======
# 🧠 AI Development Workflow – Assignment Repository

This repository contains the implementation for the **PLP Week 5 Assignment: Understanding the AI Development Workflow**, which explores the end-to-end pipeline for building and deploying machine learning solutions using simulated real-world problems.

---

## 📁 Structure

```bash
ai-workflow-assignment/
├── student_dropout_predictor.py           # Predicts student dropout using Random Forest
├── hospital_readmission_model.py         # Predicts 30-day patient readmission risk
├── requirements.txt                      # Project dependencies
├── utils/
│   └── preprocessing_pipeline.py         # (Optional) Common preprocessing logic
└── README.md                             # Assignment overview
```

---

## 🚀 Project Overview

This project demonstrates how to apply the AI development lifecycle, from problem definition to deployment, using two practical scenarios:

### 🧑‍🎓 Student Dropout Prediction
- **Goal:** Predict which students are at risk of dropping out based on behavioral metrics.
- **Model Used:** Random Forest
- **Evaluation:** Confusion matrix, F1-score

### 🏥 Hospital Readmission Risk
- **Goal:** Predict whether a patient will be readmitted within 30 days of discharge.
- **Model Used:** Random Forest
- **Evaluation:** Precision, Recall, Confusion Matrix

---

## 📊 Evaluation Metrics

- **Confusion Matrix**: Visual validation of prediction performance
- **F1 Score**: Balances precision and recall for dropout prediction
- **Precision & Recall**: Critical metrics in healthcare to avoid false positives/negatives

---

## 🛠 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧩 Optional Extension
- Utility functions for scaling, encoding, and cleaning data can be stored in `utils/preprocessing_pipeline.py`.
>>>>>>> 0075519 (JY)
