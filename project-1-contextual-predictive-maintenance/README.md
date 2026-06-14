# Project 1: Manufacturing & Automotive - Contextual Predictive Maintenance (IoT Edge AI)

**Team:** Palak, Isha

---

## Executive Problem Statement

Predictive maintenance can drastically reduce unexpected breakdowns and lower fleet or factory maintenance costs. Most existing ML systems rely only on **internal diagnostic signals** and fail in real-world deployment - machine failures are rarely isolated. They are heavily influenced by **external factors** such as weather, factory load, or traffic density.

**Objective:** Build an advanced **contextual data fusion framework** that integrates:
- **Internal IoT telemetry** - vibration, current, temperature, and other sensor time-series
- **External environmental signals** - weather APIs, factory load, V2X logs, etc.

The goal is to accurately predict mechanical failures **before they happen** using robust ensemble modeling.

---

## Business Objectives & KPIs

| Goal | Detail |
|------|--------|
| Strategic vision | Transition from reactive "break-fix" maintenance to proactive, high-accuracy prediction |
| Primary metric | **Macro F1 score >= 0.85** |
| Robustness target | Maintain high predictive accuracy even when sensor data has **moderate noise interference** |
| Data challenge | Machine failures are **rare** - dataset is highly **imbalanced** (< 2% failure rate) |

---

## User Personas & Workflows

| Persona | Primary Need | System Interaction |
|---------|--------------|-------------------|
| **Fleet/Plant Manager** | Minimize unexpected downtime; optimize repair schedules | Reviews predictive dashboard for machines/vehicles with **> 80% probability of failing in the next 7 days** |
| **Reliability Engineer** | Understand root cause of predicted failures | Analyzes feature importance (e.g., **SHAP values**) to see whether external temperature or internal vibration drives the anomaly |

---

## Minimum Viable Product (MVP)

### 1. Contextual Data Fusion Pipeline
- Merge **high-frequency IoT sensor time-series** with **external API data** (weather, load conditions, etc.)
- Use complex **Pandas windowing operations** for temporal alignment and feature creation

### 2. Classification Pipeline
- **LightGBM** gradient boosting classifier
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied **only within training folds** during cross-validation - prevents data leakage
- Strict **5-fold stratified cross-validation**

---

## Four-Week Engineering Roadmap

### Week 1: IoT Telemetry Ingestion & Signal Processing
- Ingest industrial dataset (e.g., **AI4I Predictive Maintenance Dataset**)
- Handle massive time-series logs in Python
- Compute rolling means, standard deviations, and signal variances over operational windows
- Create baseline time-series features

### Week 2: Contextual Data Fusion & Feature Engineering
- Simulate or integrate external context (ambient temperature, load density, etc.)
- Merge external data with internal sensor telemetry using **precise timestamps**
- Run an **ablation study** proving that external features improve predictive power

### Week 3: Imbalanced Classification & LightGBM Modeling
- Set up **5-fold stratified cross-validation**
- Implement **SMOTE inside the CV loop** (training folds only) to handle < 2% failure rate
- Train **LightGBM classifier** to predict impending faults
- Optimize for **Macro F1**

### Week 4: Noise Sensitivity Analysis & Threshold Tuning
- Inject **synthetic noise** into the test dataset to characterize model robustness
- Plot **Precision-Recall curves**
- Tune decision threshold to balance **false alarms vs. missed maintenance windows**
- Document the full pipeline on GitHub

---

## Evaluation Criteria

| Criterion | Requirement |
|-----------|-------------|
| **Macro F1 score** | >= 0.85 |
| **Noise robustness** | Empirical analysis under moderate sensor noise |
| **Ablation study** | Demonstrate external/contextual features improve performance |
| **GitHub history** | Commits across all 4 weeks; no single-day bulk upload |
| **Issue tracking** | Kanban board + semantic commits referencing issues |
| **Documentation** | Full pipeline documented and reproducible |

---

## Recommended Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas |
| Dataset | AI4I Predictive Maintenance Dataset (or equivalent) |
| External context | Weather/load APIs or simulated V2X logs |
| Modeling | LightGBM, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Evaluation | Stratified K-Fold CV, Precision-Recall curves |
| Version control | GitHub Issues, Projects (Kanban), semantic commits |

---

## Key Technical Reminders

- **Never apply SMOTE globally** - only inside training folds per CV split to avoid leakage.
- **Class imbalance** is the central modeling challenge; Macro F1 is the right metric (not accuracy).
- **Threshold tuning** in Week 4 is a deployment concern - balance operational false alarms against missed failures.
- Exclude `data/` and `models/` from Git; provide download/retrain scripts instead.
- Clear notebook outputs before every commit.

---

## Deliverables Checklist

- [ ] Contextual data fusion pipeline (IoT + external APIs)
- [ ] Feature engineering with rolling window statistics
- [ ] Ablation study (with vs. without external features)
- [ ] LightGBM + SMOTE-in-CV classification pipeline
- [ ] Macro F1 >= 0.85 on evaluation set
- [ ] Noise sensitivity analysis report
- [ ] Precision-Recall curves and threshold tuning
- [ ] SHAP / feature importance for reliability engineers
- [ ] Predictive dashboard concept (machines > 80% failure risk in 7 days)
- [ ] 4 weeks of incremental GitHub commits with linked issues