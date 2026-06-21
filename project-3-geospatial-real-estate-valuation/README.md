# Project 3: Construction & Real Estate - Geospatial Valuation via Spatial Embeddings

**Team:** Tanvish, Hemit

---

## Executive Problem Statement

Traditional Automated Valuation Models (AVMs) rely heavily on **tabular features** (bedrooms, square footage) but fail to capture complex **spatial relationships** in a housing market. A property's value is deeply influenced by:
- Neighboring properties
- Nearby amenities
- Localized socio-economic factors

Simple linear regression or tree-based models cannot natively process these geo-spatial dependencies.

**Objective:** Build a state-of-the-art **Real Estate Valuation engine** using **Spatial Embeddings**, **K-Nearest Neighbor (KNN) graphs**, or **Graph Neural Networks (GNNs)** to model how surrounding properties affect a target house's price.

---

## Business Objectives & KPIs

| Goal | Detail |
|------|--------|
| Primary objective | Hyper-accurate property appraisal for real estate firms or automated lending platforms |
| Primary metric | Minimize **Mean Absolute Percentage Error (MAPE)** on holdout data |
| Proof of value | Spatial model (GNN / spatial embeddings) must **outperform** tabular baselines (XGBoost, Linear Regression) |
| Secondary metrics | RMSE (for baseline comparison in Week 2) |

---

## User Personas & Workflows

| Persona | Primary Need | System Interaction |
|---------|--------------|-------------------|
| **Real Estate Appraiser** | Data-driven, unbiased valuations based on comparable local sales | Inputs a property address; model outputs predicted price + **top 5 most influential neighboring properties** |
| **Investment Strategist** | Identify undervalued neighborhoods or "hotspots" | Reviews **spatial heatmaps** from embeddings to guide large-scale commercial acquisitions |

---

## Minimum Viable Product (MVP)

### 1. Geospatial Data Wrangling Module
- Process latitude/longitude with **GeoPandas** and **Shapely**
- Calculate **Haversine distances**
- Construct **neighborhood graphs** connecting nearby properties

### 2. Attention-Based or Graph-Based Valuation Model
- Deep learning architecture using **PyTorch** or **DGL**
- Aggregates pricing data from a property's immediate neighbors
- **Attention mechanism** to weight certain neighbors more heavily than others

---

## Four-Week Engineering Roadmap

### Week 1: Geospatial Data Acquisition & Processing
- Acquire real estate dataset with exact lat/long (e.g., **King County Housing Data**)
- Clean data; normalize extreme price outliers
- Use **GeoPandas** for geospatial processing
- Plot data on interactive maps (**Folium** or **Kepler.gl**) to inspect spatial pricing trends

### Week 2: Feature Engineering & Baseline ML
- Engineer standard tabular features (house age, distance to city center, etc.)
- Train **XGBoost regressor** as the baseline model
- Calculate baseline **MAPE** and **RMSE**
- Document limitations (e.g., poor performance in rapidly gentrifying neighborhoods)

### Week 3: Spatial Embeddings & Graph Construction
- Convert dataset into a **Graph**: each house = node; edges = **K-nearest neighbors** by physical distance
- Generate **spatial embeddings** representing localized neighborhood context
- Prepare graph structure for deep learning input

### Week 4: GNN / Attention Modeling & Geospatial Dashboard
- Train **Graph Neural Network (GNN)** or **Attention-Based Spatial model** on the constructed graph
- Compare new **MAPE** against XGBoost baseline - prove spatial dependencies add value
- Deploy a **Streamlit app** visualizing predicted price disparities on a map
- Rigorously version-control all code on GitHub

---

## Evaluation Criteria

| Criterion | Requirement |
|-----------|-------------|
| **MAPE** | Minimized on holdout set; spatial model beats XGBoost baseline |
| **Baseline comparison** | Documented XGBoost / Linear Regression vs. GNN or attention model |
| **Spatial proof** | Ablation or comparison showing graph/spatial features improve predictions |
| **Geospatial dashboard** | Streamlit app with map-based price visualization |
| **Top-K neighbors** | Model surfaces top 5 influential neighboring properties per prediction |
| **GitHub history** | Commits across all 4 weeks; no single-day bulk upload |
| **Issue tracking** | Kanban board + semantic commits referencing issues |

---

## Recommended Tech Stack

| Layer | Tools |
|-------|-------|
| Geospatial processing | GeoPandas, Shapely |
| Distance metrics | Haversine formula |
| Dataset | King County Housing Data (or equivalent with lat/long) |
| Visualization (EDA) | Folium, Kepler.gl |
| Baseline ML | XGBoost, scikit-learn |
| Deep learning | PyTorch or DGL (Deep Graph Library) |
| Graph construction | KNN-based neighborhood graphs |
| App / dashboard | Streamlit |
| Version control | GitHub Issues, Projects (Kanban), semantic commits |

---

## Key Technical Reminders

- **Week 2 baseline is mandatory** - you need XGBoost MAPE/RMSE numbers to prove the spatial model adds value in Week 4.
- **KNN graph construction** is the bridge between tabular geodata and GNN input - get edge definitions and node features right.
- **Attention weights** can directly support the "top 5 influential neighbors" appraiser workflow.
- **Outlier normalization** in Week 1 affects both baseline and spatial models - document your approach.
- Exclude `data/` and `models/` from Git; provide download/retrain scripts instead.
- Clear notebook outputs before every commit.

---

## Deliverables Checklist

- [x] Cleaned geospatial dataset with lat/long coordinates
- [x] Interactive map visualizations (Folium / Kepler.gl)
- [x] Tabular feature engineering pipeline
- [x] XGBoost baseline with documented MAPE and RMSE
- [x] KNN neighborhood graph construction
- [x] Spatial embeddings for localized context
- [x] GNN or attention-based valuation model
- [ ] MAPE comparison: spatial model vs. XGBoost baseline
- [ ] Top-5 influential neighbor explanations per prediction
- [ ] Streamlit geospatial dashboard
- [ ] Spatial heatmaps for investment strategist persona
- [ ] 4 weeks of incremental GitHub commits with linked issues

---

## How to Run Tests

To execute unit tests across the preprocessing, feature engineering, and graph construction modules, make sure you have the dependencies installed and run:

```bash
# Using pytest via python module
python -m pytest
```