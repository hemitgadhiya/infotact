# Team Daily Workflow Guide

Simple instructions for Palak, Isha, Tanvish, and Hemit.

---

## Who works on which issues?

| Person | Project | Issue numbers |
|--------|---------|---------------|
| **Palak** | Project 1 - Predictive Maintenance | **#1 to #12** |
| **Isha** | Project 1 - Predictive Maintenance | **#1 to #12** |
| **Tanvish** | Project 3 - Geospatial Valuation | **#13 to #24** |
| **Hemit** | Project 3 - Geospatial Valuation | **#13 to #24** |

Work on issues **in order** within your project's range. Do not skip ahead unless your teammate already closed that issue.

---

## One-time setup (do this once)

### 1. Install Git
Download from https://git-scm.com/download/win

### 2. Clone the repo (first time only)
```powershell
cd C:\Users\YourName\Desktop
git clone https://github.com/hemitgadhiya/infotact.git
cd infotact
```

### 3. Tell Git who you are (first time only)
```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

---

## Daily workflow (repeat every work day)

### Step 1 - Open PowerShell and go to the project folder
```powershell
cd "C:\path\to\infotact"
```

### Step 2 - Pull latest code from GitHub
Always do this first so you have everyone's latest changes.
```powershell
git pull
```

### Step 3 - Pick your next issue
- **Palak / Isha:** Go to https://github.com/hemitgadhiya/infotact/issues and find the next open issue labeled `project-1` (issues #1-#12).
- **Tanvish / Hemit:** Find the next open issue labeled `project-3` (issues #13-#24).

Read the issue description. That is your task for today.

### Step 4 - Open the repo in your IDE and write code
- Project 1 work goes in: `project-1-contextual-predictive-maintenance/`
- Project 3 work goes in: `project-3-geospatial-real-estate-valuation/`

**Before committing notebooks:** clear all cell outputs (Kernel -> Restart & Clear Output).

**Never commit:** data files (.csv), model weights (.pt, .h5) - they stay local only.

### Step 5 - Save your work and commit
Replace `#N` with your actual issue number (e.g. `#2`).

```powershell
git add .
git status
git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "feat: short description of what you did (fixes #N)"
```

**Commit message examples:**
```
feat: load AI4I dataset and validate schema (fixes #2)
feat: add rolling mean features for vibration signal (fixes #3)
feat: plot King County prices on Folium map (fixes #15)
```

### Step 6 - Push to GitHub
```powershell
git push
```

GitHub will automatically close the issue when your commit message contains `(fixes #N)`.

### Step 7 - Repeat
Aim for **1-2 issues per day** (or 3-5 commits per active day). Spread work across all 4 weeks.

---

## Quick reference - all commands in order

```powershell
cd "C:\path\to\infotact"
git pull
# ... do your coding work in the IDE ...
git add .
git commit --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "feat: what you did (fixes #N)"
git push
```

---

## Issue list

### Project 1 (Palak & Isha) - Issues #1 to #12

| # | Week | Task |
|---|------|------|
| 1 | 1 | Set up project scaffolding and .gitignore |
| 2 | 1 | Ingest AI4I Predictive Maintenance dataset |
| 3 | 1 | Compute rolling window sensor features |
| 4 | 2 | Integrate or simulate external context data |
| 5 | 2 | Merge IoT telemetry with external context by timestamp |
| 6 | 2 | Run ablation study for external features |
| 7 | 3 | Set up 5-fold stratified cross-validation |
| 8 | 3 | Implement SMOTE inside CV training folds only |
| 9 | 3 | Train LightGBM classifier and optimize Macro F1 |
| 10 | 4 | Inject synthetic noise and test robustness |
| 11 | 4 | Plot Precision-Recall curves and tune threshold |
| 12 | 4 | Add SHAP explainability and finalize documentation |

### Project 3 (Tanvish & Hemit) - Issues #13 to #24

| # | Week | Task |
|---|------|------|
| 13 | 1 | Set up project scaffolding and .gitignore |
| 14 | 1 | Acquire and clean King County housing dataset |
| 15 | 1 | Create interactive geospatial EDA maps |
| 16 | 2 | Engineer tabular property features |
| 17 | 2 | Train XGBoost baseline and report MAPE/RMSE |
| 18 | 2 | Document baseline limitations by neighborhood |
| 19 | 3 | Build KNN neighborhood graph with Haversine distance |
| 20 | 3 | Generate spatial embeddings for neighborhood context |
| 21 | 4 | Train GNN or attention-based spatial valuation model |
| 22 | 4 | Compare spatial model MAPE against XGBoost baseline |
| 23 | 4 | Build Streamlit geospatial valuation dashboard |
| 24 | 4 | Surface top-5 influential neighbors per prediction |

---

## Rules to remember

1. **Always `git pull` before you start** - avoids conflicts.
2. **Always reference the issue number** in your commit: `(fixes #N)`.
3. **Commit 3-5 times per active day** when you are making progress.
4. **Work steadily across 4 weeks** - do not dump everything on the last day.
5. **Coordinate with your project partner** - split issues so you are not both working on the same one.
6. **Ask for help** in the team group if you are stuck.

---

## Kanban board (set up once, then use weekly)

On GitHub: repo -> **Projects** -> create a board with columns **To Do**, **In Progress**, **Done**. Add all issues to the board. Drag issues across columns as you work. We will set this up together as a team.