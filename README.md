# Infotact DS/ML Internship - Team Repository

This repository contains work for the **Infotact Technical Internship Program** (Advanced Data Science & Machine Learning). Our team of four is executing **two** of the three available projects in parallel.

---

## Projects & Team Assignments

| Project | Domain | Folder | Team Members |
|---------|--------|--------|--------------|
| **Project 1** | Manufacturing & Automotive - Contextual Predictive Maintenance (IoT Edge AI) | [project-1-contextual-predictive-maintenance/](project-1-contextual-predictive-maintenance/) | **Palak**, **Isha** |
| **Project 3** | Construction & Real Estate - Geospatial Valuation via Spatial Embeddings | [project-3-geospatial-real-estate-valuation/](project-3-geospatial-real-estate-valuation/) | **Tanvish**, **Hemit** |

> **Note:** Project 2 (Travel & Hospitality - Reinforcement Learning for Dynamic Pricing) is **not** in scope for this team.

Each project folder contains a detailed README with problem statement, MVP specs, 4-week roadmap, evaluation criteria, and technical stack.

---

## General Guidelines

### Evaluation Requirements (from DSML Project Spec)

- **Evaluation is only performed if there are GitHub commits and contributions across all 4 weeks.** A single bulk upload on the final day (e.g., one massive notebook with everything) results in **immediate disqualification**.
- Machine learning is assessed as an **iterative, documented process** - not just the final trained model.
- Deliverables must demonstrate **steady week-over-week experimentation**, not a compressed commit history.

### GitHub & Version Control (Mandatory)

1. **GitHub Projects (Kanban board)**
   - Create a Kanban board at the start of Week 1: **To Do -> In Progress -> Done**.
   - Break the 4-week roadmap into individual **GitHub Issues** (e.g., "Code Haversine distance function", "Implement SMOTE inside CV folds").
   - GitHub timestamps issue creation, movement, and closure - managers audit this for weekly cadence.

2. **Commit frequency & semantic messages**
   - Commit frequently: **3-5 times per active development day**.
   - Every commit must reference the GitHub Issue it addresses.
   - **Format:** `feat: implement SMOTE inside cross-validation folds (fixes #4)` or `model: tune LightGBM hyperparameters (fixes #6)`.

3. **Notebooks & large files**
   - **Before committing notebooks:** clear all cell outputs (Kernel -> Restart & Clear Output) or use a pre-commit hook like `nbstripout`.
   - **Do not push** raw `.csv` files or trained model weights (`.pt`, `.h5`) to GitHub.
   - Use `.gitignore` from Day 1 to exclude `data/` and `models/` directories.
   - Track large artifacts externally (e.g., DVC) or provide scripts to re-download data and re-train from scratch.

### Collaboration Guidelines (from Project Instructions)

- **Mutual respect:** Treat all colleagues with respect, dignity, and kindness.
- **Professional behavior:** Maintain professionalism; avoid harassment, bullying, or inappropriate conduct.
- **Active engagement:** Contribute ideas, volunteer for tasks, and participate in discussions.
- **Responsibility:** Own your portion of the work; meet deadlines and deliver quality output.
- **Open communication:** Share progress updates, ask for help when needed, and support teammates.
- **Conflict resolution:** Resolve disagreements through dialogue and compromise; escalate to a supervisor or mentor if needed.
- **Accountability:** Non-compliance can lead to disciplinary action, up to removal from the internship program.
- **Reporting:** Report misconduct to a supervisor or designated contact; reports are treated confidentially.

**Support contact:** support@infotact.in

---

## Repository Structure

```
infotact/
├── README.md
├── project-1-contextual-predictive-maintenance/
│   └── README.md
├── project-3-geospatial-real-estate-valuation/
│   └── README.md
├── DSML Project.pdf
└── PROJECT_INSTRUCTIONS .pdf
```

---

## Quick Links

- [Project 1 README](project-1-contextual-predictive-maintenance/README.md)
- [Project 3 README](project-3-geospatial-real-estate-valuation/README.md)

---

## Team Workflow

See [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md) for simple daily Git commands and the full issue list (#1-#24).
