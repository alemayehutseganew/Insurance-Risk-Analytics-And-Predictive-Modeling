# End-to-End Insurance Risk Analytics & Predictive Modeling

Project for 10 Academy: Artificial Intelligence Mastery — Insurance Risk Analytics (03 Dec - 09 Dec 2025).

Overview
- Use historical insurance data to discover low-risk segments and build predictive models to optimize premiums.

Repository structure
- `data/` - (not tracked) place raw CSV datasets here. Use DVC to version large data files.
- `notebooks/` - exploration and analysis notebooks (EDA, hypothesis testing, modeling)
- `src/` - reusable Python modules and scripts
- `scripts/` - helper scripts (DVC setup, smoke tests)
- `.github/workflows/` - CI workflows
- `requirements.txt` - Python dependencies

Quickstart (PowerShell)
1. Create a virtual env and install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. (Optional) Initialize DVC and add a local remote:
```powershell
pip install dvc
dvc init
mkdir data_storage
dvc remote add -d localstorage ./data_storage
```

3. Add your dataset and track with DVC:
```powershell
mkdir data
copy path\to\your\dataset.csv data\dataset.csv
dvc add data\dataset.csv
git add data\dataset.csv.dvc .gitignore
git commit -m "chore: add raw dataset via DVC"
dvc push
```

Project workflow suggestions
- Work in feature branches: `task-1`, `task-2`, `task-3`, `task-4`.
- Commit frequently with descriptive messages.
- Use DVC to version all large data files and intermediate artifacts.

Deliverables
- Interim (07 Dec 2025): merged work for Task-1 and Task-2, EDA findings, DVC setup.
- Final (09 Dec 2025): polished report (Medium-style blog post), code and notebooks.

Contact / Team
- Facilitators: Kerod, Mahbubah, Filimon
