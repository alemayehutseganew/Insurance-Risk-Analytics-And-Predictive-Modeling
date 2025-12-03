# End-to-End Insurance Risk Analytics & Predictive Modeling

Project for 10 Academy: Artificial Intelligence Mastery — Insurance Risk Analytics (03 Dec - 09 Dec 2025).

Overview
- Use historical insurance data to discover low-risk segments and build predictive models to optimize premiums.

Repository structure
- `MachineLearningRating_v3/` - raw historical dataset tracked by DVC (`MachineLearningRating_v3.txt.dvc` keeps Git history small)
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

2. Sync DVC-tracked data (Task 2):
```powershell
# create the shared cache/remote once (sits next to the repo)
New-Item -ItemType Directory ..\dvc-storage -Force | Out-Null

# pull the raw dataset into MachineLearningRating_v3/
dvc pull
```

3. Update / add new raw data via DVC:
```powershell
copy path\to\new_file.txt MachineLearningRating_v3\MachineLearningRating_v3.txt
dvc add MachineLearningRating_v3\MachineLearningRating_v3.txt
git add MachineLearningRating_v3\MachineLearningRating_v3.txt.dvc MachineLearningRating_v3\.gitignore
git commit -m "data: refresh MachineLearningRating_v3 sample"
dvc push
```

Project workflow suggestions
- Work in feature branches: `task-1`, `task-2`, `task-3`, `task-4`.
- Commit frequently with descriptive messages.
- Use DVC to version all large data files and intermediate artifacts.

DVC reference
- Default remote `localstore` lives at `../dvc-storage` (relative to the repo root). Run `dvc remote modify localstore url <new_path>` if you prefer cloud/object storage.
- `dvc pull` restores the raw dataset. `dvc status` shows whether anything changed.
- After editing or replacing `MachineLearningRating_v3/MachineLearningRating_v3.txt`, run `dvc add ... && dvc push` so collaborators can reproduce the exact raw inputs.

Deliverables
- Interim (07 Dec 2025): merged work for Task-1 and Task-2, EDA findings, DVC setup.
- Final (09 Dec 2025): polished report (Medium-style blog post), code and notebooks.

