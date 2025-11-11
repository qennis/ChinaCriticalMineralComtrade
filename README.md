# China IR Paper – Reproducible Research Repo

This repository contains code and data scaffolding for the *China IR Paper* project.

## Layout
```
/code                  # analysis & helpers (Python packages/modules)
/data_raw              # immutable raw data (not tracked by Git; see .gitignore)
/data_work             # intermediate, derived data (not tracked by Git; prefer LFS if needed)
/figures               # generated figures
/notes                 # scratch notes, drafts
/outputs               # final outputs (tables, slides, exports)
/paper                 # manuscript text, refs
/.github/workflows     # CI
```
## Getting started
1. Create environment
   ```bash
   conda env create -f environment.yml
   conda activate china-ir
   pre-commit install
   ```
2. Run checks
   ```bash
   make lint && make test
   ```
3. Start coding in `code/` and keep raw data in `data_raw/`.

## Conventions
- **Branching**: `main` is stable; feature branches as `feat/<topic>` or `fix/<topic>`.
- **Commits**: Conventional commits (e.g., `feat: add partner-share HHI`).
- **Data**: Avoid committing large/binary data. If necessary, use Git LFS (see below).
- **Repro**: Prefer notebooks only for exploration; put production code in `code/` with tests.

## Git LFS (optional)
If you truly must version big artifacts (e.g., `.parquet`), enable LFS:
```bash
git lfs install
git lfs track "*.parquet"
git add .gitattributes
```
