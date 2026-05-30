# EDGE Pass Rusher Evaluation Framework
### 2025 NFL Big Data Bowl — Baltimore Ravens Data Scientist Project

A multi-dimensional evaluation framework for EDGE pass rushers using NFL player tracking data. Classifies every rush attempt by technique (speed, power, counter, loss), computes a context-adjusted pressure rate that accounts for blocking difficulty, and produces player rankings with scheme-fit tiers.

---

## Setup

### 1. Create a virtual environment and install dependencies
```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Then in VS Code, select the `.venv` interpreter (Ctrl+Shift+P → "Python: Select Interpreter").

### 2. Get the data
Download the dataset from Kaggle:  
**https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data**

Unzip the files into the `data/` folder so the structure looks like:

```
nfl-big-data-bowl-2025/
├── data/
│   ├── games.csv
│   ├── plays.csv
│   ├── players.csv
│   ├── player_play.csv
│   ├── tracking_week_1.csv
│   ├── tracking_week_2.csv
│   │   ...
│   └── tracking_week_9.csv
├── notebooks/
└── outputs/
```

The CSV files are not committed to this repo due to file size. All outputs in `outputs/` are also excluded.

### 3. Run notebooks in order

Open each notebook in VS Code (Jupyter extension) and run all cells top to bottom:

| Notebook | What it does | Output |
|---|---|---|
| `01_data_pipeline.ipynb` | Load, filter, standardize, and clip tracking data | `outputs/tracking_clipped.parquet`, `outputs/edge_player_play.csv` |
| `02_getoff_burst.ipynb` | Get-off time and burst speed per rush attempt | `outputs/getoff_metrics.csv` |
| `03_blocker_metrics.ipynb` | Blocker separation, displacement, hip turn, blocking context | `outputs/blocker_metrics.csv` |
| `04_win_type_classification.ipynb` | Win type cascade + context-adjusted pressure rate | `outputs/win_types.csv`, `outputs/player_win_type_summary.csv` |
| `05_model_validation.ipynb` | ROC-AUC validation + player rankings with technique indices | `outputs/edge_rush_player_rankings.csv` |
| `06_visualizations.ipynb` | Technique profile, quadrant scatter, double-team rate, rankings | `outputs/figures/` |

Each notebook must be run before the next — they share intermediate outputs.

---

## Dataset

**Source:** [NFL Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data)  
**Season:** 2022 NFL season, Weeks 1–9  
**Scope:** Passing plays only (`isDropback == True`); EDGE rushers (`DE`, `OLB`) with `wasInitialPassRusher == True`  
**Tracking:** 10 frames/second player position, speed, orientation, direction
