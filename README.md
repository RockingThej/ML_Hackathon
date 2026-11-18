# PathSnap: GPS map matching and route reconstruction

PathSnap takes a vehicle's noisy GPS trace and works out which road arcs it actually drove on.

Pipeline:
1. **Candidate generation**: the 15 nearest road arcs for every GPS point.
2. **Classifier**: scores each candidate as P(true arc).
3. **HMM + Viterbi**: picks the most likely *connected* sequence of arcs across the whole trip.
4. **API** and **UI**: serve and visualise the result.

## Results (10 held-out test trips, 19,133 labelled GPS points)

**Classifiers**, from `results/model_comparison.csv`. Miss-rate is the share of points whose top candidate isn't the true arc; lower is better.

| Model | Miss-rate | Accuracy |
|---|---|---|
| PyTorch MLP (listwise loss) | **0.1868** | **81.3%** |
| **XGBoost (GPU), final model** | 0.1890 | 81.1% |
| Random Forest | 0.1911 | 80.9% |
| LightGBM | 0.1925 | 80.7% |
| Gaussian Naive Bayes | 0.2288 | 77.1% |
| KNN | 0.2519 | 74.8% |
| Logistic Regression | 0.3263 | 67.4% |

**Route reconstruction**, from `results/hmm_ml_comparison.csv`:

| Method | Accuracy |
|---|---|
| **HMM with XGBoost emissions** | **82.4%** (better than the classifier alone on 8 of 10 trips) |
| XGBoost alone (top-1 per point) | 81.1% |
| HMM with hand-made Gaussian emissions | 65.1% |

XGBoost is the final model because the HMM is built on its scores. The MLP is the best single classifier, but the gap is small (0.2 percentage points). MLP loss curves are in `results/plots/`.

## Installation

Python 3.10+.

```bash
pip install -r requirements.txt
```

For GPU training, install PyTorch with CUDA (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu126`).

## Data setup

Raw data is git-ignored. Place it like this:
```
data/raw/train/<8-digit trip id>/<id>.track  .nodes  .arcs  .route
data/raw/student_test/<8-digit trip id>/...
```
Build the feature tables, labels and trip-level CV folds:
```bash
python scripts/build_dataset.py
```
This writes `data/processed/features_train.parquet`, `features_test.parquet`, `cv_folds.parquet`, `point_truth.parquet` and `candidate_miss_rate.csv`.

### Feature Rationale

| Feature | Description |
|---|---|
| `dist_to_arc` | Strongest cue: true road is usually nearest; metres in per-trip UTM |
| `dist_rank` | Relative nearness, comparable across dense/sparse networks |
| `bearing_diff` | Travel heading vs arc direction (0-180 deg); separates parallel and opposite-direction roads. 90 when heading unknown |
| `heading_valid` | Flag: 0 when the vehicle moved <1 m so heading is unreliable |
| `speed` | Low speed = noisy GPS / stop, bearing less trustworthy |
| `arc_length` | Short arcs at junctions behave differently from long ones |
| `proj_pos` | Where along the arc the point projects (0-1): continuity and junction cues |
| `dt_prev` | Time gap since previous point (not always 1 s), larger gap = more uncertainty |

Labels are assigned via monotone DP matching of points to `.route` arcs (min total distance). Points with best route arc >50 m away have no truth and are excluded.

## Training

| Step | Command | Output |
|---|---|---|
| Baseline models (LR, NB, KNN, RF, LightGBM) | `python run_experiments.py --save_models` | rows in `results/model_comparison.csv`, `artifacts/*.joblib` |
| PyTorch MLP | `python train_mlp.py` | `artifacts/mlp_listwise.joblib`, loss curve in `results/plots/` |
| XGBoost on GPU | `python train_xgb_gpu.py` | `artifacts/xgb.joblib`, `data/processed/scores_xgb_test.parquet` |
| Bayesian tuning (Optuna) | `python tune_optuna.py --model xgb --trials 60` | `artifacts/best_model.joblib`, `results/optuna_*` |
| HMM evaluation on all test trips | `python evaluate_hmm_ml.py --scores data/processed/scores_xgb_test.parquet` | `results/hmm_ml_comparison.csv` |
| Tests | `python -m pytest tests -q` | |

Every script uses the trip-level folds in `cv_folds.parquet`, so no trip is ever in both training and validation.

## Running the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
The API loads the first model it finds in `artifacts/`, in this order: `best_model` → `xgb` → `mlp_listwise` → `lgbm` → `rf` → `logreg`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | status and loaded model |
| `GET /trips` | available train and test trips |
| `POST /predict` `{"trip_id": 7, "top_k": 15, "use_hmm": true}` | classifier route, HMM route and server-side accuracy |
| `GET /trips/{id}/track` | raw GPS points |
| `POST /trips/{id}/arcs` `{"arc_ids": [...]}` | arc geometry for drawing the route |

Ground truth never leaves the server; only accuracy numbers are returned.

## Running the UI

```bash
streamlit run ui/app.py
```
Enter the API URL in the sidebar, choose a trip and click **Run map matching**. The map shows the raw GPS trace (dashed blue), the classifier route (orange) and the HMM route (green). The second tab shows the model leaderboard and the training plots.

## Deployment (two laptops)

| Laptop 1: API | Laptop 2: UI |
|---|---|
| Needs `data/raw/` and `artifacts/best_model.joblib` | Needs only the repo |
| `uvicorn api.main:app --host 0.0.0.0 --port 8000` | `streamlit run ui/app.py` |
| Find its IP with `ipconfig` (Windows) or `ip a` (Linux) | Put `http://<laptop-1-ip>:8000` in the sidebar |

Both laptops must be on the same network, and the firewall on Laptop 1 must allow port 8000.

## Repository layout

```
api/                       FastAPI backend
src/data/                  loaders, projection to metric CRS
src/features/              candidate generation, features, labels
src/cv/                    trip-grouped CV splits
src/models/                MLP + shared helpers
src/hmm/                   emissions, transitions, Viterbi, matcher
src/route_reconstruction/  road graph
src/viz/                   Folium maps
src/evaluation/            HMM evaluation
ui/                        Streamlit UI
scripts/                   dataset build, EDA
notebooks/                 EDA notebook
results/                   leaderboard, HMM comparison, plots
maps/                      generated HTML maps
```
