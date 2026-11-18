# Person B files — EDA then HMM

This package is designed to be copied into the existing PathSnap repository. It follows the supplied team assignment for Person B:

1. EDA notebook: dataset inventory, missing values/timestamp gaps, network size/connectivity, GPS plots, quality notes, and a Folium map showing GPS observations, sampled road network and reference route.
2. Reusable Folium function in `src/viz/`.
3. Viterbi decoder in `src/hmm/`, smoke-tested on a toy graph.

## Files in this package

- `notebooks/02_person_b_eda.ipynb` — Person B EDA notebook. The provided repository ZIP already contains `notebooks/01_eda.ipynb`; this package deliberately does not overwrite it.
- `src/viz/folium_map.py` — reusable Folium map function.
- `src/hmm/viterbi.py` — log-space Viterbi decoder and toy graph demo.
- `tests/test_viterbi.py` — small tests.
- `maps/person_b/` — destination folder for generated HTML maps.

## Copying into the repository

Extract this ZIP at the repository root, allowing the `src`, `notebooks`, `tests`, `maps`, and `report` folders to merge with existing folders. Do not replace the entire repository.

## Run from repository root (PowerShell)

```powershell
python -m pip install -r requirements.txt
python -m src.hmm.viterbi
python -m pytest tests/test_viterbi.py -q
```

Open `notebooks/02_person_b_eda.ipynb` in VS Code/Jupyter and run all cells. Raw data must be present in `data/raw/train/<8-digit-trip-id>/` and/or `data/raw/student_test/<8-digit-trip-id>/`.

The map is saved to `maps/person_b/`. The road-network layer is sampled (up to the configured edge limit) so large trip networks remain usable in a browser.

## Important integration boundary

The Viterbi decoder is a tested generic decoder, not yet the complete road-matching HMM. The next integration work is to construct emission log-probabilities and transition log-probabilities from Person A's candidate rows for each trip, then call `viterbi_decode` in GPS point order. Candidate features in the supplied repo include `point_idx`, `candidate_arc_id`, `dist_to_arc`, `bearing_diff`, `speed`, `arc_length`, `proj_pos`, and `dt_prev`.

The supplied repository snapshot has a potential interface inconsistency: `src/features/pipeline.py` imports `build_trip_features`, while `src/features/candidates.py` in that snapshot defines `build_features`. Person B's notebook avoids calling that pipeline and reads trip files through `src.data.loaders` instead. Coordinate with Person A before attempting to run the shared feature pipeline.
