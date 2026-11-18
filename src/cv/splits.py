import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def create_trip_grouped_cv_splits(trip_ids: List[int], n_splits: int=5, random_state: int=42) -> List[Dict[str, Any]]:
    unique_trips = np.array(sorted(list(set(trip_ids))))
    dummy_y = np.zeros(len(unique_trips))
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(unique_trips, dummy_y, groups=unique_trips)):
        train_trips = unique_trips[train_idx].tolist()
        val_trips = unique_trips[val_idx].tolist()
        assert set(train_trips).isdisjoint(set(val_trips)), f'Leakage detected in fold {fold_idx}!'
        folds.append({'fold': fold_idx, 'train_trip_ids': train_trips, 'val_trip_ids': val_trips, 'num_train_trips': len(train_trips), 'num_val_trips': len(val_trips)})
    return folds

def save_cv_folds(folds: List[Dict[str, Any]], output_dir: str='data/processed') -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'cv_folds.json')
    csv_path = os.path.join(output_dir, 'cv_folds.csv')
    with open(json_path, 'w') as f:
        json.dump(folds, f, indent=2)
    records = []
    for f in folds:
        fold_num = f['fold']
        for tid in f['val_trip_ids']:
            records.append({'trip_id': int(tid), 'val_fold': int(fold_num)})
    df_folds = pd.DataFrame(records).sort_values('trip_id').reset_index(drop=True)
    df_folds.to_csv(csv_path, index=False)
    return (json_path, csv_path)
