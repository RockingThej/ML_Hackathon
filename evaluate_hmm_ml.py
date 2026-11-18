import argparse
import numpy as np
import pandas as pd
from src.data.loaders import load_trip
from src.features.candidates import build_features, prepare
from src.route_reconstruction.road_graph import RoadGraph
from src.hmm.viterbi import viterbi_decode_variable
from src.hmm.transitions import build_transition_log_probs
from src.hmm.matcher import build_emission_log_probs
from src.hmm.ml_emissions import load_scores, build_emission_log_probs_ml

def decode(groups, emissions, transitions):
    idx, _ = viterbi_decode_variable(emissions, transitions)
    return pd.DataFrame({'point_idx': [int(g['point_idx'].iloc[0]) for g in groups], 'predicted_arc_id': [int(groups[t].iloc[int(i)]['candidate_arc_id']) for t, i in enumerate(idx)]})

def acc(pred, truth):
    m = truth.merge(pred, on='point_idx', how='inner')
    m = m[m['has_truth']]
    return (int((m['true_arc_id'] == m['predicted_arc_id']).sum()), len(m))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='data/raw/student_test')
ap.add_argument('--scores', required=True)
ap.add_argument('--truth', default='data/processed/point_truth.parquet')
ap.add_argument('--weights', type=float, nargs='+', default=[1.0])
ap.add_argument('--trips', type=int, nargs='*', default=None)
ap.add_argument('--out', default='results/hmm_ml_comparison.csv')
a = ap.parse_args()
scores = load_scores(a.scores)
truth_all = pd.read_parquet(a.truth)
trips = a.trips or sorted((int(t) for t in scores['trip_id'].unique()))
rows = []
for n_done, tid in enumerate(trips, 1):
    print(f'\n=== trip {tid} ({n_done}/{len(trips)}) ===', flush=True)
    truth = truth_all[truth_all.trip_id == tid][['point_idx', 'true_arc_id', 'has_truth']]
    trip = load_trip(a.root, tid)
    features = build_features(trip, K=15, with_labels=False)
    xy, _, _, _, _ = prepare(trip)
    graph = RoadGraph(trip)
    transitions = build_transition_log_probs(features, xy, graph, beta=50.0)
    groups = [g.sort_values('dist_rank').reset_index(drop=True) for _, g in features.groupby('point_idx', sort=True)]
    gaussian = build_emission_log_probs(features)
    c, n = acc(decode(groups, gaussian, transitions), truth)
    rows.append(dict(trip_id=tid, method='gaussian_hmm', correct=c, n=n))
    top = scores[scores.trip_id == tid].sort_values('point_prob', ascending=False).drop_duplicates('point_idx').rename(columns={'candidate_arc_id': 'predicted_arc_id'})
    c, n = acc(top[['point_idx', 'predicted_arc_id']], truth)
    rows.append(dict(trip_id=tid, method='ml_top1', correct=c, n=n))
    for w in a.weights:
        em = build_emission_log_probs_ml(features, scores, tid, weight=w, fallback=gaussian)
        c, n = acc(decode(groups, em, transitions), truth)
        rows.append(dict(trip_id=tid, method=f'ml_hmm_w{w}', correct=c, n=n))
    last = pd.DataFrame(rows)
    print(last[last.trip_id == tid].assign(acc=lambda d: d.correct / d.n)[['method', 'correct', 'n', 'acc']].to_string(index=False), flush=True)
    last.to_csv(a.out, index=False)
df = pd.DataFrame(rows)
summary = df.groupby('method')[['correct', 'n']].sum()
summary['accuracy'] = summary.correct / summary.n
print('\n===== OVERALL (all trips) =====')
print(summary.sort_values('accuracy', ascending=False))
print('saved', a.out)
