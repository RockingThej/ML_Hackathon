import numpy as np
import pandas as pd
from src.hmm.ml_emissions import build_emission_log_probs_ml

def _toy():
    feats = pd.DataFrame({'point_idx': [0, 0, 0, 1, 1, 1], 'candidate_arc_id': [10, 11, 12, 11, 12, 13], 'dist_rank': [0, 1, 2, 0, 1, 2]})
    scores = pd.DataFrame({'trip_id': 5, 'point_idx': [0, 0, 0, 1, 1, 1], 'candidate_arc_id': [12, 10, 11, 13, 11, 12], 'point_prob': [0.7, 0.2, 0.1, 0.6, 0.3, 0.1]})
    return (feats, scores)

def test_alignment_and_normalisation():
    feats, scores = _toy()
    em = build_emission_log_probs_ml(feats, scores, 5)
    assert np.allclose(em[0], np.log([0.2, 0.1, 0.7]))
    assert np.allclose(em[1], np.log([0.3, 0.1, 0.6]))
    assert all((np.isclose(np.exp(e).sum(), 1) for e in em))

def test_zero_prob_is_finite():
    feats, scores = _toy()
    scores.loc[0, 'point_prob'] = 0.0
    em = build_emission_log_probs_ml(feats, scores, 5)
    assert np.isfinite(np.concatenate(em)).all()

def test_scores_from_candidate_probs_sums_to_one():
    from src.hmm.ml_emissions import scores_from_candidate_probs
    feats, _ = _toy()
    out = scores_from_candidate_probs(feats, [0.9, 0.05, 0.05, 0.2, 0.7, 0.1], trip_id=5)
    assert np.allclose(out.groupby('point_idx')['point_prob'].sum(), 1.0)
    assert out.loc[0, 'point_prob'] > out.loc[1, 'point_prob']
