import numpy as np
from src.hmm.viterbi import viterbi_decode, demo_toy_graph

def test_toy_graph_returns_three_states():
    path, score = demo_toy_graph()
    assert path.shape == (3,)
    assert np.isfinite(score)

def test_viterbi_prefers_consistent_path():
    emission = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    transition = np.log(np.array([[0.2, 0.8], [0.1, 0.9]]))
    start = np.log([0.9, 0.1])
    path, score = viterbi_decode(emission, transition, start)
    assert path.tolist() == [0, 1]
    assert np.isfinite(score)
