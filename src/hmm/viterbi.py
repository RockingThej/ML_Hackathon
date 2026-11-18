import numpy as np

def viterbi_decode(log_emission_probs, log_transition_probs, log_start_probs=None):
    emission = np.asarray(log_emission_probs, dtype=float)
    transition = np.asarray(log_transition_probs, dtype=float)
    if emission.ndim != 2 or emission.shape[0] == 0 or emission.shape[1] == 0:
        raise ValueError('log_emission_probs must be a non-empty 2D array')
    n_steps, n_states = emission.shape
    if transition.shape != (n_states, n_states):
        raise ValueError('transition matrix must have shape (n_states, n_states)')
    if np.isnan(emission).any() or np.isnan(transition).any():
        raise ValueError('Probabilities must not contain NaN')
    if log_start_probs is None:
        start = np.full(n_states, -np.log(n_states), dtype=float)
    else:
        start = np.asarray(log_start_probs, dtype=float)
        if start.shape != (n_states,):
            raise ValueError('log_start_probs must have one value per state')
        if np.isnan(start).any():
            raise ValueError('Start probabilities must not contain NaN')
    score = np.full((n_steps, n_states), -np.inf, dtype=float)
    backpointer = np.zeros((n_steps, n_states), dtype=np.int64)
    score[0] = start + emission[0]
    for t in range(1, n_steps):
        candidates = score[t - 1][:, None] + transition
        backpointer[t] = np.argmax(candidates, axis=0)
        score[t] = np.max(candidates, axis=0) + emission[t]
    last_state = int(np.argmax(score[-1]))
    best_log_probability = float(score[-1, last_state])
    if not np.isfinite(best_log_probability):
        raise ValueError('No valid state sequence exists for these probabilities')
    path = np.empty(n_steps, dtype=np.int64)
    path[-1] = last_state
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backpointer[t, path[t]]
    return (path, best_log_probability)

def viterbi_decode_variable(emissions, transitions, start_probs=None):
    emissions = [np.asarray(e, dtype=float) for e in emissions]
    transitions = [np.asarray(a, dtype=float) for a in transitions]
    T = len(emissions)
    if T == 0:
        raise ValueError('At least one observation is required')
    if any((e.ndim != 1 or len(e) == 0 for e in emissions)):
        raise ValueError('Each emission array must be non-empty and 1D')
    if len(transitions) != T - 1:
        raise ValueError('There must be one transition matrix between points')
    for t, matrix in enumerate(transitions):
        expected = (len(emissions[t]), len(emissions[t + 1]))
        if matrix.shape != expected:
            raise ValueError(f'Transition {t} must have shape {expected}')
    if any((np.isnan(e).any() for e in emissions)):
        raise ValueError('Emissions cannot contain NaN')
    if any((np.isnan(a).any() for a in transitions)):
        raise ValueError('Transitions cannot contain NaN')
    if start_probs is None:
        start = np.full(len(emissions[0]), -np.log(len(emissions[0])), dtype=float)
    else:
        start = np.asarray(start_probs, dtype=float)
    if start.shape != emissions[0].shape:
        raise ValueError('Start probabilities have incorrect shape')
    if np.isnan(start).any():
        raise ValueError('Start probabilities cannot contain NaN')
    scores = start + emissions[0]
    backpointers = []
    for t in range(1, T):
        candidates = scores[:, None] + transitions[t - 1]
        back = np.argmax(candidates, axis=0)
        scores = np.max(candidates, axis=0) + emissions[t]
        backpointers.append(back)
    last = int(np.argmax(scores))
    best_score = float(scores[last])
    if not np.isfinite(best_score):
        raise ValueError('No valid state sequence exists')
    path = [0] * T
    path[-1] = last
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(backpointers[t - 1][path[t]])
    return (np.asarray(path, dtype=int), best_score)

def demo_toy_graph():
    transition = np.log(np.array([[0.8, 0.2, 0.0], [0.05, 0.8, 0.15], [0.0, 0.1, 0.9]]).clip(1e-300))
    emission = np.log(np.array([[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]]))
    start = np.log(np.array([0.8, 0.15, 0.05]))
    return viterbi_decode(emission, transition, start)
if __name__ == '__main__':
    path, log_prob = demo_toy_graph()
    labels = ['A', 'B', 'C']
    print('Toy graph Viterbi test')
    print('Best state sequence:', [labels[i] for i in path])
    print('Best log probability:', round(log_prob, 6))
    emissions = [np.log([0.8, 0.2]), np.log([0.1, 0.9, 0.2]), np.log([0.1, 0.8])]
    transitions = [np.log([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]]), np.log([[0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])]
    variable_path, variable_score = viterbi_decode_variable(emissions, transitions)
    print('\nVariable-candidate Viterbi test')
    print('Decoded candidate indices:', variable_path.tolist())
    print('Best log probability:', round(variable_score, 6))
