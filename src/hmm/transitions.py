
"""Fast transition probabilities for HMM map matching."""

import numpy as np


def build_transition_log_probs(
    features,
    xy,
    graph,
    beta=50.0
):
    """Build normalized transition matrices for consecutive GPS points."""

    grouped = [
        group.sort_values("dist_rank").reset_index(drop=True)
        for _, group in features.groupby("point_idx", sort=True)
    ]

    transitions = []

    # Cache arc endpoints and lengths.
    arc_starts = graph.arcs[:, 0].astype(int)
    arc_ends = graph.arcs[:, 1].astype(int)
    arc_lengths = graph.arc_lengths

    # Cache one-hop road connections.
    outgoing_by_node = graph.outgoing

    for t in range(len(grouped) - 1):
        current = grouped[t]
        following = grouped[t + 1]

        current_arcs = current["candidate_arc_id"].to_numpy(dtype=int)
        next_arcs = following["candidate_arc_id"].to_numpy(dtype=int)

        current_proj = current["proj_pos"].to_numpy(dtype=float)
        next_proj = following["proj_pos"].to_numpy(dtype=float)

        gps_distance = float(np.linalg.norm(xy[t + 1] - xy[t]))
        local_beta = max(float(beta), gps_distance * 0.25)

        log_matrix = np.empty(
            (len(current_arcs), len(next_arcs)),
            dtype=np.float64
        )

        for i, arc1 in enumerate(current_arcs):
            end1 = arc_ends[arc1]
            length1 = arc_lengths[arc1]

            remaining = (1.0 - current_proj[i]) * length1

            # Find one-intermediate-arc connections.
            one_hop = {}

            for middle_arc in outgoing_by_node[int(end1)]:
                middle_arc = int(middle_arc)
                middle_end = arc_ends[middle_arc]

                one_hop.setdefault(int(middle_end), []).append(
                    float(arc_lengths[middle_arc])
                )

            for j, arc2 in enumerate(next_arcs):
                start2 = arc_starts[arc2]
                length2 = arc_lengths[arc2]
                travelled = next_proj[j] * length2

                if arc1 == arc2 and next_proj[j] >= current_proj[i]:
                    road_distance = (
                        next_proj[j] - current_proj[i]
                    ) * length1

                elif end1 == start2:
                    road_distance = remaining + travelled

                elif int(start2) in one_hop:
                    middle_length = min(one_hop[int(start2)])
                    road_distance = (
                        remaining + middle_length + travelled
                    )

                else:
                    # Approximate distant transitions and penalize them.
                    endpoint_gap = float(
                        np.linalg.norm(graph.B[arc1] - graph.A[arc2])
                    )

                    road_distance = (
                        remaining + endpoint_gap + travelled + 500.0
                    )

                difference = abs(road_distance - gps_distance)
                log_matrix[i, j] = -difference / local_beta

        # Normalize each row into log probabilities.
        row_max = np.max(log_matrix, axis=1, keepdims=True)
        log_matrix -= row_max

        row_sum = np.log(
            np.sum(np.exp(log_matrix), axis=1, keepdims=True)
        )

        log_matrix -= row_sum
        transitions.append(log_matrix)

    return transitions