# Feature rationale (Person A)
| feature | why |
|---|---|
| dist_to_arc | strongest cue: true road is usually nearest; metres in per-trip UTM |
| dist_rank | relative nearness, comparable across dense/sparse networks |
| bearing_diff | travel heading vs arc direction (0-180 deg); separates parallel and opposite-direction roads. 90 when heading unknown |
| heading_valid | flag: 0 when the vehicle moved <1 m so heading is unreliable |
| speed | low speed = noisy GPS / stop, bearing less trustworthy |
| arc_length | short arcs at junctions behave differently from long ones |
| proj_pos | where along the arc the point projects (0-1): continuity and junction cues |
| dt_prev | time gap since previous point (not always 1 s), larger gap = more uncertainty |
Labels: monotone DP assignment of points to .route arcs (min total distance). Points with best route arc >50 m away have no truth and are excluded (see point_truth.parquet).
