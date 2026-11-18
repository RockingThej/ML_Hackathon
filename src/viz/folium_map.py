
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import folium

from src.data.loaders import load_trip


def get_route_locations(arc_ids, arcs, nodes):
    """Convert road arc IDs into latitude-longitude coordinates."""
    locations = []

    for arc_id in arc_ids:
        arc_id = int(arc_id)

        if arc_id < 0 or arc_id >= len(arcs):
            continue

        u, v = arcs[arc_id]

        if 0 <= u < len(nodes) and 0 <= v < len(nodes):
            if not locations:
                locations.append([
                    float(nodes[u, 1]),
                    float(nodes[u, 0])
                ])

            locations.append([
                float(nodes[v, 1]),
                float(nodes[v, 0])
            ])

    return locations


def create_trip_map(
    trip,
    output_path=None,
    max_road_edges=5000,
    zoom_start=13,
    predicted_route=None
):
    """Create a map with road network, GPS, actual and predicted routes."""

    track = np.asarray(trip["track"], dtype=float)
    nodes = np.asarray(trip["nodes"], dtype=float)
    arcs = np.asarray(trip["arcs"], dtype=int)
    route = trip.get("route")

    if track.ndim != 2 or track.shape[1] < 2 or len(track) == 0:
        raise ValueError("Invalid GPS track")

    center = [
        float(track[:, 1].mean()),
        float(track[:, 0].mean())
    ]

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        control_scale=True
    )

    # Road network
    road_group = folium.FeatureGroup(
        name="Road network",
        show=False
    )

    step = max(
        1,
        int(np.ceil(len(arcs) / max(1, max_road_edges)))
    )

    for arc_id in range(0, len(arcs), step):
        u, v = arcs[arc_id]

        if 0 <= u < len(nodes) and 0 <= v < len(nodes):
            folium.PolyLine(
                [
                    [float(nodes[u, 1]), float(nodes[u, 0])],
                    [float(nodes[v, 1]), float(nodes[v, 0])]
                ],
                color="gray",
                weight=1,
                opacity=0.35
            ).add_to(road_group)

    road_group.add_to(m)

    # GPS observations
    gps_group = folium.FeatureGroup(
        name="GPS observations",
        show=True
    )

    gps_locations = [
        [float(lat), float(lon)]
        for lon, lat in track[:, :2]
    ]

    folium.PolyLine(
        gps_locations,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="GPS track"
    ).add_to(gps_group)

    marker_step = max(1, len(gps_locations) // 100)

    for i in range(0, len(gps_locations), marker_step):
        folium.CircleMarker(
            gps_locations[i],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"GPS point {i}"
        ).add_to(gps_group)

    gps_group.add_to(m)

    # Actual ground-truth route
    if route is not None and len(route):
        actual_locations = get_route_locations(
            route, arcs, nodes
        )

        if actual_locations:
            actual_group = folium.FeatureGroup(
                name="Actual ground-truth route",
                show=True
            )

            folium.PolyLine(
                actual_locations,
                color="green",
                weight=5,
                opacity=0.9,
                tooltip="Actual route"
            ).add_to(actual_group)

            actual_group.add_to(m)

    # HMM predicted route
    if predicted_route is not None:
        predicted_locations = get_route_locations(
            predicted_route, arcs, nodes
        )

        if predicted_locations:
            predicted_group = folium.FeatureGroup(
                name="HMM predicted route",
                show=True
            )

            folium.PolyLine(
                predicted_locations,
                color="red",
                weight=4,
                opacity=0.9,
                tooltip="HMM predicted route"
            ).add_to(predicted_group)

            predicted_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(path))

    return m


def main():
    trip_id = 1

    trip = load_trip("data/raw/train", trip_id)

    predictions = pd.read_csv(
        "hmm_trip_1_predictions.csv"
    )

    predicted_route = (
        predictions["predicted_arc_id"]
        .astype(int)
        .tolist()
    )

    output_path = "maps/person_b/trip_1_hmm_map.html"

    create_trip_map(
        trip,
        output_path=output_path,
        predicted_route=predicted_route
    )

    print("\n========== FOLIUM VISUALIZATION ==========")
    print("Trip ID:", trip_id)
    print("GPS points:", len(trip["track"]))
    print("Actual route arcs:", len(trip["route"]))
    print("Predicted route arcs:", len(predicted_route))
    print("Map saved:", output_path)


if __name__ == "__main__":
    main()