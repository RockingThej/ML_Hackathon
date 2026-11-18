"""
ui/app.py  -  PathSnap demo UI (Part D, runs on Laptop 2)
=========================================================
Talks to C's FastAPI on Laptop 1:
  GET  /health, GET /trips, POST /predict, GET /trips/{id}/track
  POST /trips/{id}/arcs  (optional, see docs/D_notes_for_team.md) -> arc geometry

If /trips/{id}/arcs doesn't exist yet, the UI draws the route from local raw data
(data/raw/...) when it's present, otherwise it shows the GPS trace only.

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(base, path, timeout=10):
    r = requests.get(f"{base}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(base, path, payload, timeout=300):
    r = requests.post(f"{base}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def get_trips(base):
    return api_get(base, "/trips")


@st.cache_data(show_spinner=False)
def get_track(base, trip_id):
    return pd.DataFrame(api_get(base, f"/trips/{trip_id}/track", timeout=60)["points"])


def collapse(route):
    """[5,5,5,9,9,5] -> [5,9,5]: one entry per arc visit, in travel order."""
    out = []
    for a in route:
        if not out or out[-1] != a:
            out.append(a)
    return out


@st.cache_data(show_spinner=False)
def get_arc_geometry(base, trip_id, arc_ids: tuple):
    """{arc_id: [[lat, lon], [lat, lon]]}. API first, local raw data as fallback."""
    try:
        res = api_post(base, f"/trips/{trip_id}/arcs", {"arc_ids": list(arc_ids)}, timeout=60)
        return {int(a["arc_id"]): a["coords"] for a in res["arcs"]}, "api"
    except Exception:
        pass
    try:
        from src.data.loaders import load_arcs, load_nodes
        for part in ["student_test", "train"]:
            root = ROOT / "data" / "raw" / part
            if (root / f"{trip_id:08d}").exists():
                nodes, arcs = load_nodes(root, trip_id), load_arcs(root, trip_id)
                return {int(a): [[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]],
                                 [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]]
                        for a in arc_ids}, "local"
    except Exception:
        pass
    return {}, "none"


def build_map(track: pd.DataFrame, route_coords: list, show_points: bool):
    center = [track["lat"].mean(), track["lon"].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron", control_scale=True)
    gps = folium.FeatureGroup("Raw GPS trace")
    folium.PolyLine(track[["lat", "lon"]].values.tolist(), color="#2a78d6", weight=2,
                    opacity=0.6, dash_array="4 4").add_to(gps)
    if show_points:
        step = max(1, len(track) // 1500)  # keep the browser responsive on long trips
        for _, p in track.iloc[::step].iterrows():
            folium.CircleMarker([p.lat, p.lon], radius=2, color="#2a78d6",
                                fill=True, fill_opacity=0.8, weight=0).add_to(gps)
    gps.add_to(m)
    if route_coords:
        rt = folium.FeatureGroup("Matched route (model)")
        for seg in route_coords:
            folium.PolyLine(seg, color="#eb6834", weight=5, opacity=0.9).add_to(rt)
        rt.add_to(m)
    folium.Marker(track[["lat", "lon"]].iloc[0].tolist(), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(track[["lat", "lon"]].iloc[-1].tolist(), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])
    return m


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🗺️ PathSnap")
api = st.sidebar.text_input("API URL (Laptop 1)", value="http://localhost:8000").rstrip("/")
try:
    h = api_get(api, "/health", timeout=3)
    st.sidebar.success(f"API online · model: **{h.get('model_loaded')}**")
    online = True
except Exception as e:
    st.sidebar.error(f"API not reachable at {api}\n\n{e.__class__.__name__}")
    online = False
st.sidebar.caption("Ground truth never leaves the API - metrics are computed server-side.")

tab_match, tab_models = st.tabs(["Map matching", "Model comparison"])

# ---------------------------------------------------------------------------
# Tab 1 - map matching
# ---------------------------------------------------------------------------
with tab_match:
    st.header("Reconstruct a trip's route from noisy GPS")
    if not online:
        st.info("Start the API on Laptop 1 (`uvicorn api.main:app --host 0.0.0.0 --port 8000`) "
                "and put its IP in the sidebar, e.g. http://192.168.1.20:8000")
    else:
        trips = get_trips(api)
        c1, c2, c3 = st.columns([2, 1, 1])
        group = c1.radio("Trip set", ["student_test", "train"], horizontal=True)
        ids = trips.get(f"{group}_trips", [])
        trip_id = c2.selectbox("Trip ID", ids) if ids else None
        top_k = c3.slider("Candidates per point (K)", 5, 30, 15)
        show_points = st.checkbox("Show individual GPS points", value=True)

        if trip_id is not None and st.button("Run map matching", type="primary"):
            with st.spinner(f"Matching trip {trip_id} ..."):
                try:
                    pred = api_post(api, "/predict", {"trip_id": int(trip_id), "top_k": int(top_k)})
                    st.session_state["pred"] = pred
                except requests.HTTPError as e:
                    st.error(f"/predict failed: {e.response.text[:400]}")

        pred = st.session_state.get("pred")
        if pred and trip_id is not None and pred["trip_id"] == trip_id:
            m1, m2, m3, m4 = st.columns(4)
            met = pred.get("metrics") or {}
            m1.metric("Point accuracy", f"{met['point_accuracy']:.1%}" if met else "n/a")
            m2.metric("Miss-rate", f"{met['candidate_miss_rate']:.1%}" if met else "n/a",
                      help="Fraction of GPS points matched to the wrong arc (lower is better)")
            m3.metric("GPS points", f"{pred['n_points']:,}")
            m4.metric("Latency", f"{pred['inference_time_ms']:.0f} ms")
            st.caption(f"Model: **{pred['model_name']}**"
                       + ("" if met else " · no route file for this trip, so no accuracy"))

            track = get_track(api, trip_id)
            visits = collapse(pred["predicted_route"])
            geom, src = get_arc_geometry(api, trip_id, tuple(sorted(set(visits))))
            route_coords = [geom[a] for a in visits if a in geom]
            if src == "none":
                st.warning("Route geometry unavailable: ask C to add POST /trips/{id}/arcs "
                           "(snippet in docs/D_notes_for_team.md). Showing GPS only.")
            st_folium(build_map(track, route_coords, show_points), height=600,
                      use_container_width=True, returned_objects=[])
            with st.expander(f"Matched arcs in travel order ({len(visits)} visits)"):
                st.write(visits)

# ---------------------------------------------------------------------------
# Tab 2 - model comparison
# ---------------------------------------------------------------------------
with tab_models:
    st.header("Model comparison (held-out test trips)")
    csv = RESULTS / "model_comparison.csv"
    if not csv.exists():
        st.info("results/model_comparison.csv not found - pull the latest repo.")
    else:
        df = pd.read_csv(csv)
        df = df[df["n_samples"] == df["n_samples"].max()]          # drop smoke runs
        df = df.sort_values("timestamp").groupby("model").tail(1)  # latest run per model
        df = df.sort_values("miss_rate").reset_index(drop=True)
        best = df.iloc[0]
        st.success(f"Best: **{best.model}** - miss-rate {best.miss_rate:.2%} "
                   f"(accuracy {1 - best.miss_rate:.2%})")
        show = df[["model", "miss_rate", "accuracy_at_1", "f1_macro", "log_loss", "timestamp"]]
        st.dataframe(show.style.format({"miss_rate": "{:.4f}", "accuracy_at_1": "{:.4f}",
                                        "f1_macro": "{:.4f}", "log_loss": "{:.4f}"}),
                     width="stretch", hide_index=True)
        st.bar_chart(df.set_index("model")["miss_rate"], horizontal=True,
                     color="#2a78d6", x_label="miss-rate (lower is better)")

    imgs = sorted(PLOTS.glob("*.png")) if PLOTS.exists() else []
    if imgs:
        st.subheader("Training curves and tuning")
        for i in range(0, len(imgs), 2):
            cols = st.columns(2)
            for col, p in zip(cols, imgs[i:i + 2]):
                col.image(str(p), caption=p.stem, width="stretch")
