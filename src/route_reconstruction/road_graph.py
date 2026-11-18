
"""Build and query a directed road network for PathSnap."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.data.loaders import load_trip
from src.features.candidates import prepare


class RoadGraph:
    """Directed graph representing the road network of one trip."""

    def __init__(self, trip):
        self.trip_id = trip["trip_id"]
        self.nodes = trip["nodes"]
        self.arcs = trip["arcs"]

        # Project geographic coordinates into metres.
        _, _, self.A, self.B, self.epsg = prepare(trip)

        self.num_nodes = len(self.nodes)
        self.num_arcs = len(self.arcs)

        # Calculate length of each directed arc.
        differences = self.B - self.A
        self.arc_lengths = np.hypot(
            differences[:, 0],
            differences[:, 1]
        )

        # Build directed adjacency matrix.
        rows = self.arcs[:, 0]
        cols = self.arcs[:, 1]

        self.adjacency = csr_matrix(
            (self.arc_lengths, (rows, cols)),
            shape=(self.num_nodes, self.num_nodes)
        )

        # Keep outgoing arc IDs for each node.
        self.outgoing = [[] for _ in range(self.num_nodes)]

        for arc_id, (u, v) in enumerate(self.arcs):
            self.outgoing[int(u)].append(int(arc_id))

    def get_arc_length(self, arc_id):
        """Return the length of a road arc in metres."""
        return float(self.arc_lengths[int(arc_id)])

    def get_arc_endpoints(self, arc_id):
        """Return start and end node IDs of an arc."""
        return tuple(self.arcs[int(arc_id)])

    def get_successors(self, arc_id):
        """Return arcs that can directly follow the given arc."""
        _, end_node = self.get_arc_endpoints(arc_id)
        return self.outgoing[int(end_node)]

    def are_connected(self, arc1, arc2):
        """Check whether two directed arcs are directly connected."""
        _, end_node = self.get_arc_endpoints(arc1)
        start_node, _ = self.get_arc_endpoints(arc2)

        return int(end_node) == int(start_node)

    def shortest_distance(self, source_node, target_node):
        """Calculate shortest directed road distance in metres."""
        distances = dijkstra(
            self.adjacency,
            directed=True,
            indices=int(source_node)
        )

        return float(distances[int(target_node)])

    def shortest_path(self, source_node, target_node):
        """Return node sequence of the shortest directed path."""
        distances, predecessors = dijkstra(
            self.adjacency,
            directed=True,
            indices=int(source_node),
            return_predecessors=True
        )

        target_node = int(target_node)

        if not np.isfinite(distances[target_node]):
            return []

        path = []
        current = target_node

        while current != -9999:
            path.append(int(current))

            if current == int(source_node):
                break

            current = int(predecessors[current])

        if path[-1] != int(source_node):
            return []

        return path[::-1]

    def get_arc_sequence(self, node_path):
        """Convert consecutive nodes into directed arc IDs."""
        result = []

        for u, v in zip(node_path[:-1], node_path[1:]):
            matching = np.flatnonzero(
                (self.arcs[:, 0] == u) &
                (self.arcs[:, 1] == v)
            )

            if len(matching) == 0:
                raise ValueError(f"No directed arc found from {u} to {v}")

            result.append(int(matching[0]))

        return result

    def summary(self):
        """Print basic road network information."""
        print("\n========== ROAD GRAPH ==========")
        print("Trip ID:", self.trip_id)
        print("Number of nodes:", self.num_nodes)
        print("Number of arcs:", self.num_arcs)
        print("Average arc length:", round(float(np.mean(self.arc_lengths)), 2), "m")
        print("Minimum arc length:", round(float(np.min(self.arc_lengths)), 2), "m")
        print("Maximum arc length:", round(float(np.max(self.arc_lengths)), 2), "m")
        print("Total directed arc length:", round(float(np.sum(self.arc_lengths)), 2), "m")


def build_road_graph(root, trip_id):
    """Load one trip and build its directed road graph."""
    trip = load_trip(root, trip_id)
    return RoadGraph(trip)


if __name__ == "__main__":
    graph = build_road_graph("data/raw/train", 1)
    graph.summary()

    test_arc = 0
    successors = graph.get_successors(test_arc)

    print("\nExample arc:", test_arc)
    print("Arc endpoints:", graph.get_arc_endpoints(test_arc))
    print("Arc length:", round(graph.get_arc_length(test_arc), 2), "m")
    print("Number of directly connected successor arcs:", len(successors))
    print("First few successor arc IDs:", successors[:10])