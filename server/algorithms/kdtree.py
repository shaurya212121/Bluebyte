"""
server/algorithms/kdtree.py
KD-Tree Spatial Index for O(log N) Fish Cluster and Biodiversity Lookups.
"""

from __future__ import annotations
import math
import heapq
from typing import List, Tuple, Optional, Any, Dict

# Earth radius in kilometers for geospatial distance calculations
EARTH_RADIUS_KM = 6371.0088


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Computes great-circle distance between two (lat, lon) coordinates in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def euclidean_distance(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
    """Computes standard n-dimensional Euclidean distance."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


class KDNode:
    """Node in the KD-Tree holding spatial point, arbitrary payload, and split axis."""
    __slots__ = ('point', 'payload', 'axis', 'left', 'right')

    def __init__(
        self,
        point: Tuple[float, ...],
        payload: Any = None,
        axis: int = 0,
        left: Optional[KDNode] = None,
        right: Optional[KDNode] = None
    ):
        self.point = point
        self.payload = payload
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """
    Balanced k-d tree spatial indexing structure.
    
    Supports:
    - Balanced recursive median construction: O(N log N)
    - Nearest Neighbor Search: O(log N) average
    - k-Nearest Neighbors (k-NN) with bounded max-heap: O(k log N)
    - Radius / Range search: O(k + log N)
    """

    def __init__(self, points_with_payload: List[Tuple[Tuple[float, ...], Any]], dimensions: int = 2, is_geospatial: bool = True):
        self.k = dimensions
        self.is_geospatial = is_geospatial
        self.root = self._build_tree(points_with_payload, depth=0)

    def _distance(self, p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        if self.is_geospatial and self.k == 2:
            return haversine_distance((p1[0], p1[1]), (p2[0], p2[1]))
        return euclidean_distance(p1, p2)

    def _build_tree(self, points: List[Tuple[Tuple[float, ...], Any]], depth: int) -> Optional[KDNode]:
        if not points:
            return None

        axis = depth % self.k
        points.sort(key=lambda item: item[0][axis])
        median_idx = len(points) // 2

        median_point, median_payload = points[median_idx]

        return KDNode(
            point=median_point,
            payload=median_payload,
            axis=axis,
            left=self._build_tree(points[:median_idx], depth + 1),
            right=self._build_tree(points[median_idx + 1:], depth + 1)
        )

    def nearest_neighbor(self, target: Tuple[float, ...]) -> Tuple[Optional[Tuple[float, ...]], Optional[Any], float]:
        """
        Finds the single nearest point in O(log N) average time.
        Returns: (nearest_point, payload, distance)
        """
        if not self.root:
            return None, None, float('inf')

        best_node: Optional[KDNode] = None
        best_dist = float('inf')

        def _search(node: Optional[KDNode]):
            nonlocal best_node, best_dist
            if node is None:
                return

            d = self._distance(target, node.point)
            if d < best_dist:
                best_dist = d
                best_node = node

            axis = node.axis
            axis_delta = target[axis] - node.point[axis]

            near_branch = node.left if axis_delta < 0 else node.right
            far_branch = node.right if axis_delta < 0 else node.left

            _search(near_branch)

            # Pruning rule: check if hyper-plane crosses the bounding sphere
            if self.is_geospatial and self.k == 2:
                # 1 deg lat ~ 111.12 km
                plane_dist = abs(axis_delta) * 111.12 if axis == 0 else abs(axis_delta) * (111.12 * math.cos(math.radians(target[0])))
            else:
                plane_dist = abs(axis_delta)

            if plane_dist < best_dist:
                _search(far_branch)

        _search(self.root)
        return (best_node.point, best_node.payload, best_dist) if best_node else (None, None, float('inf'))

    def k_nearest_neighbors(self, target: Tuple[float, ...], k: int) -> List[Dict[str, Any]]:
        """
        Finds k-nearest points using a bounded Max-Heap.
        Returns sorted list of dicts: [{"point": ..., "payload": ..., "distance": ...}]
        """
        if not self.root or k <= 0:
            return []

        # Max-heap storing (-distance, counter, node)
        heap: List[Tuple[float, int, KDNode]] = []
        counter = 0

        def _search(node: Optional[KDNode]):
            nonlocal counter
            if node is None:
                return

            dist = self._distance(target, node.point)
            counter += 1

            if len(heap) < k:
                heapq.heappush(heap, (-dist, counter, node))
            else:
                if dist < -heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (-dist, counter, node))

            axis = node.axis
            axis_delta = target[axis] - node.point[axis]
            near_branch = node.left if axis_delta < 0 else node.right
            far_branch = node.right if axis_delta < 0 else node.left

            _search(near_branch)

            current_worst_dist = -heap[0][0] if len(heap) == k else float('inf')
            if self.is_geospatial and self.k == 2:
                plane_dist = abs(axis_delta) * 111.12 if axis == 0 else abs(axis_delta) * (111.12 * math.cos(math.radians(target[0])))
            else:
                plane_dist = abs(axis_delta)

            if plane_dist < current_worst_dist:
                _search(far_branch)

        _search(self.root)

        results = []
        while heap:
            neg_d, _, n = heapq.heappop(heap)
            results.append({
                "point": n.point,
                "payload": n.payload,
                "distance_km": -neg_d if self.is_geospatial else -neg_d
            })
        results.reverse()
        return results

    def radius_search(self, target: Tuple[float, ...], radius_km: float) -> List[Dict[str, Any]]:
        """
        Retrieves all points within a specified radius (in km or Euclidean units).
        """
        results: List[Dict[str, Any]] = []

        def _search(node: Optional[KDNode]):
            if node is None:
                return

            dist = self._distance(target, node.point)
            if dist <= radius_km:
                results.append({
                    "point": node.point,
                    "payload": node.payload,
                    "distance_km": dist
                })

            axis = node.axis
            axis_delta = target[axis] - node.point[axis]
            near_branch = node.left if axis_delta < 0 else node.right
            far_branch = node.right if axis_delta < 0 else node.left

            _search(near_branch)

            if self.is_geospatial and self.k == 2:
                plane_dist = abs(axis_delta) * 111.12 if axis == 0 else abs(axis_delta) * (111.12 * math.cos(math.radians(target[0])))
            else:
                plane_dist = abs(axis_delta)

            if plane_dist <= radius_km:
                _search(far_branch)

        _search(self.root)
        results.sort(key=lambda x: x["distance_km"])
        return results