"""
server/algorithms/clustering.py
Fish Cluster Dataclass and Spatial Aggregator.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from .kdtree import KDTree


@dataclass
class FishCluster:
    cluster_id: str
    lat: float
    lon: float
    species: str
    biomass_density_mt: float  # Metric Tons per km^2
    depth_m: float
    sea_surface_temp_c: float
    salinity_psu: float
    confidence_score: float

    def to_spatial_tuple(self) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        return ((self.lat, self.lon), asdict(self))


class FishClusterSpatialIndex:
    """Manages spatial lookups over active fish clusters."""

    def __init__(self, clusters: List[FishCluster]):
        self.clusters = clusters
        data = [c.to_spatial_tuple() for c in clusters]
        self.index = KDTree(data, dimensions=2, is_geospatial=True)

    def find_nearest_cluster(self, vessel_lat: float, vessel_lon: float) -> Dict[str, Any]:
        pt, payload, dist = self.index.nearest_neighbor((vessel_lat, vessel_lon))
        return {
            "target_coordinate": (vessel_lat, vessel_lon),
            "nearest_cluster": payload,
            "distance_km": round(dist, 3) if dist != float('inf') else None
        }

    def find_top_k_clusters(self, vessel_lat: float, vessel_lon: float, k: int = 5) -> List[Dict[str, Any]]:
        return self.index.k_nearest_neighbors((vessel_lat, vessel_lon), k=k)

    def find_clusters_in_zone(self, center_lat: float, center_lon: float, radius_km: float) -> List[Dict[str, Any]]:
        return self.index.radius_search((center_lat, center_lon), radius_km=radius_km)