from typing import List, Dict
from pydantic import BaseModel

class EdgeDetail(BaseModel):
    node: str
    predicate: str

class NodeMetrics(BaseModel):
    node_id: str
    degree_centrality: float
    pagerank: float
    betweenness_centrality: float = 0.0
    participation_coefficient: float = 0.0
    modularity_vitality: float = 0.0
    is_hub: bool
    is_orphan: bool
    in_edges: List[EdgeDetail] = []
    out_edges: List[EdgeDetail] = []

class HubPartition(BaseModel):
    hub_cluster_id: int
    nodes: List[str]

class CommunityPartition(BaseModel):
    community_id: int
    nodes: List[str]

class StructuralCluster(BaseModel):
    cluster_id: int
    nodes: List[str]

class ThemeInheritance(BaseModel):
    parent_theme: str
    child_theme: str
    overlap_score: float

class TopologyResult(BaseModel):
    global_hubs: List[str]
    hub_partitions: List[HubPartition] = []
    communities: List[CommunityPartition]
    structural_clusters: List[StructuralCluster]
    orphans: List[str]
    node_metrics: Dict[str, NodeMetrics]
    theme_inheritance: List[ThemeInheritance]
