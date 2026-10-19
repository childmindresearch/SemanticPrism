from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict

class EdgeDetail(BaseModel):
    model_config = ConfigDict(extra='ignore')
    node: str
    predicate: str

class NodeMetrics(BaseModel):
    model_config = ConfigDict(extra='ignore')
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
    model_config = ConfigDict(extra='ignore')
    hub_cluster_id: int
    nodes: List[str]

class CommunityPartition(BaseModel):
    model_config = ConfigDict(extra='ignore')
    community_id: int
    nodes: List[str]
    merged_from: List[int] = []

class TopologyResult(BaseModel):
    model_config = ConfigDict(extra='ignore')
    global_hubs: List[str]
    hub_partitions: List[HubPartition] = []
    communities: List[CommunityPartition]
    orphans: List[str]
    node_metrics: Dict[str, NodeMetrics]
