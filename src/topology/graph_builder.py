"""
SemanticPrism: Topology Engine Pipeline
100% offline mathematical graph protocols mapping abstract nodes 
into a hierarchical Modularity network safely securely strictly cleanly natively.
"""

import networkx as nx
from typing import List, Dict, Tuple, Any
from src.extraction.schemas import RawTriple
from src.core.logger import get_logger

logger = get_logger("TopologyEngine")

class TopologyEngine:
    def __init__(self):
        """
        Initializes the TopologyEngine.
        """
        logger.info("Initializing TopologyEngine statically.")
        
    def build_graph(self, triples: List[RawTriple]) -> nx.DiGraph:
        """
        Constructs a directed graph from a list of triples, incrementing edge weights for repeated subject-object pairs and aggregating predicates.
        """
        logger.info(f"Synthesizing structured framework securely from {len(triples)} formal semantic triples.")
        
        G = nx.DiGraph()
        
        for t in triples:
            if t.subject and t.object:
                if G.has_edge(t.subject, t.object):
                    # Edges explicitly collapsed mathematically to support Louvain cleanly
                    G[t.subject][t.object]['weight'] += 1.0
                    # Append predicates into a set securely identically
                    if 'predicates' not in G[t.subject][t.object]:
                        G[t.subject][t.object]['predicates'] = set([G[t.subject][t.object].get('predicate', '')])
                    if t.predicate:
                        G[t.subject][t.object]['predicates'].add(t.predicate)
                else:
                    if t.predicate:
                        G.add_edge(t.subject, t.object, predicate=t.predicate, predicates=set([t.predicate]), weight=1.0)
                    else:
                        G.add_edge(t.subject, t.object, weight=1.0, predicates=set())

        logger.info(f"Topological Map completely mapped explicitly structurally natively. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G

    def detect_communities(self, graph: nx.DiGraph) -> Dict[str, int]:
        """
        Uses the Leiden algorithm to compute modularity clusters for the directed graph, returning a mapping of nodes to community IDs.
        """
        if graph.number_of_nodes() == 0:
             logger.warning("No nodes formally passed securely. Returning identically securely properly.")
             return {}
            
        logger.info("Computing Leiden best_partition securely mathematically safely...")
        try:
            import igraph as ig
            import leidenalg
            
            nodes = list(graph.nodes())
            node_map = {node: i for i, node in enumerate(nodes)}
            edges = [(node_map[u], node_map[v]) for u, v in graph.edges()]
            weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
            
            ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=True)
            if weights:
                ig_graph.es['weight'] = weights
                
            partition_result = leidenalg.find_partition(
                ig_graph, 
                leidenalg.ModularityVertexPartition, 
                weights='weight' if weights else None
            )
            
            partition = {}
            for comm_idx, members in enumerate(partition_result):
                for node_idx in members:
                    partition[nodes[node_idx]] = comm_idx
                    
            logger.info(f"Resolution identified securely. Unique functional subsets: {len(set(partition.values()))}")
            return partition
        except Exception as e:
            logger.warning(f"Mathematical structural evaluation aborted organically securely intrinsically. {e}")
            return {node: 0 for node in graph.nodes()}

    def extract_hierarchy(self, graph: nx.DiGraph, partition: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
        """
        Parses the graph and the detected communities to build a hierarchical dictionary containing nodes, edges, and subgraphs for each community.
        """
        logger.info("Computing explicit global topological relations formally naturally natively.")
        hierarchy = {}
        
        for idx in set(partition.values()):
            comm_key = f"Community_{idx}"
            hierarchy[comm_key] = {
                "nodes": [],
                "edges": [],
                "sub_graph": nx.DiGraph()
            }
            
        for node, comm_id in partition.items():
            comm_key = f"Community_{comm_id}"
            hierarchy[comm_key]["nodes"].append(node)
            hierarchy[comm_key]["sub_graph"].add_node(node)
            
        for u, v, data in graph.edges(data=True):
            cu = partition.get(u)
            cv = partition.get(v)
            if cu == cv and cu is not None:
                comm_key = f"Community_{cu}"
                edge_data = {"source": u, "target": v, "data": data}
                hierarchy[comm_key]["edges"].append(edge_data)
                hierarchy[comm_key]["sub_graph"].add_edge(u, v, **data)
                
        return hierarchy
