"""
SemanticPrism: Topology Engine Pipeline
100% offline mathematical graph protocols mapping abstract nodes 
into a hierarchical Modularity network safely securely strictly cleanly natively.
"""

import networkx as nx
import numpy as np
from pyvis.network import Network
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
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

    def detect_communities(self, graph: nx.DiGraph, resolution: float = 1.0) -> Dict[str, int]:
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
                leidenalg.RBConfigurationVertexPartition, 
                weights='weight' if weights else None,
                resolution_parameter=resolution
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

    def extract_hierarchy(self, graph: nx.DiGraph, partition: Dict[str, int], min_size: int = 1) -> Dict[str, Dict[str, Any]]:
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
                
        filtered_hierarchy = {}
        for comm_key, comm_data in hierarchy.items():
            if len(comm_data["nodes"]) >= min_size:
                filtered_hierarchy[comm_key] = comm_data
            else:
                logger.info(f"Pruning micro-community {comm_key} natively mathematically (size {len(comm_data['nodes'])} < {min_size}).")
                
        return filtered_hierarchy

    def build_hypergraph_topology(self, triples: List[RawTriple], overlap_threshold: float = 0.80) -> Dict[str, Any]:
        """
        Implements: 1. Identity Guard, 2. N-ary Grouping, 
        3. Spectral Math (H & L) based on the bipartite hyperedge logic.
        """
        logger.info("Initializing n-ary hypergraph system topologically.")
        
        # --- Step 1: Identity Guard / Neighborhood Scoping ---
        # Scoping entities by their local topology to distinguish generic tags
        entity_neighborhoods = defaultdict(set)
        for t in triples:
            entity_neighborhoods[t.subject].add((t.predicate, t.object))
            entity_neighborhoods[t.object].add((t.subject, t.predicate))

        # --- Step 2: N-ary Hyperedge Grouping (Bipartite Model) ---
        # Themes act as hyperedges connecting entities
        B = nx.Graph()
        hyperedge_map = defaultdict(list)
        all_entities = set()

        for t in triples:
            theme = t.theme_association if t.theme_association else "Other"
            hyperedge_map[theme].append(t)
            all_entities.update([t.subject, t.object])

        # Add nodes to the Bipartite Graph for math and viz
        for theme, tri_list in hyperedge_map.items():
            he_node = f"HE: {theme}"
            # Store original predicates as attributes on the hyperedge
            B.add_node(he_node, label=theme, is_hyperedge=True, 
                       predicates=[t.predicate for t in tri_list])
            
            for t in tri_list:
                for ent in [t.subject, t.object]:
                    if ent not in B:
                        B.add_node(ent, label=ent, is_hyperedge=False)
                    B.add_edge(ent, he_node)

        # --- Step 3: Construct Incidence Matrix (H) & Laplacian (L) ---
        entities_sorted = sorted(list(all_entities))
        themes_sorted = sorted(list(hyperedge_map.keys()))
        ent_idx = {ent: i for i, ent in enumerate(entities_sorted)}
        thm_idx = {thm: j for j, thm in enumerate(themes_sorted)}

        H = np.zeros((len(entities_sorted), len(themes_sorted)))
        for theme, tri_list in hyperedge_map.items():
            participants = set([t.subject for t in tri_list] + [t.object for t in tri_list])
            for ent in participants:
                H[ent_idx[ent], thm_idx[theme]] = 1

        # Laplacian L = D - H*H.T
        Dv = np.diag(np.sum(H, axis=1))
        L = Dv - np.dot(H, H.T)

        # --- Step 4: Theme Overlap Matrix (O = H^T * H) ---
        O = np.dot(H.T, H)
        theme_inheritance_map = defaultdict(list)
        
        # Calculate Subsets/Inheritance based on the overlap threshold
        for i, theme_a in enumerate(themes_sorted):
            for j, theme_b in enumerate(themes_sorted):
                if i != j:
                    shared_nodes = O[i, j]
                    total_b = O[j, j]  # Total nodes in theme_b
                    
                    if total_b > 0 and (shared_nodes / total_b) >= overlap_threshold:
                        # Theme B overlaps >85% with Theme A -> Theme B subclasses/inherits Theme A
                        theme_inheritance_map[theme_a].append(theme_b)

        return {
            "B": B,
            "H": H, 
            "L": L, 
            "entities": len(entities_sorted), 
            "themes": len(themes_sorted),
            "entity_neighborhoods": dict(entity_neighborhoods),
            "theme_inheritance_map": dict(theme_inheritance_map)
        }

    def visualize_hypergraph(self, B: nx.Graph, output_dir: str = "outputs/05_topology") -> str:
        """
        Executes community detection and Centrality for Viz, 
        and generates a high-fidelity Pyvis visualization.
        """
        logger.info("Generating high-fidelity hypergraph visualization.")
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Community Detection & Centrality for Viz ---
        communities = nx.community.louvain_communities(B)
        community_map = {node: i for i, comm in enumerate(communities) for node in comm}
        centrality = nx.degree_centrality(B)

        # --- High-Fidelity Pyvis Visualization ---
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
        colors = [mcolors.to_hex(cm.tab20(i % 20)) for i in range(len(communities))]

        for node, attrs in B.nodes(data=True):
            is_he = attrs.get('is_hyperedge', False)
            comm_id = community_map.get(node, 0)
            
            shape = "box" if is_he else "dot"
            color = colors[comm_id % len(colors)]
            
            size = centrality.get(node, 0.0) * 200 + (35 if is_he else 15)
            
            title = f"Role: {'Hyperedge' if is_he else 'Entity'}\nCommunity: {comm_id}"
            if is_he:
                title += f"\nPredicates: {', '.join(set(attrs.get('predicates', [])))}"

            net.add_node(node, label=attrs.get('label', node), color=color, shape=shape, 
                         size=size, title=title, borderWidth=2, shadow=True)

        for source, target in B.edges():
            net.add_edge(source, target, color="rgba(200,200,200,0.3)", width=1)

        # ForceAtlas2 Physics for Celestial layout
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {"gravitationalConstant": -80, "centralGravity": 0.005, "springLength": 200, "springConstant": 0.08},
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          }
        }
        """)
        
        output_file = os.path.join(output_dir, "04_hypergraph_topology_graph.html")
        net.save_graph(output_file)
        logger.info(f"Hypergraph visualization saved to {output_file}")
        return output_file
