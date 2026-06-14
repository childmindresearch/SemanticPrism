import json
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from src.topology.schemas import NodeMetrics, CommunityPartition, StructuralCluster, ThemeInheritance, TopologyResult

try:
    from cdlib import algorithms
except ImportError:
    algorithms = None

try:
    import numpy as np
    from node2vec import Node2Vec
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except ImportError:
    np = None
    Node2Vec = None
    KMeans = None
    silhouette_score = None

class TopologyPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('topology', {})
        self.spectral_variance = self.config.get('spectral_variance_retention', 0.05)
        self.leiden_res = self.config.get('leiden_resolution', 1.0)
        self.inheritance_threshold = self.config.get('inheritance_overlap_threshold', 0.3)

    def execute(self, refined_triplets: List[Dict[str, Any]]) -> TopologyResult:
        print("[Topology] Starting Stage 3 Pipeline...")
        
        # 3.1 NetworkX Graph Construction
        print("   -> Constructing Network Graphs...")
        dg = nx.DiGraph()
        
        for t in refined_triplets:
            subj = t.get('subject')
            obj = t.get('object')
            pred = t.get('predicate', '')
            
            if not subj or not obj:
                continue
            
            # Ensure all entities are perfectly lowercase before graph ingestion
            subj = str(subj).lower().strip()
            obj = str(obj).lower().strip()
                
            if dg.has_edge(subj, obj):
                dg[subj][obj]['weight'] = dg[subj][obj].get('weight', 1) + 1
            else:
                dg.add_edge(subj, obj, predicate=pred, weight=1)
                
        ug = dg.to_undirected()
        
        # 3.2 Spectral Hub & Orphan Identification
        print("   -> Running Spectral Centrality...")
        pagerank = nx.pagerank(dg) if len(dg) > 0 else {}
        degree_cent = nx.degree_centrality(dg) if len(dg) > 0 else {}
        
        # Calculate Hubs (Top N% of PageRank)
        pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        hub_cutoff = max(1, int(len(pr_sorted) * self.spectral_variance))
        global_hubs = [node for node, score in pr_sorted[:hub_cutoff]]
        
        # Calculate Orphans (Degree == 1 or 0)
        orphans = [node for node in ug.nodes() if ug.degree(node) <= 1 and node not in global_hubs]
        
        # Populate Metrics
        node_metrics = {}
        for node in dg.nodes():
            in_edges_list = [{"node": u, "predicate": data.get("predicate", "")} for u, v, data in dg.in_edges(node, data=True)]
            out_edges_list = [{"node": v, "predicate": data.get("predicate", "")} for u, v, data in dg.out_edges(node, data=True)]
            
            node_metrics[node] = NodeMetrics(
                node_id=node,
                degree_centrality=degree_cent.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0),
                is_hub=(node in global_hubs),
                is_orphan=(node in orphans),
                in_edges=in_edges_list,
                out_edges=out_edges_list
            )

        # 3.3 Leiden Modularity (Community Detection)
        print("   -> Pruning Graph & Running Leiden Clustering...")
        sub_graph = ug.copy()
        sub_graph.remove_nodes_from(global_hubs)
        sub_graph.remove_nodes_from(orphans)
        
        communities = []
        if algorithms and len(sub_graph.nodes()) > 0:
            try:
                # cdlib implementation of leiden
                coms = algorithms.leiden(sub_graph)
                for i, community_nodes in enumerate(coms.communities):
                    communities.append(CommunityPartition(
                        community_id=i,
                        nodes=list(community_nodes)
                    ))
            except Exception as e:
                print(f"[Topology] Error in Leiden clustering: {e}. Falling back to connected components...")
                for i, comp in enumerate(nx.connected_components(sub_graph)):
                    communities.append(CommunityPartition(community_id=i, nodes=list(comp)))
        else:
            print("[Topology] CDlib not installed or sub_graph empty, falling back to connected components...")
            for i, comp in enumerate(nx.connected_components(sub_graph)):
                communities.append(CommunityPartition(community_id=i, nodes=list(comp)))

        # 3.4 Structural Equivalence (Node2Vec + KMeans + Silhouette)
        print("   -> Running Node2Vec Structural Equivalence...")
        structural_clusters = []
        if Node2Vec and KMeans and len(sub_graph.nodes()) >= 3:
            try:
                # 1. Generate Embeddings
                node2vec_model = Node2Vec(sub_graph, dimensions=64, walk_length=10, num_walks=100, workers=1, quiet=True)
                model = node2vec_model.fit(window=5, min_count=1, batch_words=4)
                
                # Extract nodes and their corresponding vectors
                node_list = list(sub_graph.nodes())
                embeddings = np.array([model.wv[node] for node in node_list])
                
                # 2. Dynamic K Search via Silhouette Score
                max_clusters = min(self.config.get('max_structural_clusters', 10), len(node_list) - 1)
                best_k = 2
                best_score = -1.0
                
                if max_clusters > 2:
                    for k in range(2, max_clusters + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                        labels = kmeans.fit_predict(embeddings)
                        score = silhouette_score(embeddings, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                            
                    print(f"      -> Optimal K found: {best_k} (Silhouette Score: {best_score:.3f})")
                else:
                    best_k = max_clusters
                    
                # 3. Final Clustering
                final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
                final_labels = final_kmeans.fit_predict(embeddings)
                
                # Group nodes by cluster
                cluster_dict = defaultdict(list)
                for node, label in zip(node_list, final_labels):
                    cluster_dict[label].append(node)
                    
                for c_id, nodes in cluster_dict.items():
                    structural_clusters.append(StructuralCluster(cluster_id=int(c_id), nodes=nodes))
                    
            except Exception as e:
                print(f"[Topology] Error in Node2Vec clustering: {e}")
        else:
            print("[Topology] node2vec/scikit-learn not installed or graph too small. Skipping Structural Equivalence.")

        # 3.5 Hypergraph Theme Inheritance
        print("   -> Calculating Semantic Inheritance...")
        theme_inheritance = []
        theme_sets = defaultdict(set)
        
        for t in refined_triplets:
            subj = t.get('subject')
            obj = t.get('object')
            theme = t.get('theme_association')
            if theme and theme != "Other":
                if subj: theme_sets[theme].add(str(subj).lower().strip())
                if obj: theme_sets[theme].add(str(obj).lower().strip())
                
        for child_theme, child_nodes in theme_sets.items():
            for parent_theme, parent_nodes in theme_sets.items():
                if child_theme == parent_theme:
                    continue
                if not child_nodes:
                    continue
                overlap = len(child_nodes.intersection(parent_nodes)) / len(child_nodes)
                if overlap >= self.inheritance_threshold:
                    theme_inheritance.append(ThemeInheritance(
                        parent_theme=parent_theme,
                        child_theme=child_theme,
                        overlap_score=overlap
                    ))

        # Assemble Final Payload
        result = TopologyResult(
            global_hubs=global_hubs,
            communities=communities,
            structural_clusters=structural_clusters,
            orphans=orphans,
            node_metrics=node_metrics,
            theme_inheritance=theme_inheritance
        )

        out_dir = Path("outputs/03_topology")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "topology_partitions.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        # 3.6 PyVis Native Visualizations
        print("   -> Generating Interactive HTML Visualizations...")
        self._generate_visuals(dg, result, theme_sets)

        print("[Topology] Pipeline complete.")
        return result

    def _generate_visuals(self, dg: nx.DiGraph, result: TopologyResult, theme_sets: dict):
        vis_dir = Path("outputs/visuals")
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Extended high-contrast hex palette (matches matplotlib tab20)
        colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
            "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
            "#17becf", "#9edae5"
        ]
        
        # 1. Standard Topology Visual
        net = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white")
        
        community_map = {}
        for comm in result.communities:
            for node in comm.nodes:
                community_map[node] = comm.community_id
                
        for node in dg.nodes():
            metrics = result.node_metrics.get(node)
            if not metrics:
                continue
                
            comm_id = community_map.get(node, 0)
            centrality = metrics.degree_centrality
            
            is_hub = metrics.is_hub
            is_orphan = metrics.is_orphan
            
            if is_hub:
                color = "red"
                shape = "star"
            elif is_orphan:
                color = "gray"
                shape = "dot"
            else:
                color = colors[comm_id % len(colors)]
                shape = "dot"
                
            # Dynamic scalar logic
            size = (centrality * 200) + (35 if is_hub else 15)
            
            title = f"Role: {'Hub' if is_hub else 'Orphan' if is_orphan else 'Entity'}\n"
            title += f"Community: {comm_id}\n"
            title += f"Centrality: {centrality:.4f}"
            
            net.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=2, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net.get_nodes() and v in net.get_nodes():
                net.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
                
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {"gravitationalConstant": -80, "centralGravity": 0.005, "springLength": 200, "springConstant": 0.08},
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          }
        }
        """)
        net.save_graph(str(vis_dir / "interactive_topology_graph.html"))
        
        # 2. Hypergraph Visual
        hyper = Network(height="1000px", width="100%", directed=True, bgcolor="#111111", font_color="white")
        
        for theme in theme_sets.keys():
            title = f"Role: Hyperedge (Theme)\nEntities Connected: {len(theme_sets[theme])}"
            hyper.add_node(f"THEME_{theme}", label=f"Theme: {theme}", color="blue", size=60, shape="star", title=title, shadow=True)
            
        for inh in result.theme_inheritance:
            hyper.add_edge(f"THEME_{inh.child_theme}", f"THEME_{inh.parent_theme}", title=f"Inherits ({inh.overlap_score:.2f})", color="rgba(255,255,255,0.8)", width=3)
            
        added_entities = set()
        for theme, entities in theme_sets.items():
            for entity in entities:
                if entity not in added_entities:
                    metrics = result.node_metrics.get(entity)
                    centrality = metrics.degree_centrality if metrics else 0.0
                    size = (centrality * 150) + 10
                    comm_id = community_map.get(entity, 0)
                    color = colors[comm_id % len(colors)]
                    
                    title = f"Role: Entity\nCommunity: {comm_id}\nCentrality: {centrality:.4f}"
                    
                    hyper.add_node(entity, label=entity, color=color, size=size, shape="dot", title=title, borderWidth=1)
                    added_entities.add(entity)
                
                hyper.add_edge(entity, f"THEME_{theme}", color="rgba(200,200,200,0.2)", width=1)
                
        hyper.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {"gravitationalConstant": -80, "centralGravity": 0.005, "springLength": 200, "springConstant": 0.08},
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          }
        }
        """)
        hyper.save_graph(str(vis_dir / "interactive_hypergraph.html"))
