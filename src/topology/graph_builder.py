import json
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from src.topology.schemas import NodeMetrics, CommunityPartition, StructuralCluster, ThemeInheritance, TopologyResult

try:
    from cdlib import algorithms
except ImportError:
    algorithms = None

# Compatibility patch for scipy 1.13+ and gensim: inject triu into scipy.linalg if missing
try:
    import numpy as np
    import scipy
    import scipy.linalg
    scipy.linalg.triu = np.triu
except Exception:
    pass

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


def _calculate_participation_coefficient(dg: nx.DiGraph, theme_sets: Dict[str, set]) -> Dict[str, float]:
    """
    Computes Participation Coefficient P_i for each node across themes:
    P_i = 1 - sum( (k_{i,m} / k_i)^2 )
    where k_i is total degree of node i, and k_{i,m} is connection count to theme m.
    """
    p_scores = {}
    for node in dg.nodes():
        total_deg = dg.in_degree(node) + dg.out_degree(node)
        if total_deg == 0:
            p_scores[node] = 0.0
            continue
        
        neighbors = set(dg.predecessors(node)).union(set(dg.successors(node)))
        theme_counts = defaultdict(int)
        for theme_name, theme_nodes in theme_sets.items():
            overlap = len(neighbors.intersection(theme_nodes))
            if node in theme_nodes:
                overlap += 1
            if overlap > 0:
                theme_counts[theme_name] = overlap
                
        sum_sq = 0.0
        total_theme_connections = sum(theme_counts.values())
        if total_theme_connections > 0:
            for theme_name, count in theme_counts.items():
                frac = count / total_theme_connections
                sum_sq += frac * frac
            p_scores[node] = float(max(0.0, 1.0 - sum_sq))
        else:
            p_scores[node] = 0.0
    return p_scores


def _calculate_betweenness(dg: nx.DiGraph) -> Dict[str, float]:
    """Computes shortest-path Betweenness Centrality for directed graph."""
    if len(dg) == 0:
        return {}
    return nx.betweenness_centrality(dg)


def _calculate_modularity_vitality(ug: nx.Graph, theme_sets: Dict[str, set]) -> Dict[str, float]:
    """
    Computes Delta Q(v) = Q(G) - Q(G \ v) for each node v in ug.
    Negative value means removing node v INCREASES modularity Q (node blurs community boundaries).
    """
    vitality = {}
    if len(ug) < 3 or not theme_sets:
        for n in ug.nodes(): vitality[n] = 0.0
        return vitality
    try:
        comms = [list(nodes.intersection(set(ug.nodes()))) for nodes in theme_sets.values() if len(nodes.intersection(set(ug.nodes()))) > 0]
        if len(comms) < 2:
            for n in ug.nodes(): vitality[n] = 0.0
            return vitality
            
        base_q = nx.community.modularity(ug, comms)
        for n in ug.nodes():
            sub_g = ug.copy()
            sub_g.remove_node(n)
            sub_comms = [[x for x in comm if x != n] for comm in comms]
            sub_comms = [c for c in sub_comms if len(c) > 0]
            if len(sub_comms) >= 2 and len(sub_g.edges()) > 0:
                sub_q = nx.community.modularity(sub_g, sub_comms)
                vitality[n] = float(base_q - sub_q)
            else:
                vitality[n] = 0.0
    except Exception:
        for n in ug.nodes(): vitality[n] = 0.0
    return vitality


class TopologyPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('topology', {})
        self.mode = self.config.get('execution_mode', 'both')
        self.inheritance_threshold = self.config.get('inheritance_overlap_threshold', 0.3)
        
        # Path 1 Config (Approach A: Community Path)
        self.comm_cfg = self.config.get('community_path', {})
        self.comm_p_thresh = self.comm_cfg.get('participation_threshold', 0.5)
        self.comm_betw_percentile = self.comm_cfg.get('betweenness_percentile', 0.90)
        self.leiden_res = self.comm_cfg.get('leiden_resolution', 1.0)
        
        # Path 2 Config (Approach B: Embedding Path)
        self.emb_cfg = self.config.get('embedding_path', {})
        self.emb_p_thresh = self.emb_cfg.get('participation_threshold', 0.5)
        self.enable_mod_vitality = self.emb_cfg.get('enable_modularity_vitality_pruning', True)
        self.max_structural_clusters = self.emb_cfg.get('max_structural_clusters', 10)
        self.n2v_dim = self.emb_cfg.get('node2vec_dimensions', 64)
        self.n2v_walk_len = self.emb_cfg.get('node2vec_walk_length', 10)
        self.n2v_num_walks = self.emb_cfg.get('node2vec_num_walks', 100)

    def execute(self, refined_triplets: List[Dict[str, Any]]) -> Dict[str, TopologyResult]:
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
            
            subj = str(subj).lower().strip()
            obj = str(obj).lower().strip()
                
            if dg.has_edge(subj, obj):
                dg[subj][obj]['weight'] = dg[subj][obj].get('weight', 1) + 1
            else:
                dg.add_edge(subj, obj, predicate=pred, weight=1)
                
        ug = dg.to_undirected()
        
        # Build Theme Sets
        theme_sets = defaultdict(set)
        for t in refined_triplets:
            subj = t.get('subject')
            obj = t.get('object')
            theme = t.get('theme_association')
            if theme and theme != "Other":
                if subj: theme_sets[theme].add(str(subj).lower().strip())
                if obj: theme_sets[theme].add(str(obj).lower().strip())

        # 3.2 Compute Advanced Metrics
        print("   -> Calculating Centrality Metrics (PageRank, Betweenness, Participation, Modularity Vitality)...")
        pagerank = nx.pagerank(dg) if len(dg) > 0 else {}
        degree_cent = nx.degree_centrality(dg) if len(dg) > 0 else {}
        betweenness = _calculate_betweenness(dg)
        participation = _calculate_participation_coefficient(dg, theme_sets)
        mod_vitality = _calculate_modularity_vitality(ug, theme_sets)

        results = {}
        
        # Execute Path 1: Community Path (Approach A)
        if self.mode in ("both", "community"):
            print("   ==> Executing Path 1: Community Path (Approach A: P_i + Betweenness -> Leiden)...")
            results["community"] = self.execute_community_path(dg, ug, refined_triplets, theme_sets, pagerank, degree_cent, betweenness, participation, mod_vitality)

        # Execute Path 2: Embedding Path (Approach B)
        if self.mode in ("both", "embedding"):
            print("   ==> Executing Path 2: Embedding Path (Approach B: P_i + Modularity Vitality -> Node2Vec)...")
            results["embedding"] = self.execute_embedding_path(dg, ug, refined_triplets, theme_sets, pagerank, degree_cent, betweenness, participation, mod_vitality)

        print("[Topology] Stage 3 Pipeline complete.")
        return results

    def _build_node_metrics(self, dg: nx.DiGraph, ug: nx.Graph, global_hubs: List[str], pagerank: dict, degree_cent: dict, betweenness: dict, participation: dict, mod_vitality: dict) -> Dict[str, NodeMetrics]:
        orphans = [node for node in ug.nodes() if ug.degree(node) <= 1 and node not in global_hubs]
        node_metrics = {}
        for node in dg.nodes():
            in_edges_list = [{"node": u, "predicate": data.get("predicate", "")} for u, v, data in dg.in_edges(node, data=True)]
            out_edges_list = [{"node": v, "predicate": data.get("predicate", "")} for u, v, data in dg.out_edges(node, data=True)]
            
            node_metrics[node] = NodeMetrics(
                node_id=node,
                degree_centrality=degree_cent.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0),
                betweenness_centrality=betweenness.get(node, 0.0),
                participation_coefficient=participation.get(node, 0.0),
                modularity_vitality=mod_vitality.get(node, 0.0),
                is_hub=(node in global_hubs),
                is_orphan=(node in orphans),
                in_edges=in_edges_list,
                out_edges=out_edges_list
            )
        return node_metrics

    def _calculate_theme_inheritance(self, theme_sets: Dict[str, set]) -> List[ThemeInheritance]:
        theme_inheritance = []
        for child_theme, child_nodes in theme_sets.items():
            for parent_theme, parent_nodes in theme_sets.items():
                if child_theme == parent_theme or not child_nodes:
                    continue
                overlap = len(child_nodes.intersection(parent_nodes)) / len(child_nodes)
                if overlap >= self.inheritance_threshold:
                    theme_inheritance.append(ThemeInheritance(
                        parent_theme=parent_theme,
                        child_theme=child_theme,
                        overlap_score=overlap
                    ))
        return theme_inheritance

    # =========================================================================
    # PATH 1: COMMUNITY PATH (Approach A)
    # =========================================================================
    def execute_community_path(self, dg: nx.DiGraph, ug: nx.Graph, refined_triplets: List[dict], theme_sets: dict, pagerank: dict, degree_cent: dict, betweenness: dict, participation: dict, mod_vitality: dict) -> TopologyResult:
        # Hub Selection: High Participation Coefficient (P_i > threshold) OR Top 10% Betweenness Centrality
        betweenness_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        betw_cutoff = max(1, int(len(betweenness_sorted) * (1.0 - self.comm_betw_percentile)))
        top_betweenness_nodes = set([n for n, score in betweenness_sorted[:betw_cutoff] if score > 0])
        
        global_hubs = []
        for node in dg.nodes():
            p_val = participation.get(node, 0.0)
            is_chokepoint = node in top_betweenness_nodes
            if p_val >= self.comm_p_thresh or is_chokepoint:
                global_hubs.append(node)
                
        orphans = [node for node in ug.nodes() if ug.degree(node) <= 1 and node not in global_hubs]
        node_metrics = self._build_node_metrics(dg, ug, global_hubs, pagerank, degree_cent, betweenness, participation, mod_vitality)

        # Subgraph Pruning
        sub_graph = ug.copy()
        sub_graph.remove_nodes_from(global_hubs)
        sub_graph.remove_nodes_from(orphans)
        
        # Leiden Clustering
        communities = []
        if algorithms and len(sub_graph.nodes()) > 0:
            try:
                coms = algorithms.leiden(sub_graph)
                for i, community_nodes in enumerate(coms.communities):
                    communities.append(CommunityPartition(community_id=i, nodes=list(community_nodes)))
            except Exception as e:
                print(f"[Topology Community] Error in Leiden clustering: {e}. Falling back to connected components...")
                for i, comp in enumerate(nx.connected_components(sub_graph)):
                    communities.append(CommunityPartition(community_id=i, nodes=list(comp)))
        else:
            for i, comp in enumerate(nx.connected_components(sub_graph)):
                communities.append(CommunityPartition(community_id=i, nodes=list(comp)))

        theme_inheritance = self._calculate_theme_inheritance(theme_sets)
        
        result = TopologyResult(
            global_hubs=global_hubs,
            communities=communities,
            structural_clusters=[], # Path 1 focuses on communities
            orphans=orphans,
            node_metrics=node_metrics,
            theme_inheritance=theme_inheritance
        )

        out_dir = Path("outputs/03_topology/community")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "topology_partitions.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        # Visualizations
        vis_dir = Path("outputs/visuals/community")
        self._generate_path_visuals(dg, result, theme_sets, refined_triplets, vis_dir=vis_dir, path_type="community", top_betweenness=top_betweenness_nodes)
        
        return result

    # =========================================================================
    # PATH 2: EMBEDDING PATH (Approach B)
    # =========================================================================
    def execute_embedding_path(self, dg: nx.DiGraph, ug: nx.Graph, refined_triplets: List[dict], theme_sets: dict, pagerank: dict, degree_cent: dict, betweenness: dict, participation: dict, mod_vitality: dict) -> TopologyResult:
        # Hub Selection: High Participation Coefficient (P_i > threshold) OR Negative Modularity Vitality (Delta Q < 0)
        global_hubs = []
        for node in dg.nodes():
            p_val = participation.get(node, 0.0)
            q_val = mod_vitality.get(node, 0.0)
            is_mod_pruned = self.enable_mod_vitality and (q_val < 0.0)
            if p_val >= self.emb_p_thresh or is_mod_pruned:
                global_hubs.append(node)
                
        orphans = [node for node in ug.nodes() if ug.degree(node) <= 1 and node not in global_hubs]
        node_metrics = self._build_node_metrics(dg, ug, global_hubs, pagerank, degree_cent, betweenness, participation, mod_vitality)

        # Subgraph Pruning
        sub_graph = ug.copy()
        sub_graph.remove_nodes_from(global_hubs)
        sub_graph.remove_nodes_from(orphans)

        # Node2Vec Embeddings + Dynamic K-Means Silhouette
        structural_clusters = []
        node_embeddings = {}
        if Node2Vec and KMeans and len(sub_graph.nodes()) >= 3:
            try:
                node2vec_model = Node2Vec(sub_graph, dimensions=self.n2v_dim, walk_length=self.n2v_walk_len, num_walks=self.n2v_num_walks, workers=1, quiet=True)
                model = node2vec_model.fit(window=5, min_count=1, batch_words=4)
                
                node_list = list(sub_graph.nodes())
                embeddings = np.array([model.wv[node] for node in node_list])
                for node in node_list:
                    node_embeddings[node] = [float(x) for x in model.wv[node]]
                
                max_clusters = min(self.max_structural_clusters, len(node_list) - 1)
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
                    print(f"      -> Path 2 Optimal K found: {best_k} (Silhouette Score: {best_score:.3f})")
                else:
                    best_k = max_clusters
                    
                final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
                final_labels = final_kmeans.fit_predict(embeddings)
                
                cluster_dict = defaultdict(list)
                for node, label in zip(node_list, final_labels):
                    cluster_dict[label].append(node)
                    
                for c_id, nodes in cluster_dict.items():
                    structural_clusters.append(StructuralCluster(cluster_id=int(c_id), nodes=nodes))
            except Exception as e:
                print(f"[Topology Embedding] Error in Node2Vec clustering: {e}")

        theme_inheritance = self._calculate_theme_inheritance(theme_sets)

        result = TopologyResult(
            global_hubs=global_hubs,
            communities=[], # Path 2 focuses on structural clusters
            structural_clusters=structural_clusters,
            orphans=orphans,
            node_metrics=node_metrics,
            theme_inheritance=theme_inheritance
        )

        out_dir = Path("outputs/03_topology/embedding")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "topology_partitions.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        # Visualizations
        vis_dir = Path("outputs/visuals/embedding")
        self._generate_path_visuals(dg, result, theme_sets, refined_triplets, vis_dir=vis_dir, path_type="embedding", node_embeddings=node_embeddings)

        return result

    # =========================================================================
    # VISUALIZATION ENGINE FOR BOTH PATHS
    # =========================================================================
    def _generate_path_visuals(self, dg: nx.DiGraph, result: TopologyResult, theme_sets: dict, refined_triplets: List[dict], vis_dir: Path, path_type: str, node_embeddings: dict = None, top_betweenness: set = None):
        vis_dir.mkdir(parents=True, exist_ok=True)
        root_vis_dir = Path("outputs/visuals")
        root_vis_dir.mkdir(parents=True, exist_ok=True)
        
        colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
            "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
            "#17becf", "#9edae5"
        ]

        # 1. Standard Topology Graph
        net = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading=f"Standard Topology Graph ({path_type.title()} Path)")
        for node in dg.nodes():
            metrics = result.node_metrics.get(node)
            if not metrics: continue
            is_hub = metrics.is_hub
            is_orphan = metrics.is_orphan
            
            color = "red" if is_hub else "gray" if is_orphan else "#aec7e8"
            shape = "star" if is_hub else "dot"
            size = (metrics.degree_centrality * 200) + (35 if is_hub else 15)
            title = f"Role: {'Hub' if is_hub else 'Orphan' if is_orphan else 'Entity'}\nDegree Cent: {metrics.degree_centrality:.4f}\nP_i: {metrics.participation_coefficient:.3f}"
            net.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=2, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net.get_nodes() and v in net.get_nodes():
                net.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
                
        net.set_options("""{"physics": {"forceAtlas2Based": {"gravitationalConstant": -80, "centralGravity": 0.005, "springLength": 200, "springConstant": 0.08}, "solver": "forceAtlas2Based", "stabilization": {"iterations": 150}}}""")
        net.save_graph(str(vis_dir / "interactive_topology_graph.html"))
        if path_type == "community":
            net.save_graph(str(root_vis_dir / "interactive_topology_graph.html"))

        # 2. Hubs & Ego Networks Visual
        net_hubs = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading=f"Global Hubs & Ego Networks ({path_type.title()} Path)")
        hubs_set = set(result.global_hubs)
        hub_connected_nodes = set(hubs_set)
        for u, v, data in dg.edges(data=True):
            if u in hubs_set or v in hubs_set:
                hub_connected_nodes.add(u); hub_connected_nodes.add(v)
                
        for node in hub_connected_nodes:
            metrics = result.node_metrics.get(node)
            is_hub = node in hubs_set
            centrality = metrics.degree_centrality if metrics else 0.0
            color = "red" if is_hub else "#aec7e8"
            shape = "star" if is_hub else "dot"
            size = (centrality * 200) + (35 if is_hub else 15)
            title = f"Role: {'Global Hub' if is_hub else 'Spoke Node'}\nCentrality: {centrality:.4f}"
            net_hubs.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=2, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net_hubs.get_nodes() and v in net_hubs.get_nodes():
                if u in hubs_set or v in hubs_set:
                    is_inter_hub = u in hubs_set and v in hubs_set
                    net_hubs.add_edge(u, v, title=data.get("predicate", ""), color="rgba(255, 100, 100, 0.8)" if is_inter_hub else "rgba(200,200,200,0.4)", width=3 if is_inter_hub else 1)
        net_hubs.save_graph(str(vis_dir / "interactive_hubs_ego_network.html"))

        # PATH 1 SPECIFIC VISUALIZATIONS
        if path_type == "community":
            # Communities Only
            comm_map = {}
            for comm in result.communities:
                if len(comm.nodes) >= 3:
                    for n in comm.nodes: comm_map[n] = comm.community_id

            net_comm = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading="Communities Topology (Intra-Community Edges Only)")
            for node in dg.nodes():
                if node not in comm_map: continue
                metrics = result.node_metrics.get(node)
                comm_id = comm_map[node]
                color = colors[comm_id % len(colors)]
                net_comm.add_node(node, label=node, color=color, shape="dot", size=25, title=f"Community: {comm_id}", borderWidth=2, shadow=True)
            for u, v, data in dg.edges(data=True):
                if u in net_comm.get_nodes() and v in net_comm.get_nodes() and comm_map.get(u) == comm_map.get(v):
                    net_comm.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_comm.save_graph(str(vis_dir / "interactive_communities_only.html"))

            # NEW: Workflow Narratives (Betweenness Chokepoints)
            net_flow = Network(height="1000px", width="100%", directed=True, bgcolor="#111122", font_color="white", heading="Workflow Narratives & Betweenness Chokepoints")
            for node in dg.nodes():
                metrics = result.node_metrics.get(node)
                if not metrics: continue
                betw = metrics.betweenness_centrality
                is_choke = top_betweenness and (node in top_betweenness)
                color = "#ff4444" if is_choke else "#44aaff"
                shape = "diamond" if is_choke else "dot"
                size = 15 + (betw * 300)
                title = f"Node: {node}\nBetweenness Centrality: {betw:.4f}\nIs Narrative Chokepoint: {is_choke}"
                net_flow.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=3 if is_choke else 1, shadow=True)
            for u, v, data in dg.edges(data=True):
                if u in net_flow.get_nodes() and v in net_flow.get_nodes():
                    net_flow.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,255,0.4)", width=1)
            net_flow.save_graph(str(vis_dir / "interactive_workflow_narratives.html"))

            # NEW: Participation Dispersion Map
            net_part = Network(height="1000px", width="100%", directed=True, bgcolor="#1a1a1a", font_color="white", heading="Participation Coefficient (P_i) Dispersion Map")
            for node in dg.nodes():
                metrics = result.node_metrics.get(node)
                if not metrics: continue
                p_val = metrics.participation_coefficient
                # Scale color from blue (low P_i) to bright orange/yellow (high P_i)
                color = f"hsl({int((1.0 - p_val) * 200)}, 80%, 50%)"
                size = 15 + (p_val * 40)
                title = f"Node: {node}\nParticipation Coefficient (P_i): {p_val:.4f}\nCross-Community Hub: {p_val >= self.comm_p_thresh}"
                net_part.add_node(node, label=node, color=color, shape="star" if p_val >= self.comm_p_thresh else "dot", size=size, title=title, borderWidth=2)
            for u, v, data in dg.edges(data=True):
                if u in net_part.get_nodes() and v in net_part.get_nodes():
                    net_part.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_part.save_graph(str(vis_dir / "interactive_participation_dispersion.html"))

        # PATH 2 SPECIFIC VISUALIZATIONS
        if path_type == "embedding":
            # Structural Clusters Only
            struct_map = {}
            for sc in result.structural_clusters:
                if len(sc.nodes) >= 3:
                    for n in sc.nodes: struct_map[n] = sc.cluster_id

            net_struct = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading="Structural Clusters Topology (Intra-Cluster Edges Only)")
            for node in dg.nodes():
                if node not in struct_map: continue
                sc_id = struct_map[node]
                color = colors[sc_id % len(colors)]
                net_struct.add_node(node, label=node, color=color, shape="dot", size=25, title=f"Structural Cluster: {sc_id}", borderWidth=2, shadow=True)
            for u, v, data in dg.edges(data=True):
                if u in net_struct.get_nodes() and v in net_struct.get_nodes() and struct_map.get(u) == struct_map.get(v):
                    net_struct.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_struct.save_graph(str(vis_dir / "interactive_structural_clusters_only.html"))

            # NEW: Modularity Vitality Landscape
            net_vit = Network(height="1000px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white", heading="Modularity Vitality (Delta Q) Landscape")
            for node in dg.nodes():
                metrics = result.node_metrics.get(node)
                if not metrics: continue
                q_val = metrics.modularity_vitality
                # Color code: Red for negative Delta Q (blurs boundaries), Green for positive Delta Q (strengthens boundaries)
                color = "#ff4444" if q_val < 0 else "#44ff44" if q_val > 0 else "#aaaaaa"
                title = f"Node: {node}\nModularity Vitality (Delta Q): {q_val:.4f}\nPruned as Hub: {q_val < 0}"
                net_vit.add_node(node, label=node, color=color, shape="box" if q_val < 0 else "dot", size=25, title=title, borderWidth=2)
            for u, v, data in dg.edges(data=True):
                if u in net_vit.get_nodes() and v in net_vit.get_nodes():
                    net_vit.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_vit.save_graph(str(vis_dir / "interactive_modularity_vitality_landscape.html"))

            # Node2Vec 2D Embeddings Scatter Plot
            if node_embeddings and len(node_embeddings) >= 2:
                try:
                    from sklearn.decomposition import PCA
                    nodes_list = list(node_embeddings.keys())
                    embeddings_matrix = np.array([node_embeddings[n] for n in nodes_list])
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(embeddings_matrix)
                    
                    node_clusters = {}
                    for sc in result.structural_clusters:
                        for n in sc.nodes: node_clusters[n] = sc.cluster_id
                    
                    plotly_data = []
                    for idx, name in enumerate(nodes_list):
                        c_id = node_clusters.get(name, -1)
                        plotly_data.append({"name": name, "x": float(coords[idx, 0]), "y": float(coords[idx, 1]), "cluster": int(c_id)})
                    
                    plotly_json = json.dumps(plotly_data)
                    html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Node2Vec 2D Embedding Space</title><script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script><style>body {{ background-color: #222222; color: #ffffff; font-family: sans-serif; margin: 0; padding: 20px; }} #plot {{ width: 100%; height: 800px; }}</style></head><body><h1>Node2Vec 2D Embedding Space</h1><div id="plot"></div><script>const data = {plotly_json}; const tracesMap = {{}}; data.forEach(item => {{ const c = item.cluster; if (!tracesMap[c]) {{ tracesMap[c] = {{ x: [], y: [], text: [], mode: 'markers+text', textposition: 'top center', name: 'Cluster ' + c, type: 'scatter', marker: {{ size: 14, opacity: 0.8 }} }}; }} tracesMap[c].x.push(item.x); tracesMap[c].y.push(item.y); tracesMap[c].text.push(item.name); }}); Plotly.newPlot('plot', Object.values(tracesMap), {{ paper_bgcolor: '#222222', plot_bgcolor: '#222222', xaxis: {{ gridcolor: '#444' }}, yaxis: {{ gridcolor: '#444' }} }});</script></body></html>"""
                    with open(vis_dir / "interactive_node2vec_embeddings.html", "w") as f:
                        f.write(html_content)
                except Exception as e:
                    print(f"Warning: Failed to generate Node2Vec Scatter Plot: {e}")
