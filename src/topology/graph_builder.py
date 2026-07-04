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


def _calculate_participation_coefficient(dg: nx.DiGraph, refined_triplets: List[dict]) -> Dict[str, float]:
    """
    Computes Participation Coefficient P_i for each node based on incident edge themes:
    P_i = 1 - sum( (k_{i,m} / k_i)^2 )
    where k_i is total degree of node i, and k_{i,m} is incident edge count associated with theme m.
    """
    node_edge_themes = defaultdict(list)
    for t in refined_triplets:
        subj = t.get('subject')
        obj = t.get('object')
        theme = t.get('theme_association', 'Other')
        if not theme or theme == 'Other':
            theme = 'Unassigned'
        if subj:
            node_edge_themes[str(subj).lower().strip()].append(theme)
        if obj:
            node_edge_themes[str(obj).lower().strip()].append(theme)
            
    p_scores = {}
    for node in dg.nodes():
        themes = node_edge_themes.get(node, [])
        k_i = len(themes)
        if k_i < 3:
            p_scores[node] = 0.0
            continue
        counts = defaultdict(int)
        for th in themes:
            counts[th] += 1
        sum_sq = sum((c / k_i) ** 2 for c in counts.values())
        p_scores[node] = float(max(0.0, 1.0 - sum_sq))
    return p_scores


def _calculate_betweenness(dg: nx.DiGraph) -> Dict[str, float]:
    """Computes shortest-path Betweenness Centrality for directed graph."""
    if len(dg) == 0:
        return {}
    if len(dg) > 500:
        k_sample = min(100, max(50, int(len(dg) * 0.1)))
        return nx.betweenness_centrality(dg, k=k_sample)
    return nx.betweenness_centrality(dg)


def _calculate_modularity_vitality(ug: nx.Graph, refined_triplets: List[dict]) -> Dict[str, float]:
    """
    Computes Delta Q(v) = Q(G) - Q(G \ v) for each node v in ug relative to primary theme partition.
    Negative value means removing node v INCREASES modularity Q (node blurs primary theme boundaries).
    """
    vitality = {}
    if len(ug) < 3:
        for n in ug.nodes(): vitality[n] = 0.0
        return vitality
        
    node_edge_themes = defaultdict(list)
    for t in refined_triplets:
        subj = t.get('subject')
        obj = t.get('object')
        theme = t.get('theme_association', 'Other')
        if not theme or theme == 'Other':
            theme = 'Unassigned'
        if subj:
            node_edge_themes[str(subj).lower().strip()].append(theme)
        if obj:
            node_edge_themes[str(obj).lower().strip()].append(theme)
            
    primary_theme = {}
    for node in ug.nodes():
        themes = node_edge_themes.get(node, [])
        if not themes:
            primary_theme[node] = 'Unassigned'
        else:
            counts = defaultdict(int)
            for th in themes: counts[th] += 1
            primary_theme[node] = max(counts.items(), key=lambda x: x[1])[0]
            
    theme_groups = defaultdict(list)
    for node, p_th in primary_theme.items():
        theme_groups[p_th].append(node)
        
    comms = [nodes for nodes in theme_groups.values() if len(nodes) > 0]
    if len(comms) < 2:
        for n in ug.nodes(): vitality[n] = 0.0
        return vitality
        
    try:
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

        # Visual Pruning & Rendering Config (HTML files ONLY - 0 impact on JSON partitions)
        self.vis_cfg = self.config.get('visualizations', {})
        self.min_node_degree = self.vis_cfg.get('min_node_degree', 0)
        self.max_nodes_per_cluster = self.vis_cfg.get('max_nodes_per_cluster', 0)
        self.max_total_visual_nodes = self.vis_cfg.get('max_total_visual_nodes', 0)
        self.freeze_physics = self.vis_cfg.get('freeze_physics', True)

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
        participation = _calculate_participation_coefficient(dg, refined_triplets)
        mod_vitality = _calculate_modularity_vitality(ug, refined_triplets)

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
        betweenness_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        betw_cutoff = max(1, int(len(betweenness_sorted) * (1.0 - self.comm_betw_percentile)))
        top_betweenness_nodes = set([n for n, score in betweenness_sorted[:betw_cutoff] if score > 0.02])
        
        global_hubs = []
        for node in dg.nodes():
            p_val = participation.get(node, 0.0)
            is_chokepoint = node in top_betweenness_nodes
            if p_val >= self.comm_p_thresh or is_chokepoint:
                global_hubs.append(node)
                
        orphans = [node for node in ug.nodes() if ug.degree(node) <= 1 and node not in global_hubs]
        node_metrics = self._build_node_metrics(dg, ug, global_hubs, pagerank, degree_cent, betweenness, participation, mod_vitality)

        sub_graph = ug.copy()
        sub_graph.remove_nodes_from(global_hubs)
        sub_graph.remove_nodes_from(orphans)
        
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
            structural_clusters=[],
            orphans=orphans,
            node_metrics=node_metrics,
            theme_inheritance=theme_inheritance
        )

        out_dir = Path("outputs/03_topology/community")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "topology_partitions.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        # Visualizations (Filtered ONLY for HTML output)
        vis_dir = Path("outputs/visuals/community")
        self._generate_path_visuals(dg, result, theme_sets, refined_triplets, vis_dir=vis_dir, path_type="community", top_betweenness=top_betweenness_nodes)
        
        return result

    # =========================================================================
    # PATH 2: EMBEDDING PATH (Approach B)
    # =========================================================================
    def execute_embedding_path(self, dg: nx.DiGraph, ug: nx.Graph, refined_triplets: List[dict], theme_sets: dict, pagerank: dict, degree_cent: dict, betweenness: dict, participation: dict, mod_vitality: dict) -> TopologyResult:
        mod_v_sorted = sorted(mod_vitality.items(), key=lambda x: x[1])
        negative_mod_v_hubs = set([n for n, val in mod_v_sorted[:5] if val < -0.005 and dg.degree(n) >= 3])
        
        global_hubs = []
        for node in dg.nodes():
            p_val = participation.get(node, 0.0)
            is_mod_pruned = self.enable_mod_vitality and (node in negative_mod_v_hubs)
            if p_val >= self.emb_p_thresh or is_mod_pruned:
                global_hubs.append(node)
                
        orphans = [node for node in ug.nodes() if ug.degree(node) <= 1 and node not in global_hubs]
        node_metrics = self._build_node_metrics(dg, ug, global_hubs, pagerank, degree_cent, betweenness, participation, mod_vitality)

        sub_graph = ug.copy()
        sub_graph.remove_nodes_from(global_hubs)
        sub_graph.remove_nodes_from(orphans)

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
            communities=[],
            structural_clusters=structural_clusters,
            orphans=orphans,
            node_metrics=node_metrics,
            theme_inheritance=theme_inheritance
        )

        out_dir = Path("outputs/03_topology/embedding")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "topology_partitions.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        # Visualizations (Filtered ONLY for HTML output)
        vis_dir = Path("outputs/visuals/embedding")
        self._generate_path_visuals(dg, result, theme_sets, refined_triplets, vis_dir=vis_dir, path_type="embedding", node_embeddings=node_embeddings)

        return result

    # =========================================================================
    # VISUALIZATION ENGINE FOR BOTH PATHS (HTML OUTPUTS ONLY)
    # =========================================================================
    def _generate_path_visuals(self, dg: nx.DiGraph, result: TopologyResult, theme_sets: dict, refined_triplets: List[dict], vis_dir: Path, path_type: str, node_embeddings: dict = None, top_betweenness: set = None):
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        path_label = "Path 1: Community Workflow Path" if path_type == "community" else "Path 2: Embedding Categorical Path"
        
        colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
            "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
            "#17becf", "#9edae5"
        ]

        # Apply Visual Degree Pruning (HTML ONLY)
        candidate_nodes = set()
        for node in dg.nodes():
            metrics = result.node_metrics.get(node)
            if not metrics: continue
            is_hub = metrics.is_hub
            deg = dg.degree(node)
            if is_hub or self.min_node_degree == 0 or deg >= self.min_node_degree:
                candidate_nodes.add(node)

        # Apply Max Visual Nodes Cap if configured (Supports fraction e.g. 0.25 = top 25%, or hard integer e.g. 200)
        if self.max_total_visual_nodes > 0:
            if isinstance(self.max_total_visual_nodes, float) and self.max_total_visual_nodes <= 1.0:
                target_max = max(10, int(len(dg) * self.max_total_visual_nodes))
            else:
                target_max = int(self.max_total_visual_nodes)
                
            if len(candidate_nodes) > target_max:
                hubs_in_cand = set(result.global_hubs)
                non_hubs = candidate_nodes - hubs_in_cand
                sorted_non_hubs = sorted(non_hubs, key=lambda n: result.node_metrics[n].degree_centrality, reverse=True)
                candidate_nodes = hubs_in_cand.union(set(sorted_non_hubs[:max(0, target_max - len(hubs_in_cand))]))

        options_json = json.dumps({
            "physics": {
                "enabled": True,
                "forceAtlas2Based": {
                    "gravitationalConstant": -80,
                    "centralGravity": 0.005,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": True,
                    "iterations": 150,
                    "fit": True
                }
            }
        })

        # 1. Standard Topology Graph
        net = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading=f"[{path_label}] Standard Topology Graph")
        for node in candidate_nodes:
            metrics = result.node_metrics.get(node)
            if not metrics: continue
            is_hub = metrics.is_hub
            is_orphan = metrics.is_orphan
            color = "red" if is_hub else "gray" if is_orphan else "#aec7e8"
            shape = "star" if is_hub else "dot"
            size = (metrics.degree_centrality * 200) + (35 if is_hub else 15)
            title = f"[{path_label}]\nRole: {'Hub' if is_hub else 'Orphan' if is_orphan else 'Entity'}\nDegree Cent: {metrics.degree_centrality:.4f}\nP_i: {metrics.participation_coefficient:.3f}"
            net.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=2, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net.get_nodes() and v in net.get_nodes():
                net.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
                
        net.set_options(options_json)
        net.save_graph(str(vis_dir / "interactive_topology_graph.html"))

        # 2. Hubs & Ego Networks Visual
        net_hubs = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading=f"[{path_label}] Global Hubs & Ego Networks")
        hubs_set = set(result.global_hubs)
        hub_connected_nodes = set(hubs_set)
        for u, v, data in dg.edges(data=True):
            if u in hubs_set or v in hubs_set:
                if u in candidate_nodes and v in candidate_nodes:
                    hub_connected_nodes.add(u); hub_connected_nodes.add(v)
                
        for node in hub_connected_nodes:
            metrics = result.node_metrics.get(node)
            is_hub = node in hubs_set
            centrality = metrics.degree_centrality if metrics else 0.0
            color = "red" if is_hub else "#aec7e8"
            shape = "star" if is_hub else "dot"
            size = (centrality * 200) + (35 if is_hub else 15)
            title = f"[{path_label}]\nRole: {'Global Hub' if is_hub else 'Spoke Node'}\nCentrality: {centrality:.4f}"
            net_hubs.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=2, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net_hubs.get_nodes() and v in net_hubs.get_nodes():
                if u in hubs_set or v in hubs_set:
                    is_inter_hub = u in hubs_set and v in hubs_set
                    net_hubs.add_edge(u, v, title=data.get("predicate", ""), color="rgba(255, 100, 100, 0.8)" if is_inter_hub else "rgba(200,200,200,0.4)", width=3 if is_inter_hub else 1)
        net_hubs.set_options(options_json)
        net_hubs.save_graph(str(vis_dir / "interactive_hubs_ego_network.html"))

        # 2b. Global Hubs Only Topology Graph (Inter-Hub Edges Only)
        net_hubs_only = Network(height="1000px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white", heading=f"[{path_label}] Global Hubs Only Topology")
        for node in hubs_set:
            metrics = result.node_metrics.get(node)
            centrality = metrics.degree_centrality if metrics else 0.0
            p_val = metrics.participation_coefficient if metrics else 0.0
            size = (centrality * 200) + 35
            title = f"[{path_label}]\nRole: Global Hub\nDegree Centrality: {centrality:.4f}\nParticipation Coeff (P_i): {p_val:.4f}"
            net_hubs_only.add_node(node, label=node, color="#ff4444", shape="star", size=size, title=title, borderWidth=3, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in hubs_set and v in hubs_set:
                net_hubs_only.add_edge(u, v, title=data.get("predicate", ""), color="rgba(255, 215, 0, 0.8)", width=3)
                
        net_hubs_only.set_options(options_json)
        net_hubs_only.save_graph(str(vis_dir / "interactive_global_hubs.html"))

        # PATH 1 SPECIFIC VISUALIZATIONS
        if path_type == "community":
            comm_map = {}
            for comm in result.communities:
                nodes_in_c = [n for n in comm.nodes if n in candidate_nodes]
                if self.max_nodes_per_cluster > 0 and len(nodes_in_c) > self.max_nodes_per_cluster:
                    nodes_in_c = sorted(nodes_in_c, key=lambda n: result.node_metrics[n].degree_centrality, reverse=True)[:self.max_nodes_per_cluster]
                for n in nodes_in_c: comm_map[n] = comm.community_id

            net_comm = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading=f"[{path_label}] Communities Topology (Intra-Community Edges Only)")
            for node in candidate_nodes:
                if node not in comm_map: continue
                comm_id = comm_map[node]
                color = colors[comm_id % len(colors)]
                net_comm.add_node(node, label=node, color=color, shape="dot", size=25, title=f"[{path_label}]\nCommunity: {comm_id}", borderWidth=2, shadow=True)
            for u, v, data in dg.edges(data=True):
                if u in net_comm.get_nodes() and v in net_comm.get_nodes() and comm_map.get(u) == comm_map.get(v):
                    net_comm.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_comm.set_options(options_json)
            net_comm.save_graph(str(vis_dir / "interactive_communities_only.html"))

            # Workflow Narratives (Betweenness Chokepoints)
            net_flow = Network(height="1000px", width="100%", directed=True, bgcolor="#111122", font_color="white", heading=f"[{path_label}] Workflow Narratives & Betweenness Chokepoints")
            for node in candidate_nodes:
                metrics = result.node_metrics.get(node)
                if not metrics: continue
                betw = metrics.betweenness_centrality
                is_choke = top_betweenness and (node in top_betweenness)
                color = "#ff4444" if is_choke else "#44aaff"
                shape = "diamond" if is_choke else "dot"
                size = 15 + (betw * 300)
                title = f"[{path_label}]\nNode: {node}\nBetweenness Centrality: {betw:.4f}\nIs Narrative Chokepoint: {is_choke}"
                net_flow.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=3 if is_choke else 1, shadow=True)
            for u, v, data in dg.edges(data=True):
                if u in net_flow.get_nodes() and v in net_flow.get_nodes():
                    net_flow.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,255,0.4)", width=1)
            net_flow.set_options(options_json)
            net_flow.save_graph(str(vis_dir / "interactive_workflow_narratives.html"))

            # Participation Dispersion Map
            net_part = Network(height="1000px", width="100%", directed=True, bgcolor="#1a1a1a", font_color="white", heading=f"[{path_label}] Participation Coefficient (P_i) Dispersion Map")
            for node in candidate_nodes:
                metrics = result.node_metrics.get(node)
                if not metrics: continue
                p_val = metrics.participation_coefficient
                color = f"hsl({int((1.0 - p_val) * 200)}, 80%, 50%)"
                size = 15 + (p_val * 40)
                title = f"[{path_label}]\nNode: {node}\nParticipation Coefficient (P_i): {p_val:.4f}\nCross-Community Hub: {p_val >= self.comm_p_thresh}"
                net_part.add_node(node, label=node, color=color, shape="star" if p_val >= self.comm_p_thresh else "dot", size=size, title=title, borderWidth=2)
            for u, v, data in dg.edges(data=True):
                if u in net_part.get_nodes() and v in net_part.get_nodes():
                    net_part.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_part.set_options(options_json)
            net_part.save_graph(str(vis_dir / "interactive_participation_dispersion.html"))

        # PATH 2 SPECIFIC VISUALIZATIONS
        if path_type == "embedding":
            struct_map = {}
            for sc in result.structural_clusters:
                nodes_in_sc = [n for n in sc.nodes if n in candidate_nodes]
                if self.max_nodes_per_cluster > 0 and len(nodes_in_sc) > self.max_nodes_per_cluster:
                    nodes_in_sc = sorted(nodes_in_sc, key=lambda n: result.node_metrics[n].degree_centrality, reverse=True)[:self.max_nodes_per_cluster]
                for n in nodes_in_sc: struct_map[n] = sc.cluster_id

            net_struct = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading=f"[{path_label}] Structural Clusters Topology (Intra-Cluster Edges Only)")
            for node in candidate_nodes:
                if node not in struct_map: continue
                sc_id = struct_map[node]
                color = colors[sc_id % len(colors)]
                net_struct.add_node(node, label=node, color=color, shape="dot", size=25, title=f"[{path_label}]\nStructural Cluster: {sc_id}", borderWidth=2, shadow=True)
            for u, v, data in dg.edges(data=True):
                if u in net_struct.get_nodes() and v in net_struct.get_nodes() and struct_map.get(u) == struct_map.get(v):
                    net_struct.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_struct.set_options(options_json)
            net_struct.save_graph(str(vis_dir / "interactive_structural_clusters_only.html"))

            # Modularity Vitality Landscape
            net_vit = Network(height="1000px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white", heading=f"[{path_label}] Modularity Vitality (Delta Q) Landscape")
            for node in candidate_nodes:
                metrics = result.node_metrics.get(node)
                if not metrics: continue
                q_val = metrics.modularity_vitality
                color = "#ff4444" if q_val < 0 else "#44ff44" if q_val > 0 else "#aaaaaa"
                title = f"[{path_label}]\nNode: {node}\nModularity Vitality (Delta Q): {q_val:.4f}\nPruned as Hub: {q_val < 0}"
                net_vit.add_node(node, label=node, color=color, shape="box" if q_val < 0 else "dot", size=25, title=title, borderWidth=2)
            for u, v, data in dg.edges(data=True):
                if u in net_vit.get_nodes() and v in net_vit.get_nodes():
                    net_vit.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)
            net_vit.set_options(options_json)
            net_vit.save_graph(str(vis_dir / "interactive_modularity_vitality_landscape.html"))

            # Node2Vec 2D Embeddings Scatter Plot
            if node_embeddings and len(node_embeddings) >= 2:
                try:
                    from sklearn.decomposition import PCA
                    nodes_list = [n for n in node_embeddings.keys() if n in candidate_nodes]
                    if len(nodes_list) >= 2:
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
                        html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>[{path_label}] Node2Vec 2D Embedding Space</title><script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script><style>body {{ background-color: #222222; color: #ffffff; font-family: sans-serif; margin: 0; padding: 20px; }} #plot {{ width: 100%; height: 800px; }}</style></head><body><h1>[{path_label}] Node2Vec 2D Embedding Space</h1><div id="plot"></div><script>const data = {plotly_json}; const tracesMap = {{}}; data.forEach(item => {{ const c = item.cluster; if (!tracesMap[c]) {{ tracesMap[c] = {{ x: [], y: [], text: [], mode: 'markers+text', textposition: 'top center', name: 'Cluster ' + c, type: 'scatter', marker: {{ size: 14, opacity: 0.8 }} }}; }} tracesMap[c].x.push(item.x); tracesMap[c].y.push(item.y); tracesMap[c].text.push(item.name); }}); Plotly.newPlot('plot', Object.values(tracesMap), {{ paper_bgcolor: '#222222', plot_bgcolor: '#222222', xaxis: {{ gridcolor: '#444' }}, yaxis: {{ gridcolor: '#444' }} }});</script></body></html>"""
                        with open(vis_dir / "interactive_node2vec_embeddings.html", "w") as f:
                            f.write(html_content)
                except Exception as e:
                    print(f"Warning: Failed to generate Node2Vec Scatter Plot: {e}")

        # NEW VISUALIZATIONS:
        # A. Stage 4 LLM Payload Gallery (Isolated Cluster Cards)
        self._generate_llm_payload_gallery(dg, result, refined_triplets, vis_dir, path_label, path_type)
        
        # B. Collapsed Module Architecture Diagram
        self._generate_collapsed_modules_graph(dg, result, vis_dir, path_label, path_type)

    def _generate_llm_payload_gallery(self, dg: nx.DiGraph, result: TopologyResult, refined_triplets: List[dict], vis_dir: Path, path_label: str, path_type: str):
        clusters = result.communities if path_type == "community" else result.structural_clusters
        agent_name = "leiden_schema_agent" if path_type == "community" else "node2vec_schema_agent"
        cluster_type_label = "Community" if path_type == "community" else "Structural Cluster"
        
        payload_data = []
        for cluster in clusters:
            c_id = getattr(cluster, 'community_id', getattr(cluster, 'cluster_id', 0))
            nodes = cluster.nodes
            
            # Find triplets associated with this cluster
            nodes_set = set(nodes)
            associated_triplets = []
            for t in refined_triplets:
                s = str(t.get('subject', '')).lower().strip()
                o = str(t.get('object', '')).lower().strip()
                if s in nodes_set or o in nodes_set:
                    associated_triplets.append({"subject": s, "predicate": t.get('predicate', ''), "object": o})
                    
            connected_hubs = [h for h in result.global_hubs if any(dg.has_edge(h, n) or dg.has_edge(n, h) for n in nodes)]
            
            payload_data.append({
                "id": c_id,
                "module_name": f"01_{path_type}_{c_id}.py",
                "nodes": nodes,
                "triplet_count": len(associated_triplets),
                "triplets": associated_triplets, # full triplets array for vis.js edges
                "sample_triplets": associated_triplets[:15], # top 15 for sidebar text preview
                "connected_hubs": connected_hubs
            })
            
        payload_json = json.dumps(payload_data)
        hubs_json = json.dumps(result.global_hubs)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>[{path_label}] Stage 4 LLM Payload Gallery</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
        body {{ background-color: #121212; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ margin-top: 0; color: #44aaff; font-size: 24px; }}
        .subtitle {{ color: #aaaaaa; margin-bottom: 20px; font-size: 14px; }}
        .container {{ display: flex; gap: 20px; height: 800px; }}
        .sidebar {{ width: 350px; background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; overflow-y: auto; }}
        .canvas-container {{ flex: 1; background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; position: relative; }}
        #network {{ width: 100%; height: 100%; }}
        select {{ width: 100%; padding: 10px; background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; font-size: 15px; margin-bottom: 15px; cursor: pointer; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px; background: #333; color: #44aaff; }}
        .card {{ background: #262626; padding: 12px; border-radius: 6px; margin-bottom: 10px; border-left: 4px solid #44aaff; }}
        .triplet-item {{ font-size: 12px; color: #cccccc; padding: 4px 0; border-bottom: 1px solid #333; }}
        code {{ color: #ff9933; font-family: monospace; }}
    </style>
</head>
<body>
    <h1>[{path_label}] Stage 4 LLM Payload Gallery</h1>
    <div class="subtitle">Direct visualization of individual cluster subgraphs as passed to <code>{agent_name}</code> for Stage 4 Pydantic schema synthesis.</div>
    
    <div class="container">
        <div class="sidebar">
            <label for="clusterSelect"><b>Select Partition Payload:</b></label>
            <select id="clusterSelect" onchange="loadCluster(this.value)"></select>
            
            <div id="payloadDetails"></div>
        </div>
        <div class="canvas-container">
            <div id="network"></div>
        </div>
    </div>

    <script>
        const payloads = {payload_json};
        const globalHubs = new Set({hubs_json});
        const selectEl = document.getElementById('clusterSelect');
        const detailsEl = document.getElementById('payloadDetails');
        let network = null;

        const defaultOpt = document.createElement('option');
        defaultOpt.value = "";
        defaultOpt.textContent = "-- Select a Partition Payload --";
        defaultOpt.disabled = true;
        defaultOpt.selected = true;
        selectEl.appendChild(defaultOpt);

        payloads.forEach((p, idx) => {{
            const opt = document.createElement('option');
            opt.value = idx;
            opt.textContent = `{cluster_type_label} ${{p.id}} (${{p.nodes.length}} entities)`;
            selectEl.appendChild(opt);
        }});

        function loadCluster(idx) {{
            if (idx === "" || idx === null || idx === undefined) return;
            const payload = payloads[idx];
            
            // Render Details Panel
            detailsEl.innerHTML = `
                <div class="card">
                    <span class="badge">LLM Agent</span> <code>{agent_name}</code><br>
                    <span class="badge">Output Schema</span> <code>${{payload.module_name}}</code><br>
                    <span class="badge">Entities</span> <b>${{payload.nodes.length}}</b> | <span class="badge">Triplets</span> <b>${{payload.triplet_count}}</b>
                </div>
                <h4>Global Hub Anchors:</h4>
                <p>${{payload.connected_hubs.length > 0 ? payload.connected_hubs.map(h => `<span class="badge" style="color:#ff5555;">${{h}}</span>`).join(' ') : '<i>None</i>'}}</p>
                <h4>Sample Incident Triplets:</h4>
                ${{payload.sample_triplets.map(t => `<div class="triplet-item"><b>${{t.subject}}</b> <i>--[${{t.predicate}}]--></i> <b>${{t.object}}</b></div>`).join('')}}
            `;

            // Draw Subgraph
            const nodes = [];
            const edges = [];
            const addedNodes = new Set();

            payload.nodes.forEach(n => {{
                addedNodes.add(n);
                nodes.push({{ id: n, label: n, color: '#44aaff', shape: 'dot', size: 25 }});
            }});

            payload.connected_hubs.forEach(h => {{
                if (!addedNodes.has(h)) {{
                    addedNodes.add(h);
                    nodes.push({{ id: h, label: h, color: '#ff4444', shape: 'star', size: 35, title: 'Global Hub Anchor' }});
                }}
            }});

            payload.triplets.forEach(t => {{
                if (addedNodes.has(t.subject) && addedNodes.has(t.object)) {{
                    edges.push({{ from: t.subject, to: t.object, title: t.predicate, arrows: 'to', color: 'rgba(200,200,200,0.5)' }});
                }}
            }});

            const container = document.getElementById('network');
            const data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
            const options = {{ physics: {{ enabled: true, solver: 'forceAtlas2Based' }} }};
            
            if (network) network.destroy();
            network = new vis.Network(container, data, options);
        }}

        detailsEl.innerHTML = '<div style="color: #888888; font-style: italic; padding: 20px 0;">Select a partition payload from the dropdown above to view its discrete subgraph and LLM metadata.</div>';
    </script>
</body>
</html>"""
        with open(vis_dir / "interactive_llm_payload_gallery.html", "w") as f:
            f.write(html_content)

    def _generate_collapsed_modules_graph(self, dg: nx.DiGraph, result: TopologyResult, vis_dir: Path, path_label: str, path_type: str):
        clusters = result.communities if path_type == "community" else result.structural_clusters
        cluster_label = "Community" if path_type == "community" else "Cluster"
        
        net = Network(height="1000px", width="100%", directed=True, bgcolor="#181824", font_color="white", heading=f"[{path_label}] Collapsed Module Architecture Diagram")
        
        # Add Global Hubs as Star Nodes
        hubs_set = set(result.global_hubs)
        for h in hubs_set:
            metrics = result.node_metrics.get(h)
            cent = metrics.degree_centrality if metrics else 0.0
            net.add_node(h, label=f"HUB: {h}", color="#ff4444", shape="star", size=40, title=f"Global Hub\nCentrality: {cent:.4f}", borderWidth=2, shadow=True)
            
        # Add Clusters as Collapsed Box Nodes
        for cluster in clusters:
            c_id = getattr(cluster, 'community_id', getattr(cluster, 'cluster_id', 0))
            nodes = cluster.nodes
            mod_id = f"Module_{c_id}"
            label = f"{cluster_label} {c_id}\n({len(nodes)} entities)"
            title = f"{cluster_label} {c_id} Members:\n" + "\n".join(nodes[:10]) + (f"\n... and {len(nodes)-10} more" if len(nodes) > 10 else "")
            net.add_node(mod_id, label=label, color="#44aaff", shape="box", size=30, title=title, borderWidth=2, shadow=True)
            
            # Connect Hubs to Modules
            for h in hubs_set:
                if any(dg.has_edge(h, n) or dg.has_edge(n, h) for n in nodes):
                    net.add_edge(h, mod_id, color="rgba(255, 100, 100, 0.7)", width=2)
                    
        net.set_options("""{"physics": {"forceAtlas2Based": {"gravitationalConstant": -120, "centralGravity": 0.01, "springLength": 250}, "solver": "forceAtlas2Based"}}""")
        net.save_graph(str(vis_dir / "interactive_collapsed_modules.html"))
