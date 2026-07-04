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
        node_embeddings = {}
        if Node2Vec and KMeans and len(sub_graph.nodes()) >= 3:
            try:
                # 1. Generate Embeddings
                node2vec_model = Node2Vec(sub_graph, dimensions=64, walk_length=10, num_walks=100, workers=1, quiet=True)
                model = node2vec_model.fit(window=5, min_count=1, batch_words=4)
                
                # Extract nodes and their corresponding vectors
                node_list = list(sub_graph.nodes())
                embeddings = np.array([model.wv[node] for node in node_list])
                
                for node in node_list:
                    node_embeddings[node] = [float(x) for x in model.wv[node]]
                
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
        self._generate_visuals(dg, result, theme_sets, refined_triplets, node_embeddings=node_embeddings)

        print("[Topology] Pipeline complete.")
        return result

    def _generate_visuals(self, dg: nx.DiGraph, result: TopologyResult, theme_sets: dict, refined_triplets: List[dict], node_embeddings: dict = None):
        vis_dir = Path("outputs/visuals")
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Extended high-contrast hex palette (matches matplotlib tab20)
        colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
            "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
            "#17becf", "#9edae5"
        ]

        # Load normalized triplets to retrieve subject nodes and theme associations
        norm_triplets_path = Path("outputs/02_refinement/normalized_triplets.json")
        if not norm_triplets_path.exists():
            norm_triplets_path = Path(__file__).parent.parent.parent / "outputs" / "02_refinement" / "normalized_triplets.json"

        norm_triplets = []
        if norm_triplets_path.exists():
            try:
                with open(norm_triplets_path, "r") as f:
                    norm_triplets = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load normalized triplets for theme mapping: {e}")

        # Load theme_mapping_clusters.json mapping "master theme" -> list of associated themes
        theme_map_path = Path("outputs/02_refinement/theme_mapping_clusters.json")
        if not theme_map_path.exists():
            theme_map_path = Path(__file__).parent.parent.parent / "outputs" / "02_refinement" / "theme_mapping_clusters.json"

        theme_clusters = {}
        if theme_map_path.exists():
            try:
                with open(theme_map_path, "r") as f:
                    theme_clusters = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load theme mapping clusters: {e}")

        def get_master_theme(orig_theme: str) -> str:
            if not orig_theme:
                return None
            orig_clean = orig_theme.strip().lower()
            for master, low_levels in theme_clusters.items():
                for low in low_levels:
                    if isinstance(low, str) and low.strip().lower() == orig_clean:
                        return master
                if master.strip().lower() == orig_clean:
                    return master
            # Substring fallback for robustness
            for master, low_levels in theme_clusters.items():
                for low in low_levels:
                    if isinstance(low, str):
                        low_clean = low.strip().lower()
                        if low_clean in orig_clean or orig_clean in low_clean:
                            return master
            # Word overlap fallback for spelling/formatting variations
            orig_words = set(orig_clean.replace("&", "and").replace(",", "").split())
            if len(orig_words) > 1:
                for master, low_levels in theme_clusters.items():
                    master_words = set(master.strip().lower().replace("&", "and").replace(",", "").split())
                    if len(orig_words.intersection(master_words)) / len(orig_words) >= 0.7:
                        return master
                    for low in low_levels:
                        if isinstance(low, str):
                            low_words = set(low.strip().lower().replace("&", "and").replace(",", "").split())
                            if len(orig_words.intersection(low_words)) / len(orig_words) >= 0.7:
                                return master
            return None

        # Retrieve normalized_triplets.json subject nodes and "theme association"
        # and resolve them to "master theme" for counter building
        node_themes = defaultdict(list)
        for idx, ref_t in enumerate(refined_triplets):
            if idx < len(norm_triplets):
                orig_t = norm_triplets[idx]
                orig_theme = orig_t.get('theme_association')
                if orig_theme and orig_theme != "Other":
                    master_theme = get_master_theme(orig_theme)
                    if master_theme:
                        ref_subj = str(ref_t.get('subject', '')).lower().strip()
                        node_themes[ref_subj].append(master_theme)

        # Pre-calculate node mappings for structural clusters
        struct_map = {}
        if result.structural_clusters:
            for sc in result.structural_clusters:
                # Apply size filter for structural clusters similar to communities
                if len(sc.nodes) >= 3:
                    for node in sc.nodes:
                        struct_map[node] = sc.cluster_id

        # Pre-calculate representative terms for communities (size >= 3)
        community_reps = {}
        for comm in result.communities:
            if len(comm.nodes) < 3:
                continue
            comm_id = comm.community_id
            comm_nodes = comm.nodes
            
            node_centralities = []
            for n in comm_nodes:
                m = result.node_metrics.get(n)
                cent = m.pagerank if m else 0.0
                node_centralities.append((n, cent))
            node_centralities.sort(key=lambda x: x[1], reverse=True)
            
            top_nodes = [n for n, _ in node_centralities[:5]]
            community_reps[comm_id] = {
                "top_nodes": top_nodes,
                "top_nodes_str": ", ".join(top_nodes)
            }
        
        # 1. Standard Topology Visual
        net = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white")
        
        community_map = {}
        for comm in result.communities:
            if len(comm.nodes) >= 3:
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
            
            title = f"Role: {'Hub' if is_hub else 'Orphan' if is_orphan else 'Entity'}\nCommunity: {comm_id}\nCentrality: {centrality:.4f}"
            
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



        # 5. Communities Only Visual (Option A) - with Core Representatives Highlighted
        net_comm = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading="Communities Topology (Intra-Community Edges Only)")
        
        for node in dg.nodes():
            if node not in community_map:
                continue
            metrics = result.node_metrics.get(node)
            if not metrics:
                continue
            comm_id = community_map[node]
            centrality = metrics.degree_centrality
            
            reps_info = community_reps.get(comm_id, {"top_nodes": [], "top_nodes_str": ""})
            is_rep = node in reps_info["top_nodes"][:3] # Highlight top 3 central nodes
            
            color = colors[comm_id % len(colors)]
            shape = "star" if is_rep else "dot"
            size = (centrality * 200) + (25 if is_rep else 15)
            border_width = 4 if is_rep else 2
            
            role_str = "Core Representative" if is_rep else "Community Member"
            title = f"Role: {role_str}\nCommunity: {comm_id}\nCentrality: {centrality:.4f}\nCommunity Reps: {reps_info['top_nodes_str']}"
            
            net_comm.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=border_width, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net_comm.get_nodes() and v in net_comm.get_nodes():
                # Only add edge if they are in the same community (intra-community)
                if community_map.get(u) == community_map.get(v):
                    net_comm.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)

        net_comm.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -250,
              "centralGravity": 0.002,
              "springLength": 300,
              "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          }
        }
        """)
        net_comm.save_graph(str(vis_dir / "interactive_communities_only.html"))

        # 6. Collapsed Communities Visual (Option C) - with Size Filtering and Theme Dominance Counts
        net_collapsed = Network(height="1000px", width="100%", directed=False, bgcolor="#222222", font_color="white", heading="Collapsed Communities")
        
        for comm in result.communities:
            comm_nodes = comm.nodes
            if len(comm_nodes) < 3: # Remove clusters with nodes < 3
                continue
            
            comm_id = comm.community_id
            
            # Use Theme Dominance to list associated themes with counts
            comm_themes = defaultdict(int)
            for n in comm_nodes:
                for theme in node_themes.get(n, []):
                    comm_themes[theme] += 1
                    
            sorted_comm_themes = sorted(comm_themes.items(), key=lambda x: x[1], reverse=True)
            themes_str = "\n".join([f"{theme} ({count})" for theme, count in sorted_comm_themes])
            if not themes_str:
                themes_str = "(No associated themes)"
                
            reps_info = community_reps.get(comm_id, {"top_nodes": [], "top_nodes_str": ""})
            
            # Use community ID for coloring
            color = colors[comm_id % len(colors)]
            size = 30 + min(len(comm_nodes) * 2, 70) 
            
            label = f"Community {comm_id}\n({len(comm_nodes)} nodes)\n{themes_str}"
            title = f"Community {comm_id}\nTotal Nodes: {len(comm_nodes)}\nRepresentative Terms: {reps_info['top_nodes_str']}\nThemes:\n" + "\n".join([f"- {theme}: {count}" for theme, count in sorted_comm_themes])
            
            net_collapsed.add_node(f"COMMUNITY_{comm_id}", label=label, color=color, shape="dot", size=size, title=title, borderWidth=2, shadow=True)
            
        net_collapsed.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -12000,
              "centralGravity": 0.02,
              "springLength": 300,
              "springConstant": 0.02
            },
            "solver": "barnesHut"
          }
        }
        """)
        net_collapsed.save_graph(str(vis_dir / "interactive_collapsed_communities.html"))


        # 9. Dual Perspective Visual (Option 2)
        net_dual = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading="Dual Perspective Topology (Colors: Leiden Communities, Shapes: Node2Vec Clusters)")
        
        shapes = ["dot", "square", "triangle", "diamond", "star", "hexagon", "triangleDown"]
        
        # We display the pruned subgraph nodes (both community and structural cluster mapped nodes)
        # to focus on the active communities.
        for node in dg.nodes():
            if node not in community_map:
                continue
            metrics = result.node_metrics.get(node)
            if not metrics:
                continue
            
            comm_id = community_map[node]
            sc_id = struct_map.get(node)
            
            color = colors[comm_id % len(colors)]
            
            # Map structural cluster to shape
            if sc_id is not None:
                shape = shapes[sc_id % len(shapes)]
                shape_str = f"Cluster {sc_id}"
            else:
                shape = "dot"
                shape_str = "None (Unclustered)"
                
            centrality = metrics.degree_centrality
            size = (centrality * 200) + 15
            
            title = f"Node: {node}\nLeiden Community (Color): {comm_id}\nNode2Vec Cluster (Shape): {shape_str}\nCentrality: {centrality:.4f}"
            
            net_dual.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=2, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net_dual.get_nodes() and v in net_dual.get_nodes():
                # Render ALL edges between the community nodes as requested by Option A
                net_dual.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)

        net_dual.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -250,
              "centralGravity": 0.002,
              "springLength": 300,
              "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          }
        }
        """)
        net_dual.save_graph(str(vis_dir / "interactive_dual_perspective.html"))

        # 10. Node2Vec 2D Embedding Space Scatter Plot (Option 1)
        if node_embeddings and len(node_embeddings) >= 2:
            try:
                from sklearn.decomposition import PCA
                nodes_list = list(node_embeddings.keys())
                embeddings_matrix = np.array([node_embeddings[n] for n in nodes_list])
                
                # Project 64D vectors down to 2D
                pca = PCA(n_components=2)
                coords = pca.fit_transform(embeddings_matrix)
                
                # Build mapping of node to its cluster ID (or default to 0)
                node_clusters = {}
                for sc in result.structural_clusters:
                    for n in sc.nodes:
                        node_clusters[n] = sc.cluster_id
                
                # Construct plotly data array
                plotly_data = []
                for idx, name in enumerate(nodes_list):
                    c_id = node_clusters.get(name, -1)
                    plotly_data.append({
                        "name": name,
                        "x": float(coords[idx, 0]),
                        "y": float(coords[idx, 1]),
                        "cluster": int(c_id)
                    })
                
                # Generate self-contained HTML page using Plotly CDN
                plotly_json = json.dumps(plotly_data)
                
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Node2Vec 2D Embedding Space</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{
            background-color: #222222;
            color: #ffffff;
            font-family: 'Open Sans', sans-serif;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 5px;
        }}
        p {{
            text-align: center;
            color: #aaaaaa;
            margin-bottom: 20px;
        }}
        #plot {{
            width: 100%;
            height: 800px;
            background-color: #222222;
            border: 1px solid #444444;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <h1>Node2Vec 2D Embedding Space</h1>
    <p>Each point represents a node. Coordinates are projected from 64D Node2Vec space to 2D using PCA. Colors represent K-Means structural clusters.</p>
    <div id="plot"></div>

    <script>
        const data = {plotly_json};
        
        // Group by cluster
        const tracesMap = {{}};
        data.forEach(item => {{
            const clusterId = item.cluster;
            if (!tracesMap[clusterId]) {{
                tracesMap[clusterId] = {{
                    x: [],
                    y: [],
                    text: [],
                    mode: 'markers+text',
                    textposition: 'top center',
                    name: 'Cluster ' + (clusterId === -1 ? 'Unclustered' : clusterId),
                    type: 'scatter',
                    marker: {{
                        size: 14,
                        opacity: 0.8,
                        line: {{ width: 1, color: '#ffffff' }}
                    }}
                }};
            }}
            tracesMap[clusterId].x.push(item.x);
            tracesMap[clusterId].y.push(item.y);
            tracesMap[clusterId].text.push(item.name);
        }});
        
        const traces = Object.values(tracesMap);
        
        const layout = {{
            paper_bgcolor: '#222222',
            plot_bgcolor: '#222222',
            xaxis: {{
                gridcolor: '#444444',
                zerolinecolor: '#666666',
                tickfont: {{ color: '#aaaaaa' }}
            }},
            yaxis: {{
                gridcolor: '#444444',
                zerolinecolor: '#666666',
                tickfont: {{ color: '#aaaaaa' }}
            }},
            legend: {{
                font: {{ color: '#ffffff' }},
                bgcolor: 'rgba(0,0,0,0)'
            }},
            margin: {{ t: 40, b: 40, l: 40, r: 40 }},
            hovermode: 'closest'
        }};
        
        Plotly.newPlot('plot', traces, layout);
    </script>
</body>
</html>
"""
                with open(vis_dir / "interactive_node2vec_embeddings.html", "w") as f:
                    f.write(html_content)
                    
            except Exception as e:
                print(f"Warning: Failed to generate Node2Vec 2D Embedding Space visual: {e}")

        # 11. Role-Based Network Graph (Option 3)
        if result.structural_clusters and struct_map:
            net_struct = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading="Structural Clusters Topology (Intra-Cluster Edges Only)")
            
            for node in dg.nodes():
                if node not in struct_map:
                    continue
                metrics = result.node_metrics.get(node)
                if not metrics:
                    continue
                sc_id = struct_map[node]
                centrality = metrics.degree_centrality
                
                # Draw top central nodes with a distinct shape to designate them as representative
                sc_nodes = [n for n in struct_map.keys() if struct_map[n] == sc_id]
                node_centralities = []
                for n in sc_nodes:
                    m = result.node_metrics.get(n)
                    cent = m.pagerank if m else 0.0
                    node_centralities.append((n, cent))
                node_centralities.sort(key=lambda x: x[1], reverse=True)
                top_nodes = [n for n, _ in node_centralities[:3]]
                is_rep = node in top_nodes
                
                color = colors[sc_id % len(colors)]
                shape = "star" if is_rep else "dot"
                size = (centrality * 200) + (25 if is_rep else 15)
                border_width = 4 if is_rep else 2
                
                role_str = "Core Representative" if is_rep else "Cluster Member"
                title = f"Role: {role_str}\nStructural Cluster: {sc_id}\nCentrality: {centrality:.4f}"
                
                net_struct.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=border_width, shadow=True)

            for u, v, data in dg.edges(data=True):
                if u in net_struct.get_nodes() and v in net_struct.get_nodes():
                    # Only add edge if they are in the same structural cluster
                    if struct_map.get(u) == struct_map.get(v):
                        net_struct.add_edge(u, v, title=data.get("predicate", ""), color="rgba(200,200,200,0.3)", width=1)

            net_struct.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -250,
                  "centralGravity": 0.002,
                  "springLength": 300,
                  "springConstant": 0.05
                },
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 150}
              }
            }
            """)
            net_struct.save_graph(str(vis_dir / "interactive_structural_clusters_only.html"))

        # 12. Hubs & Ego Networks Visual (Hub Subgraph)
        net_hubs = Network(height="1000px", width="100%", directed=True, bgcolor="#222222", font_color="white", heading="Global Hubs & Ego Networks Topology")
        
        hubs_set = set(result.global_hubs)
        hub_connected_nodes = set(hubs_set)
        
        for u, v, data in dg.edges(data=True):
            if u in hubs_set or v in hubs_set:
                hub_connected_nodes.add(u)
                hub_connected_nodes.add(v)
                
        for node in hub_connected_nodes:
            metrics = result.node_metrics.get(node)
            is_hub = node in hubs_set
            centrality = metrics.degree_centrality if metrics else 0.0
            
            color = "red" if is_hub else "#aec7e8"
            shape = "star" if is_hub else "dot"
            size = (centrality * 200) + (35 if is_hub else 15)
            border_width = 4 if is_hub else 2
            
            role_str = "Global Hub" if is_hub else "Spoke Node"
            title = f"Role: {role_str}\nCentrality: {centrality:.4f}"
            
            net_hubs.add_node(node, label=node, color=color, shape=shape, size=size, title=title, borderWidth=border_width, shadow=True)

        for u, v, data in dg.edges(data=True):
            if u in net_hubs.get_nodes() and v in net_hubs.get_nodes():
                if u in hubs_set or v in hubs_set:
                    is_inter_hub = u in hubs_set and v in hubs_set
                    net_hubs.add_edge(u, v, title=data.get("predicate", ""), color="rgba(255, 100, 100, 0.8)" if is_inter_hub else "rgba(200,200,200,0.4)", width=3 if is_inter_hub else 1)

        net_hubs.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -150,
              "centralGravity": 0.005,
              "springLength": 250,
              "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          }
        }
        """)
        net_hubs.save_graph(str(vis_dir / "interactive_hubs_ego_network.html"))

