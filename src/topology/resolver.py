import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

class TargetResolver:
    """
    Target qualification and triplet filtering criteria used by Stage 4 for LLM synthesis.
    Provides methods to resolve qualified spoke targets, rank/cap global hubs, aggregate enum nodes,
    and filter/cap triplet payloads based on PageRank and intra-cluster edge priority.
    """
    
    @staticmethod
    def qualify_and_resolve_targets(topology: Dict[str, Any], path_type: str, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Resolves qualified spoke targets (>= min_cluster_size), global hub targets (top max_hub_targets),
        and aggregates low-density cluster nodes, orphans, and demoted hubs into all_enum_nodes.
        
        Returns:
            (targets, all_enum_nodes)
        """
        min_cluster_size = config.get('synthesis', {}).get('min_cluster_size', 32)
        max_hub_targets = config.get('synthesis', {}).get('max_hub_targets', 4)
        
        # 1. Resolve Hub Targets
        raw_hubs = topology.get("global_hubs", [])
        metrics = topology.get("node_metrics", {})
        
        def get_hub_sort_key(node_id):
            node_m = metrics.get(node_id, {})
            if isinstance(node_m, dict):
                return (node_m.get("betweenness_centrality", 0.0), node_m.get("degree_centrality", 0.0))
            return (getattr(node_m, "betweenness_centrality", 0.0), getattr(node_m, "degree_centrality", 0.0))
            
        sorted_hubs = sorted(raw_hubs, key=get_hub_sort_key, reverse=True)
        hubs = sorted_hubs[:max_hub_targets]
        discarded_hubs = sorted_hubs[max_hub_targets:]
        
        # 2. Phase 1: Aggregate Orphans + Low-Density Nodes + Discarded Hubs (Enums)
        orphans = list(topology.get("orphans", []))
        small_cluster_nodes = []
        
        if path_type == "embedding":
            for cluster in topology.get("structural_clusters", []):
                nodes = cluster.get("nodes", [])
                if len(nodes) < min_cluster_size:
                    small_cluster_nodes.extend(nodes)
        else:
            for comm in topology.get("communities", []):
                nodes = comm.get("nodes", [])
                if len(nodes) < min_cluster_size:
                    small_cluster_nodes.extend(nodes)

        all_enum_nodes = sorted(list(set(orphans + small_cluster_nodes + discarded_hubs)))
        
        # 3. Phase 2: Consolidate Qualified Targets (>= min_cluster_size)
        targets = []
        if path_type == "embedding":
            for cluster in topology.get("structural_clusters", []):
                nodes_in_cluster = cluster.get("nodes", [])
                if len(nodes_in_cluster) >= min_cluster_size:
                    targets.append({"type": "structural_cluster", "id": cluster.get("cluster_id"), "nodes": nodes_in_cluster})
        else:
            for comm in topology.get("communities", []):
                nodes_in_comm = comm.get("nodes", [])
                if len(nodes_in_comm) >= min_cluster_size:
                    targets.append({"type": "community", "id": comm.get("community_id"), "nodes": nodes_in_comm})

        targets.sort(key=lambda x: len(x["nodes"]), reverse=True)
        
        # Append Hub Targets
        for hub in hubs:
            targets.append({"type": "hub", "id": hub, "nodes": [hub]})
            
        return targets, all_enum_nodes

    @staticmethod
    def _compute_node_specificity_map(topology: Dict[str, Any], triplets: List[Dict[str, Any]], path_type: str) -> Tuple[Dict[str, float], Dict[Tuple[Any, str], float]]:
        """
        Computes Inverse Global Concept Rarity and TF-IDF Cluster Concentration Ratio.
        Returns:
            (global_rarity_map, cluster_node_spec_map)
        """
        import math
        from collections import Counter, defaultdict
        
        global_node_counts = Counter()
        for t in triplets:
            subj = str(t.get('subject', '')).lower().strip()
            obj = str(t.get('object', '')).lower().strip()
            if subj: global_node_counts[subj] += 1
            if obj: global_node_counts[obj] += 1
            
        total_global_nodes = len(global_node_counts) or 1
        global_rarity_map = {}
        for node, cnt in global_node_counts.items():
            global_rarity_map[node] = math.log(1.0 + (total_global_nodes / (cnt + 1.0)))
            
        cluster_node_counts = defaultdict(Counter)
        if path_type == "embedding":
            clusters = topology.get("structural_clusters", [])
            for c in clusters:
                c_id = c.get("cluster_id")
                for n in c.get("nodes", []):
                    cluster_node_counts[c_id][str(n).lower().strip()] += 1
        else:
            communities = topology.get("communities", [])
            for comm in communities:
                c_id = comm.get("community_id")
                for n in comm.get("nodes", []):
                    cluster_node_counts[c_id][str(n).lower().strip()] += 1

        cluster_node_spec_map = {}
        for cid, c_counts in cluster_node_counts.items():
            for node, c_cnt in c_counts.items():
                global_cnt = global_node_counts.get(node, c_cnt)
                other_cnt = global_cnt - c_cnt
                cluster_spec = math.log(1.0 + (c_cnt / (other_cnt + 1.0)))
                g_rarity = global_rarity_map.get(node, 1.0)
                cluster_node_spec_map[(cid, node)] = g_rarity * (1.0 + cluster_spec)
                
        return global_rarity_map, cluster_node_spec_map

    @staticmethod
    def filter_and_cap_triplets(
        target_nodes: set,
        triplets: List[Dict[str, Any]],
        topology: Dict[str, Any],
        max_triplets_cap: int,
        target_id: Any = "global",
        path_type: str = "community"
    ) -> List[Dict[str, Any]]:
        """
        Filters triplets incident to target_nodes, scores each statement using
        pre-computed PageRank centrality and intra-cluster connectedness,
        sorts descending by rank score, and caps at max_triplets_cap.
        """
        metrics = topology.get("node_metrics", {})
        
        def get_pagerank(node_name: str) -> float:
            m = metrics.get(node_name, {})
            if isinstance(m, dict):
                return m.get("pagerank", 0.01)
            return getattr(m, "pagerank", 0.01)

        scored_triplets = []
        for t in triplets:
            subj = str(t.get('subject', '')).lower().strip()
            obj = str(t.get('object', '')).lower().strip()
            
            if subj in target_nodes or obj in target_nodes:
                intra_bonus = 1.5 if (subj in target_nodes and obj in target_nodes) else 1.0
                pr_score = (get_pagerank(subj) + get_pagerank(obj)) * intra_bonus
                
                t_copy = dict(t)
                t_copy["triplet_rank_score"] = round(float(pr_score), 6)
                t_copy["pagerank_score"] = round(float(pr_score), 6)
                scored_triplets.append((pr_score, t_copy))
                
        scored_triplets.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_triplets[:max_triplets_cap]]

    @staticmethod
    def export_resolved_targets_json(
        topology: Dict[str, Any],
        path_type: str,
        refined_triplets: List[Dict[str, Any]],
        config: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """
        Executes target qualification and triplet PageRank capping, and exports
        resolved_<path_type>_targets.json to output_dir with explicit rank scores.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        synth_cfg = config.get('synthesis', {})
        min_cluster_size = synth_cfg.get('min_cluster_size', 32)
        max_hub_targets = synth_cfg.get('max_hub_targets', 4)
        max_triplets_cap = synth_cfg.get('max_triplets_per_target', 1000)
        metrics = topology.get("node_metrics", {})
        
        # 1. Resolve target qualification & enum nodes
        targets, all_enum_nodes = TargetResolver.qualify_and_resolve_targets(topology, path_type, config)
        
        # 2. Build target payload records with sorted & capped triplets
        target_payloads = []
        for i, target in enumerate(targets, start=1):
            target_type = target["type"]
            c_id = target["id"]
            target_nodes = set(target["nodes"])
            
            # Filter, score, and cap triplets with Asymmetric Synergy Engine
            capped_triplets = TargetResolver.filter_and_cap_triplets(
                target_nodes, refined_triplets, topology, max_triplets_cap, target_id=c_id, path_type=path_type
            )
            
            filename_suffix = f"{path_type}_{c_id}" if target_type != "hub" else f"hub_{c_id}"
            output_module_name = f"{i:02d}_{filename_suffix}.py"
            
            # Compute target-level ranking metrics
            if target_type == "hub":
                hub_m = metrics.get(c_id, {})
                betw_score = hub_m.get("betweenness_centrality", 0.0) if isinstance(hub_m, dict) else getattr(hub_m, "betweenness_centrality", 0.0)
                deg_score = hub_m.get("degree_centrality", 0.0) if isinstance(hub_m, dict) else getattr(hub_m, "degree_centrality", 0.0)
                rank_score = round(float(betw_score), 6)
                target_metrics = {
                    "rank_score": rank_score,
                    "betweenness_centrality": round(float(betw_score), 6),
                    "degree_centrality": round(float(deg_score), 6)
                }
            else:
                rank_score = float(len(target_nodes))
                target_metrics = {
                    "rank_score": rank_score
                }
            
            payload_item = {
                "target_index": i,
                "target_type": target_type,
                "target_id": c_id,
                "output_module_name": output_module_name,
                "rank_score": rank_score,
                "nodes_count": len(target_nodes),
                "nodes": sorted(list(target_nodes)),
                "triplets_count": len(capped_triplets),
                "triplets": capped_triplets
            }
            if target_type == "hub":
                payload_item["betweenness_centrality"] = target_metrics["betweenness_centrality"]
                payload_item["degree_centrality"] = target_metrics["degree_centrality"]
                
            target_payloads.append(payload_item)
            
        # 3. Construct master audit record
        audit_payload = {
            "path_type": path_type,
            "min_cluster_size": min_cluster_size,
            "max_hub_targets": max_hub_targets,
            "max_triplets_per_target": max_triplets_cap,
            "enum_nodes_count": len(all_enum_nodes),
            "enum_nodes": all_enum_nodes,
            "targets_count": len(target_payloads),
            "targets": target_payloads
        }
        
        if path_type == "community":
            target_filename = "resolved_community_targets.json"
        elif path_type == "embedding":
            target_filename = "resolved_embedded_targets.json"
        else:
            target_filename = f"resolved_{path_type}_targets.json"
            
        target_file = output_dir / target_filename
        with open(target_file, "w") as f:
            json.dump(audit_payload, f, indent=2)
            
        print(f"   [OK] Resolved Targets Saved: {target_file}")
        return target_file
