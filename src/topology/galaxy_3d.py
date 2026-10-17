"""
SemanticPrism Stage 3 Topology: 3D Galaxy Cluster & Embedding Visualizer
========================================================================
Interactive 3D WebGL (Three.js) visualization engine for exploring concept
embeddings, Fréchet mean community centroids, inter-cluster highways, and
cross-border semantic refugee trajectories in high-dimensional space.

Features:
  - Dual-Engine 3D Projections: Real-time animated switching between UMAP-3D
    (non-linear manifold cluster separation) and PCA-3D (global variance).
  - Hierarchical Galaxy View:
      * 3D Community Centroids (glowing celestial cores with title rings).
      * Orbiting Concept Point Clouds (~1,600 nodes scaled and colored by
        Architectural Tier: Nucleus, Mechanics, Modifiers).
      * Translucent Iridescent Cluster Density Hulls (isochrone bounding spheres).
      * 3D Topological Highway Conduits with animated energy pulse particles.
      * Cross-Border "Semantic Refugee" attraction vector rays.
  - Interactive HUD & Deep Inspector:
      * Real-time Raycaster tooltips and cinematic camera fly-to focusing.
      * Search and filter by concept, community, architectural tier, or health.
      * Prototypicality, Centrality, and Grounding Fidelity gauges.
      * Full source quote context card.
  - Performance & Caching:
      * Precomputes and caches 3D coordinates to JSON for instant reloads.
      * Standalone, self-contained HTML running smoothly at 60 FPS in modern browsers.
"""

import json
import time
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.decomposition import PCA


class Galaxy3DVisualizer:
    """
    Computes 3D coordinates (PCA-3D and UMAP-3D), spatial cluster hulls,
    highway conduits, and renders the interactive Three.js 3D Galaxy Dashboard.
    """

    COMMUNITY_COLORS = [
        "#00f0ff",  # 0: Electric Cyan
        "#3b82f6",  # 1: Cobalt Blue
        "#10b981",  # 2: Emerald Green
        "#f43f5e",  # 3: Rose Crimson
        "#a855f7",  # 4: Neon Purple
        "#eab308",  # 5: Luminous Amber
        "#06b6d4",  # 6: Turquoise
        "#ec4899",  # 7: Hot Pink
        "#f97316",  # 8: Bright Orange
        "#8b5cf6",  # 9: Electric Violet
        "#14b8a6",  # 10: Teal
        "#6366f1",  # 11: Indigo
        "#e11d48",  # 12: Ruby
        "#84cc16",  # 13: Lime
        "#d946ef",  # 14: Fuchsia
        "#0284c7",  # 15: Sky Blue
        "#f59e0b",  # 16: Amber
        "#10b981",  # 17: Mint
        "#64748b",  # 18: Slate
        "#8b5cf6",  # 19: Purple
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def _build_partition_subpack(
        self,
        comm_concept_indices: Dict[int, List[int]],
        comm_labels: Dict[int, str],
        community_theme_counts: Dict[int, Counter],
        concepts_payload: List[Dict[str, Any]],
        coords_pca: np.ndarray,
        coords_umap: np.ndarray,
        proj_vecs: np.ndarray,
        highways_raw: List[Dict[str, Any]],
        weighting_mode: str,
        hull_percentile: float,
        is_baseline: bool = False
    ) -> Dict[str, Any]:
        """
        Builds the complete spatial and topological payload for a specific partition
        (either agglomerated macro-clusters or baseline granular communities).
        """
        centroids_dict: Dict[int, Dict[str, Any]] = {}

        for cid, indices in sorted(comm_concept_indices.items(), key=lambda x: x[0]):
            if not indices:
                continue

            if weighting_mode == "tier_1_only":
                core_indices = [idx for idx in indices if concepts_payload[idx]["tier_key"] == "tier_1_nucleus"]
                if not core_indices:
                    core_indices = [idx for idx in indices if concepts_payload[idx]["tier_key"] in ("tier_1_nucleus", "tier_2_mechanics")]
                if not core_indices:
                    core_indices = indices
                core_weights = np.array([max(0.01, float(concepts_payload[idx]["composite_score"])) for idx in core_indices])
                core_weights /= np.sum(core_weights)
            elif weighting_mode == "uniform":
                core_indices = indices
                core_weights = np.ones(len(core_indices)) / len(core_indices)
            else:  # "core_tiers" (default)
                core_indices = [idx for idx in indices if concepts_payload[idx]["tier_key"] in ("tier_1_nucleus", "tier_2_mechanics")]
                if not core_indices:
                    core_indices = indices
                core_weights = np.array([max(0.01, float(concepts_payload[idx]["composite_score"])) for idx in core_indices])
                core_weights /= np.sum(core_weights)

            core_sub_pca = coords_pca[core_indices]
            core_sub_umap = coords_umap[core_indices]

            c_pca = np.average(core_sub_pca, axis=0, weights=core_weights)
            c_umap = np.average(core_sub_umap, axis=0, weights=core_weights)

            # Radii: Tight cohesive shells calculated on core community members
            dists_pca = np.linalg.norm(core_sub_pca - c_pca, axis=1)
            dists_umap = np.linalg.norm(core_sub_umap - c_umap, axis=1)

            rad_pca = float(np.percentile(dists_pca, hull_percentile)) if len(dists_pca) > 0 else 10.0
            rad_umap = float(np.percentile(dists_umap, hull_percentile)) if len(dists_umap) > 0 else 10.0

            # Count architectural tiers and top tier 1 exemplars
            tier_counts = {"tier_1_nucleus": 0, "tier_2_mechanics": 0, "tier_3_modifiers": 0}
            top_t1 = []
            for idx in indices:
                t_key = concepts_payload[idx].get("tier_key", "tier_2_mechanics")
                if t_key in tier_counts:
                    tier_counts[t_key] += 1
                if t_key == "tier_1_nucleus" and len(top_t1) < 5:
                    top_t1.append(concepts_payload[idx]["entity"])

            if not top_t1 and indices:
                sorted_core = sorted(indices, key=lambda i: concepts_payload[i]["composite_score"], reverse=True)
                top_t1 = [concepts_payload[i]["entity"] for i in sorted_core[:5]]

            label = comm_labels.get(cid, f"Community #{cid}")
            t1_theme = ", ".join(top_t1[:3]) if top_t1 else label

            theme_counter = community_theme_counts.get(cid, Counter())
            total_triplets = sum(theme_counter.values())
            top_themes = [
                {
                    "theme": name,
                    "count": count,
                    "percentage": round((count / total_triplets * 100.0), 1) if total_triplets > 0 else 0.0
                }
                for name, count in theme_counter.most_common(6)
            ]

            color = self.COMMUNITY_COLORS[cid % len(self.COMMUNITY_COLORS)]

            centroids_dict[cid] = {
                "community_id": cid,
                "label": label,
                "headline_theme": t1_theme,
                "color": color,
                "total_concepts": len(indices),
                "total_triplets": total_triplets,
                "tier_counts": tier_counts,
                "top_themes": top_themes,
                "pca_3d": [round(float(c_pca[0]), 2), round(float(c_pca[1]), 2), round(float(c_pca[2]), 2)],
                "umap_3d": [round(float(c_umap[0]), 2), round(float(c_umap[1]), 2), round(float(c_umap[2]), 2)],
                "radius_pca": round(max(4.0, rad_pca), 2),
                "radius_umap": round(max(4.0, rad_umap), 2),
                "exemplar_concepts": top_t1
            }

        # Cross-Border Semantic Refugees
        high_d_centroids = {}
        for cid, indices in comm_concept_indices.items():
            if not indices:
                continue
            if weighting_mode == "tier_1_only":
                core_indices = [idx for idx in indices if concepts_payload[idx]["tier_key"] == "tier_1_nucleus"]
                if not core_indices:
                    core_indices = [idx for idx in indices if concepts_payload[idx]["tier_key"] in ("tier_1_nucleus", "tier_2_mechanics")]
                if not core_indices:
                    core_indices = indices
                core_weights = np.array([max(0.01, float(concepts_payload[idx]["composite_score"])) for idx in core_indices])
                core_weights /= np.sum(core_weights)
            elif weighting_mode == "uniform":
                core_indices = indices
                core_weights = np.ones(len(core_indices)) / len(core_indices)
            else:
                core_indices = [idx for idx in indices if concepts_payload[idx]["tier_key"] in ("tier_1_nucleus", "tier_2_mechanics")]
                if not core_indices:
                    core_indices = indices
                core_weights = np.array([max(0.01, float(concepts_payload[idx]["composite_score"])) for idx in core_indices])
                core_weights /= np.sum(core_weights)

            c_vec = np.average(proj_vecs[core_indices], axis=0, weights=core_weights)
            c_vec /= max(1e-9, np.linalg.norm(c_vec))
            high_d_centroids[cid] = c_vec

        cids_list = list(high_d_centroids.keys())
        refugee_links: List[Dict[str, Any]] = []
        refugee_map: Dict[int, Dict[str, Any]] = {}

        if len(cids_list) > 1:
            centroid_matrix = np.array([high_d_centroids[c] for c in cids_list])
            for cid, indices in comm_concept_indices.items():
                if not indices:
                    continue
                curr_c_idx = cids_list.index(cid)
                sub_vecs = proj_vecs[indices]
                dists = 1.0 - np.clip(sub_vecs @ centroid_matrix.T, -1.0, 1.0)
                own_dists = dists[:, curr_c_idx]
                min_other_indices = np.argmin(dists, axis=1)

                for local_i, global_idx in enumerate(indices):
                    best_c_idx = min_other_indices[local_i]
                    target_cid = cids_list[best_c_idx]
                    if target_cid != cid and dists[local_i, best_c_idx] < own_dists[local_i] - 0.05:
                        pull_delta = float(own_dists[local_i] - dists[local_i, best_c_idx])
                        ref_info = {
                            "concept_id": global_idx,
                            "source_community": cid,
                            "target_community": target_cid,
                            "pull_delta": round(pull_delta, 4)
                        }
                        refugee_links.append(ref_info)
                        refugee_map[global_idx] = ref_info

        # Filter Highways
        highways_payload: List[Dict[str, Any]] = []
        for h in highways_raw:
            sc = h.get("source_community")
            tc = h.get("target_community")
            cnt = h.get("directed_edge_count", 0)
            if sc in centroids_dict and tc in centroids_dict and sc != tc and cnt >= 5:
                highways_payload.append({
                    "source": sc,
                    "target": tc,
                    "edge_count": cnt,
                    "top_predicates": [p.get("predicate") for p in h.get("top_predicates", [])[:3]]
                })
        highways_payload.sort(key=lambda x: x["edge_count"], reverse=True)

        # Build Concept Partition Mapping
        concept_community_map: Dict[int, Dict[str, Any]] = {}
        for cid, indices in comm_concept_indices.items():
            if cid not in centroids_dict:
                continue
            c_meta = centroids_dict[cid]
            for idx in indices:
                is_ref = idx in refugee_map
                concept_community_map[idx] = {
                    "community_id": cid,
                    "community_label": c_meta["headline_theme"],
                    "color": c_meta["color"],
                    "tier": concepts_payload[idx]["tier"],
                    "tier_key": concepts_payload[idx]["tier_key"],
                    "is_refugee": is_ref,
                    "refugee_target_id": refugee_map[idx]["target_community"] if is_ref else None,
                    "refugee_delta": refugee_map[idx]["pull_delta"] if is_ref else 0.0
                }

        return {
            "meta": {
                "total_concepts": len(concepts_payload),
                "total_communities": len(centroids_dict),
                "total_highways": len(highways_payload),
                "total_refugees": len(refugee_links),
                "partition_name": "Baseline Communities (Granular)" if is_baseline else "Agglomerated Clusters (Macro)"
            },
            "communities": list(centroids_dict.values()),
            "highways": highways_payload,
            "refugees": refugee_links,
            "concept_community_map": concept_community_map
        }

    def compute_3d_galaxy_data(
        self,
        base_dir: Path = Path("outputs/03_topology"),
        refinement_dir: Path = Path("outputs/02_refinement"),
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Loads embeddings and community partitions, computing both PCA-3D and
        UMAP-3D projection coordinates for concepts, centroids, hulls, and highways.
        Caches the computed 3D dataset to avoid repeated dimensionality reduction.
        """
        cache_dir = base_dir / "03_paths" / "community" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "galaxy_3d_data.json"

        g_cfg = self.config.get("topology", {}).get("galaxy_3d", {}) if "topology" in self.config else self.config.get("galaxy_3d", {})
        desired_debiased = bool(g_cfg.get("debias_embeddings", True))
        desired_weighting = str(g_cfg.get("centroid_weighting", "core_tiers"))
        desired_hull = int(g_cfg.get("density_hull_percentile", 75))

        comm_dir = base_dir / "03_paths" / "community"
        baseline_file_check = comm_dir / "topology_partitions_baseline.json"

        if not force_recompute and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cached_meta = data.get("meta", {})
                cached_debiased = cached_meta.get("debiased")
                cached_weighting = cached_meta.get("centroid_weighting")
                cached_hull = cached_meta.get("density_hull_percentile")
                has_themes = any("top_themes" in c for c in data.get("communities", []))
                if (
                    "concepts" in data
                    and "agglom" in data
                    and len(data["concepts"]) > 100
                    and cached_debiased == desired_debiased
                    and cached_weighting == desired_weighting
                    and cached_hull == desired_hull
                    and has_themes
                    and (not baseline_file_check.exists() or data.get("has_baseline", False))
                ):
                    print(f"   -> [3D Galaxy] Loaded cached 3D projections from: {cache_file.name}")
                    return data
            except Exception as e:
                print(f"   -> [3D Galaxy] Cache read failed ({e}), recomputing...")

        print("   -> [3D Galaxy] Computing dual-engine 3D projections (PCA-3D & UMAP-3D)...")

        # 1. Load Concept Embeddings & Metadata
        ref_cache = refinement_dir / "cache"
        ref_cfg = self.config.get('refinement', {}) if 'refinement' in self.config else self.config
        model_name = ref_cfg.get('embedding_model')

        # Auto-detect model if not explicitly specified, defaulting to all-MiniLM-L6-v2
        if not model_name:
            if list(ref_cache.glob("subject_all_MiniLM_L6_v2*.npy")) or list(ref_cache.glob("concept_all_MiniLM_L6_v2*.npy")):
                model_name = "all-MiniLM-L6-v2"
            elif list(ref_cache.glob("concept_*_embeddings.npy")):
                found = list(ref_cache.glob("concept_*_embeddings.npy"))[0].stem
                slug = found.replace("concept_", "").replace("_embeddings", "")
                model_name = slug.replace("_", "/")
            elif list(ref_cache.glob("subject_*_embeddings.npy")):
                found = list(ref_cache.glob("subject_*_embeddings.npy"))[0].stem
                slug = found.replace("subject_", "").replace("_embeddings", "")
                model_name = slug.replace("_", "/")
            else:
                model_name = "all-MiniLM-L6-v2"

        model_slug = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        concept_npy = ref_cache / f"concept_{model_slug}_embeddings.npy"
        concept_meta = ref_cache / f"concept_{model_slug}_meta.json"

        if not concept_npy.exists() or not concept_meta.exists():
            subject_npy = ref_cache / f"subject_{model_slug}_embeddings.npy"
            subject_meta = ref_cache / f"subject_{model_slug}_meta.json"
            if subject_npy.exists() and subject_meta.exists():
                concept_npy, concept_meta = subject_npy, subject_meta
            else:
                avail = list(ref_cache.glob("concept_*_embeddings.npy"))
                if avail:
                    concept_npy = avail[0]
                    concept_meta = ref_cache / (concept_npy.stem.replace("_embeddings", "_meta.json"))
                else:
                    raise FileNotFoundError(f"Concept embeddings not found in '{ref_cache}'.")

        with open(concept_meta, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        vocab_map = {str(t).lower().strip(): idx for idx, t in enumerate(meta_data.get("vocab", []))}
        embedding_matrix = np.load(concept_npy)

        # 2. Load Community Tiers, Partitions & Highways
        comm_dir = base_dir / "03_paths" / "community"
        tiers_json = comm_dir / "community_architectural_tiers.json"
        partitions_json = comm_dir / "topology_partitions.json"
        baseline_partitions_json = comm_dir / "topology_partitions_baseline.json"
        highways_json = comm_dir / "community_highways.json"

        if tiers_json.exists():
            with open(tiers_json, "r", encoding="utf-8") as f:
                tiers_data = json.load(f)
        elif partitions_json.exists():
            with open(partitions_json, "r", encoding="utf-8") as f:
                part_raw = json.load(f)
            tiers_data = []
            for c in part_raw.get("communities", []):
                cid = c.get("community_id")
                nodes = c.get("nodes", [])
                tiers_data.append({
                    "community_id": cid,
                    "community_label": f"Community #{cid}",
                    "tier_counts": {"tier_1_nucleus": 0, "tier_2_mechanics": len(nodes), "tier_3_modifiers": 0},
                    "tier_1_nucleus": [],
                    "tier_2_mechanics": [{"entity": n, "prototypicality": 0.5, "centrality": 0.5, "fidelity": 0.2, "composite_score": 0.4, "exemplar_quote": ""} for n in nodes],
                    "tier_3_modifiers": []
                })
        else:
            raise FileNotFoundError("No community partitions or architectural tiers found.")

        # Ingest baseline partition if available
        has_baseline = False
        baseline_data = None
        node_to_baseline_cid: Dict[str, int] = {}
        baseline_comm_labels: Dict[int, str] = {}
        if baseline_partitions_json.exists():
            try:
                with open(baseline_partitions_json, "r", encoding="utf-8") as bf:
                    baseline_data = json.load(bf)
                if baseline_data.get("communities"):
                    has_baseline = True
                    for bc in baseline_data.get("communities", []):
                        bcid = int(bc.get("community_id", 0))
                        blabel = bc.get("label", f"Community #{bcid}")
                        baseline_comm_labels[bcid] = blabel
                        for node in bc.get("nodes", []):
                            node_to_baseline_cid[str(node).lower().strip()] = bcid
            except Exception as e:
                print(f"   -> [Warning] Failed to load baseline partitions ({e})")

        # Load highways
        highways_raw = []
        if highways_json.exists():
            try:
                with open(highways_json, "r", encoding="utf-8") as hf:
                    h_obj = json.load(hf)
                    highways_raw = h_obj.get("highways", [])
            except Exception:
                pass

        # Load theme associations per community from resolved targets or refined triplets
        community_theme_counts: Dict[int, Counter] = defaultdict(Counter)
        baseline_theme_counts: Dict[int, Counter] = defaultdict(Counter)
        resolved_targets_json = comm_dir / "resolved_community_targets.json"
        refined_triplets_json = refinement_dir / "refined_triplets.json"

        if resolved_targets_json.exists():
            try:
                with open(resolved_targets_json, "r", encoding="utf-8") as rf:
                    res_data = json.load(rf)
                for tgt in res_data.get("targets", []):
                    t_cid = int(tgt.get("target_id", 0))
                    for tr in tgt.get("triplets", []):
                        th = tr.get("theme_association")
                        if th:
                            community_theme_counts[t_cid][th] += 1
            except Exception as e:
                print(f"   -> [Warning] Failed to extract themes from resolved targets: {e}")

        r_triplets = []
        if refined_triplets_json.exists():
            try:
                with open(refined_triplets_json, "r", encoding="utf-8") as rf:
                    r_triplets = json.load(rf)
            except Exception:
                r_triplets = []

        if not community_theme_counts and r_triplets:
            try:
                node_to_cid = {}
                for c_info in tiers_data:
                    actual_cid = int(c_info.get("community_id", 0))
                    for tier_k in ["tier_1_nucleus", "tier_2_mechanics", "tier_3_modifiers"]:
                        for item in c_info.get(tier_k, []):
                            ent = str(item.get("entity", "")).lower().strip()
                            if ent:
                                node_to_cid[ent] = actual_cid
                for tr in r_triplets:
                    th = tr.get("theme_association")
                    if not th:
                        continue
                    s = str(tr.get("subject", "")).lower().strip()
                    o = str(tr.get("object", "")).lower().strip()
                    if s in node_to_cid:
                        community_theme_counts[node_to_cid[s]][th] += 1
                    elif o in node_to_cid:
                        community_theme_counts[node_to_cid[o]][th] += 1
            except Exception:
                pass

        if has_baseline and r_triplets:
            try:
                for tr in r_triplets:
                    th = tr.get("theme_association")
                    if not th:
                        continue
                    s = str(tr.get("subject", "")).lower().strip()
                    o = str(tr.get("object", "")).lower().strip()
                    if s in node_to_baseline_cid:
                        baseline_theme_counts[node_to_baseline_cid[s]][th] += 1
                    elif o in node_to_baseline_cid:
                        baseline_theme_counts[node_to_baseline_cid[o]][th] += 1
            except Exception:
                pass

        # Compute baseline inter-community highways from refined triplets if available
        baseline_highways_raw: List[Dict[str, Any]] = []
        if has_baseline and r_triplets:
            try:
                b_highways = defaultdict(lambda: {"edge_count": 0, "predicates": defaultdict(int)})
                for tr in r_triplets:
                    s = str(tr.get("subject", "")).lower().strip()
                    o = str(tr.get("object", "")).lower().strip()
                    if s in node_to_baseline_cid and o in node_to_baseline_cid:
                        cu, cv = node_to_baseline_cid[s], node_to_baseline_cid[o]
                        if cu != cv:
                            b_highways[(cu, cv)]["edge_count"] += 1
                            pred = str(tr.get("predicate", "related_to"))
                            b_highways[(cu, cv)]["predicates"][pred] += 1
                for (cu, cv), h_data in sorted(b_highways.items(), key=lambda x: x[1]["edge_count"], reverse=True):
                    top_preds = sorted(h_data["predicates"].items(), key=lambda x: x[1], reverse=True)[:3]
                    baseline_highways_raw.append({
                        "source_community": cu,
                        "target_community": cv,
                        "directed_edge_count": h_data["edge_count"],
                        "top_predicates": [{"predicate": p, "count": cnt} for p, cnt in top_preds]
                    })
            except Exception:
                pass

        # 3. Collect Concepts and Vectors
        concepts_payload: List[Dict[str, Any]] = []
        concept_vectors_list: List[np.ndarray] = []
        community_concept_indices: Dict[int, List[int]] = defaultdict(list)
        community_meta: Dict[int, Dict[str, Any]] = {}

        for c_info in tiers_data:
            cid = int(c_info.get("community_id", 0))
            label = c_info.get("community_label", f"Community #{cid}")
            tier_counts = c_info.get("tier_counts", {})

            community_meta[cid] = {
                "community_id": cid,
                "label": label,
                "color": self.COMMUNITY_COLORS[cid % len(self.COMMUNITY_COLORS)],
                "tier_counts": tier_counts,
                "top_t1": []
            }

            for tier_name, tier_key in [("Tier 1: Nucleus", "tier_1_nucleus"), ("Tier 2: Mechanics", "tier_2_mechanics"), ("Tier 3: Modifiers", "tier_3_modifiers")]:
                for item in c_info.get(tier_key, []):
                    entity = str(item.get("entity", "")).strip()
                    if not entity:
                        continue
                    e_lower = entity.lower()
                    if e_lower not in vocab_map:
                        continue

                    vec = embedding_matrix[vocab_map[e_lower]]
                    item_idx = len(concepts_payload)

                    if tier_key == "tier_1_nucleus" and len(community_meta[cid]["top_t1"]) < 5:
                        community_meta[cid]["top_t1"].append(entity)

                    concepts_payload.append({
                        "id": item_idx,
                        "entity": entity,
                        "community_id": cid,
                        "tier": tier_name,
                        "tier_key": tier_key,
                        "composite_score": round(float(item.get("composite_score", 0.0)), 4),
                        "prototypicality": round(float(item.get("prototypicality", 0.0)), 4),
                        "centrality": round(float(item.get("centrality", 0.0)), 4),
                        "fidelity": round(float(item.get("fidelity", 0.0)), 4),
                        "exemplar_quote": str(item.get("exemplar_quote", "")),
                        "pca_3d": [0.0, 0.0, 0.0],
                        "umap_3d": [0.0, 0.0, 0.0],
                        "is_refugee": False,
                        "refugee_target_id": None,
                        "refugee_delta": 0.0
                    })
                    concept_vectors_list.append(vec)
                    community_concept_indices[cid].append(item_idx)

        total_nodes = len(concepts_payload)
        if total_nodes == 0:
            raise ValueError("No valid concept vectors mapped to community tiers.")

        all_vecs = np.array(concept_vectors_list)
        print(f"   -> [3D Galaxy] Processing {total_nodes} nodes across {len(community_meta)} communities.")

        # 3.5 In-Memory Concept Vector De-biasing (Background Subtraction)
        # Strips out anisotropic background components (corpus mean and dominant principal component)
        # to maximize inter-cluster geometric separation in PCA and UMAP manifolds.
        g_cfg = self.config.get("topology", {}).get("galaxy_3d", {}) if "topology" in self.config else self.config.get("galaxy_3d", {})
        debias_enabled = g_cfg.get("debias_embeddings", True)
        hull_percentile = g_cfg.get("density_hull_percentile", 75)

        if debias_enabled:
            t0 = time.time()
            mean_vec = np.mean(all_vecs, axis=0)
            centered_vecs = all_vecs - mean_vec
            pca_bg = PCA(n_components=2, random_state=42)
            pca_bg.fit(centered_vecs)
            u_bg = pca_bg.components_[:1]
            clean_vecs = centered_vecs - (centered_vecs @ u_bg.T) @ u_bg
            norm_clean = np.linalg.norm(clean_vecs, axis=1, keepdims=True)
            proj_vecs = clean_vecs / np.maximum(norm_clean, 1e-9)
            print(f"   -> [3D Galaxy] Applied All-but-the-Top vector de-biasing in {time.time() - t0:.2f}s")
        else:
            proj_vecs = all_vecs

        # 4. Dimensionality Reduction: PCA 3D
        t0 = time.time()
        pca_3d = PCA(n_components=3, random_state=42)
        coords_pca_raw = pca_3d.fit_transform(proj_vecs)
        pca_var = float(np.sum(pca_3d.explained_variance_ratio_))
        scale_pca = 100.0 / max(float(np.max(np.abs(coords_pca_raw))), 1e-6)
        coords_pca = coords_pca_raw * scale_pca
        print(f"   -> [3D Galaxy] PCA-3D fit in {time.time() - t0:.2f}s (Explained Var: {pca_var * 100:.1f}%)")

        # 5. Dimensionality Reduction: UMAP 3D
        t0 = time.time()
        try:
            import umap
            umap_3d = umap.UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.12,
                metric='cosine',
                random_state=42,
                verbose=False
            )
            coords_umap_raw = umap_3d.fit_transform(proj_vecs)
            coords_umap_centered = coords_umap_raw - np.mean(coords_umap_raw, axis=0)
            scale_umap = 100.0 / max(float(np.max(np.abs(coords_umap_centered))), 1e-6)
            coords_umap = coords_umap_centered * scale_umap
            print(f"   -> [3D Galaxy] UMAP-3D fit in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"   -> [Warning] UMAP-3D failed ({e}); falling back to PCA-3D.")
            coords_umap = coords_pca.copy()

        # Populate coordinates
        for idx in range(total_nodes):
            concepts_payload[idx]["pca_3d"] = [
                round(float(coords_pca[idx, 0]), 2),
                round(float(coords_pca[idx, 1]), 2),
                round(float(coords_pca[idx, 2]), 2)
            ]
            concepts_payload[idx]["umap_3d"] = [
                round(float(coords_umap[idx, 0]), 2),
                round(float(coords_umap[idx, 1]), 2),
                round(float(coords_umap[idx, 2]), 2)
            ]

        # 6. Build Dual-Partition Packs (Agglomerated Macro-Clusters & Baseline Granular Communities)
        weighting_mode = g_cfg.get("centroid_weighting", "core_tiers")

        # Map concepts to baseline community indices
        baseline_community_concept_indices: Dict[int, List[int]] = defaultdict(list)
        if has_baseline:
            for idx, c_item in enumerate(concepts_payload):
                ent = c_item["entity"].lower().strip()
                b_cid = node_to_baseline_cid.get(ent, c_item["community_id"])
                baseline_community_concept_indices[b_cid].append(idx)

        # Build Agglomerated Partition Pack
        agglom_pack = self._build_partition_subpack(
            comm_concept_indices=community_concept_indices,
            comm_labels={cid: meta["label"] for cid, meta in community_meta.items()},
            community_theme_counts=community_theme_counts,
            concepts_payload=concepts_payload,
            coords_pca=coords_pca,
            coords_umap=coords_umap,
            proj_vecs=proj_vecs,
            highways_raw=highways_raw,
            weighting_mode=weighting_mode,
            hull_percentile=hull_percentile,
            is_baseline=False
        )

        # Build Baseline Partition Pack (if available)
        baseline_pack = None
        if has_baseline and baseline_community_concept_indices:
            baseline_pack = self._build_partition_subpack(
                comm_concept_indices=baseline_community_concept_indices,
                comm_labels=baseline_comm_labels,
                community_theme_counts=baseline_theme_counts,
                concepts_payload=concepts_payload,
                coords_pca=coords_pca,
                coords_umap=coords_umap,
                proj_vecs=proj_vecs,
                highways_raw=baseline_highways_raw,
                weighting_mode=weighting_mode,
                hull_percentile=hull_percentile,
                is_baseline=True
            )

        # Synchronize default refugee annotations on concepts_payload to match agglom
        for idx, concept in enumerate(concepts_payload):
            node_info = agglom_pack["concept_community_map"].get(idx)
            if node_info:
                concept["is_refugee"] = node_info["is_refugee"]
                concept["refugee_target_id"] = node_info["refugee_target_id"]
                concept["refugee_delta"] = node_info["refugee_delta"]

        galaxy_data = {
            "has_baseline": bool(has_baseline and baseline_pack is not None),
            "meta": {
                "total_concepts": total_nodes,
                "total_communities": agglom_pack["meta"]["total_communities"],
                "total_highways": agglom_pack["meta"]["total_highways"],
                "total_refugees": agglom_pack["meta"]["total_refugees"],
                "pca_explained_variance_pct": round(pca_var * 100.0, 2),
                "debiased": debias_enabled,
                "centroid_weighting": weighting_mode,
                "density_hull_percentile": hull_percentile,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "agglom": agglom_pack,
            "baseline": baseline_pack,
            "communities": agglom_pack["communities"],
            "concepts": concepts_payload,
            "highways": agglom_pack["highways"],
            "refugees": agglom_pack["refugees"]
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(galaxy_data, f, indent=2)
        print(f"   -> [3D Galaxy] Exported and cached 3D dataset to: {cache_file.name}")

        return galaxy_data

    # =========================================================================
    # THREE.JS 3D HTML DASHBOARD GENERATOR
    # =========================================================================

    def generate_3d_galaxy_dashboard(
        self,
        base_dir: Path = Path("outputs/03_topology"),
        refinement_dir: Path = Path("outputs/02_refinement"),
        output_path: Optional[Path] = None,
        force_recompute: bool = False
    ) -> Path:
        """
        Computes 3D projections and generates a standalone, high-performance
        Three.js WebGL 3D Galaxy visualization HTML file.
        """
        galaxy_data = self.compute_3d_galaxy_data(base_dir=base_dir, refinement_dir=refinement_dir, force_recompute=force_recompute)

        if output_path is None:
            output_path = base_dir / "03_paths" / "community" / "cluster_3d_galaxy_visualization.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_json_str = json.dumps(galaxy_data)

        meta = galaxy_data["meta"]
        comms = galaxy_data["communities"]

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SemanticPrism // 3D Hyper-Space Cluster &amp; Embedding Galaxy</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  
  <!-- Three.js + OrbitControls from reliable CDNs -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

  <style>
    :root {{
      --bg-dark: #060813;
      --bg-card: rgba(12, 18, 34, 0.82);
      --bg-card-border: rgba(255, 255, 255, 0.09);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-cyan: #00f0ff;
      --accent-blue: #3b82f6;
      --accent-purple: #a855f7;
      --accent-rose: #f43f5e;
      --accent-gold: #eab308;
      --glow-cyan: 0 0 20px rgba(0, 240, 255, 0.45);
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      user-select: none;
    }}

    /* WebGL Canvas Container */
    #webgl-container {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }}

    /* Floating Top HUD Header */
    header.hud-header {{
      position: absolute;
      top: 16px;
      left: 20px;
      right: 20px;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(9, 14, 28, 0.75);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--bg-card-border);
      border-radius: 14px;
      padding: 10px 22px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      pointer-events: auto;
    }}

    .logo-group {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    .logo-badge {{
      background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(168, 85, 247, 0.2));
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
      font-family: 'Orbitron', monospace;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      box-shadow: var(--glow-cyan);
    }}

    .title-text {{
      display: flex;
      flex-direction: column;
    }}

    .title-text h1 {{
      font-family: 'Orbitron', sans-serif;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.8px;
      background: linear-gradient(90deg, #fff, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .title-text p {{
      font-size: 11px;
      color: var(--text-muted);
    }}

    /* Partition Toggle Button & Badges */
    .partition-badge {{
      display: inline-block;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      padding: 3px 8px;
      border-radius: 6px;
      background: rgba(168, 85, 247, 0.2);
      border: 1px solid var(--accent-purple);
      color: #c084fc;
      margin-left: 8px;
      vertical-align: middle;
      box-shadow: 0 0 10px rgba(168, 85, 247, 0.25);
    }}

    .partition-badge.baseline {{
      background: rgba(0, 240, 255, 0.15);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.25);
    }}

    .partition-toggle-btn {{
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(168, 85, 247, 0.5);
      color: #e2e8f0;
      font-family: 'Outfit', sans-serif;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      box-shadow: 0 0 10px rgba(168, 85, 247, 0.2);
    }}

    .partition-toggle-btn:hover {{
      background: rgba(168, 85, 247, 0.25);
      border-color: var(--accent-purple);
      color: #fff;
      box-shadow: 0 0 16px rgba(168, 85, 247, 0.45);
      transform: translateY(-1px);
    }}

    /* Engine Switcher (UMAP-3D vs PCA-3D) */
    .engine-switcher {{
      display: flex;
      align-items: center;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--bg-card-border);
      border-radius: 10px;
      padding: 3px;
      gap: 4px;
    }}

    .engine-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: 'Outfit', sans-serif;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 7px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    .engine-btn.active {{
      background: linear-gradient(135deg, rgba(0, 240, 255, 0.25), rgba(59, 130, 246, 0.25));
      border: 1px solid rgba(0, 240, 255, 0.4);
      color: var(--accent-cyan);
      box-shadow: 0 0 14px rgba(0, 240, 255, 0.3);
    }}

    .engine-btn:hover:not(.active) {{
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
    }}

    /* Global Stats Ribbon */
    .stats-ribbon {{
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .stat-badge {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }}

    .stat-badge .val {{
      font-family: 'Orbitron', monospace;
      font-size: 14px;
      font-weight: 700;
      color: #fff;
    }}

    .stat-badge .lbl {{
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }}

    /* Left Drawer: Community Navigator */
    aside.left-drawer {{
      position: absolute;
      top: 86px;
      left: 20px;
      bottom: 20px;
      width: 320px;
      z-index: 10;
      background: var(--bg-card);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid var(--bg-card-border);
      border-radius: 14px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 15px 35px rgba(0,0,0,0.6);
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      pointer-events: auto;
    }}

    .drawer-header {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--bg-card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .drawer-header h3 {{
      font-family: 'Orbitron', sans-serif;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .search-box {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--bg-card-border);
      background: rgba(0, 0, 0, 0.2);
    }}

    .search-box input {{
      width: 100%;
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--bg-card-border);
      border-radius: 8px;
      padding: 7px 12px;
      color: #fff;
      font-family: 'Outfit', sans-serif;
      font-size: 12px;
      outline: none;
      transition: border 0.2s ease;
    }}

    .search-box input:focus {{
      border-color: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }}

    .community-list {{
      flex: 1;
      overflow-y: auto;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .community-list::-webkit-scrollbar {{
      width: 4px;
    }}
    .community-list::-webkit-scrollbar-thumb {{
      background: rgba(255,255,255,0.15);
      border-radius: 4px;
    }}

    .comm-item {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 9px;
      padding: 9px 12px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 4px;
      transition: all 0.2s ease;
    }}

    .comm-item:hover {{
      background: rgba(30, 41, 59, 0.8);
      border-color: rgba(255, 255, 255, 0.15);
      transform: translateX(2px);
    }}

    .comm-item.active {{
      background: rgba(15, 23, 42, 0.95);
      border-color: var(--accent-cyan);
      box-shadow: 0 0 12px rgba(0, 240, 255, 0.2);
    }}

    .comm-item.comm-hidden {{
      opacity: 0.35;
      filter: grayscale(60%);
    }}

    .comm-item.comm-hidden:hover {{
      opacity: 0.70;
      filter: grayscale(10%);
    }}

    .comm-counter-badge {{
      font-size: 10.5px;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-cyan);
      background: rgba(0, 240, 255, 0.12);
      border: 1px solid rgba(0, 240, 255, 0.25);
      border-radius: 4px;
      padding: 2px 7px;
    }}

    .comm-actions-toolbar {{
      display: flex;
      gap: 6px;
      padding: 6px 14px 10px 14px;
      border-bottom: 1px solid var(--bg-card-border);
      background: rgba(0, 0, 0, 0.15);
    }}

    .comm-tool-btn {{
      flex: 1;
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      padding: 5px 6px;
      font-size: 11px;
      font-weight: 500;
      color: var(--text-muted);
      cursor: pointer;
      font-family: 'Outfit', sans-serif;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }}

    .comm-tool-btn:hover {{
      background: rgba(51, 65, 85, 0.9);
      color: #fff;
      border-color: var(--accent-cyan);
    }}

    .comm-item-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .comm-title {{
      font-size: 12px;
      font-weight: 600;
      color: #f1f5f9;
      display: flex;
      align-items: center;
      gap: 8px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }}

    .color-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      flex-shrink: 0;
      box-shadow: 0 0 6px currentColor;
    }}

    .comm-item-actions {{
      display: flex;
      align-items: center;
      gap: 5px;
      flex-shrink: 0;
    }}

    .comm-count {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--text-muted);
      margin-right: 2px;
    }}

    .comm-btn-icon {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      cursor: pointer;
      font-size: 12px;
      padding: 3px 5px;
      border-radius: 5px;
      transition: all 0.15s ease;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      user-select: none;
    }}

    .comm-btn-icon:hover {{
      background: rgba(0, 240, 255, 0.2);
      border-color: var(--accent-cyan);
      transform: scale(1.15);
    }}

    .comm-btn-icon.off {{
      opacity: 0.45;
      filter: grayscale(100%);
      background: rgba(0, 0, 0, 0.2);
    }}

    .comm-theme {{
      font-size: 11px;
      color: #94a3b8;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    /* Bottom Control Bar */
    footer.bottom-bar {{
      position: absolute;
      bottom: 20px;
      left: 360px;
      right: 360px;
      z-index: 10;
      background: rgba(9, 14, 28, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--bg-card-border);
      border-radius: 12px;
      padding: 8px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      pointer-events: auto;
    }}

    .mode-pills {{
      display: flex;
      gap: 6px;
    }}

    .mode-pill {{
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--bg-card-border);
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 500;
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .mode-pill:hover {{
      color: #fff;
      background: rgba(255, 255, 255, 0.08);
    }}

    .mode-pill.active {{
      background: rgba(0, 240, 255, 0.15);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }}

    .cam-actions {{
      display: flex;
      gap: 8px;
    }}

    .cam-btn {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--bg-card-border);
      color: #cbd5e1;
      font-size: 11px;
      padding: 6px 12px;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s ease;
    }}

    .cam-btn:hover {{
      background: rgba(255, 255, 255, 0.15);
      color: #fff;
    }}

    /* Right Deep Inspector Drawer */
    aside.right-drawer {{
      position: absolute;
      top: 86px;
      right: 20px;
      bottom: 20px;
      width: 320px;
      z-index: 10;
      background: var(--bg-card);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid var(--bg-card-border);
      border-radius: 14px;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      box-shadow: 0 15px 35px rgba(0,0,0,0.6);
      pointer-events: auto;
      padding: 16px;
      gap: 14px;
    }}

    .inspector-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--bg-card-border);
      padding-bottom: 10px;
    }}

    .inspector-header h3 {{
      font-family: 'Orbitron', sans-serif;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--accent-cyan);
      text-transform: uppercase;
    }}

    .node-title-card {{
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--bg-card-border);
      border-radius: 10px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .node-name {{
      font-size: 15px;
      font-weight: 700;
      color: #fff;
      word-break: break-word;
    }}

    .node-tier-badge {{
      display: inline-block;
      align-self: flex-start;
      padding: 3px 8px;
      border-radius: 5px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .tier-nucleus {{
      background: rgba(234, 179, 8, 0.2);
      border: 1px solid var(--accent-gold);
      color: var(--accent-gold);
    }}
    .tier-mechanics {{
      background: rgba(0, 240, 255, 0.2);
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
    }}
    .tier-modifiers {{
      background: rgba(148, 163, 184, 0.2);
      border: 1px solid #94a3b8;
      color: #cbd5e1;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }}

    .metric-cell {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 8px;
      display: flex;
      flex-direction: column;
    }}

    .metric-cell .lbl {{
      font-size: 9px;
      color: var(--text-muted);
      text-transform: uppercase;
    }}

    .metric-cell .val {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      font-weight: 700;
      color: #fff;
    }}

    .quote-box {{
      background: rgba(0, 0, 0, 0.35);
      border-left: 3px solid var(--accent-cyan);
      border-radius: 0 8px 8px 0;
      padding: 10px;
      font-size: 11px;
      color: #cbd5e1;
      line-height: 1.5;
      max-height: 140px;
      overflow-y: auto;
      font-style: italic;
    }}

    .refugee-alert {{
      background: rgba(244, 63, 94, 0.15);
      border: 1px solid var(--accent-rose);
      border-radius: 8px;
      padding: 10px;
      font-size: 11px;
      color: #fecdd3;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    /* Cluster Extracted Theme Breakdown */
    .theme-breakdown-card {{
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--bg-card-border);
      border-radius: 10px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 9px;
      max-height: 190px;
      overflow-y: auto;
    }}

    .theme-item {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .theme-item-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
    }}

    .theme-name {{
      color: #e2e8f0;
      font-weight: 500;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 175px;
    }}

    .theme-stat {{
      color: var(--accent-cyan);
      font-weight: 700;
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
    }}

    .theme-bar-bg {{
      width: 100%;
      height: 4px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 2px;
      overflow: hidden;
    }}

    .theme-bar-fill {{
      height: 100%;
      border-radius: 2px;
      transition: width 0.4s ease;
    }}

    /* Floating Tooltip */
    #hover-tooltip {{
      position: absolute;
      z-index: 100;
      background: rgba(9, 14, 28, 0.92);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--accent-cyan);
      border-radius: 8px;
      padding: 8px 12px;
      pointer-events: none;
      display: none;
      box-shadow: 0 8px 24px rgba(0,0,0,0.6);
      max-width: 260px;
    }}

    #hover-tooltip .tt-title {{
      font-size: 12px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 2px;
    }}

    #hover-tooltip .tt-sub {{
      font-size: 10px;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

  <!-- 3D Canvas -->
  <div id="webgl-container"></div>

  <!-- Top Floating HUD -->
  <header class="hud-header">
    <div class="logo-group">
      <div class="logo-badge">STAGE 3 TOPOLOGY</div>
      <div class="title-text">
        <div style="display:flex; align-items:center;">
          <h1>SemanticPrism // 3D Embedding Galaxy</h1>
          <span id="partition-badge" class="partition-badge">Agglomerated Clusters (Macro)</span>
        </div>
        <p>Fréchet Mean Centroids &bull; Hierarchical Concept Clouds &bull; High-Dimensional Manifolds</p>
      </div>
    </div>

    <!-- Controls Container (Toggle + Engine) -->
    <div style="display:flex; align-items:center; gap:10px;">
      <button id="btn-toggle-partition" class="partition-toggle-btn" style="display:none;" title="Toggle between Agglomerated Macro Clusters and Baseline Granular Communities">
        <span>🔄</span>
        <span id="partition-toggle-text">Switch to Baseline Communities</span>
      </button>

      <!-- Engine Switcher -->
      <div class="engine-switcher">
        <button id="btn-engine-umap" class="engine-btn active" title="Non-linear manifold preserving local &amp; global clusters">
          <span>✨</span>
          <span>UMAP 3D</span>
        </button>
        <button id="btn-engine-pca" class="engine-btn" title="Deterministic global variance linear projection">
          <span>📐</span>
          <span>PCA 3D ({meta["pca_explained_variance_pct"]}%)</span>
        </button>
      </div>
    </div>

    <!-- Stats Ribbon -->
    <div class="stats-ribbon">
      <div class="stat-badge">
        <span class="val" id="stat-total-comms">{meta["total_communities"]}</span>
        <span class="lbl">Centroids</span>
      </div>
      <div class="stat-badge">
        <span class="val" id="stat-total-concepts">{meta["total_concepts"]}</span>
        <span class="lbl">Concepts</span>
      </div>
      <div class="stat-badge">
        <span class="val" id="stat-total-highways">{meta["total_highways"]}</span>
        <span class="lbl">Highways</span>
      </div>
      <div class="stat-badge">
        <span class="val" id="stat-total-refugees" style="color:var(--accent-rose);">{meta["total_refugees"]}</span>
        <span class="lbl">Refugees</span>
      </div>
    </div>
  </header>

  <!-- Left Drawer: Celestial Communities -->
  <aside class="left-drawer">
    <div class="drawer-header">
      <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
        <h3><span>🪐</span> Community Galaxies</h3>
        <span id="visible-comm-counter" class="comm-counter-badge">{len(comms)} / {len(comms)} Visible</span>
      </div>
    </div>
    <div class="search-box">
      <input type="text" id="filter-input" placeholder="Search concepts or clusters...">
    </div>
    <div class="comm-actions-toolbar">
      <button id="btn-show-all-comms" class="comm-tool-btn" title="Show all communities">👁️ All</button>
      <button id="btn-hide-all-comms" class="comm-tool-btn" title="Hide all communities">🚫 None</button>
      <button id="btn-invert-comms" class="comm-tool-btn" title="Invert visible communities">🔄 Invert</button>
    </div>
    <div class="community-list" id="community-list">
      <!-- Injected dynamically -->
    </div>
  </aside>

  <!-- Bottom Control Bar -->
  <footer class="bottom-bar">
    <div class="mode-pills">
      <button class="mode-pill active" data-mode="all">
        <span>🌌</span> All Elements
      </button>
      <button class="mode-pill" data-mode="centroids">
        <span>🪐</span> Centroids &amp; Highways
      </button>
      <button class="mode-pill" data-mode="hulls">
        <span>🛡️</span> Density Hulls
      </button>
      <button class="mode-pill" data-mode="refugees">
        <span>🚨</span> Semantic Refugees
      </button>
      <button class="mode-pill" data-mode="tier1">
        <span>👑</span> Tier 1 Nuclei
      </button>
    </div>

    <div class="cam-actions">
      <button id="btn-reset-cam" class="cam-btn" title="Reset camera to default orbital perspective">Reset View</button>
      <button id="btn-top-view" class="cam-btn" title="Align camera to top-down 2D plane">Top-Down</button>
      <button id="btn-auto-rotate" class="cam-btn" title="Toggle gentle cinematic orbit">Orbit: ON</button>
    </div>
  </footer>

  <!-- Right Drawer: Deep Concept Inspector -->
  <aside class="right-drawer" id="right-drawer">
    <div class="inspector-header">
      <h3>Entity Inspector</h3>
      <span id="inspector-source" style="font-size:10px; color:var(--text-muted);">Hover or Click</span>
    </div>

    <div class="node-title-card">
      <div id="inspector-name" class="node-name">Select an Entity</div>
      <div id="inspector-tier" class="node-tier-badge tier-nucleus">Tier 1: Nucleus</div>
      <div id="inspector-community" style="font-size:11px; color:var(--accent-cyan);">Community #0</div>
    </div>

    <div class="metric-grid">
      <div class="metric-cell">
        <span class="lbl">Composite Score</span>
        <span id="m-composite" class="val">--</span>
      </div>
      <div class="metric-cell">
        <span class="lbl">Prototypicality</span>
        <span id="m-proto" class="val">--</span>
      </div>
      <div class="metric-cell">
        <span class="lbl">Graph Centrality</span>
        <span id="m-cent" class="val">--</span>
      </div>
      <div class="metric-cell">
        <span class="lbl">Context Fidelity</span>
        <span id="m-fid" class="val">--</span>
      </div>
    </div>

    <div id="refugee-alert-box" class="refugee-alert" style="display:none;">
      <strong>⚠️ Cross-Border Semantic Refugee</strong>
      <span id="refugee-alert-desc">Attracted to external community</span>
    </div>

    <div>
      <span style="font-size:10px; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted); display:block; margin-bottom:6px;">Exemplar Source Grounding</span>
      <div id="inspector-quote" class="quote-box">Hover or click a node in the 3D space to inspect its empirical evidence quote.</div>
    </div>

    <!-- Cluster Original Extracted Themes -->
    <div id="themes-section">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <span style="font-size:10px; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted);">Cluster Extracted Themes</span>
        <span id="themes-total-count" style="font-size:10px; color:var(--accent-cyan); font-weight:700; font-family:'JetBrains Mono',monospace;">--</span>
      </div>
      <div id="inspector-themes" class="theme-breakdown-card">
        <span style="font-size:11px; color:var(--text-muted); font-style:italic;">Select a cluster to inspect its original theme distribution.</span>
      </div>
    </div>
  </aside>

  <!-- Hover Floating Tooltip -->
  <div id="hover-tooltip">
    <div class="tt-title" id="tt-title"></div>
    <div class="tt-sub" id="tt-sub"></div>
  </div>

  <!-- Embedded Dataset -->
  <script>
    const GALAXY_DATA = {data_json_str};
  </script>

  <!-- Three.js 3D Engine Script -->
  <script>
    // =========================================================================
    // STATE & ENGINE VARIABLES
    // =========================================================================
    let currentEngine = 'umap'; // 'umap' or 'pca'
    let currentFilterMode = 'all'; // 'all', 'centroids', 'hulls', 'refugees', 'tier1'
    let selectedCommunityId = null;
    let autoRotate = true;

    // Dual-Partition State
    let isAgglomerated = true;
    const hasBaseline = Boolean(GALAXY_DATA.has_baseline && GALAXY_DATA.baseline);
    let activeData = (GALAXY_DATA.agglom || GALAXY_DATA);

    const container = document.getElementById('webgl-container');
    const width = window.innerWidth;
    const height = window.innerHeight;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060813, 0.0022);

    const camera = new THREE.PerspectiveCamera(52, width / height, 0.1, 2000);
    camera.position.set(0, 45, 175);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false, powerPreference: 'high-performance' }});
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    // 2. OrbitControls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 500;
    controls.minDistance = 10;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 0.45;

    // 3. Lighting & Celestial Starfield
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x00f0ff, 0.6);
    dirLight.position.set(100, 150, 100);
    scene.add(dirLight);

    // Starfield Particle Background
    const starCount = 2800;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i += 3) {{
      starPos[i] = (Math.random() - 0.5) * 1200;
      starPos[i+1] = (Math.random() - 0.5) * 1200;
      starPos[i+2] = (Math.random() - 0.5) * 1200;
    }}
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({{
      color: 0x94a3b8,
      size: 1.4,
      transparent: true,
      opacity: 0.6,
      sizeAttenuation: true
    }});
    const starPoints = new THREE.Points(starGeo, starMat);
    scene.add(starPoints);

    // Subtle Ground Grid Plane
    const grid = new THREE.GridHelper(260, 26, 0x1e293b, 0x0f172a);
    grid.position.y = -65;
    grid.material.opacity = 0.45;
    grid.material.transparent = true;
    scene.add(grid);

    // =========================================================================
    // 3D OBJECT GROUPS & MESHES
    // =========================================================================
    const centroidGroup = new THREE.Group();
    const conceptGroup = new THREE.Group();
    const hullGroup = new THREE.Group();
    const highwayGroup = new THREE.Group();
    const refugeeGroup = new THREE.Group();

    scene.add(centroidGroup);
    scene.add(conceptGroup);
    scene.add(hullGroup);
    scene.add(highwayGroup);
    scene.add(refugeeGroup);

    // Mapping arrays
    const centroidMeshes = [];
    const conceptMeshes = [];
    const hullMeshes = [];
    const highwayLines = [];
    const refugeeLines = [];

    // Community Lookup Map
    let communityMap = {{}};
    function rebuildCommunityMap() {{
      communityMap = {{}};
      activeData.communities.forEach(c => {{ communityMap[c.community_id] = c; }});
    }}
    rebuildCommunityMap();

    // 1. Build Centroid Meshes (Celestial Stars)
    const centroidGeo = new THREE.SphereGeometry(3.6, 32, 32);
    function rebuildCentroids(engine) {{
      while (centroidGroup.children.length > 0) {{
        const obj = centroidGroup.children[0];
        centroidGroup.remove(obj);
      }}
      centroidMeshes.length = 0;

      activeData.communities.forEach(c => {{
        const color = new THREE.Color(c.color);
        const mat = new THREE.MeshStandardMaterial({{
          color: color,
          emissive: color,
          emissiveIntensity: 0.85,
          roughness: 0.2,
          metalness: 0.8
        }});
        const mesh = new THREE.Mesh(centroidGeo, mat);
        mesh.userData = {{
          type: 'centroid',
          community_id: c.community_id,
          label: c.label,
          theme: c.headline_theme,
          total_concepts: c.total_concepts,
          pca_pos: new THREE.Vector3(...c.pca_3d),
          umap_pos: new THREE.Vector3(...c.umap_3d),
          color: c.color,
          exemplar_concepts: c.exemplar_concepts
        }};

        mesh.position.copy(engine === 'umap' ? mesh.userData.umap_pos : mesh.userData.pca_pos);

        // Centroid Outer Glow Ring
        const ringGeo = new THREE.RingGeometry(4.8, 5.4, 32);
        const ringMat = new THREE.MeshBasicMaterial({{
          color: color,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.65
        }});
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.rotation.x = Math.PI / 2;
        mesh.add(ringMesh);

        centroidGroup.add(mesh);
        centroidMeshes.push(mesh);
      }});
    }}
    rebuildCentroids('umap');

    // 2. Build Concept Meshes (Orbiting Entities)
    const tierGeoNucleus = new THREE.SphereGeometry(1.6, 16, 16);
    const tierGeoMechanics = new THREE.SphereGeometry(1.1, 14, 14);
    const tierGeoModifiers = new THREE.SphereGeometry(0.75, 10, 10);

    GALAXY_DATA.concepts.forEach(item => {{
      const nodeInfo = (activeData.concept_community_map && activeData.concept_community_map[item.id]) || item;
      const cMeta = communityMap[nodeInfo.community_id] || {{ color: '#00f0ff', headline_theme: `Community #${{nodeInfo.community_id}}` }};
      const baseColor = new THREE.Color(cMeta.color);

      let geo = tierGeoMechanics;
      let emissiveIntensity = 0.35;
      if (item.tier_key === 'tier_1_nucleus') {{
        geo = tierGeoNucleus;
        emissiveIntensity = 0.75;
      }} else if (item.tier_key === 'tier_3_modifiers') {{
        geo = tierGeoModifiers;
        emissiveIntensity = 0.15;
      }}

      const mat = new THREE.MeshStandardMaterial({{
        color: baseColor,
        emissive: baseColor,
        emissiveIntensity: emissiveIntensity,
        roughness: 0.35,
        metalness: 0.5
      }});

      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData = {{
        type: 'concept',
        ...item,
        community_id: nodeInfo.community_id,
        community_label: nodeInfo.community_label || cMeta.headline_theme || `Community #${{nodeInfo.community_id}}`,
        color: cMeta.color,
        is_refugee: Boolean(nodeInfo.is_refugee),
        refugee_target_id: nodeInfo.refugee_target_id,
        refugee_delta: nodeInfo.refugee_delta || 0.0,
        pca_pos: new THREE.Vector3(...item.pca_3d),
        umap_pos: new THREE.Vector3(...item.umap_3d)
      }};

      mesh.position.copy(mesh.userData.umap_pos);
      conceptGroup.add(mesh);
      conceptMeshes.push(mesh);
    }});

    function updateConceptPartitions() {{
      if (!activeData.concept_community_map) return;
      conceptMeshes.forEach(mesh => {{
        const nodeInfo = activeData.concept_community_map[mesh.userData.id];
        if (nodeInfo) {{
          mesh.userData.community_id = nodeInfo.community_id;
          mesh.userData.community_label = nodeInfo.community_label;
          mesh.userData.color = nodeInfo.color;
          mesh.userData.is_refugee = Boolean(nodeInfo.is_refugee);
          mesh.userData.refugee_target_id = nodeInfo.refugee_target_id;
          mesh.userData.refugee_delta = nodeInfo.refugee_delta || 0.0;

          const baseColor = new THREE.Color(nodeInfo.color);
          mesh.material.color = baseColor;
          mesh.material.emissive = baseColor;
        }}
      }});
    }}

    // 3. Build Cluster Density Hulls (Translucent Glowing Isochrones)
    function rebuildHulls(engine) {{
      while (hullGroup.children.length > 0) {{
        const obj = hullGroup.children[0];
        hullGroup.remove(obj);
      }}
      hullMeshes.length = 0;

      activeData.communities.forEach(c => {{
        const color = new THREE.Color(c.color);
        const hullGeo = new THREE.IcosahedronGeometry(1.0, 3);
        const hullMat = new THREE.MeshBasicMaterial({{
          color: color,
          wireframe: true,
          transparent: true,
          opacity: 0.14
        }});
        const hullMesh = new THREE.Mesh(hullGeo, hullMat);
        hullMesh.userData = {{
          community_id: c.community_id,
          radius_pca: c.radius_pca,
          radius_umap: c.radius_umap,
          pca_pos: new THREE.Vector3(...c.pca_3d),
          umap_pos: new THREE.Vector3(...c.umap_3d)
        }};
        hullMesh.position.copy(engine === 'umap' ? hullMesh.userData.umap_pos : hullMesh.userData.pca_pos);
        hullMesh.scale.setScalar(engine === 'umap' ? c.radius_umap : c.radius_pca);

        hullGroup.add(hullMesh);
        hullMeshes.push(hullMesh);
      }});
    }}
    rebuildHulls('umap');

    // 4. Build Highway Conduits (3D Spline Tubes)
    function rebuildHighways(engine) {{
      while (highwayGroup.children.length > 0) {{
        const obj = highwayGroup.children[0];
        highwayGroup.remove(obj);
      }}
      highwayLines.length = 0;

      activeData.highways.forEach(hw => {{
        const sMeta = communityMap[hw.source];
        const tMeta = communityMap[hw.target];
        if (!sMeta || !tMeta) return;

        const p1 = engine === 'umap' ? new THREE.Vector3(...sMeta.umap_3d) : new THREE.Vector3(...sMeta.pca_3d);
        const p2 = engine === 'umap' ? new THREE.Vector3(...tMeta.umap_3d) : new THREE.Vector3(...tMeta.pca_3d);

        // Arc mid-point for aesthetic celestial conduit
        const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        const dist = p1.distanceTo(p2);
        mid.y += dist * 0.12;

        const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
        const tubeRadius = Math.min(1.4, Math.max(0.3, Math.log10(hw.edge_count + 1) * 0.7));
        const tubeGeo = new THREE.TubeGeometry(curve, 24, tubeRadius, 8, false);

        const color = new THREE.Color(sMeta.color).lerp(new THREE.Color(tMeta.color), 0.5);
        const tubeMat = new THREE.MeshBasicMaterial({{
          color: color,
          transparent: true,
          opacity: 0.35
        }});

        const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
        tubeMesh.userData = {{
          type: 'highway',
          source: hw.source,
          target: hw.target,
          count: hw.edge_count
        }};
        highwayGroup.add(tubeMesh);
        highwayLines.push(tubeMesh);
      }});
    }}
    rebuildHighways('umap');

    // 5. Build Cross-Border Semantic Refugee Rays
    function rebuildRefugees(engine) {{
      while (refugeeGroup.children.length > 0) {{
        const obj = refugeeGroup.children[0];
        refugeeGroup.remove(obj);
      }}
      refugeeLines.length = 0;

      activeData.refugees.forEach(ref => {{
        const concept = GALAXY_DATA.concepts[ref.concept_id];
        const targetComm = communityMap[ref.target_community];
        if (!concept || !targetComm) return;

        const p1 = engine === 'umap' ? new THREE.Vector3(...concept.umap_3d) : new THREE.Vector3(...concept.pca_3d);
        const p2 = engine === 'umap' ? new THREE.Vector3(...targetComm.umap_3d) : new THREE.Vector3(...targetComm.pca_3d);

        const geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
        const mat = new THREE.LineDashedMaterial({{
          color: 0xf43f5e,
          dashSize: 1.5,
          gapSize: 0.8,
          transparent: true,
          opacity: 0.65
        }});
        const line = new THREE.Line(geo, mat);
        line.computeLineDistances();
        line.userData = {{ type: 'refugee_ray', ...ref }};

        refugeeGroup.add(line);
        refugeeLines.push(line);
      }});
    }}
    rebuildRefugees('umap');

    // =========================================================================
    // DUAL-ENGINE ANIMATED INTERPOLATION (TWEENING)
    // =========================================================================
    let isTransitioning = false;
    let transitionStartTime = 0;
    const transitionDuration = 1200; // ms

    function switchEngine(newEngine) {{
      if (newEngine === currentEngine || isTransitioning) return;
      currentEngine = newEngine;
      isTransitioning = true;
      transitionStartTime = performance.now();

      document.getElementById('btn-engine-umap').classList.toggle('active', currentEngine === 'umap');
      document.getElementById('btn-engine-pca').classList.toggle('active', currentEngine === 'pca');

      // Cache start positions for tweening
      centroidMeshes.forEach(m => {{
        m.userData.start_pos = m.position.clone();
        m.userData.target_pos = currentEngine === 'umap' ? m.userData.umap_pos : m.userData.pca_pos;
      }});

      conceptMeshes.forEach(m => {{
        m.userData.start_pos = m.position.clone();
        m.userData.target_pos = currentEngine === 'umap' ? m.userData.umap_pos : m.userData.pca_pos;
      }});

      hullMeshes.forEach(m => {{
        m.userData.start_pos = m.position.clone();
        m.userData.target_pos = currentEngine === 'umap' ? m.userData.umap_pos : m.userData.pca_pos;
        m.userData.start_rad = m.scale.x;
        m.userData.target_rad = currentEngine === 'umap' ? m.userData.radius_umap : m.userData.radius_pca;
      }});
    }}

    function updateTransition(timeNow) {{
      if (!isTransitioning) return;
      const elapsed = timeNow - transitionStartTime;
      const progress = Math.min(1.0, elapsed / transitionDuration);
      // Smooth cubic-bezier easeInOut
      const ease = progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2;

      centroidMeshes.forEach(m => {{
        m.position.lerpVectors(m.userData.start_pos, m.userData.target_pos, ease);
      }});

      conceptMeshes.forEach(m => {{
        m.position.lerpVectors(m.userData.start_pos, m.userData.target_pos, ease);
      }});

      hullMeshes.forEach(m => {{
        m.position.lerpVectors(m.userData.start_pos, m.userData.target_pos, ease);
        const rad = THREE.MathUtils.lerp(m.userData.start_rad, m.userData.target_rad, ease);
        m.scale.setScalar(rad);
      }});

      if (progress >= 1.0) {{
        isTransitioning = false;
        rebuildHighways(currentEngine);
        rebuildRefugees(currentEngine);
        if (typeof updateVisibility === 'function') updateVisibility();
      }}
    }}

    document.getElementById('btn-engine-umap').addEventListener('click', () => switchEngine('umap'));
    document.getElementById('btn-engine-pca').addEventListener('click', () => switchEngine('pca'));

    // =========================================================================
    // RAYCASTER & INTERACTION ENGINE
    // =========================================================================
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 2.0;
    const mouse = new THREE.Vector2(-999, -999);

    const tooltip = document.getElementById('hover-tooltip');
    const ttTitle = document.getElementById('tt-title');
    const ttSub = document.getElementById('tt-sub');

    let hoveredMesh = null;
    let selectedMesh = null;

    window.addEventListener('mousemove', (e) => {{
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY + 14) + 'px';
    }});

    function checkRaycast() {{
      if (isTransitioning) return;
      raycaster.setFromCamera(mouse, camera);

      // Check centroids first, then concepts
      const activeObjects = [];
      if (currentFilterMode !== 'hulls') {{
        centroidMeshes.forEach(m => {{ if (m.visible) activeObjects.push(m); }});
        conceptMeshes.forEach(m => {{ if (m.visible) activeObjects.push(m); }});
      }}

      const intersects = raycaster.intersectObjects(activeObjects, false);

      if (intersects.length > 0) {{
        const target = intersects[0].object;
        if (hoveredMesh !== target) {{
          resetHover();
          hoveredMesh = target;
          target.scale.multiplyScalar(1.4);
          document.body.style.cursor = 'pointer';

          tooltip.style.display = 'block';
          if (target.userData.type === 'centroid') {{
            ttTitle.innerText = `🪐 ${{target.userData.label}}`;
            ttSub.innerText = `${{target.userData.theme}} (${{target.userData.total_concepts}} concepts)`;
          }} else {{
            ttTitle.innerText = target.userData.entity;
            ttSub.innerText = `${{target.userData.tier}} &bull; ${{target.userData.community_label}}`;
          }}
        }}
      }} else {{
        resetHover();
      }}
    }}

    function resetHover() {{
      if (hoveredMesh) {{
        hoveredMesh.scale.divideScalar(1.4);
        hoveredMesh = null;
        document.body.style.cursor = 'default';
        tooltip.style.display = 'none';
      }}
    }}

    // Click handler to select and inspect
    window.addEventListener('click', (e) => {{
      if (e.target.closest('header') || e.target.closest('aside') || e.target.closest('footer')) return;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects([...centroidMeshes, ...conceptMeshes], false);

      if (intersects.length > 0) {{
        const hit = intersects[0].object;
        selectEntity(hit.userData);
      }}
    }});

    function selectEntity(data) {{
      const nameEl = document.getElementById('inspector-name');
      const tierEl = document.getElementById('inspector-tier');
      const commEl = document.getElementById('inspector-community');
      const mComp = document.getElementById('m-composite');
      const mProto = document.getElementById('m-proto');
      const mCent = document.getElementById('m-cent');
      const mFid = document.getElementById('m-fid');
      const quoteEl = document.getElementById('inspector-quote');
      const refAlert = document.getElementById('refugee-alert-box');
      const refDesc = document.getElementById('refugee-alert-desc');

      if (data.type === 'centroid') {{
        nameEl.innerText = `🪐 ${{data.label}}`;
        tierEl.innerText = 'COMMUNITY CORE';
        tierEl.className = 'node-tier-badge tier-nucleus';
        commEl.innerText = data.theme;
        mComp.innerText = `${{data.total_concepts}} Nodes`;
        mProto.innerText = 'Centroid';
        mCent.innerText = '--';
        mFid.innerText = '--';
        quoteEl.innerText = `Fréchet Mean semantic center of mass for ${{data.label}}. Exemplar nucleus concepts include: ${{data.exemplar_concepts ? data.exemplar_concepts.join(', ') : 'N/A'}}.`;
        refAlert.style.display = 'none';
      }} else {{
        nameEl.innerText = data.entity;
        tierEl.innerText = data.tier;
        tierEl.className = `node-tier-badge ${{data.tier_key === 'tier_1_nucleus' ? 'tier-nucleus' : (data.tier_key === 'tier_2_mechanics' ? 'tier-mechanics' : 'tier-modifiers')}}`;
        commEl.innerText = data.community_label;
        mComp.innerText = (data.composite_score || 0).toFixed(4);
        mProto.innerText = (data.prototypicality || 0).toFixed(4);
        mCent.innerText = (data.centrality || 0).toFixed(4);
        mFid.innerText = (data.fidelity || 0).toFixed(4);
        quoteEl.innerText = data.exemplar_quote ? `"${{data.exemplar_quote}}"` : 'No source quote context associated.';

        if (data.is_refugee) {{
          refAlert.style.display = 'flex';
          refDesc.innerText = `Attracted to Community #${{data.refugee_target_id}} (Pull Delta: +${{data.refugee_delta}})`;
        }} else {{
          refAlert.style.display = 'none';
        }}
      }}

      // Populate Original Extracted Themes
      const themesEl = document.getElementById('inspector-themes');
      const themesCountEl = document.getElementById('themes-total-count');
      const targetCid = (data.community_id !== undefined && data.community_id !== null) ? data.community_id : selectedCommunityId;
      const cMeta = (targetCid !== null && targetCid !== undefined) ? communityMap[targetCid] : null;

      if (cMeta && cMeta.top_themes && cMeta.top_themes.length > 0) {{
        if (themesCountEl) {{
          themesCountEl.innerText = `${{cMeta.total_triplets || 0}} Triplets`;
        }}
        const clusterColor = cMeta.color || 'var(--accent-cyan)';
        themesEl.innerHTML = cMeta.top_themes.map(th => `
          <div class="theme-item" title="${{th.theme}}: ${{th.count}} triplets (${{th.percentage}}%)">
            <div class="theme-item-header">
              <span class="theme-name">${{th.theme}}</span>
              <span class="theme-stat">${{th.count}} <span style="color:var(--text-muted); font-weight:400; font-size:9px;">(${{th.percentage}}%)</span></span>
            </div>
            <div class="theme-bar-bg">
              <div class="theme-bar-fill" style="width: ${{Math.min(100, th.percentage)}}%; background: ${{clusterColor}};"></div>
            </div>
          </div>
        `).join('');
      }} else {{
        if (themesCountEl) themesCountEl.innerText = '--';
        themesEl.innerHTML = `<span style="font-size:11px; color:var(--text-muted); font-style:italic;">No extracted themes available for this cluster.</span>`;
      }}
    }}

    // =========================================================================
    // UI POPULATION & EVENT CONTROLS
    // =========================================================================
    // Track visible communities (all visible by default)
    let visibleCommunityIds = new Set(activeData.communities.map(c => c.community_id));

    // 1. Populate Community List with Visibility Toggles and Solo Focus
    function populateCommunityList() {{
      const commListEl = document.getElementById('community-list');
      if (!commListEl) return;
      commListEl.innerHTML = '';
      visibleCommunityIds = new Set(activeData.communities.map(c => c.community_id));

      activeData.communities.forEach(c => {{
        const item = document.createElement('div');
        item.className = 'comm-item';
        item.dataset.cid = c.community_id;
        item.innerHTML = `
          <div class="comm-item-top">
            <span class="comm-title">
              <span class="color-dot" style="background:${{c.color}};"></span>
              ${{c.label}}
            </span>
            <div class="comm-item-actions">
              <span class="comm-count">${{c.total_concepts}}</span>
              <button class="comm-btn-icon comm-solo-btn" title="Solo / Isolate only this community" data-cid="${{c.community_id}}">🎯</button>
              <button class="comm-btn-icon comm-eye-btn" title="Toggle visibility of this community" data-cid="${{c.community_id}}">👁️</button>
            </div>
          </div>
          <div class="comm-theme">${{c.headline_theme}}</div>
        `;

        // Eye button: Toggle visibility of this specific community
        const eyeBtn = item.querySelector('.comm-eye-btn');
        eyeBtn.addEventListener('click', (e) => {{
          e.stopPropagation();
          if (visibleCommunityIds.has(c.community_id)) {{
            visibleCommunityIds.delete(c.community_id);
          }} else {{
            visibleCommunityIds.add(c.community_id);
          }}
          updateVisibility();
        }});

        // Solo button: Isolate only this community and fly to it
        const soloBtn = item.querySelector('.comm-solo-btn');
        soloBtn.addEventListener('click', (e) => {{
          e.stopPropagation();
          visibleCommunityIds.clear();
          visibleCommunityIds.add(c.community_id);
          flyToCommunity(c.community_id);
          updateVisibility();
        }});

        // Card click: If multi-select modifier (Shift/Ctrl/Cmd), toggle; else ensure visible & fly to it
        item.addEventListener('click', (e) => {{
          if (e.shiftKey || e.ctrlKey || e.metaKey) {{
            if (visibleCommunityIds.has(c.community_id)) {{
              visibleCommunityIds.delete(c.community_id);
            }} else {{
              visibleCommunityIds.add(c.community_id);
            }}
          }} else {{
            if (!visibleCommunityIds.has(c.community_id)) {{
              visibleCommunityIds.add(c.community_id);
            }}
            flyToCommunity(c.community_id);
          }}
          updateVisibility();
        }});

        commListEl.appendChild(item);
      }});
    }}
    populateCommunityList();

    // Toolbar buttons for batch visibility operations
    const btnShowAll = document.getElementById('btn-show-all-comms');
    if (btnShowAll) {{
      btnShowAll.addEventListener('click', () => {{
        activeData.communities.forEach(c => visibleCommunityIds.add(c.community_id));
        updateVisibility();
      }});
    }}

    const btnHideAll = document.getElementById('btn-hide-all-comms');
    if (btnHideAll) {{
      btnHideAll.addEventListener('click', () => {{
        visibleCommunityIds.clear();
        updateVisibility();
      }});
    }}

    const btnInvert = document.getElementById('btn-invert-comms');
    if (btnInvert) {{
      btnInvert.addEventListener('click', () => {{
        activeData.communities.forEach(c => {{
          if (visibleCommunityIds.has(c.community_id)) {{
            visibleCommunityIds.delete(c.community_id);
          }} else {{
            visibleCommunityIds.add(c.community_id);
          }}
        }});
        updateVisibility();
      }});
    }}

    // 2. Cinematic Camera Fly-To Animation
    function flyToCommunity(cid) {{
      const targetCentroid = centroidMeshes.find(m => m.userData.community_id === cid);
      if (!targetCentroid) return;

      selectedCommunityId = cid;

      // Highlight active in list
      document.querySelectorAll('.comm-item').forEach(el => {{
        el.classList.toggle('active', parseInt(el.dataset.cid) === cid);
      }});

      // Select in inspector
      selectEntity(targetCentroid.userData);

      // Camera Fly-To animation
      const targetPos = targetCentroid.position.clone();
      const cameraOffset = new THREE.Vector3(targetPos.x, targetPos.y + 18, targetPos.z + 42);

      const startCam = camera.position.clone();
      const startTarget = controls.target.clone();
      const startTime = performance.now();
      const duration = 1400;

      function stepCam(t) {{
        const elapsed = t - startTime;
        const p = Math.min(1.0, elapsed / duration);
        const ease = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;

        camera.position.lerpVectors(startCam, cameraOffset, ease);
        controls.target.lerpVectors(startTarget, targetPos, ease);

        if (p < 1.0) {{
          requestAnimationFrame(stepCam);
        }}
      }}
      requestAnimationFrame(stepCam);
    }}

    // 3. Mode Pills Filter
    document.querySelectorAll('.mode-pill').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.mode-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilterMode = btn.dataset.mode;
        updateVisibility();
      }});
    }});

    // 4. Centralized Multi-Level Visibility Engine
    function updateVisibility() {{
      const mode = currentFilterMode;
      const filterInputEl = document.getElementById('filter-input');
      const q = filterInputEl ? filterInputEl.value.toLowerCase().trim() : '';

      // Ensure parent groups remain active containers
      centroidGroup.visible = true;
      highwayGroup.visible = true;
      hullGroup.visible = true;
      refugeeGroup.visible = true;
      conceptGroup.visible = true;

      // 1. Concept Meshes
      conceptMeshes.forEach(mesh => {{
        const cid = mesh.userData.community_id;
        const isCommVis = visibleCommunityIds.has(cid);

        if (!isCommVis) {{
          mesh.visible = false;
          return;
        }}

        if (q) {{
          const match = mesh.userData.entity.toLowerCase().includes(q) ||
                        mesh.userData.community_label.toLowerCase().includes(q);
          if (!match) {{
            mesh.visible = false;
            return;
          }}
        }}

        if (mode === 'all') {{
          mesh.visible = true;
        }} else if (mode === 'centroids') {{
          mesh.visible = false;
        }} else if (mode === 'hulls') {{
          mesh.visible = true;
        }} else if (mode === 'refugees') {{
          mesh.visible = Boolean(mesh.userData.is_refugee);
        }} else if (mode === 'tier1') {{
          mesh.visible = (mesh.userData.tier_key === 'tier_1_nucleus');
        }}
      }});

      // 2. Centroid Meshes
      centroidMeshes.forEach(mesh => {{
        const cid = mesh.userData.community_id;
        const isCommVis = visibleCommunityIds.has(cid);
        const modeAllows = (mode === 'all' || mode === 'centroids' || mode === 'tier1');
        mesh.visible = isCommVis && modeAllows;
      }});

      // 3. Density Hull Meshes
      hullMeshes.forEach(mesh => {{
        const cid = mesh.userData.community_id;
        const isCommVis = visibleCommunityIds.has(cid);
        const modeAllows = (mode === 'all' || mode === 'hulls');
        mesh.visible = isCommVis && modeAllows;
      }});

      // 4. Highway Conduits (visible if both connected communities are active)
      highwayLines.forEach(line => {{
        const sVis = visibleCommunityIds.has(line.userData.source);
        const tVis = visibleCommunityIds.has(line.userData.target);
        const modeAllows = (mode === 'all' || mode === 'centroids');
        line.visible = (sVis && tVis) && modeAllows;
      }});

      // 5. Refugee Rays (visible if either source or target community is active)
      refugeeLines.forEach(line => {{
        const sVis = visibleCommunityIds.has(line.userData.source_community);
        const tVis = visibleCommunityIds.has(line.userData.target_community);
        const modeAllows = (mode === 'all' || mode === 'refugees');
        line.visible = (sVis || tVis) && modeAllows;
      }});

      // 6. Update Community List Cards & Eye Button State
      document.querySelectorAll('.comm-item').forEach(el => {{
        const cid = parseInt(el.dataset.cid);
        const isVis = visibleCommunityIds.has(cid);
        el.classList.toggle('comm-hidden', !isVis);
        const eyeBtn = el.querySelector('.comm-eye-btn');
        if (eyeBtn) {{
          eyeBtn.innerHTML = isVis ? '👁️' : '🚫';
          eyeBtn.title = isVis ? 'Click to hide this community' : 'Click to show this community';
          eyeBtn.classList.toggle('off', !isVis);
        }}
      }});

      // 7. Update Visible Communities Counter Badge
      const counterEl = document.getElementById('visible-comm-counter');
      if (counterEl) {{
        counterEl.innerText = `${{visibleCommunityIds.size}} / ${{activeData.communities.length}} Visible`;
      }}
    }}

    // 5. Search Filter
    const filterInput = document.getElementById('filter-input');
    if (filterInput) {{
      filterInput.addEventListener('input', () => updateVisibility());
    }}

    // 6. Camera Control Buttons
    document.getElementById('btn-reset-cam').addEventListener('click', () => {{
      camera.position.set(0, 45, 175);
      controls.target.set(0, 0, 0);
      selectedCommunityId = null;
      document.querySelectorAll('.comm-item').forEach(el => el.classList.remove('active'));
      activeData.communities.forEach(c => visibleCommunityIds.add(c.community_id));
      updateVisibility();
    }});

    document.getElementById('btn-top-view').addEventListener('click', () => {{
      camera.position.set(0, 220, 0);
      controls.target.set(0, 0, 0);
    }});

    const autoBtn = document.getElementById('btn-auto-rotate');
    autoBtn.addEventListener('click', () => {{
      autoRotate = !autoRotate;
      controls.autoRotate = autoRotate;
      autoBtn.innerText = `Orbit: ${{autoRotate ? 'ON' : 'OFF'}}`;
    }});

    // 7. Partition Toggle Mode Engine
    function togglePartitionMode() {{
      if (!hasBaseline) return;
      isAgglomerated = !isAgglomerated;
      activeData = isAgglomerated ? (GALAXY_DATA.agglom || GALAXY_DATA) : GALAXY_DATA.baseline;

      // 1. Update Badge & Button UI
      const badgeEl = document.getElementById('partition-badge');
      const btnText = document.getElementById('partition-toggle-text');

      if (isAgglomerated) {{
        if (badgeEl) {{
          badgeEl.innerText = 'Agglomerated Clusters (Macro)';
          badgeEl.classList.remove('baseline');
        }}
        if (btnText) {{
          btnText.innerText = 'Switch to Baseline Communities';
        }}
      }} else {{
        if (badgeEl) {{
          badgeEl.innerText = 'Baseline Communities (Granular)';
          badgeEl.classList.add('baseline');
        }}
        if (btnText) {{
          btnText.innerText = 'Switch to Agglomerated Clusters';
        }}
      }}

      // 2. Update Stats Ribbon
      const meta = activeData.meta || {{}};
      const commsEl = document.getElementById('stat-total-comms');
      const hwsEl = document.getElementById('stat-total-highways');
      const refsEl = document.getElementById('stat-total-refugees');
      if (commsEl) commsEl.innerText = meta.total_communities ?? activeData.communities.length;
      if (hwsEl) hwsEl.innerText = meta.total_highways ?? activeData.highways.length;
      if (refsEl) refsEl.innerText = meta.total_refugees ?? activeData.refugees.length;

      // 3. Rebuild Community Lookup & 3D Objects
      rebuildCommunityMap();
      updateConceptPartitions();
      rebuildCentroids(currentEngine);
      rebuildHulls(currentEngine);
      rebuildHighways(currentEngine);
      rebuildRefugees(currentEngine);
      populateCommunityList();

      // 4. Reset inspector and selections
      selectedCommunityId = null;
      if (conceptMeshes.length > 0) {{
        selectEntity(conceptMeshes[0].userData);
      }}
      updateVisibility();
    }}

    // Initialize button visibility if baseline exists
    const toggleBtnEl = document.getElementById('btn-toggle-partition');
    if (toggleBtnEl) {{
      if (hasBaseline) {{
        toggleBtnEl.style.display = 'inline-flex';
        toggleBtnEl.addEventListener('click', togglePartitionMode);
      }} else {{
        toggleBtnEl.style.display = 'none';
      }}
    }}

    // Window Resize
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // Select initial exemplar
    if (conceptMeshes.length > 0) {{
      selectEntity(conceptMeshes[0].userData);
    }}

    // =========================================================================
    // MAIN RENDER LOOP
    // =========================================================================
    function animate(timeNow) {{
      requestAnimationFrame(animate);

      updateTransition(timeNow);
      checkRaycast();
      controls.update();

      // Gentle Starfield Drift
      starPoints.rotation.y += 0.00015;

      renderer.render(scene, camera);
    }}
    requestAnimationFrame(animate);

  </script>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_template)

        print(f"   -> [3D Galaxy] Successfully generated 3D Galaxy visualization: {output_path}")
        return output_path


if __name__ == "__main__":
    vis = Galaxy3DVisualizer()
    out = vis.generate_3d_galaxy_dashboard()
    print(f"✅ Generated 3D Galaxy HTML at: {out}")
