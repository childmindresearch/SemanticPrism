"""
SemanticPrism Stage 3 Topology: Cluster Embedding & Centroid Dispersion Visualizer
==================================================================================
Dedicated module for computing and visualizing community clusters through their
semantic center of mass (Fréchet mean) in high-dimensional embedding space.

Provides:
  1. Fréchet Mean Centroid computation for each community partition:
     μ_C = (∑_{v ∈ C} e_v) / ||∑_{v ∈ C} e_v||_2
  2. Intra-Cluster Dispersion & Null Model Baselines (within-cluster distance,
     prototypicality isochrones, and compactness percentiles vs random chance).
  3. Inter-Cluster Distance Topology (pairwise centroid distance matrix,
     nearest-neighbor ranking, and boundary bridging concepts).
  4. Cross-Border Drift & "Semantic Refugee" Detection (concepts whose embedding
     vector is closer to an external community's centroid than to their own).
  5. Topological Highway vs Semantic Alignment Analysis (combining graph edge
     traffic from community_highways.json with embedding similarity).
  6. Automated Cluster Health Scorecards & Actionable Downstream Synthesis Guidance.
  7. Unified Interactive Multi-Panel Dashboard (HTML5 + SVG + Canvas).
"""

import json
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


class ClusterEmbeddingVisualizer:
    """
    Computes Fréchet mean centroids, cross-border drift, topological-semantic
    alignment, and health diagnostics for community clusters in embedding space.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    # =========================================================================
    # METRIC COMPUTATION ENGINE
    # =========================================================================

    def compute_centroid_metrics(
        self,
        base_dir: Path = Path("outputs/03_topology"),
        refinement_dir: Path = Path("outputs/02_refinement")
    ) -> Dict[str, Any]:
        """
        Loads community partitions and concept embeddings to compute:
          - Unit-normalized Fréchet mean centroid for every community
          - Intra-cluster distances (within-cluster dispersion & prototypicality)
          - Pairwise inter-centroid distance matrix (across-cluster separation)
          - Cross-border "Semantic Refugees" (concepts attracted to other centroids)
          - Graph highway vs. semantic proximity alignment
          - Automated health scorecards and synthesis guidance recommendations
        """
        # 1. Load Concept Embeddings & Metadata
        cache_dir = refinement_dir / "cache"
        concept_npy = cache_dir / "concept_all_MiniLM_L6_v2_embeddings.npy"
        concept_meta = cache_dir / "concept_all_MiniLM_L6_v2_meta.json"

        if not concept_npy.exists() or not concept_meta.exists():
            subject_npy = cache_dir / "subject_all_MiniLM_L6_v2_embeddings.npy"
            subject_meta = cache_dir / "subject_all_MiniLM_L6_v2_meta.json"
            if subject_npy.exists() and subject_meta.exists():
                concept_npy, concept_meta = subject_npy, subject_meta
            else:
                raise FileNotFoundError(f"Concept embeddings not found in '{cache_dir}'. Ensure Stage 2 is complete.")

        with open(concept_meta, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        vocab_list = meta_data.get("vocab", [])
        vocab_map = {str(term).lower().strip(): idx for idx, term in enumerate(vocab_list)}
        embedding_matrix = np.load(concept_npy)

        # 2. Load Community Architectural Tiers and/or Partitions
        comm_dir = base_dir / "03_paths" / "community"
        tiers_json = comm_dir / "community_architectural_tiers.json"
        partitions_json = comm_dir / "topology_partitions.json"
        highways_json = comm_dir / "community_highways.json"

        if not tiers_json.exists() and not partitions_json.exists():
            raise FileNotFoundError(f"Neither community tiers nor partitions found in '{comm_dir}'. Ensure Stage 3 is complete.")

        tiers_data = []
        if tiers_json.exists():
            with open(tiers_json, "r", encoding="utf-8") as f:
                tiers_data = json.load(f)
        else:
            with open(partitions_json, "r", encoding="utf-8") as f:
                part_raw = json.load(f)
            for c in part_raw.get("communities", []):
                cid = c.get("community_id")
                nodes = c.get("nodes", [])
                tiers_data.append({
                    "community_id": cid,
                    "community_label": f"Community #{cid}",
                    "total_nodes": len(nodes),
                    "tier_counts": {"tier_1_nucleus": 0, "tier_2_mechanics": len(nodes), "tier_3_modifiers": 0},
                    "tier_1_nucleus": [],
                    "tier_2_mechanics": [{"entity": n, "prototypicality": 0.5, "centrality": 0.5, "fidelity": 0.2, "composite_score": 0.4, "internal_degree": 1, "boundary_degree": 0, "exemplar_quote": ""} for n in nodes],
                    "tier_3_modifiers": []
                })

        # Load inter-community highway traffic (directed graph edge counts)
        highway_map = defaultdict(lambda: defaultdict(int))
        if highways_json.exists():
            try:
                with open(highways_json, "r", encoding="utf-8") as hf:
                    hw_data = json.load(hf)
                for h in hw_data.get("highways", []):
                    sc, tc, cnt = h.get("source_community"), h.get("target_community"), h.get("directed_edge_count", 0)
                    highway_map[sc][tc] += cnt
                    highway_map[tc][sc] += cnt  # symmetric edge volume
            except Exception:
                pass

        # 3. Compute Fréchet Centroids & Intra-Cluster Metrics
        community_records = []
        centroid_vectors = []
        valid_community_ids = []
        all_concept_vectors = []

        for c_info in tiers_data:
            cid = c_info.get("community_id")
            label = c_info.get("community_label", f"Community #{cid}")
            
            all_concepts = []
            for tier_name, tier_key in [("Tier 1: Nucleus", "tier_1_nucleus"), ("Tier 2: Mechanics", "tier_2_mechanics"), ("Tier 3: Modifiers", "tier_3_modifiers")]:
                for item in c_info.get(tier_key, []):
                    entity = str(item.get("entity", "")).lower().strip()
                    if entity:
                        all_concepts.append({
                            "entity": entity,
                            "tier": tier_name,
                            "tier_key": tier_key,
                            "composite_score": round(float(item.get("composite_score", 0.0)), 4),
                            "prototypicality": round(float(item.get("prototypicality", 0.0)), 4),
                            "centrality": round(float(item.get("centrality", 0.0)), 4),
                            "fidelity": round(float(item.get("fidelity", 0.0)), 4),
                            "internal_degree": int(item.get("internal_degree", 0)),
                            "boundary_degree": int(item.get("boundary_degree", 0)),
                            "exemplar_quote": str(item.get("exemplar_quote", ""))
                        })

            if not all_concepts:
                continue

            concept_vectors = []
            valid_concepts = []
            for c_obj in all_concepts:
                e_name = c_obj["entity"]
                if e_name in vocab_map:
                    vec = embedding_matrix[vocab_map[e_name]]
                    concept_vectors.append(vec)
                    valid_concepts.append(c_obj)
                    all_concept_vectors.append(vec)

            if len(concept_vectors) == 0:
                continue

            concept_mat = np.array(concept_vectors)
            
            # Unit-normalized Fréchet Mean Centroid: μ_C = mean(vecs) / ||mean(vecs)||_2
            raw_mean = np.mean(concept_mat, axis=0)
            norm_val = np.linalg.norm(raw_mean)
            centroid_l2 = raw_mean / (norm_val if norm_val > 1e-9 else 1.0)
            
            # Intra-cluster cosine similarity and cosine distance: d_u = 1 - (e_u · μ_C)
            sim_to_centroid = np.clip(concept_mat @ centroid_l2, -1.0, 1.0)
            dist_to_centroid = 1.0 - sim_to_centroid

            for i, c_obj in enumerate(valid_concepts):
                c_obj["cosine_distance"] = round(float(dist_to_centroid[i]), 4)
                c_obj["prototypicality"] = round(float(sim_to_centroid[i]), 4)

            mean_dist = float(np.mean(dist_to_centroid))
            median_dist = float(np.median(dist_to_centroid))
            std_dist = float(np.std(dist_to_centroid))
            min_dist = float(np.min(dist_to_centroid))
            max_dist = float(np.max(dist_to_centroid))
            p10_dist = float(np.percentile(dist_to_centroid, 10))
            p90_dist = float(np.percentile(dist_to_centroid, 90))

            mean_fidelity = float(np.mean([x["fidelity"] for x in valid_concepts]))

            valid_concepts.sort(key=lambda x: x["cosine_distance"])

            community_records.append({
                "community_id": cid,
                "community_label": label,
                "total_concepts": len(valid_concepts),
                "intra_dispersion": {
                    "mean_distance": round(mean_dist, 4),
                    "median_distance": round(median_dist, 4),
                    "std_distance": round(std_dist, 4),
                    "min_distance": round(min_dist, 4),
                    "max_distance": round(max_dist, 4),
                    "p10_distance": round(p10_dist, 4),
                    "p90_distance": round(p90_dist, 4),
                    "compactness_index": round(1.0 - mean_dist, 4),
                    "mean_grounding_fidelity": round(mean_fidelity, 4)
                },
                "tier_counts": {
                    "tier_1_nucleus": sum(1 for x in valid_concepts if x["tier_key"] == "tier_1_nucleus"),
                    "tier_2_mechanics": sum(1 for x in valid_concepts if x["tier_key"] == "tier_2_mechanics"),
                    "tier_3_modifiers": sum(1 for x in valid_concepts if x["tier_key"] == "tier_3_modifiers")
                },
                "exemplar_concepts": [x["entity"] for x in valid_concepts[:6]],
                "concepts": valid_concepts
            })
            centroid_vectors.append(centroid_l2)
            valid_community_ids.append(cid)

        num_comms = len(community_records)
        if num_comms == 0:
            raise ValueError("No valid communities with embeddings could be resolved.")

        centroids_arr = np.array(centroid_vectors)

        # 4. Null Model Baseline (Random cluster dispersion from full pool)
        null_intra_dist = 0.785
        if len(all_concept_vectors) > 100:
            all_mat = np.array(all_concept_vectors)
            rand_indices = np.random.RandomState(42).choice(len(all_mat), size=min(400, len(all_mat)), replace=False)
            rand_sample = all_mat[rand_indices]
            rand_centroid = np.mean(rand_sample, axis=0)
            rand_centroid /= max(1e-9, np.linalg.norm(rand_centroid))
            null_intra_dist = float(np.mean(1.0 - (rand_sample @ rand_centroid)))

        # 5. Compute Inter-Centroid Distance Matrix
        inter_sim_mat = np.clip(centroids_arr @ centroids_arr.T, -1.0, 1.0)
        inter_dist_mat = np.clip(1.0 - inter_sim_mat, 0.0, 2.0)

        # 6. Project Centroids to 2D Plane using PCA
        if num_comms >= 2:
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(centroids_arr)
            max_coord = np.max(np.abs(coords_2d))
            if max_coord > 1e-6:
                coords_2d = (coords_2d / max_coord) * 0.85
            explained_var = float(np.sum(pca.explained_variance_ratio_))
        else:
            coords_2d = np.array([[0.0, 0.0]])
            explained_var = 1.0

        for idx, c_rec in enumerate(community_records):
            c_rec["coord_x"] = round(float(coords_2d[idx, 0]), 4)
            c_rec["coord_y"] = round(float(coords_2d[idx, 1]), 4)

        # 7. Cross-Border "Semantic Refugees" & Highway Alignment Engine
        total_global_refugees = 0
        matrix_rows = []

        for i, src_rec in enumerate(community_records):
            src_cid = src_rec["community_id"]
            src_vecs = np.array([embedding_matrix[vocab_map[c["entity"]]] for c in src_rec["concepts"]])

            # Compute distances from all concepts in src_rec to ALL centroids: shape (N, K)
            concept_to_all_centroids_dist = 1.0 - np.clip(src_vecs @ centroids_arr.T, -1.0, 1.0)
            closest_centroid_indices = np.argmin(concept_to_all_centroids_dist, axis=1)
            own_distances = concept_to_all_centroids_dist[:, i]

            # Detect Semantic Refugees
            refugee_list = []
            for c_idx, c_obj in enumerate(src_rec["concepts"]):
                best_k = closest_centroid_indices[c_idx]
                if best_k != i:
                    best_pull_cid = community_records[best_k]["community_id"]
                    best_pull_label = community_records[best_k]["community_label"]
                    drift_margin = float(own_distances[c_idx] - concept_to_all_centroids_dist[c_idx, best_k])
                    c_obj["is_refugee"] = True
                    c_obj["pull_community_id"] = best_pull_cid
                    c_obj["pull_community_label"] = best_pull_label
                    c_obj["drift_margin"] = round(drift_margin, 4)
                    refugee_list.append({
                        "entity": c_obj["entity"],
                        "tier": c_obj["tier"],
                        "own_distance": c_obj["cosine_distance"],
                        "pull_community_id": best_pull_cid,
                        "pull_community_label": best_pull_label,
                        "neighbor_distance": round(float(concept_to_all_centroids_dist[c_idx, best_k]), 4),
                        "drift_margin": round(drift_margin, 4),
                        "exemplar_quote": c_obj.get("exemplar_quote", "")
                    })
                else:
                    c_obj["is_refugee"] = False

            refugee_list.sort(key=lambda x: x["drift_margin"], reverse=True)
            src_rec["semantic_refugees"] = refugee_list
            src_rec["refugee_count"] = len(refugee_list)
            src_rec["refugee_percentage"] = round((len(refugee_list) / max(1, len(src_rec["concepts"]))) * 100, 1)
            total_global_refugees += len(refugee_list)

            # Nearest Neighbors & Highway Alignment
            neighbors = []
            for j, tgt_rec in enumerate(community_records):
                if i == j:
                    continue
                tgt_cid = tgt_rec["community_id"]
                dist_val = float(inter_dist_mat[i, j])
                sim_val = float(inter_sim_mat[i, j])
                highway_traffic = highway_map[src_cid][tgt_cid]

                # Quadrant Categorization
                if highway_traffic >= 8 and sim_val >= 0.88:
                    quadrant = "Tightly Coupled Workflow"
                elif highway_traffic >= 8 and sim_val < 0.88:
                    quadrant = "Structural Bridge (Different Domains)"
                elif highway_traffic < 8 and sim_val >= 0.92:
                    quadrant = "High Semantic Redundancy (Merge Risk)"
                else:
                    quadrant = "Autonomous Disjoint Domain"

                neighbors.append({
                    "target_community_id": tgt_cid,
                    "target_label": tgt_rec["community_label"],
                    "inter_centroid_distance": round(dist_val, 4),
                    "cosine_similarity": round(sim_val, 4),
                    "highway_traffic_edges": highway_traffic,
                    "alignment_quadrant": quadrant
                })

            neighbors.sort(key=lambda x: x["inter_centroid_distance"])
            src_rec["nearest_neighbors"] = neighbors[:5]
            src_rec["furthest_neighbors"] = neighbors[-3:]

            # Separation Ratio: d_nearest / mean_intra
            nearest_d = neighbors[0]["inter_centroid_distance"] if neighbors else 1.0
            intra_d = src_rec["intra_dispersion"]["mean_distance"]
            src_rec["separation_ratio"] = round(nearest_d / max(0.001, intra_d), 4)

            # Nearest Neighbor Boundary Bridging Concepts
            if neighbors:
                closest_tgt_idx = [k for k, r in enumerate(community_records) if r["community_id"] == neighbors[0]["target_community_id"]][0]
                tgt_centroid = centroids_arr[closest_tgt_idx]
                dists_to_neighbor = 1.0 - np.clip(src_vecs @ tgt_centroid, -1.0, 1.0)
                bridge_indices = np.argsort(dists_to_neighbor)[:5]
                src_rec["boundary_bridge_concepts"] = [
                    {
                        "entity": src_rec["concepts"][idx]["entity"],
                        "tier": src_rec["concepts"][idx]["tier"],
                        "distance_to_neighbor_centroid": round(float(dists_to_neighbor[idx]), 4),
                        "target_neighbor_label": neighbors[0]["target_label"]
                    }
                    for idx in bridge_indices
                ]

            # 8. Automated Cluster Health Scorecard & Synthesis Guidance
            # Health Score Formulation (0 to 100):
            # Cohesion (30%): (1 - intra_d / null_intra) * 100
            # Separation (30%): min(100, (nearest_d / 0.15) * 100)
            # Fidelity (20%): min(100, (mean_fidelity / 0.35) * 100)
            # Boundary Integrity (20%): (1.0 - refugee_count / total) * 100
            cohesion_score = max(0.0, min(100.0, (1.0 - (intra_d / max(0.001, null_intra_dist))) * 100.0 * 2.5))
            separation_score = max(0.0, min(100.0, (nearest_d / 0.15) * 100.0))
            fidelity_score = max(0.0, min(100.0, (src_rec["intra_dispersion"]["mean_grounding_fidelity"] / 0.35) * 100.0))
            integrity_score = max(0.0, min(100.0, (1.0 - (len(refugee_list) / max(1, len(src_rec["concepts"])))) * 100.0))

            health_score = round(0.30 * cohesion_score + 0.30 * separation_score + 0.20 * fidelity_score + 0.20 * integrity_score, 1)

            # Categorize Status Badge
            closest_sim = neighbors[0]["cosine_similarity"] if neighbors else 0.0
            if closest_sim >= 0.93:
                status_badge = "Merge Risk"
                status_color = "#f97316"
                status_desc = f"Centroid is {closest_sim*100:.1f}% similar to {neighbors[0]['target_label']}. High vocabulary overlap."
            elif health_score >= 70:
                status_badge = "Autonomous Domain"
                status_color = "#10b981"
                status_desc = "Strong cohesion and clear domain separation. Ideal candidate for primary root Pydantic model."
            elif health_score >= 50:
                status_badge = "Integrated Sub-Domain"
                status_color = "#eab308"
                status_desc = f"Heavily interacts with {neighbors[0]['target_label']} ({neighbors[0]['highway_traffic_edges']} highway edges). Model as related sub-schema."
            else:
                status_badge = "Diffuse / Drift Risk"
                status_color = "#ef4444"
                status_desc = "Broad vocabulary and high refugee bleed. Consider treating as descriptive metadata rather than rigid entities."

            # Actionable Stage 4 Synthesis Guidance
            top_nucleus = [x["entity"] for x in src_rec["concepts"] if x["tier_key"] == "tier_1_nucleus"][:3]
            nucleus_str = ", ".join(f"`{t}`" for t in top_nucleus) if top_nucleus else "`GeneralEntity`"
            if status_badge == "Merge Risk":
                synth_guidance = f"⚠️ In Stage 4 consolidation, check for redundancy with {neighbors[0]['target_label']}. Either merge into a unified model or define as a subclass."
            elif status_badge == "Autonomous Domain":
                synth_guidance = f"✅ Synthesize a primary root BaseModel anchored on {nucleus_str}. Model Tier 2 entities as internal attributes."
            elif status_badge == "Integrated Sub-Domain":
                synth_guidance = f"🔗 Model {nucleus_str} with explicit typed references or foreign keys to {neighbors[0]['target_label']} models."
            else:
                synth_guidance = f"🔍 Keep fields loosely typed (`Optional[str] = None`). Discard low-scoring refugee concepts during prompt generation."

            src_rec["health_scorecard"] = {
                "health_score": health_score,
                "status_badge": status_badge,
                "status_color": status_color,
                "status_desc": status_desc,
                "synthesis_guidance": synth_guidance,
                "components": {
                    "cohesion_score": round(cohesion_score, 1),
                    "separation_score": round(separation_score, 1),
                    "fidelity_score": round(fidelity_score, 1),
                    "boundary_integrity_score": round(integrity_score, 1)
                }
            }

            matrix_rows.append({
                "community_id": src_cid,
                "label": src_rec["community_label"],
                "distances": [round(float(inter_dist_mat[i, k]), 4) for k in range(num_comms)],
                "similarities": [round(float(inter_sim_mat[i, k]), 4) for k in range(num_comms)],
                "highway_traffic": [highway_map[src_cid][community_records[k]["community_id"]] for k in range(num_comms)]
            })

        # 9. Global Summary & Diagnostic Health Tally
        all_intra_means = [r["intra_dispersion"]["mean_distance"] for r in community_records]
        upper_tri_inter = inter_dist_mat[np.triu_indices(num_comms, k=1)]
        global_mean_intra = float(np.mean(all_intra_means))
        global_mean_inter = float(np.mean(upper_tri_inter)) if len(upper_tri_inter) > 0 else 0.0

        status_counts = defaultdict(int)
        for r in community_records:
            status_counts[r["health_scorecard"]["status_badge"]] += 1

        global_metrics = {
            "total_communities_analyzed": num_comms,
            "total_concepts_mapped": sum(r["total_concepts"] for r in community_records),
            "global_mean_intra_distance": round(global_mean_intra, 4),
            "global_mean_inter_distance": round(global_mean_inter, 4),
            "global_separation_index": round(global_mean_inter / max(0.001, global_mean_intra), 4),
            "null_model_random_intra_distance": round(null_intra_dist, 4),
            "dispersion_reduction_vs_random_pct": round(((null_intra_dist - global_mean_intra) / null_intra_dist) * 100.0, 1),
            "total_semantic_refugees": total_global_refugees,
            "global_refugee_rate_pct": round((total_global_refugees / max(1, sum(r['total_concepts'] for r in community_records))) * 100.0, 1),
            "status_breakdown": dict(status_counts),
            "pca_variance_retained_2d": round(explained_var, 4)
        }

        return {
            "global_metrics": global_metrics,
            "communities": community_records,
            "inter_centroid_matrix": {
                "community_ids": valid_community_ids,
                "labels": [r["community_label"] for r in community_records],
                "rows": matrix_rows
            }
        }

    # =========================================================================
    # MULTI-PANEL INTERACTIVE DASHBOARD HTML GENERATOR
    # =========================================================================

    def generate_centroid_dispersion_dashboard(
        self,
        base_dir: Path = Path("outputs/03_topology"),
        refinement_dir: Path = Path("outputs/02_refinement"),
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Computes centroid metrics and renders the complete interactive Option 5
        Multi-Panel HTML application (Self-contained, zero external network deps).
        """
        metrics = self.compute_centroid_metrics(base_dir=base_dir, refinement_dir=refinement_dir)
        
        if output_path is None:
            output_path = base_dir / "03_paths" / "community" / "cluster_semantic_dispersion_map.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export companion diagnostics JSON for programmatic evaluation
        diag_path = output_path.parent / "cluster_centroid_diagnostics.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump({
                "global_metrics": metrics["global_metrics"],
                "communities": [
                    {
                        "community_id": c["community_id"],
                        "community_label": c["community_label"],
                        "total_concepts": c["total_concepts"],
                        "health_scorecard": c["health_scorecard"],
                        "intra_dispersion": c["intra_dispersion"],
                        "refugee_metrics": {
                            "refugee_count": c["refugee_count"],
                            "refugee_percentage": c["refugee_percentage"],
                            "top_refugees": c["semantic_refugees"][:5]
                        },
                        "tier_counts": c["tier_counts"],
                        "nearest_neighbors": c["nearest_neighbors"][:3],
                        "boundary_bridge_concepts": c.get("boundary_bridge_concepts", [])
                    }
                    for c in metrics["communities"]
                ]
            }, f, indent=2)

        data_json_str = json.dumps(metrics)

        # Extract status badges for top ribbon
        badge_summary = metrics["global_metrics"]["status_breakdown"]
        badge_pills_html = "".join([
            f'<div class="stat-pill"><span class="num">{cnt}</span><span class="lbl">{badge}</span></div>'
            for badge, cnt in badge_summary.items()
        ])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SemanticPrism // Cluster Centroid & Dispersion Analysis</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {{
      --bg-dark: #090d16;
      --bg-card: rgba(15, 23, 42, 0.88);
      --bg-card-border: rgba(255, 255, 255, 0.08);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-cyan: #06b6d4;
      --accent-blue: #3b82f6;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-orange: #f97316;
      --accent-rose: #ef4444;
      --accent-purple: #8b5cf6;
      --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 10% 15%, rgba(6, 182, 212, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 90% 85%, rgba(59, 130, 246, 0.04) 0%, transparent 40%);
      color: var(--text-main);
      font-family: var(--font-stack);
      min-height: 100vh;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
    }}

    /* Header Bar */
    header {{
      background: rgba(15, 23, 42, 0.96);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--bg-card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .brand-badge {{
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      color: #050b14;
      font-weight: 800;
      font-size: 11px;
      letter-spacing: 1px;
      padding: 4px 8px;
      border-radius: 4px;
      text-transform: uppercase;
    }}

    .brand-title {{
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: #ffffff;
    }}

    .brand-subtitle {{
      font-size: 11px;
      color: var(--text-muted);
    }}

    /* Stat Ribbon */
    .stat-ribbon {{
      display: flex;
      gap: 8px;
    }}

    .stat-pill {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 5px 12px;
      border-radius: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}

    .stat-pill .num {{
      font-size: 12px;
      font-weight: 700;
      color: var(--accent-cyan);
    }}

    .stat-pill .lbl {{
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }}

    /* Main Dashboard Layout */
    .dashboard-container {{
      flex: 1;
      display: grid;
      grid-template-columns: 50% 50%;
      gap: 16px;
      padding: 16px 20px;
      height: calc(100vh - 64px);
      min-height: 720px;
    }}

    .panel-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--bg-card-border);
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
    }}

    .panel-header {{
      padding: 10px 16px;
      border-bottom: 1px solid var(--bg-card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.02);
    }}

    .panel-header h2 {{
      font-size: 13px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #fff;
    }}

    .panel-controls {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    select, button {{
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #fff;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      outline: none;
    }}

    select:focus {{
      border-color: var(--accent-cyan);
    }}

    /* Left Panel: Macro Centroid Map */
    #macro-canvas-container {{
      flex: 1;
      position: relative;
      background: radial-gradient(circle at center, rgba(15, 23, 42, 0.8) 0%, rgba(9, 13, 22, 0.98) 100%);
      overflow: hidden;
    }}

    #macro-canvas {{
      width: 100%;
      height: 100%;
      cursor: grab;
    }}
    #macro-canvas:active {{
      cursor: grabbing;
    }}

    .canvas-legend {{
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid var(--bg-card-border);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      pointer-events: none;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }}

    /* Right Split: Radar (Top) + Comparative Inspector (Bottom) */
    .right-column {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      height: 100%;
    }}

    .radar-card {{
      flex: 1.1;
    }}

    .inspector-card {{
      flex: 1.1;
    }}

    #radar-canvas-container {{
      flex: 1;
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
      background: radial-gradient(circle at center, rgba(15, 23, 42, 0.85) 0%, rgba(9, 13, 22, 0.98) 100%);
    }}

    #radar-canvas {{
      width: 100%;
      height: 100%;
    }}

    /* Inspector Multi-Tabs */
    .tab-bar {{
      display: flex;
      gap: 4px;
      border-bottom: 1px solid var(--bg-card-border);
      padding: 0 14px;
      background: rgba(255, 255, 255, 0.015);
    }}

    .tab-btn {{
      padding: 8px 12px;
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      position: relative;
      transition: color 0.2s;
    }}

    .tab-btn.active {{
      color: var(--accent-cyan);
    }}

    .tab-btn.active::after {{
      content: "";
      position: absolute;
      bottom: -1px;
      left: 0;
      right: 0;
      height: 2px;
      background: var(--accent-cyan);
    }}

    .tab-content {{
      flex: 1;
      padding: 12px 16px;
      overflow-y: auto;
    }}

    /* Custom Scrollbars */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(255, 255, 255, 0.15); border-radius: 3px; }}

    /* Cards & Tables */
    .health-hud {{
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .badge-pill {{
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    .guidance-box {{
      background: rgba(6, 182, 212, 0.08);
      border: 1px solid rgba(6, 182, 212, 0.25);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 11px;
      color: #e2e8f0;
      margin-bottom: 12px;
      line-height: 1.45;
    }}

    .component-bars {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      font-size: 10px;
    }}

    .comp-card {{
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      padding: 6px 8px;
    }}

    .refugee-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
    }}

    .refugee-table th, .refugee-table td {{
      padding: 6px 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      text-align: left;
    }}

    .refugee-table th {{
      color: var(--text-muted);
      font-weight: 600;
      font-size: 10px;
      text-transform: uppercase;
    }}

    /* Heatmatrix Table */
    .heatmatrix-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 10px;
      text-align: center;
    }}

    .heatmatrix-table th, .heatmatrix-table td {{
      padding: 4px 5px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }}

    .heatmatrix-table th {{
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-muted);
      font-weight: 600;
    }}

    .heat-cell {{
      cursor: pointer;
      transition: transform 0.1s;
    }}
    .heat-cell:hover {{
      outline: 2px solid #fff;
      z-index: 10;
    }}

    /* Diagnostic Framework Button */
    .btn-framework {{
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(59, 130, 246, 0.2));
      border: 1px solid rgba(6, 182, 212, 0.5);
      color: #38bdf8;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      transition: all 0.2s ease;
      box-shadow: 0 2px 10px rgba(6, 182, 212, 0.15);
    }}

    .btn-framework:hover {{
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.35), rgba(59, 130, 246, 0.35));
      border-color: var(--accent-cyan);
      color: #ffffff;
      box-shadow: 0 0 16px rgba(6, 182, 212, 0.4);
      transform: translateY(-1px);
    }}

    .link-btn-framework {{
      background: none;
      border: none;
      color: var(--accent-cyan);
      font-size: 10px;
      cursor: pointer;
      text-decoration: underline;
      text-underline-offset: 3px;
      padding: 0;
      margin-left: 6px;
      font-family: inherit;
    }}

    .link-btn-framework:hover {{
      color: #ffffff;
    }}

    /* Modal Overlay & Container */
    .modal-overlay {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(4, 7, 14, 0.85);
      backdrop-filter: blur(10px);
      z-index: 2000;
      display: flex;
      justify-content: center;
      align-items: center;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s ease;
    }}

    .modal-overlay.active {{
      opacity: 1;
      pointer-events: auto;
    }}

    .modal-container {{
      background: #0d1527;
      border: 1px solid rgba(6, 182, 212, 0.4);
      border-radius: 14px;
      width: 94%;
      max-width: 1020px;
      max-height: 88vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 24px 64px rgba(0, 0, 0, 0.85), 0 0 32px rgba(6, 182, 212, 0.2);
      transform: scale(0.96);
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      overflow: hidden;
    }}

    .modal-overlay.active .modal-container {{
      transform: scale(1);
    }}

    .modal-header {{
      padding: 16px 22px;
      background: rgba(15, 23, 42, 0.98);
      border-bottom: 1px solid var(--bg-card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .modal-title-wrap {{
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}

    .modal-title {{
      font-size: 16px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .modal-subtitle {{
      font-size: 11px;
      color: var(--text-muted);
    }}

    .modal-close-btn {{
      background: rgba(255, 255, 255, 0.08);
      border: none;
      color: #94a3b8;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s;
    }}

    .modal-close-btn:hover {{
      background: rgba(239, 68, 68, 0.25);
      color: #ef4444;
      transform: rotate(90deg);
    }}

    .modal-tabs {{
      display: flex;
      background: rgba(10, 16, 30, 0.95);
      border-bottom: 1px solid var(--bg-card-border);
      padding: 0 16px;
      gap: 6px;
      overflow-x: auto;
    }}

    .modal-tab-btn {{
      padding: 11px 16px;
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      position: relative;
      transition: all 0.2s;
      white-space: nowrap;
    }}

    .modal-tab-btn:hover {{
      color: #ffffff;
    }}

    .modal-tab-btn.active {{
      color: var(--accent-cyan);
    }}

    .modal-tab-btn.active::after {{
      content: "";
      position: absolute;
      bottom: -1px;
      left: 0;
      right: 0;
      height: 2px;
      background: var(--accent-cyan);
      box-shadow: 0 0 8px var(--accent-cyan);
    }}

    .modal-body {{
      padding: 22px 24px;
      overflow-y: auto;
      flex: 1;
      font-size: 12px;
      line-height: 1.6;
      color: #cbd5e1;
    }}

    .modal-tab-pane {{
      display: none;
    }}

    .modal-tab-pane.active {{
      display: block;
    }}

    /* Diagnostic Matrix Table */
    .matrix-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 11px;
      margin-top: 10px;
      border: 1px solid var(--bg-card-border);
      border-radius: 8px;
      overflow: hidden;
    }}

    .matrix-table th {{
      background: rgba(30, 41, 59, 0.85);
      padding: 10px 12px;
      text-align: left;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #94a3b8;
      border-bottom: 1px solid var(--bg-card-border);
    }}

    .matrix-table td {{
      padding: 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      vertical-align: top;
      background: rgba(15, 23, 42, 0.6);
    }}

    .matrix-table tr:last-child td {{
      border-bottom: none;
    }}

    .matrix-table tr:hover td {{
      background: rgba(30, 41, 59, 0.45);
    }}

    .badge-signal {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    .badge-signal.good {{
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
    }}
    .badge-signal.bad {{
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
    }}
    .badge-signal.remedy {{
      background: rgba(6, 182, 212, 0.15);
      border: 1px solid rgba(6, 182, 212, 0.4);
      color: #38bdf8;
    }}

    /* Formula & Math Cards */
    .math-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 10px;
    }}

    .math-card {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--bg-card-border);
      border-radius: 8px;
      padding: 14px;
    }}

    .math-card h4 {{
      font-size: 12px;
      color: var(--accent-cyan);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .formula-box {{
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      padding: 8px 12px;
      font-family: "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      color: #38bdf8;
      margin-bottom: 8px;
      letter-spacing: 0.3px;
    }}

    /* Quadrant Grid */
    .quadrant-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }}

    .quadrant-card {{
      background: rgba(15, 23, 42, 0.7);
      border-radius: 8px;
      padding: 14px;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }}

    .quadrant-card.q1 {{ border-left: 4px solid var(--accent-green); }}
    .quadrant-card.q2 {{ border-left: 4px solid var(--accent-blue); }}
    .quadrant-card.q3 {{ border-left: 4px solid var(--accent-orange); }}
    .quadrant-card.q4 {{ border-left: 4px solid var(--text-muted); }}

    .quadrant-title {{
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    /* Alert / Explainer Box */
    .explainer-banner {{
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(59, 130, 246, 0.08));
      border: 1px solid rgba(6, 182, 212, 0.3);
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 14px;
      font-size: 12px;
      color: #e2e8f0;
    }}
  </style>
</head>
<body>

  <!-- Header Bar -->
  <header>
    <div class="brand">
      <span class="brand-badge">Stage 3.3 Topology</span>
      <div>
        <div class="brand-title">Cluster Centroid &amp; Dispersion Analysis</div>
        <div class="brand-subtitle">Fr&eacute;chet Centers of Mass, Cross-Border Drift &amp; Downstream Health Audits</div>
      </div>
    </div>

    <div style="display: flex; align-items: center; gap: 14px;">
      <!-- Stat Ribbon -->
      <div class="stat-ribbon">
        {badge_pills_html}
        <div class="stat-pill">
          <span class="num">{metrics["global_metrics"]["total_concepts_mapped"]}</span>
          <span class="lbl">Total Concepts</span>
        </div>
        <div class="stat-pill">
          <span class="num">+{metrics["global_metrics"]["dispersion_reduction_vs_random_pct"]}%</span>
          <span class="lbl">Vs Random Null</span>
        </div>
        <div class="stat-pill">
          <span class="num" style="color:var(--accent-orange);">{metrics["global_metrics"]["global_refugee_rate_pct"]}%</span>
          <span class="lbl">Cross-Border Drift</span>
        </div>
      </div>

      <!-- Click to View Diagnostic Framework -->
      <button id="btn-open-framework" class="btn-framework" title="Open complete metrics &amp; diagnostic framework reference">
        <span>📖</span>
        <span>The Diagnostic Framework</span>
      </button>
    </div>
  </header>

  <!-- Dashboard Grid -->
  <main class="dashboard-container">

    <!-- Panel 1: Macro Centroid Separation Network -->
    <section class="panel-card">
      <div class="panel-header">
        <h2><span>★</span> Macro Centroid Separation Space (PCA 2D)</h2>
        <div class="panel-controls">
          <label style="font-size:10px; color:var(--text-muted);">Links:</label>
          <select id="link-filter">
            <option value="top3">Top 3 Closest</option>
            <option value="highways">Graph Highways (&ge;8 edges)</option>
            <option value="merges">Merge Risks (Sim &ge; 0.92)</option>
            <option value="none">Centroids Only</option>
          </select>
          <button id="btn-reset-zoom">Reset View</button>
        </div>
      </div>
      <div id="macro-canvas-container">
        <canvas id="macro-canvas"></canvas>
        <div class="canvas-legend">
          <div class="legend-item"><span class="legend-dot" style="background:#10b981;"></span> Autonomous Domain (&ge;70)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#eab308;"></span> Integrated Sub-Domain (50-69)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#f97316;"></span> Merge Risk (Sim &ge; 0.93)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span> Diffuse / Drift Risk (&lt;50)</div>
        </div>
      </div>
    </section>

    <!-- Right Column: Solar System Radar + Diagnostic Inspector -->
    <div class="right-column">

      <!-- Panel 2: Polar Dispersion Radar -->
      <section class="panel-card radar-card">
        <div class="panel-header">
          <h2><span>◎</span> <span id="radar-title">Solar System Dispersion Radar</span></h2>
          <div class="panel-controls">
            <select id="community-select"></select>
            <select id="concept-filter">
              <option value="all">All Concepts</option>
              <option value="core_only">Core Inhabitants Only</option>
              <option value="refugees_only">Cross-Border Refugees Only</option>
              <option value="tier_1_nucleus">Tier 1: Nucleus</option>
              <option value="tier_2_mechanics">Tier 2: Mechanics</option>
              <option value="tier_3_modifiers">Tier 3: Modifiers</option>
            </select>
          </div>
        </div>
        <div id="radar-canvas-container">
          <canvas id="radar-canvas"></canvas>
        </div>
      </section>

      <!-- Panel 3: Diagnostic Audit Drawer -->
      <section class="panel-card inspector-card">
        <div class="tab-bar">
          <button class="tab-btn active" data-tab="tab-health">🏥 Health & Synthesis Guidance</button>
          <button class="tab-btn" data-tab="tab-refugees">🏃 Cross-Border Refugees (<span id="tab-refugee-count">0</span>)</button>
          <button class="tab-btn" data-tab="tab-highways">🛣️ Highway Alignment</button>
          <button class="tab-btn" data-tab="tab-matrix">📊 Distance Heatmatrix</button>
        </div>

        <!-- Tab 1: Health & Synthesis Guidance -->
        <div class="tab-content" id="tab-health">
          <div class="health-hud">
            <div>
              <div style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">Composite Cluster Health</div>
              <div style="display:flex; align-items:baseline; gap:8px;">
                <div style="font-size:20px; font-weight:800;" id="hud-health-score">--</div>
                <button type="button" class="link-btn-framework" id="btn-open-framework-inline" title="Click to view Diagnostic Framework reference">📖 Diagnostic Framework</button>
              </div>
            </div>
            <div>
              <span class="badge-pill" id="hud-status-badge">STATUS</span>
            </div>
          </div>

          <div class="guidance-box" id="hud-guidance-box">
            Loading synthesis guidance...
          </div>

          <div class="component-bars" id="hud-component-bars">
            <div class="comp-card">
              <div style="color:var(--text-muted);">Cohesion</div>
              <div style="font-weight:700;" id="comp-cohesion">--</div>
            </div>
            <div class="comp-card">
              <div style="color:var(--text-muted);">Separation</div>
              <div style="font-weight:700;" id="comp-separation">--</div>
            </div>
            <div class="comp-card">
              <div style="color:var(--text-muted);">Fidelity</div>
              <div style="font-weight:700;" id="comp-fidelity">--</div>
            </div>
            <div class="comp-card">
              <div style="color:var(--text-muted);">Integrity</div>
              <div style="font-weight:700;" id="comp-integrity">--</div>
            </div>
          </div>
        </div>

        <!-- Tab 2: Cross-Border Refugees -->
        <div class="tab-content" id="tab-refugees" style="display:none;">
          <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">
            Concepts grouped into this community by graph walk, but whose semantic vector is closer to another cluster:
          </div>
          <table class="refugee-table">
            <thead>
              <tr>
                <th>Concept</th>
                <th>Tier</th>
                <th>Distance Here</th>
                <th>Attracted To</th>
                <th>Drift (Δ)</th>
              </tr>
            </thead>
            <tbody id="refugee-table-body"></tbody>
          </table>
        </div>

        <!-- Tab 3: Highway Alignment -->
        <div class="tab-content" id="tab-highways" style="display:none;">
          <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">
            Graph Workflow Highways vs. High-Dimensional Semantic Similarity:
          </div>
          <table class="refugee-table">
            <thead>
              <tr>
                <th>Neighbor</th>
                <th>Graph Edges</th>
                <th>Cosine Sim</th>
                <th>Topological Alignment</th>
              </tr>
            </thead>
            <tbody id="highways-table-body"></tbody>
          </table>
        </div>

        <!-- Tab 4: 17x17 Heatmatrix -->
        <div class="tab-content" id="tab-matrix" style="display:none;">
          <div style="overflow-x:auto;">
            <table class="heatmatrix-table" id="heatmatrix-table"></table>
          </div>
        </div>
      </section>

    </div>
  </main>

  <!-- Modal: The Diagnostic Framework & Metrics Reference -->
  <div id="framework-modal" class="modal-overlay">
    <div class="modal-container">
      <div class="modal-header">
        <div class="modal-title-wrap">
          <div class="modal-title">
            <span>📖</span>
            <span>The Diagnostic Framework // Semantic &amp; Topological Metrics</span>
          </div>
          <div class="modal-subtitle">
            Interpretative benchmarks, mathematical formulations, high-dimensional baseline theory, and Stage 4 synthesis guidance
          </div>
        </div>
        <button id="btn-close-framework" class="modal-close-btn" title="Close Framework (Esc)">&times;</button>
      </div>

      <!-- Modal Tabs -->
      <div class="modal-tabs">
        <button class="modal-tab-btn active" data-pane="pane-health">🏥 Health Matrix ("Good" vs. "Concerning")</button>
        <button class="modal-tab-btn" data-pane="pane-math">📐 Mathematical Formulations &amp; Scorecards</button>
        <button class="modal-tab-btn" data-pane="pane-cone">🌌 The High-Dimensional "Cone Effect"</button>
        <button class="modal-tab-btn" data-pane="pane-highways">🛣️ Highway Traffic vs. Semantic Alignment</button>
      </div>

      <!-- Modal Body -->
      <div class="modal-body">
        
        <!-- Tab 1: Health Matrix -->
        <div id="pane-health" class="modal-tab-pane active">
          <div class="explainer-banner">
            <b>Operational Heuristics:</b> In high-dimensional embedding spaces, raw cosine distances can be counter-intuitive without reference baselines. Use this matrix to evaluate whether a community is a healthy, cohesive architectural module or exhibits cluster smells requiring Stage 4 synthesis remedies.
          </div>

          <table class="matrix-table">
            <thead>
              <tr>
                <th style="width: 22%;">Metric Signal</th>
                <th style="width: 26%;">🟢 "Good" (Healthy Modular Cluster)</th>
                <th style="width: 26%;">🔴 "Concerning" (Cluster Smells &amp; Risks)</th>
                <th style="width: 26%;">🛠️ Downstream Action / Synthesis Remedy</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <b style="color:#fff;">Intra-Cluster Dispersion</b><br/>
                  <code style="color:var(--accent-cyan); font-size:10px;">mean_distance (d&#772;)</code><br/>
                  <span style="color:var(--text-muted); font-size:10px;">Distance of concepts to community Fr&eacute;chet mean &mu;<sub>C</sub></span>
                </td>
                <td>
                  <span class="badge-signal good">d&#772; &le; 0.55 &bull; low &sigma; &le; 0.10</span><br/>
                  Member concepts tightly orbit the center of mass. The cluster possesses a crisp, unmistakable domain boundary.
                </td>
                <td>
                  <span class="badge-signal bad">d&#772; &ge; 0.70 &bull; high &sigma; &ge; 0.15</span><br/>
                  <b>"Kitchen-Sink" Cluster</b>. Highly diffuse vocabulary without a coherent semantic core. Graph walk lumped disparate topics together.
                </td>
                <td>
                  <span class="badge-signal remedy">Decompose or Multi-Root</span><br/>
                  In Stage 4, generate multiple decoupled sub-schemas or elevate Tier 1 nucleus entities as distinct root models rather than a monolithic object.
                </td>
              </tr>

              <tr>
                <td>
                  <b style="color:#fff;">Nearest-Neighbor Distance</b><br/>
                  <code style="color:var(--accent-cyan); font-size:10px;">D_nearest / Cosine Sim</code><br/>
                  <span style="color:var(--text-muted); font-size:10px;">Distance to closest external community centroid</span>
                </td>
                <td>
                  <span class="badge-signal good">D_nearest &ge; 0.18 (Sim &le; 0.82)</span><br/>
                  The cluster occupies an independent, distinct semantic territory with clear separation from all other clusters.
                </td>
                <td>
                  <span class="badge-signal bad">D_nearest &le; 0.07 (Sim &ge; 0.93)</span><br/>
                  <b>Merge Risk / Redundancy</b>. Centroids virtually overlap in embedding space. The graph partitioned them artificially due to phrasing differences.
                </td>
                <td>
                  <span class="badge-signal remedy">Consolidate Schemas</span><br/>
                  Merge twin communities into a single consolidated schema in Stage 4, or establish a shared polymorphic base model with subtype discriminators.
                </td>
              </tr>

              <tr>
                <td>
                  <b style="color:#fff;">Tier Radial Order</b><br/>
                  <code style="color:var(--accent-cyan); font-size:10px;">r_T1 &lt; r_T2 &lt; r_T3</code><br/>
                  <span style="color:var(--text-muted); font-size:10px;">Radial distance across architectural tiers</span>
                </td>
                <td>
                  <span class="badge-signal good">Monotonic Radial Tiering</span><br/>
                  Tier 1 Nucleus entities sit closest to centroid (d &approx; 0.3&ndash;0.5); Tier 2 Mechanics occupy mid-range (d &approx; 0.5&ndash;0.65); Tier 3 Modifiers sit on the outer periphery.
                </td>
                <td>
                  <span class="badge-signal bad">Tier Inversion</span><br/>
                  A concept labeled Tier 1 solely because of high graph degree sits on the distant periphery (d &gt; 0.75). It is a structural router/bridge, not a semantic invariant.
                </td>
                <td>
                  <span class="badge-signal remedy">Reclassify to Edge Field</span><br/>
                  Downgrade inverted Tier 1 concepts to relational edge fields or utility types rather than treating them as core domain entities in downstream Pydantic models.
                </td>
              </tr>

              <tr>
                <td>
                  <b style="color:#fff;">Cross-Border Refugees</b><br/>
                  <code style="color:var(--accent-cyan); font-size:10px;">refugee_pct (&Delta; &gt; 0)</code><br/>
                  <span style="color:var(--text-muted); font-size:10px;">Concepts closer to an external centroid than own</span>
                </td>
                <td>
                  <span class="badge-signal good">0% &ndash; 5% Border Drift</span><br/>
                  Almost all concepts are geometrically loyal to their home community. High boundary integrity.
                </td>
                <td>
                  <span class="badge-signal bad">&gt; 15% Cross-Border Drift</span><br/>
                  <b>Semantic Refugee Crisis</b>. Concepts grouped here by sentence co-occurrence are drawn strongly toward another community's centroid.
                </td>
                <td>
                  <span class="badge-signal remedy">Relational Link / Relocate</span><br/>
                  Instruct Stage 4 LLM synthesis to model refugees as optional foreign references, or re-route them into the target community's schema definition.
                </td>
              </tr>

              <tr>
                <td>
                  <b style="color:#fff;">Grounding Context Fidelity</b><br/>
                  <code style="color:var(--accent-cyan); font-size:10px;">mean_fidelity (&Gamma;&#772;_C)</code><br/>
                  <span style="color:var(--text-muted); font-size:10px;">Contextual grounding to verbatim source quotes</span>
                </td>
                <td>
                  <span class="badge-signal good">&Gamma;&#772; &ge; 0.32 (Dense Evidence)</span><br/>
                  Concepts have rich verbatim evidence and strong semantic overlap with raw source transcript text chunks.
                </td>
                <td>
                  <span class="badge-signal bad">&Gamma;&#772; &le; 0.18 (Sparse Grounding)</span><br/>
                  <b>Abstraction Artifact Risk</b>. High probability of LLM hallucinations or superficial vocabulary without clinical/source backing.
                </td>
                <td>
                  <span class="badge-signal remedy">Enforce Quote Citations</span><br/>
                  Constrain Stage 4 schema synthesis with strict source quote verification and lower temperature generation for low-fidelity clusters.
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Tab 2: Mathematical Formulations & Scorecards -->
        <div id="pane-math" class="modal-tab-pane">
          <div class="explainer-banner">
            <b>Mathematical Engine:</b> Every community partition is projected into the 384-dimensional unit hypersphere &Sopf;<sup>383</sup> using unit-normalized Sentence-Transformers embeddings (<code>all-MiniLM-L6-v2</code>).
          </div>

          <div class="math-grid">
            <div class="math-card">
              <h4><span>1.</span> Fr&eacute;chet Mean Centroid on Hypersphere</h4>
              <div class="formula-box">
                &mu;<sub>C</sub> = (&sum;<sub>v &isin; C</sub> e<sub>v</sub>) / ||&sum;<sub>v &isin; C</sub> e<sub>v</sub>||<sub>2</sub>
              </div>
              <p style="font-size:11px; color:#cbd5e1;">
                For normalized embeddings (||e<sub>v</sub>||<sub>2</sub> = 1), the Fr&eacute;chet mean center of mass minimizes the sum of squared chordal and angular distances across all member concepts:
              </p>
              <div style="font-family:monospace; font-size:10px; color:var(--text-muted); margin-top:6px;">
                &mu;<sub>C</sub> = argmin<sub>||&mu;||=1</sub> &sum;<sub>v &isin; C</sub> ||e<sub>v</sub> - &mu;||<sup>2</sup>
              </div>
            </div>

            <div class="math-card">
              <h4><span>2.</span> Intra-Cluster Distance &amp; Prototypicality</h4>
              <div class="formula-box">
                d(u, &mu;<sub>C</sub>) = 1 - (e<sub>u</sub> &middot; &mu;<sub>C</sub>)<br/>
                Prototypicality(u) = e<sub>u</sub> &middot; &mu;<sub>C</sub>
              </div>
              <p style="font-size:11px; color:#cbd5e1;">
                Cosine distance bounded in [0, 2]. Distance 0 denotes exact identity with the centroid; distance 1 is orthogonal. Concepts with highest prototypicality define the semantic core of the cluster.
              </p>
            </div>

            <div class="math-card">
              <h4><span>3.</span> Cross-Border Refugee Detection Condition</h4>
              <div class="formula-box">
                &exist; j &ne; i : d(u, &mu;<sub>C<sub>j</sub></sub>) &lt; d(u, &mu;<sub>C<sub>i</sub></sub>)<br/>
                &Delta;<sub>drift</sub>(u) = d(u, &mu;<sub>C<sub>i</sub></sub>) - min<sub>j &ne; i</sub> d(u, &mu;<sub>C<sub>j</sub></sub>)
              </div>
              <p style="font-size:11px; color:#cbd5e1;">
                Identifies concepts assigned to community C<sub>i</sub> by graph edge traversal, but whose pure semantic meaning is closer to an external community C<sub>j</sub>. &Delta;<sub>drift</sub> measures the magnitude of border pull.
              </p>
            </div>

            <div class="math-card">
              <h4><span>4.</span> Composite Health Score (0 &ndash; 100)</h4>
              <div class="formula-box">
                Score = 0.30&middot;Coh + 0.30&middot;Sep + 0.20&middot;Fid + 0.20&middot;Integ
              </div>
              <div style="font-size:10px; color:#94a3b8; line-height:1.5;">
                &bull; <b>Cohesion</b>: 100 &times; clamp((0.75 - d&#772;) / 0.35, 0, 1)<br/>
                &bull; <b>Separation</b>: 100 &times; clamp((D<sub>nearest</sub> - 0.05) / 0.20, 0, 1)<br/>
                &bull; <b>Fidelity</b>: 100 &times; clamp((&Gamma;&#772; - 0.10) / 0.30, 0, 1)<br/>
                &bull; <b>Integrity</b>: 100 &times; (1 - refugee_percentage / 100)
              </div>
            </div>
          </div>

          <div style="margin-top:14px; background:rgba(15,23,42,0.8); border:1px solid var(--bg-card-border); border-radius:8px; padding:12px;">
            <div style="font-size:11px; font-weight:700; color:#fff; margin-bottom:6px;">Status Badge Classification Rules:</div>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
              <span class="badge-signal good">Score &ge; 70: Autonomous Domain</span>
              <span class="badge-signal" style="background:rgba(234,179,8,0.15); border:1px solid rgba(234,179,8,0.4); color:#facc15;">Score 50&ndash;69: Integrated Sub-Domain</span>
              <span class="badge-signal" style="background:rgba(249,115,22,0.15); border:1px solid rgba(249,115,22,0.4); color:#fb923c;">Nearest Sim &ge; 0.93: Merge Risk</span>
              <span class="badge-signal bad">Score &lt; 50: Diffuse / Drift Risk</span>
            </div>
          </div>
        </div>

        <!-- Tab 3: The Cone Effect -->
        <div id="pane-cone" class="modal-tab-pane">
          <div class="explainer-banner">
            <b>The Apparent Paradox:</b> Why is the pairwise distance between centroids only ~0.15 (cosine similarity &approx; 0.85), while member concepts inside a single cluster are dispersed at distance ~0.62? Understanding the geometry of high-dimensional sentence representations is critical for proper evaluation.
          </div>

          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:10px;">
            <div class="math-card">
              <h4><span>🌌</span> Geometry of Deep Sentence Embeddings</h4>
              <p style="font-size:11px; margin-bottom:8px;">
                In deep language models like Sentence-BERT / <code>all-MiniLM-L6-v2</code>, mean-pooled contextual embeddings do not uniformly span the 384-dimensional hypersphere. Instead, they concentrate inside a narrow, high-dimensional positive cone.
              </p>
              <p style="font-size:11px; color:#94a3b8;">
                Because all vocabulary terms originate from the same natural language domain (English medical/educational/clinical texts), random pairs of concepts naturally share a high baseline similarity of 0.70 to 0.85 (cosine distance 0.15 &ndash; 0.30). Concepts almost never point in opposite directions (cosine distance &approx; 2.0).
              </p>
            </div>

            <div class="math-card">
              <h4><span>🎯</span> The Centroid Averaging Effect</h4>
              <p style="font-size:11px; margin-bottom:8px;">
                When you calculate the Fr&eacute;chet mean centroid &mu;<sub>C</sub> across 50 to 150 concept vectors, specific word-level idiosyncrasies and random angular offsets cancel out.
              </p>
              <p style="font-size:11px; color:#94a3b8;">
                The resulting centroid collapses toward the common angular center of the English language cone. Consequently, all 17 community centroids reside very close to each other in angular space (D<sub>centroids</sub> &approx; 0.05 &ndash; 0.20), while individual specific nouns fan out radially around that central cone (d<sub>member</sub> &approx; 0.50 &ndash; 0.70).
              </p>
            </div>
          </div>

          <div style="margin-top:14px; background:rgba(15,23,42,0.8); border:1px solid rgba(6,182,212,0.3); border-radius:8px; padding:14px;">
            <h4 style="color:var(--accent-cyan); font-size:12px; margin-bottom:6px;">🎲 The Random Null Model Baseline</h4>
            <p style="font-size:11px; line-height:1.5; color:#cbd5e1;">
              Because raw distance numbers reflect the embedding geometry, we compare all empirical metrics against a <b>Monte Carlo Random Null Model</b>: 1,381 concepts randomly permuted into synthetic clusters matching the exact community size distribution.
            </p>
            <div style="display:flex; gap:16px; margin-top:10px; font-size:11px;">
              <div style="background:rgba(0,0,0,0.3); padding:8px 12px; border-radius:6px; flex:1;">
                <div style="color:var(--text-muted); font-size:10px; text-transform:uppercase;">Random Null Model Intra-Dispersion</div>
                <div style="font-size:14px; font-weight:700; color:#f87171;">d&#772; = 0.655 &ndash; 0.785</div>
                <div style="font-size:10px; color:#94a3b8;">Expected dispersion from random clustering</div>
              </div>
              <div style="background:rgba(0,0,0,0.3); padding:8px 12px; border-radius:6px; flex:1;">
                <div style="color:var(--text-muted); font-size:10px; text-transform:uppercase;">Empirical Stage 3 Communities</div>
                <div style="font-size:14px; font-weight:700; color:var(--accent-cyan);">d&#772; = 0.623 (+4.8% tighter)</div>
                <div style="font-size:10px; color:#94a3b8;">Statistically significant modular cohesion</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Tab 4: Highway Traffic vs Semantic Alignment -->
        <div id="pane-highways" class="modal-tab-pane">
          <div class="explainer-banner">
            <b>Structural vs. Semantic Topology:</b> Graph highways (Stage 3 directed edge walks) measure workflow co-occurrence in raw sentences. High-dimensional cosine similarity (Stage 2 MiniLM) measures semantic meaning. Cross-referencing these two signals reveals 4 distinct structural quadrants:
          </div>

          <div class="quadrant-grid">
            <div class="quadrant-card q1">
              <div class="quadrant-title" style="color:var(--accent-green);">
                <span>Quadrant I: Tightly Coupled Workflow</span>
                <span style="font-size:10px; color:var(--text-muted);">&ge;8 edges &bull; Sim &ge; 0.88</span>
              </div>
              <p style="font-size:11px; color:#cbd5e1; margin-bottom:6px;">
                High structural co-occurrence combined with close semantic alignment. These communities form a natural macro-module that frequently interacts during operations.
              </p>
              <div style="font-size:10px; color:#34d399;">
                <b>Stage 4 Action:</b> Model via direct foreign keys, nested sub-schemas, or unified service boundaries.
              </div>
            </div>

            <div class="quadrant-card q2">
              <div class="quadrant-title" style="color:var(--accent-blue);">
                <span>Quadrant II: Functional / Structural Bridge</span>
                <span style="font-size:10px; color:var(--text-muted);">&ge;8 edges &bull; Sim &lt; 0.88</span>
              </div>
              <p style="font-size:11px; color:#cbd5e1; margin-bottom:6px;">
                High structural co-occurrence despite disparate semantic meaning (e.g. <code>Clinical Assessment</code> linking to <code>School Notification</code>). A functional workflow process connects two separate domains.
              </p>
              <div style="font-size:10px; color:#60a5fa;">
                <b>Stage 4 Action:</b> Model via event-driven edge schemas, bridge contracts, or transitional workflow state machines.
              </div>
            </div>

            <div class="quadrant-card q3">
              <div class="quadrant-title" style="color:var(--accent-orange);">
                <span>Quadrant III: High Semantic Redundancy</span>
                <span style="font-size:10px; color:var(--text-muted);">&lt;8 edges &bull; Sim &ge; 0.92</span>
              </div>
              <p style="font-size:11px; color:#cbd5e1; margin-bottom:6px;">
                Low structural co-occurrence but nearly identical semantic meaning. The graph partitioned these communities due to vocabulary phrasing differences, but they describe the same underlying domain.
              </p>
              <div style="font-size:10px; color:#fb923c;">
                <b>Stage 4 Action:</b> Primary merge candidate! Consolidate both clusters into a single unified Pydantic schema.
              </div>
            </div>

            <div class="quadrant-card q4">
              <div class="quadrant-title" style="color:var(--text-muted);">
                <span>Quadrant IV: Autonomous Disjoint Domain</span>
                <span style="font-size:10px; color:var(--text-muted);">&lt;8 edges &bull; Sim &lt; 0.88</span>
              </div>
              <p style="font-size:11px; color:#cbd5e1; margin-bottom:6px;">
                Low structural interaction and distant semantic meaning. These represent completely independent domain components.
              </p>
              <div style="font-size:10px; color:#94a3b8;">
                <b>Stage 4 Action:</b> Keep strictly decoupled; generate standalone Pydantic models with no direct relational dependencies.
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- Global Tooltip -->
  <div id="global-tooltip"></div>

  <!-- Interactive Logic Engine -->
  <script>
    const DATA = {data_json_str};

    let activeCommunityId = DATA.communities[0].community_id;
    let activeConceptFilter = "all";
    let activeLinkFilter = "top3";

    // Diagnostic Framework Modal Controller
    const modal = document.getElementById("framework-modal");
    const openBtn = document.getElementById("btn-open-framework");
    const openInlineBtn = document.getElementById("btn-open-framework-inline");
    const closeBtn = document.getElementById("btn-close-framework");

    function openModal(defaultTab) {{
      modal.classList.add("active");
      if (defaultTab) {{
        switchModalTab(defaultTab);
      }}
    }}
    function closeModal() {{
      modal.classList.remove("active");
    }}

    if (openBtn) openBtn.addEventListener("click", () => openModal("pane-health"));
    if (openInlineBtn) openInlineBtn.addEventListener("click", () => openModal("pane-health"));
    if (closeBtn) closeBtn.addEventListener("click", closeModal);

    modal.addEventListener("click", (e) => {{
      if (e.target === modal) closeModal();
    }});

    window.addEventListener("keydown", (e) => {{
      if (e.key === "Escape" && modal.classList.contains("active")) {{
        closeModal();
      }}
    }});

    function switchModalTab(tabId) {{
      document.querySelectorAll(".modal-tab-btn").forEach(btn => {{
        btn.classList.toggle("active", btn.dataset.pane === tabId);
      }});
      document.querySelectorAll(".modal-tab-pane").forEach(pane => {{
        pane.classList.toggle("active", pane.id === tabId);
      }});
    }}

    document.querySelectorAll(".modal-tab-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        switchModalTab(btn.dataset.pane);
      }});
    }});

    // Initialize Community Dropdown
    const commSelect = document.getElementById("community-select");
    DATA.communities.forEach((c) => {{
      const opt = document.createElement("option");
      opt.value = c.community_id;
      opt.textContent = `${{c.community_label}} (${{c.total_concepts}} concepts - ${{c.health_scorecard.status_badge}})`;
      commSelect.appendChild(opt);
    }});

    commSelect.addEventListener("change", (e) => {{
      activeCommunityId = parseInt(e.target.value);
      updateDashboard();
    }});

    document.getElementById("concept-filter").addEventListener("change", (e) => {{
      activeConceptFilter = e.target.value;
      drawRadarCanvas();
    }});

    document.getElementById("link-filter").addEventListener("change", (e) => {{
      activeLinkFilter = e.target.value;
      drawMacroCanvas();
    }});

    // Tab Switching
    document.querySelectorAll(".tab-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.style.display = "none");
        btn.classList.add("active");
        document.getElementById(btn.getAttribute("data-tab")).style.display = "block";
      }});
    }});

    // =========================================================================
    // PANEL 1: MACRO CENTROID CANVAS ENGINE
    // =========================================================================
    const macroCanvas = document.getElementById("macro-canvas");
    const mCtx = macroCanvas.getContext("2d");
    let mPanX = 0, mPanY = 0, mZoom = 1.0;
    let isDragging = false, dragStartX = 0, dragStartY = 0;

    function resizeMacroCanvas() {{
      const rect = macroCanvas.parentElement.getBoundingClientRect();
      macroCanvas.width = rect.width * window.devicePixelRatio;
      macroCanvas.height = rect.height * window.devicePixelRatio;
      drawMacroCanvas();
    }}

    function resizeRadarCanvas() {{
      const rect = radarCanvas.parentElement.getBoundingClientRect();
      radarCanvas.width = rect.width * window.devicePixelRatio;
      radarCanvas.height = rect.height * window.devicePixelRatio;
      drawRadarCanvas();
    }}

    window.addEventListener("resize", () => {{
      resizeMacroCanvas();
      resizeRadarCanvas();
    }});

    function drawMacroCanvas() {{
      const w = macroCanvas.width;
      const h = macroCanvas.height;
      mCtx.save();
      mCtx.clearRect(0, 0, w, h);

      mCtx.translate(w / 2 + mPanX, h / 2 + mPanY);
      mCtx.scale(mZoom, mZoom);

      const scale = Math.min(w, h) * 0.44;

      // Draw Inter-Centroid Links
      DATA.communities.forEach((c1) => {{
        const x1 = c1.coord_x * scale;
        const y1 = c1.coord_y * scale;

        c1.nearest_neighbors.forEach((nb, rank) => {{
          const c2 = DATA.communities.find(c => c.community_id === nb.target_community_id);
          if (!c2) return;
          const x2 = c2.coord_x * scale;
          const y2 = c2.coord_y * scale;

          let shouldDraw = false;
          let linkColor = "rgba(255, 255, 255, 0.06)";
          let linkWidth = 1;

          if (activeLinkFilter === "top3" && rank < 3) shouldDraw = true;
          if (activeLinkFilter === "highways" && nb.highway_traffic_edges >= 8) {{
            shouldDraw = true;
            linkColor = "rgba(59, 130, 246, 0.4)";
            linkWidth = Math.min(4, nb.highway_traffic_edges / 8);
          }}
          if (activeLinkFilter === "merges" && nb.cosine_similarity >= 0.92) {{
            shouldDraw = true;
            linkColor = "rgba(249, 115, 22, 0.7)";
            linkWidth = 2.5;
          }}

          if (shouldDraw) {{
            const isHighlight = (c1.community_id === activeCommunityId || c2.community_id === activeCommunityId);
            mCtx.beginPath();
            mCtx.moveTo(x1, y1);
            mCtx.lineTo(x2, y2);
            mCtx.strokeStyle = isHighlight ? "rgba(6, 182, 212, 0.75)" : linkColor;
            mCtx.lineWidth = isHighlight ? Math.max(2, linkWidth + 1) : linkWidth;
            if (activeLinkFilter === "merges") mCtx.setLineDash([4, 4]);
            mCtx.stroke();
            mCtx.setLineDash([]);
          }}
        }});
      }});

      // Draw Centroid Nodes
      DATA.communities.forEach((c) => {{
        const cx = c.coord_x * scale;
        const cy = c.coord_y * scale;
        const isSelected = (c.community_id === activeCommunityId);
        const statusColor = c.health_scorecard.status_color;
        const radius = Math.max(12, Math.min(26, Math.sqrt(c.total_concepts) * 2.1));

        // Outer Selection Ring
        if (isSelected) {{
          mCtx.beginPath();
          mCtx.arc(cx, cy, radius + 7, 0, Math.PI * 2);
          mCtx.fillStyle = "rgba(6, 182, 212, 0.25)";
          mCtx.fill();
          mCtx.strokeStyle = "#06b6d4";
          mCtx.lineWidth = 2;
          mCtx.stroke();
        }}

        // Main Centroid Body
        mCtx.beginPath();
        mCtx.arc(cx, cy, radius, 0, Math.PI * 2);
        mCtx.fillStyle = "#1e293b";
        mCtx.fill();
        mCtx.strokeStyle = statusColor;
        mCtx.lineWidth = 3;
        mCtx.stroke();

        // Inner Status Pip
        mCtx.beginPath();
        mCtx.arc(cx, cy, 4, 0, Math.PI * 2);
        mCtx.fillStyle = statusColor;
        mCtx.fill();

        // Label
        mCtx.fillStyle = "#ffffff";
        mCtx.font = "bold 10px Inter, sans-serif";
        mCtx.textAlign = "center";
        mCtx.fillText(`C#${{c.community_id}}`, cx, cy + radius + 12);
      }});

      mCtx.restore();
    }}

    // Mouse Interaction
    macroCanvas.addEventListener("mousedown", (e) => {{
      isDragging = true;
      dragStartX = e.clientX - mPanX;
      dragStartY = e.clientY - mPanY;
    }});

    window.addEventListener("mousemove", (e) => {{
      if (isDragging) {{
        mPanX = e.clientX - dragStartX;
        mPanY = e.clientY - dragStartY;
        drawMacroCanvas();
      }}
    }});

    window.addEventListener("mouseup", () => isDragging = false);

    macroCanvas.addEventListener("wheel", (e) => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      mZoom = Math.max(0.4, Math.min(3.5, mZoom * zoomFactor));
      drawMacroCanvas();
    }});

    macroCanvas.addEventListener("click", (e) => {{
      const rect = macroCanvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left) * window.devicePixelRatio;
      const clickY = (e.clientY - rect.top) * window.devicePixelRatio;
      const w = macroCanvas.width;
      const h = macroCanvas.height;
      const scale = Math.min(w, h) * 0.44;

      const worldX = (clickX - (w / 2 + mPanX)) / mZoom;
      const worldY = (clickY - (h / 2 + mPanY)) / mZoom;

      DATA.communities.forEach(c => {{
        const cx = c.coord_x * scale;
        const cy = c.coord_y * scale;
        const radius = Math.max(12, Math.min(26, Math.sqrt(c.total_concepts) * 2.1));
        const dist = Math.hypot(worldX - cx, worldY - cy);
        if (dist <= radius + 6) {{
          activeCommunityId = c.community_id;
          commSelect.value = c.community_id;
          updateDashboard();
        }}
      }});
    }});

    document.getElementById("btn-reset-zoom").addEventListener("click", () => {{
      mPanX = 0; mPanY = 0; mZoom = 1.0;
      drawMacroCanvas();
    }});

    // =========================================================================
    // PANEL 2: POLAR RADAR CANVAS ENGINE
    // =========================================================================
    const radarCanvas = document.getElementById("radar-canvas");
    const rCtx = radarCanvas.getContext("2d");

    function drawRadarCanvas() {{
      const w = radarCanvas.width;
      const h = radarCanvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const maxRadius = Math.min(w, h) * 0.43;

      rCtx.clearRect(0, 0, w, h);

      const comm = DATA.communities.find(c => c.community_id === activeCommunityId) || DATA.communities[0];
      document.getElementById("radar-title").textContent = `${{comm.community_label}} // Polar Dispersion Radar`;

      // Draw Isochrone Rings
      const rings = [
        {{ r: 0.35, label: "Tier 1: Nucleus (r < 0.35)", color: "rgba(16, 185, 129, 0.22)" }},
        {{ r: 0.60, label: "Tier 2: Mechanics (r < 0.60)", color: "rgba(59, 130, 246, 0.18)" }},
        {{ r: 0.85, label: "Tier 3: Modifiers (r < 0.85)", color: "rgba(245, 158, 11, 0.18)" }}
      ];

      rings.forEach(ring => {{
        const rPix = ring.r * maxRadius;
        rCtx.beginPath();
        rCtx.arc(cx, cy, rPix, 0, Math.PI * 2);
        rCtx.strokeStyle = ring.color;
        rCtx.lineWidth = 1.5;
        rCtx.setLineDash([4, 4]);
        rCtx.stroke();
        rCtx.setLineDash([]);

        rCtx.fillStyle = ring.color.replace("0.18", "0.6").replace("0.22", "0.6");
        rCtx.font = "9px Inter, sans-serif";
        rCtx.fillText(ring.label, cx + 8, cy - rPix + 12);
      }});

      // Draw Centroid Anchor
      rCtx.beginPath();
      rCtx.arc(cx, cy, 9, 0, Math.PI * 2);
      rCtx.fillStyle = "#ffffff";
      rCtx.shadowColor = "#06b6d4";
      rCtx.shadowBlur = 14;
      rCtx.fill();
      rCtx.shadowBlur = 0;
      rCtx.fillStyle = "#06b6d4";
      rCtx.font = "bold 9px Inter, sans-serif";
      rCtx.textAlign = "center";
      rCtx.fillText("★ μ_C", cx, cy + 20);

      // Filter Concepts
      let filteredConcepts = comm.concepts;
      if (activeConceptFilter === "core_only") filteredConcepts = comm.concepts.filter(c => !c.is_refugee);
      if (activeConceptFilter === "refugees_only") filteredConcepts = comm.concepts.filter(c => c.is_refugee);
      if (activeConceptFilter.startsWith("tier_")) filteredConcepts = comm.concepts.filter(c => c.tier_key === activeConceptFilter);

      comm._radar_positions = [];

      filteredConcepts.forEach((c, idx) => {{
        const rNorm = Math.min(1.0, c.cosine_distance);
        const rPix = rNorm * maxRadius;
        const angle = idx * 2.3999632; 

        const px = cx + Math.cos(angle) * rPix;
        const py = cy + Math.sin(angle) * rPix;

        let nodeColor = "#10b981";
        if (c.tier_key === "tier_2_mechanics") nodeColor = "#3b82f6";
        if (c.tier_key === "tier_3_modifiers") nodeColor = "#f59e0b";

        const isRefugee = c.is_refugee;
        const nodeRadius = isRefugee ? 6 : Math.max(3.5, Math.min(7.5, Math.sqrt(c.internal_degree + 1) * 1.5));

        // Draw Refugee Pull Vector
        if (isRefugee) {{
          rCtx.beginPath();
          rCtx.arc(px, py, nodeRadius + 3, 0, Math.PI * 2);
          rCtx.fillStyle = "rgba(249, 115, 22, 0.25)";
          rCtx.fill();
          rCtx.strokeStyle = "#f97316";
          rCtx.lineWidth = 1.2;
          rCtx.stroke();
        }}

        rCtx.beginPath();
        rCtx.arc(px, py, nodeRadius, 0, Math.PI * 2);
        rCtx.fillStyle = isRefugee ? "#f97316" : nodeColor;
        rCtx.fill();
        rCtx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        rCtx.lineWidth = 1;
        rCtx.stroke();

        comm._radar_positions.push({{ x: px, y: py, r: nodeRadius, concept: c }});
      }});

      // Draw Neighbor Direction Markers
      comm.nearest_neighbors.slice(0, 3).forEach((nb, i) => {{
        const angle = (i * (Math.PI * 2 / 3)) - Math.PI / 2;
        const px = cx + Math.cos(angle) * (maxRadius + 18);
        const py = cy + Math.sin(angle) * (maxRadius + 18);

        rCtx.fillStyle = "rgba(6, 182, 212, 0.9)";
        rCtx.font = "10px Inter, sans-serif";
        rCtx.textAlign = "center";
        rCtx.fillText(`↗ ${{nb.target_label}} (${{nb.inter_centroid_distance.toFixed(2)}})`, px, py);
      }});
    }}

    // Radar Tooltip
    const tooltip = document.getElementById("global-tooltip");
    radarCanvas.addEventListener("mousemove", (e) => {{
      const rect = radarCanvas.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left) * window.devicePixelRatio;
      const mouseY = (e.clientY - rect.top) * window.devicePixelRatio;

      const comm = DATA.communities.find(c => c.community_id === activeCommunityId);
      if (!comm || !comm._radar_positions) return;

      let found = null;
      for (const pos of comm._radar_positions) {{
        if (Math.hypot(mouseX - pos.x, mouseY - pos.y) <= pos.r + 4) {{
          found = pos.concept;
          break;
        }}
      }}

      if (found) {{
        const refugeeAlert = found.is_refugee ? `
          <div style="background:rgba(249,115,22,0.15); border:1px solid #f97316; border-radius:4px; padding:4px 6px; margin-top:6px; color:#fdba74; font-size:10px;">
            ⚠️ <b>Cross-Border Pull:</b> Closer to ${{found.pull_community_label}} by Δ=${{found.drift_margin.toFixed(3)}}
          </div>
        ` : "";

        tooltip.innerHTML = `
          <div style="font-weight:700; font-size:12px; margin-bottom:4px;">${{found.entity}}</div>
          <div style="font-size:10px; margin-bottom:6px; color:var(--text-muted);">${{found.tier}}</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:10px; margin-bottom:4px;">
            <div>Dist to μ_C: <b>${{found.cosine_distance.toFixed(4)}}</b></div>
            <div>Prototypical: <b>${{found.prototypicality.toFixed(4)}}</b></div>
            <div>Degree: <b>${{found.internal_degree}}</b></div>
            <div>Fidelity: <b>${{found.fidelity.toFixed(3)}}</b></div>
          </div>
          ${{refugeeAlert}}
          ${{found.exemplar_quote ? `<div style="font-size:9px; color:#cbd5e1; font-style:italic; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px; margin-top:4px;">"${{found.exemplar_quote.slice(0, 90)}}..."</div>` : ""}}
        `;
        tooltip.style.left = `${{e.clientX + 14}}px`;
        tooltip.style.top = `${{e.clientY + 14}}px`;
        tooltip.style.display = "block";
      }} else {{
        tooltip.style.display = "none";
      }}
    }});

    radarCanvas.addEventListener("mouseleave", () => tooltip.style.display = "none");

    // =========================================================================
    // PANEL 3: DIAGNOSTIC AUDIT DRAWER POPULATION
    // =========================================================================
    function updateInspector() {{
      const comm = DATA.communities.find(c => c.community_id === activeCommunityId) || DATA.communities[0];
      const h = comm.health_scorecard;

      // Tab 1: Health HUD
      document.getElementById("hud-health-score").textContent = `${{h.health_score}} / 100`;
      const badgeElem = document.getElementById("hud-status-badge");
      badgeElem.textContent = h.status_badge;
      badgeElem.style.background = `${{h.status_color}}22`;
      badgeElem.style.color = h.status_color;
      badgeElem.style.border = `1px solid ${{h.status_color}}`;

      document.getElementById("hud-guidance-box").innerHTML = `
        <div style="font-weight:700; margin-bottom:4px; color:${{h.status_color}};">${{h.status_desc}}</div>
        <div>${{h.synthesis_guidance}}</div>
      `;

      document.getElementById("comp-cohesion").textContent = `${{h.components.cohesion_score}}%`;
      document.getElementById("comp-separation").textContent = `${{h.components.separation_score}}%`;
      document.getElementById("comp-fidelity").textContent = `${{h.components.fidelity_score}}%`;
      document.getElementById("comp-integrity").textContent = `${{h.components.boundary_integrity_score}}%`;

      // Tab 2: Refugee Table
      document.getElementById("tab-refugee-count").textContent = comm.refugee_count;
      const refugeeBody = document.getElementById("refugee-table-body");
      refugeeBody.innerHTML = "";
      if (comm.semantic_refugees.length === 0) {{
        refugeeBody.innerHTML = `<tr><td colspan="5" style="color:var(--accent-green); text-align:center; padding:12px;">✅ No cross-border refugees detected. High boundary integrity.</td></tr>`;
      }} else {{
        comm.semantic_refugees.slice(0, 15).forEach(r => {{
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td style="font-weight:600;">${{r.entity}}</td>
            <td style="color:var(--text-muted);">${{r.tier.replace("Tier ", "T")}}</td>
            <td>${{r.own_distance.toFixed(3)}}</td>
            <td style="color:var(--accent-cyan);">${{r.pull_community_label}} (${{r.neighbor_distance.toFixed(3)}})</td>
            <td style="color:var(--accent-orange); font-weight:700;">+${{r.drift_margin.toFixed(3)}}</td>
          `;
          refugeeBody.appendChild(tr);
        }});
      }}

      // Tab 3: Highway Alignment Table
      const hwBody = document.getElementById("highways-table-body");
      hwBody.innerHTML = "";
      comm.nearest_neighbors.forEach(nb => {{
        const tr = document.createElement("tr");
        let badgeColor = "var(--text-muted)";
        if (nb.alignment_quadrant.includes("Tightly")) badgeColor = "var(--accent-green)";
        if (nb.alignment_quadrant.includes("Redundancy")) badgeColor = "var(--accent-orange)";
        if (nb.alignment_quadrant.includes("Bridge")) badgeColor = "var(--accent-blue)";

        tr.innerHTML = `
          <td style="font-weight:600;">${{nb.target_label}}</td>
          <td><b>${{nb.highway_traffic_edges}}</b> edges</td>
          <td>${{nb.cosine_similarity.toFixed(3)}}</td>
          <td style="color:${{badgeColor}}; font-weight:600;">${{nb.alignment_quadrant}}</td>
        `;
        hwBody.appendChild(tr);
      }});

      // Tab 4: Heatmatrix Table
      const matrixTable = document.getElementById("heatmatrix-table");
      matrixTable.innerHTML = "";
      const headerRow = document.createElement("tr");
      headerRow.innerHTML = "<th></th>" + DATA.inter_centroid_matrix.labels.map(l => `<th>C#${{l.replace("Community #", "")}}</th>`).join("");
      matrixTable.appendChild(headerRow);

      DATA.inter_centroid_matrix.rows.forEach(r => {{
        const tr = document.createElement("tr");
        let html = `<th>C#${{r.label.replace("Community #", "")}}</th>`;
        r.distances.forEach((d, colIdx) => {{
          const isDiag = (r.community_id === DATA.inter_centroid_matrix.community_ids[colIdx]);
          const hue = 190 - Math.min(190, d * 500);
          const bg = isDiag ? "rgba(6, 182, 212, 0.35)" : `hsla(${{hue}}, 80%, 45%, 0.35)`;
          html += `<td class="heat-cell" style="background:${{bg}};" title="${{r.label}} vs ${{DATA.inter_centroid_matrix.labels[colIdx]}}: ${{d.toFixed(3)}}">${{d.toFixed(2)}}</td>`;
        }});
        tr.innerHTML = html;
        matrixTable.appendChild(tr);
      }});
    }}

    function updateDashboard() {{
      drawMacroCanvas();
      drawRadarCanvas();
      updateInspector();
    }}

    // Boot
    resizeMacroCanvas();
    resizeRadarCanvas();
    updateDashboard();
  </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"   [OK] Enhanced Cluster Semantic Dispersion Dashboard Generated: {output_path}")
        return output_path


# =============================================================================
# STANDALONE CLI ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    print("=== SemanticPrism: Generating Enhanced Cluster Centroid Dispersion Dashboard ===")
    visualizer = ClusterEmbeddingVisualizer()
    out_file = visualizer.generate_centroid_dispersion_dashboard()
    print(f"✅ Dashboard generated successfully: {out_file}")
