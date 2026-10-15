"""
SemanticPrism Extraction: Empirical Vector Theme Clustering & Spatial Visualizer
==============================================================================
Production module to perform empirical vector clustering of raw extracted themes
(outputs/01_extraction/all_themes.json) into discrete master clusters prior to LLM synthesis.

Two Core Approaches:
  1. PRIMARY: Dynamic Topological Clustering (Adaptive kNN + Leiden Community Detection)
     - Discovers the natural number of master themes dynamically from graph modularity.
     - Adapts neighbor count k logarithmically: k = clip(2 * log2(N), 8, 30).
     - Absorbs micro-clusters (< 1% of N) into nearest dominant centroids.
  2. FIXED: Variance-Minimizing Hierarchical Clustering (Ward Linkage, K from config)
     - Balanced, non-lopsided semantic partition into exactly target_master_theme_count clusters.

Outputs:
  - 'clustering_summary.json': Full, exhaustive list of all theme groupings and metadata.
  - 'theme_cluster_galaxy_3d.html': Interactive 3D WebGL Dashboard comparing both options.
"""

import os
import sys
import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import umap
from cdlib import algorithms
from sentence_transformers import SentenceTransformer

# Load settings from SemanticPrism configuration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from src.config import settings
except ImportError:
    settings = {}


# Color palette for 3D visualizer (16 vibrant celestial neon hues)
CLUSTER_COLORS = [
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
]


class ThemeClusteringEngine:
    """
    Stage 1 Empirical Theme Clustering Engine.
    Executes vector embeddings, topological Leiden community detection (primary dynamic mode),
    and variance-minimizing Ward hierarchical clustering (fixed mode) on all extracted themes.
    Provides cluster centroid rankings, prototypical exemplars for LLM synthesis,
    and 3D spatial visualization coordinates.
    """
    def __init__(
        self,
        themes_path: Path = Path("outputs/01_extraction/all_themes.json"),
        master_themes_path: Path = Path("outputs/01_extraction/master_themes.json"),
        output_dir: Path = Path("outputs/01_extraction"),
        embedding_model_name: Optional[str] = None,
        fixed_k: Optional[int] = None,
        exemplar_count: Optional[int] = None,
        clustering_mode: Optional[str] = None,
        synthesis_strategy: Optional[str] = None,
        map_max_themes: Optional[int] = None,
        map_payload_format: Optional[str] = None,
        reduce_anchor_count: Optional[int] = None
    ):
        self.themes_path = Path(themes_path)
        self.master_themes_path = Path(master_themes_path)
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / "cache"

        # Resolve refinement settings
        ref_cfg = settings.get('refinement', {}) if hasattr(settings, 'get') else {}
        self.embedding_model_name = (
            embedding_model_name 
            or ref_cfg.get('theme_embedding_model') 
            or ref_cfg.get('embedding_model') 
            or "all-MiniLM-L6-v2"
        )

        # Resolve fixed_k from target_master_theme_count in refinement config
        if fixed_k is not None:
            self.fixed_k = fixed_k
        else:
            self.fixed_k = int(ref_cfg.get('target_master_theme_count', 8))

        # Resolve exemplar_count from theme_exemplars_per_cluster in refinement config
        if exemplar_count is not None:
            self.exemplar_count = exemplar_count
        else:
            self.exemplar_count = int(ref_cfg.get('theme_exemplars_per_cluster', 10))

        # Resolve clustering mode
        if clustering_mode is not None:
            self.clustering_mode = clustering_mode
        else:
            self.clustering_mode = str(ref_cfg.get('theme_clustering_mode', 'dynamic'))

        # Resolve confidence thresholds
        self.high_thresh = float(ref_cfg.get('theme_confidence_high_threshold', 0.40))
        self.med_thresh = float(ref_cfg.get('theme_confidence_medium_threshold', 0.25))
        self.low_thresh = float(ref_cfg.get('theme_confidence_low_threshold', 0.15))

        # Resolve Map-Reduce synthesis parameters
        if synthesis_strategy is not None:
            self.synthesis_strategy = synthesis_strategy
        else:
            self.synthesis_strategy = str(ref_cfg.get('master_synthesis_strategy', 'map_reduce'))

        if map_max_themes is not None:
            self.map_max_themes = map_max_themes
        else:
            self.map_max_themes = int(ref_cfg.get('theme_map_max_themes_per_cluster', 150))

        if map_payload_format is not None:
            self.map_payload_format = map_payload_format
        else:
            self.map_payload_format = str(ref_cfg.get('theme_map_payload_format', 'title_desc'))

        if reduce_anchor_count is not None:
            self.reduce_anchor_count = reduce_anchor_count
        else:
            self.reduce_anchor_count = int(ref_cfg.get('reduce_anchor_theme_count', 5))

        self._model = None

        # State cache for multi-phase orchestration
        self.raw_themes: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.partitions: Dict[str, Any] = {}
        self.coords_dict: Optional[Dict[str, np.ndarray]] = None
        self.active_exemplars: List[Dict[str, Any]] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        models_dir = Path("models/embeddings")
        models_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = SentenceTransformer(self.embedding_model_name, cache_folder=str(models_dir), local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(self.embedding_model_name, cache_folder=str(models_dir), local_files_only=False)
        return self._model

    def load_data(self) -> List[Dict[str, Any]]:
        print(f"-> Loading themes from: {self.themes_path}")
        if not self.themes_path.exists():
            raise FileNotFoundError(f"Missing themes file: {self.themes_path}")
        with open(self.themes_path, "r", encoding="utf-8") as f:
            raw_themes = json.load(f)
        print(f"   Loaded {len(raw_themes)} raw themes.")
        return raw_themes

    def get_or_compute_embeddings(self, raw_themes: List[Dict[str, Any]]) -> np.ndarray:
        cache_npy = self.cache_dir / "theme_embeddings.npy"
        cache_meta = self.cache_dir / "theme_embeddings_meta.json"

        # Fallback to alternate research cache if present
        if not cache_npy.exists():
            alt_candidates = [
                Path("outputs/01_extraction/research/cache"),
                Path("outputs/01_extraction/cache")
            ]
            for alt_dir in alt_candidates:
                alt_npy = alt_dir / "theme_embeddings.npy"
                alt_meta = alt_dir / "theme_embeddings_meta.json"
                if alt_npy.exists() and alt_meta.exists():
                    cache_npy = alt_npy
                    cache_meta = alt_meta
                    break

        content_hash = hashlib.md5(
            "".join(f"{t.get('title','')}_{t.get('description','')}" for t in raw_themes).encode("utf-8")
        ).hexdigest()

        if cache_npy.exists() and cache_meta.exists():
            try:
                with open(cache_meta, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                if meta.get("content_hash") == content_hash and meta.get("count") == len(raw_themes):
                    print(f"-> Loading cached theme embeddings from: {cache_npy}")
                    return np.load(cache_npy)
            except Exception as e:
                print(f"   [Warning] Cache read failed ({e}). Recomputing embeddings...")

        print(f"-> Encoding {len(raw_themes)} themes with '{self.embedding_model_name}'...")
        model = self._get_model()

        texts = []
        for t in raw_themes:
            title = str(t.get("title", "")).strip()
            desc = str(t.get("description", "")).strip()
            reason = str(t.get("reasoning", "")).strip()
            payload = f"{title}: {desc} (Reasoning: {reason})" if reason else f"{title}: {desc}"
            texts.append(payload)

        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        np.save(cache_npy, embeddings)
        with open(cache_meta, "w", encoding="utf-8") as mf:
            json.dump({
                "model": self.embedding_model_name,
                "count": len(raw_themes),
                "dim": int(embeddings.shape[1]),
                "content_hash": content_hash
            }, mf, indent=2)
        print(f"   Saved embeddings cache to: {cache_npy}")

        return embeddings

    def compute_partitions(
        self,
        raw_themes: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Computes 2 production-grade partitions:
          1. dynamic_leiden (Primary): Self-calibrating mutual kNN graph + Leiden community detection
          2. fixed_ward (Secondary): Variance-minimizing Ward Hierarchical clustering (Fixed K from config)
        """
        N = len(raw_themes)
        partitions = {}

        # -------------------------------------------------------------
        # Partition 1 (PRIMARY): Adaptive Mutual kNN Graph + Leiden
        # -------------------------------------------------------------
        print("-> Computing Partition 1 (PRIMARY): Dynamic Topological (Adaptive kNN + Leiden)...")
        
        # Self-calibrating neighborhood size k based on corpus volume
        k_neighbors = int(np.clip(round(2.0 * math.log2(max(10, N))), 8, 30))
        print(f"   -> Calibrated adaptive k-NN neighborhood size: k={k_neighbors} neighbors")

        nn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine").fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        # Build sparse semantic manifold graph
        G = nx.Graph()
        for i in range(N):
            G.add_node(i)

        edge_threshold = 0.35  # Cosine similarity lower-bound to enforce strict semantic grounding
        for i in range(N):
            for j_idx, dist in zip(indices[i][1:], distances[i][1:]):
                sim = max(0.0, 1.0 - dist)
                if sim >= edge_threshold:
                    G.add_edge(i, int(j_idx), weight=sim)

        print(f"   -> Constructed Semantic Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        comms_result = algorithms.leiden(G, weights="weight")
        raw_communities = comms_result.communities
        print(f"   -> Leiden detected {len(raw_communities)} raw communities.")

        # Micro-cluster absorption: Any community < 1% of N is merged into its nearest dominant centroid
        min_cluster_size = max(5, int(N * 0.01))
        major_raw_clusters = [list(c) for c in raw_communities if len(c) >= min_cluster_size]
        minor_raw_clusters = [list(c) for c in raw_communities if len(c) < min_cluster_size]

        major_raw_clusters.sort(key=len, reverse=True)
        print(f"   -> Major communities: {len(major_raw_clusters)} (sizes: {[len(c) for c in major_raw_clusters]})")

        # Compute initial centroids for major communities
        major_centroids = []
        for c in major_raw_clusters:
            vecs = embeddings[c]
            c_sum = np.sum(vecs, axis=0)
            c_norm = np.linalg.norm(c_sum)
            major_centroids.append(c_sum / c_norm if c_norm > 0 else c_sum)
        major_centroids = np.array(major_centroids)

        # Absorb minor clusters into nearest major centroid
        final_leiden_dict = {i: list(c) for i, c in enumerate(major_raw_clusters)}
        absorbed_count = 0
        for minor in minor_raw_clusters:
            for node_idx in minor:
                node_vec = embeddings[node_idx]
                best_major_idx = int(np.argmax(np.dot(major_centroids, node_vec)))
                final_leiden_dict[best_major_idx].append(node_idx)
                absorbed_count += 1

        if absorbed_count > 0:
            print(f"   -> Absorbed {absorbed_count} peripheral outlier themes into nearest dominant centroids.")

        leiden_meta = self._build_cluster_metadata(final_leiden_dict, raw_themes, embeddings, label_prefix="Domain")

        # Theme level assignments
        def _get_conf(sim_score: float) -> str:
            if sim_score >= self.high_thresh:
                return "high"
            elif sim_score >= self.med_thresh:
                return "medium"
            elif sim_score >= self.low_thresh:
                return "low"
            return "unassigned"

        leiden_theme_assignments = [{}] * N
        for c_id, c_data in leiden_meta.items():
            for t_item in c_data["all_themes"]:
                idx = t_item["theme_id"]
                leiden_theme_assignments[idx] = {
                    "cluster_id": c_id,
                    "cluster_label": c_data["label"],
                    "similarity": t_item["similarity_to_centroid"],
                    "confidence": _get_conf(t_item["similarity_to_centroid"])
                }

        partitions["dynamic_leiden"] = {
            "name": f"Dynamic Topological Leiden (Discovered {len(final_leiden_dict)} Domains)",
            "description": f"Self-calibrating mutual kNN graph + Leiden modularity. Automatically discovered {len(final_leiden_dict)} natural master domains.",
            "cluster_count": len(final_leiden_dict),
            "assignments": leiden_theme_assignments,
            "clusters": leiden_meta
        }

        # -------------------------------------------------------------
        # Partition 2 (FIXED): Ward's Minimum-Variance Hierarchical
        # -------------------------------------------------------------
        print(f"-> Computing Partition 2 (FIXED): Ward Hierarchical Clustering (Fixed K={self.fixed_k})...")
        ward = AgglomerativeClustering(n_clusters=self.fixed_k, linkage="ward")
        ward_labels = ward.fit_predict(embeddings)

        ward_clusters = defaultdict(list)
        for i, lbl in enumerate(ward_labels):
            ward_clusters[int(lbl)].append(i)

        # Sort clusters by size descending for clean numbering
        sorted_ward_tuples = sorted(ward_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        sorted_ward_dict = {i: nodes for i, (_, nodes) in enumerate(sorted_ward_tuples)}

        ward_meta = self._build_cluster_metadata(sorted_ward_dict, raw_themes, embeddings, label_prefix="Theme Group")
        
        ward_theme_assignments = [{}] * N
        for c_id, c_data in ward_meta.items():
            for t_item in c_data["all_themes"]:
                idx = t_item["theme_id"]
                ward_theme_assignments[idx] = {
                    "cluster_id": c_id,
                    "cluster_label": c_data["label"],
                    "similarity": t_item["similarity_to_centroid"],
                    "confidence": _get_conf(t_item["similarity_to_centroid"])
                }

        partitions["fixed_ward"] = {
            "name": f"Fixed Variance-Minimized (Ward K={self.fixed_k})",
            "description": f"Strict variance-minimizing partition into {self.fixed_k} balanced semantic categories.",
            "cluster_count": len(sorted_ward_dict),
            "assignments": ward_theme_assignments,
            "clusters": ward_meta
        }

        return partitions

    def _build_cluster_metadata(
        self,
        cluster_dict: Dict[int, List[int]],
        raw_themes: List[Dict[str, Any]],
        embeddings: np.ndarray,
        label_prefix: str = "Domain"
    ) -> Dict[int, Dict[str, Any]]:
        cluster_meta = {}

        for c_id, member_indices in sorted(cluster_dict.items()):
            member_vecs = embeddings[member_indices]
            c_sum = np.sum(member_vecs, axis=0)
            c_norm = np.linalg.norm(c_sum)
            centroid = c_sum / c_norm if c_norm > 0 else c_sum

            # Compute similarity of every single member to the centroid
            sims = np.dot(member_vecs, centroid)
            sorted_order = np.argsort(sims)[::-1]

            # Top 3 for label title
            top_3_titles = [raw_themes[member_indices[sorted_order[r]]].get("title", f"Theme_{r}") for r in range(min(3, len(sorted_order)))]
            label = f"{label_prefix} #{c_id + 1}: " + " | ".join(top_3_titles)

            color = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]

            # Build FULL list of all theme groupings for this cluster
            all_cluster_themes = []
            for r in range(len(sorted_order)):
                orig_idx = member_indices[sorted_order[r]]
                t_obj = raw_themes[orig_idx]
                all_cluster_themes.append({
                    "theme_id": orig_idx,
                    "title": t_obj.get("title", ""),
                    "description": t_obj.get("description", ""),
                    "reasoning": t_obj.get("reasoning", ""),
                    "similarity_to_centroid": round(float(sims[sorted_order[r]]), 4),
                    "rank_in_cluster": r + 1
                })

            cluster_meta[c_id] = {
                "id": c_id,
                "label": label,
                "color": color,
                "size": len(member_indices),
                "percentage": round(100.0 * len(member_indices) / len(raw_themes), 1),
                "centroid": centroid,
                "all_themes": all_cluster_themes,
                "top_exemplars": all_cluster_themes[:self.exemplar_count]
            }

        return cluster_meta

    def compute_3d_projections(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        print("-> Computing 3D Dimensionality Reductions...")

        print("   -> Running UMAP-3D (cosine metric, n_neighbors=15, min_dist=0.1)...")
        reducer_umap = umap.UMAP(
            n_components=3,
            metric="cosine",
            n_neighbors=15,
            min_dist=0.1,
            random_state=42
        )
        u_coords = reducer_umap.fit_transform(embeddings)

        print("   -> Running PCA-3D...")
        reducer_pca = PCA(n_components=3, random_state=42)
        p_coords = reducer_pca.fit_transform(embeddings)

        def normalize_coords(c: np.ndarray, span: float = 300.0) -> np.ndarray:
            c_min = c.min(axis=0)
            c_max = c.max(axis=0)
            denom = np.where((c_max - c_min) == 0, 1.0, (c_max - c_min))
            normed = (c - c_min) / denom
            return (normed - 0.5) * span

        return {
            "umap": normalize_coords(u_coords),
            "pca": normalize_coords(p_coords)
        }

    def export_clustering_summary(
        self,
        partitions: Dict[str, Any],
        raw_themes: List[Dict[str, Any]],
        master_domain: Optional[str] = None
    ) -> Path:
        summary_path = self.output_dir / "clustering_summary.json"
        print(f"-> Exporting FULL theme groupings JSON to: {summary_path}")

        export_data = {
            "total_themes": len(raw_themes),
            "embedding_model": self.embedding_model_name,
            "theme_clustering_mode": self.clustering_mode,
            "master_domain": master_domain,
            "partitions": {}
        }

        for p_key, p_val in partitions.items():
            clusters_summary = []
            for c_id, c_data in p_val["clusters"].items():
                clusters_summary.append({
                    "cluster_id": c_id,
                    "label": c_data["label"],
                    "color": c_data["color"],
                    "theme_count": c_data["size"],
                    "percentage": c_data["percentage"],
                    "all_themes": c_data["all_themes"]  # FULL list of all theme groupings (sorted by centroid similarity, rank 1 = closest)
                })
            export_data["partitions"][p_key] = {
                "name": p_val["name"],
                "description": p_val["description"],
                "cluster_count": p_val["cluster_count"],
                "clusters": clusters_summary
            }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        print(f"   Exported {len(raw_themes)} themes with full metadata across {len(partitions)} partitions.")
        return summary_path

    def generate_3d_html(
        self,
        raw_themes: List[Dict[str, Any]],
        partitions: Dict[str, Any],
        coords_dict: Dict[str, np.ndarray],
        master_domain: Optional[str] = None
    ) -> Path:
        html_path = self.output_dir / "theme_cluster_galaxy_3d.html"
        print(f"-> Building Standalone Three.js 3D Dashboard: {html_path}")

        N = len(raw_themes)
        umap_coords = coords_dict["umap"]
        pca_coords = coords_dict["pca"]

        # Build theme records for client with dynamic partition assignment map
        client_themes = []
        for i in range(N):
            t = raw_themes[i]
            p_assignments = {
                pk: partitions[pk]["assignments"][i]
                for pk in partitions
            }
            client_themes.append({
                "id": i,
                "title": t.get("title", f"Theme #{i}"),
                "description": t.get("description", ""),
                "reasoning": t.get("reasoning", ""),
                "ux": round(float(umap_coords[i, 0]), 2),
                "uy": round(float(umap_coords[i, 1]), 2),
                "uz": round(float(umap_coords[i, 2]), 2),
                "px": round(float(pca_coords[i, 0]), 2),
                "py": round(float(pca_coords[i, 1]), 2),
                "pz": round(float(pca_coords[i, 2]), 2),
                "p_assignments": p_assignments
            })

        # Build partition metadata for client
        client_partitions = {}
        for p_key, p_val in partitions.items():
            clusters_meta = {}
            for c_id, c_data in p_val["clusters"].items():
                member_idx = [i for i in range(N) if p_val["assignments"][i]["cluster_id"] == c_id]
                if member_idx:
                    u_center = [round(float(x), 2) for x in np.mean(umap_coords[member_idx], axis=0)]
                    p_center = [round(float(x), 2) for x in np.mean(pca_coords[member_idx], axis=0)]
                else:
                    u_center = [0.0, 0.0, 0.0]
                    p_center = [0.0, 0.0, 0.0]

                clusters_meta[c_id] = {
                    "id": c_id,
                    "label": c_data["label"],
                    "color": c_data["color"],
                    "size": c_data["size"],
                    "percentage": c_data["percentage"],
                    "u_center": u_center,
                    "p_center": p_center
                }
            client_partitions[p_key] = {
                "key": p_key,
                "name": p_val["name"],
                "description": p_val["description"],
                "cluster_count": p_val["cluster_count"],
                "clusters": clusters_meta
            }

        html_template = self._render_html_template(client_themes, client_partitions, master_domain=master_domain)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_template)

        print(f"   Successfully generated 3D visualizer at: {html_path.resolve()}")
        return html_path

    def _render_html_template(
        self,
        themes_data: List[Dict[str, Any]],
        partitions_data: Dict[str, Any],
        master_domain: Optional[str] = None
    ) -> str:
        themes_json_str = json.dumps(themes_data)
        partitions_json_str = json.dumps(partitions_data)
        active_partition_key = self.get_active_partition_key()
        if active_partition_key not in partitions_data:
            active_partition_key = list(partitions_data.keys())[0]

        # Options for dropdown
        dropdown_options_html = "\n".join([
            f'        <option value="{pk}"{" selected" if pk == active_partition_key else ""}>{p["name"]}</option>'
            for pk, p in partitions_data.items()
        ])

        active_part_name = partitions_data.get(active_partition_key, {}).get("name", "Active Partition")
        total_themes_count = f"{len(themes_data):,}"

        domain_badge_html = ""
        if master_domain:
            domain_badge_html = f'''    <div class="badge" style="color: #10b981; border-color: rgba(16, 185, 129, 0.35); background: rgba(16, 185, 129, 0.12);">
      Domain: {master_domain}
    </div>'''

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SemanticPrism - 3D Theme Clustering & Master Theme Explorer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #060814;
      color: #e2e8f0;
      font-family: 'Inter', -apple-system, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      user-select: none;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      z-index: 1;
    }}
    
    /* Top Header Bar */
    .top-header {{
      position: absolute;
      top: 16px;
      left: 20px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 16px;
      background: rgba(10, 15, 30, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 10px 18px;
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .brand-logo {{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: linear-gradient(135deg, #00f0ff, #a855f7);
      box-shadow: 0 0 12px #00f0ff;
    }}
    .brand-title {{
      font-size: 15px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #ffffff;
    }}
    .badge {{
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      padding: 2px 7px;
      border-radius: 6px;
      background: rgba(0, 240, 255, 0.12);
      color: #00f0ff;
      border: 1px solid rgba(0, 240, 255, 0.25);
    }}

    /* Control Bar */
    .controls-bar {{
      position: absolute;
      top: 16px;
      right: 20px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .ctrl-pill {{
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(10, 15, 30, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 6px 12px;
      border-radius: 10px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .ctrl-label {{
      font-size: 12px;
      font-weight: 500;
      color: #94a3b8;
    }}
    select.dropdown {{
      background: #0f172a;
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
      font-family: inherit;
      outline: none;
      cursor: pointer;
      transition: all 0.2s;
    }}
    select.dropdown:hover {{
      border-color: #00f0ff;
    }}
    .btn-toggle {{
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: #cbd5e1;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-toggle.active {{
      background: rgba(0, 240, 255, 0.2);
      color: #00f0ff;
      border-color: #00f0ff;
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
    }}

    /* Search Container */
    .search-box {{
      position: relative;
    }}
    .search-input {{
      background: #0f172a;
      color: #f8fafc;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 13px;
      width: 240px;
      outline: none;
      transition: all 0.2s;
    }}
    .search-input:focus {{
      width: 320px;
      border-color: #00f0ff;
      box-shadow: 0 0 14px rgba(0, 240, 255, 0.25);
    }}
    .search-results {{
      position: absolute;
      top: 42px;
      left: 0;
      width: 100%;
      max-height: 260px;
      overflow-y: auto;
      background: rgba(10, 15, 30, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      display: none;
      z-index: 20;
    }}
    .search-item {{
      padding: 8px 12px;
      font-size: 12px;
      cursor: pointer;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      transition: background 0.15s;
    }}
    .search-item:hover {{
      background: rgba(0, 240, 255, 0.15);
      color: #00f0ff;
    }}

    /* Left Sidebar: Cluster Legend */
    .sidebar-legend {{
      position: absolute;
      top: 76px;
      left: 20px;
      z-index: 10;
      width: 340px;
      max-height: calc(100vh - 120px);
      overflow-y: auto;
      background: rgba(10, 15, 30, 0.75);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }}
    .sidebar-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #94a3b8;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .cluster-item {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 8px 10px;
      border-radius: 8px;
      cursor: pointer;
      margin-bottom: 6px;
      transition: all 0.15s;
      border: 1px solid transparent;
    }}
    .cluster-item:hover {{
      background: rgba(255, 255, 255, 0.04);
      border-color: rgba(255, 255, 255, 0.1);
    }}
    .cluster-item.muted {{
      opacity: 0.30;
    }}
    .cluster-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-top: 4px;
      flex-shrink: 0;
    }}
    .cluster-text {{
      flex: 1;
      min-width: 0;
    }}
    .cluster-name {{
      font-size: 12px;
      font-weight: 600;
      color: #f1f5f9;
      line-height: 1.3;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .cluster-sub {{
      font-size: 11px;
      color: #64748b;
      margin-top: 2px;
      font-family: 'JetBrains Mono', monospace;
    }}

    /* Deep Inspector HUD Card */
    .inspector-card {{
      position: absolute;
      bottom: 24px;
      right: 20px;
      z-index: 10;
      width: 410px;
      background: rgba(10, 15, 30, 0.85);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(0, 240, 255, 0.3);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7);
      display: none;
      animation: fadeIn 0.2s ease-out;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .card-tag {{
      display: inline-block;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #00f0ff;
      margin-bottom: 6px;
    }}
    .card-title {{
      font-size: 16px;
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 10px;
      line-height: 1.3;
    }}
    .card-desc {{
      font-size: 12px;
      line-height: 1.5;
      color: #cbd5e1;
      margin-bottom: 12px;
    }}
    .card-reason {{
      font-size: 11px;
      line-height: 1.4;
      color: #94a3b8;
      background: rgba(255, 255, 255, 0.03);
      padding: 8px 10px;
      border-radius: 6px;
      border-left: 2px solid #00f0ff;
      margin-bottom: 14px;
    }}
    .card-comparisons {{
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      padding-top: 10px;
    }}
    .comp-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 11px;
      margin-bottom: 6px;
    }}
    .comp-label {{
      color: #64748b;
      flex-shrink: 0;
    }}
    .comp-val {{
      font-family: 'JetBrains Mono', monospace;
      color: #f1f5f9;
      font-weight: 500;
      text-align: right;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    /* Bottom Info Status Bar */
    .status-bar {{
      position: absolute;
      bottom: 16px;
      left: 20px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 16px;
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      color: #64748b;
      background: rgba(10, 15, 30, 0.6);
      backdrop-filter: blur(8px);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .status-item strong {{
      color: #00f0ff;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container"></div>

  <!-- Top Header -->
  <div class="top-header">
    <div class="brand">
      <div class="brand-logo"></div>
      <div class="brand-title">SemanticPrism Theme Galaxy</div>
    </div>
    <div class="badge">Research Explorer</div>
    <div class="badge" style="color: #a855f7; border-color: rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.1);">
      {total_themes_count} Themes
    </div>
{domain_badge_html}
  </div>

  <!-- Control Bar -->
  <div class="controls-bar">
    <!-- Search Bar -->
    <div class="search-box">
      <input type="text" id="searchInput" class="search-input" placeholder="Search {total_themes_count} themes..." autocomplete="off">
      <div id="searchResults" class="search-results"></div>
    </div>

    <!-- Cluster Partition Dropdown -->
    <div class="ctrl-pill">
      <span class="ctrl-label">Partition:</span>
      <select id="partitionSelect" class="dropdown">
{dropdown_options_html}
      </select>
    </div>

    <!-- 3D Projection Switch -->
    <div class="ctrl-pill">
      <span class="ctrl-label">Engine:</span>
      <button id="btnUMAP" class="btn-toggle active">UMAP-3D</button>
      <button id="btnPCA" class="btn-toggle">PCA-3D</button>
    </div>

    <!-- Hull Toggle -->
    <div class="ctrl-pill">
      <button id="btnHulls" class="btn-toggle active">Density Hulls</button>
    </div>
  </div>

  <!-- Left Sidebar Legend -->
  <div class="sidebar-legend">
    <div class="sidebar-title">
      <span id="legendTitle">Clusters</span>
      <span id="resetMute" style="cursor: pointer; color: #00f0ff; font-weight: 500;">Reset</span>
    </div>
    <div id="legendList"></div>
  </div>

  <!-- Deep Inspector Card -->
  <div id="inspectorCard" class="inspector-card">
    <div id="cardTag" class="card-tag">CLUSTER ASSIGNMENT</div>
    <div id="cardTitle" class="card-title">Theme Title</div>
    <div id="cardDesc" class="card-desc">Description</div>
    <div id="cardReason" class="card-reason">Reasoning</div>
    <div class="card-comparisons" id="cardComparisons"></div>
  </div>

  <!-- Bottom Status Bar -->
  <div class="status-bar">
    <div class="status-item">Themes: <strong>{total_themes_count}</strong></div>
    <div class="status-item">Projection: <strong id="statusProj">UMAP-3D</strong></div>
    <div class="status-item">Partition: <strong id="statusPart">{active_part_name}</strong></div>
    <div class="status-item">Controls: <strong>Drag to rotate | Scroll to zoom | Right-click to pan</strong></div>
  </div>

  <script>
    // Data Ingestion from Python
    const THEMES = {themes_json_str};
    const PARTITIONS = {partitions_json_str};

    let activePartitionKey = "{active_partition_key}";
    let activeEngine = "umap";
    let showHulls = true;
    let mutedClusters = new Set();

    // Three.js Scene Setup
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060814, 0.0018);

    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 3000);
    camera.position.set(0, 80, 420);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 1200;
    controls.minDistance = 20;

    // Ambient & Point Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x00f0ff, 1.2, 800);
    pointLight.position.set(0, 150, 200);
    scene.add(pointLight);

    // Starfield Background
    (function createStarfield() {{
      const starGeo = new THREE.BufferGeometry();
      const starCount = 1200;
      const starPos = new Float32Array(starCount * 3);
      for (let i = 0; i < starCount * 3; i++) {{
        starPos[i] = (Math.random() - 0.5) * 1600;
      }}
      starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
      const starMat = new THREE.PointsMaterial({{ color: 0x334155, size: 2, transparent: true, opacity: 0.6 }});
      scene.add(new THREE.Points(starGeo, starMat));
    }})();

    // Point Cloud & Centroid Objects
    let pointCloudMesh;
    let centroidGroup = new THREE.Group();
    let hullGroup = new THREE.Group();
    scene.add(centroidGroup);
    scene.add(hullGroup);

    // Particle Texture Generator
    function createCircleTexture() {{
      const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
      gradient.addColorStop(0, 'rgba(255,255,255,1)');
      gradient.addColorStop(0.3, 'rgba(255,255,255,0.8)');
      gradient.addColorStop(0.8, 'rgba(255,255,255,0.15)');
      gradient.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 64, 64);
      return new THREE.CanvasTexture(canvas);
    }}
    const particleTexture = createCircleTexture();

    // Interpolation State
    let targetPositions = new Float32Array(THEMES.length * 3);
    let currentPositions = new Float32Array(THEMES.length * 3);
    let colorArray = new Float32Array(THEMES.length * 3);

    // Initialize Positions
    THEMES.forEach((t, i) => {{
      const x = activeEngine === 'umap' ? t.ux : t.px;
      const y = activeEngine === 'umap' ? t.uy : t.py;
      const z = activeEngine === 'umap' ? t.uz : t.pz;
      currentPositions[i * 3] = x;
      currentPositions[i * 3 + 1] = y;
      currentPositions[i * 3 + 2] = z;
      targetPositions[i * 3] = x;
      targetPositions[i * 3 + 1] = y;
      targetPositions[i * 3 + 2] = z;
    }});

    // Build Point Cloud Geometry
    const pointGeo = new THREE.BufferGeometry();
    pointGeo.setAttribute('position', new THREE.BufferAttribute(currentPositions, 3));
    pointGeo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

    const pointMat = new THREE.PointsMaterial({{
      size: 5.5,
      map: particleTexture,
      transparent: true,
      opacity: 0.9,
      vertexColors: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    }});
    pointCloudMesh = new THREE.Points(pointGeo, pointMat);
    scene.add(pointCloudMesh);

    // Update Colors based on Active Partition
    function updateColors() {{
      const part = PARTITIONS[activePartitionKey];
      const clusters = part.clusters;

      for (let i = 0; i < THEMES.length; i++) {{
        const assign = THEMES[i].p_assignments[activePartitionKey];
        const cId = assign.cluster_id;
        const isMuted = mutedClusters.has(cId);

        let hex = (clusters[cId] && clusters[cId].color) ? clusters[cId].color : '#94a3b8';

        const color = new THREE.Color(hex);
        if (isMuted) {{
          color.multiplyScalar(0.08); // dim out
        }}

        colorArray[i * 3] = color.r;
        colorArray[i * 3 + 1] = color.g;
        colorArray[i * 3 + 2] = color.b;
      }}
      pointGeo.attributes.color.needsUpdate = true;
    }}

    // Update Centroids & Hulls
    function updateCentroidsAndHulls() {{
      while (centroidGroup.children.length > 0) {{
        centroidGroup.remove(centroidGroup.children[0]);
      }}
      while (hullGroup.children.length > 0) {{
        hullGroup.remove(hullGroup.children[0]);
      }}

      const part = PARTITIONS[activePartitionKey];
      const isUMAP = (activeEngine === 'umap');

      Object.values(part.clusters).forEach(c => {{
        const center = isUMAP ? c.u_center : c.p_center;
        const col = new THREE.Color(c.color);

        // Centroid Pulsar Sphere
        const sphereGeo = new THREE.SphereGeometry(6, 16, 16);
        const sphereMat = new THREE.MeshBasicMaterial({{
          color: col,
          transparent: true,
          opacity: 0.85
        }});
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.set(center[0], center[1], center[2]);
        centroidGroup.add(sphere);

        // Outer Glow Halo
        const haloGeo = new THREE.SphereGeometry(12, 16, 16);
        const haloMat = new THREE.MeshBasicMaterial({{
          color: col,
          transparent: true,
          opacity: 0.25,
          wireframe: true
        }});
        const halo = new THREE.Mesh(haloGeo, haloMat);
        halo.position.set(center[0], center[1], center[2]);
        centroidGroup.add(halo);

        // Density Hull Bubble
        if (showHulls) {{
          const memberPositions = [];
          for (let i = 0; i < THEMES.length; i++) {{
            const assign = THEMES[i].p_assignments[activePartitionKey];
            if (assign.cluster_id === c.id) {{
              const px = isUMAP ? THEMES[i].ux : THEMES[i].px;
              const py = isUMAP ? THEMES[i].uy : THEMES[i].py;
              const pz = isUMAP ? THEMES[i].uz : THEMES[i].pz;
              const d = Math.sqrt((px-center[0])**2 + (py-center[1])**2 + (pz-center[2])**2);
              memberPositions.push(d);
            }}
          }}
          memberPositions.sort((a,b) => a-b);
          const r80 = memberPositions.length > 0 ? memberPositions[Math.floor(memberPositions.length * 0.75)] : 30;
          const hullRadius = Math.max(25, Math.min(130, r80 * 1.1));

          const hullGeo = new THREE.SphereGeometry(hullRadius, 24, 24);
          const hullMat = new THREE.MeshBasicMaterial({{
            color: col,
            transparent: true,
            opacity: 0.05,
            depthWrite: false,
            wireframe: false
          }});
          const hullMesh = new THREE.Mesh(hullGeo, hullMat);
          hullMesh.position.set(center[0], center[1], center[2]);
          hullGroup.add(hullMesh);

          const wireMat = new THREE.MeshBasicMaterial({{
            color: col,
            transparent: true,
            opacity: 0.12,
            wireframe: true
          }});
          const wireMesh = new THREE.Mesh(hullGeo, wireMat);
          wireMesh.position.set(center[0], center[1], center[2]);
          hullGroup.add(wireMesh);
        }}
      }});
    }}

    // Render Left Sidebar Legend
    function renderLegend() {{
      const part = PARTITIONS[activePartitionKey];
      const listEl = document.getElementById('legendList');
      listEl.innerHTML = '';

      Object.values(part.clusters).forEach(c => {{
        const item = document.createElement('div');
        item.className = `cluster-item ${{mutedClusters.has(c.id) ? 'muted' : ''}}`;
        item.innerHTML = `
          <div class="cluster-swatch" style="background: ${{c.color}}; box-shadow: 0 0 8px ${{c.color}}88;"></div>
          <div class="cluster-text">
            <div class="cluster-name">${{c.label}}</div>
            <div class="cluster-sub">${{c.size}} themes (${{c.percentage}}%)</div>
          </div>
        `;
        item.addEventListener('click', () => {{
          if (mutedClusters.has(c.id)) {{
            mutedClusters.delete(c.id);
          }} else {{
            mutedClusters.add(c.id);
          }}
          updateColors();
          renderLegend();
        }});
        listEl.appendChild(item);
      }});
    }}

    // Switch Projection Targets
    function switchProjection(engine) {{
      activeEngine = engine;
      document.getElementById('statusProj').innerText = engine.toUpperCase() + '-3D';
      document.getElementById('btnUMAP').classList.toggle('active', engine === 'umap');
      document.getElementById('btnPCA').classList.toggle('active', engine === 'pca');

      for (let i = 0; i < THEMES.length; i++) {{
        targetPositions[i * 3] = engine === 'umap' ? THEMES[i].ux : THEMES[i].px;
        targetPositions[i * 3 + 1] = engine === 'umap' ? THEMES[i].uy : THEMES[i].py;
        targetPositions[i * 3 + 2] = engine === 'umap' ? THEMES[i].uz : THEMES[i].pz;
      }}
      updateCentroidsAndHulls();
    }}

    // Switch Partition
    function switchPartition(partKey) {{
      activePartitionKey = partKey;
      mutedClusters.clear();
      document.getElementById('statusPart').innerText = PARTITIONS[partKey].name;
      updateColors();
      updateCentroidsAndHulls();
      renderLegend();
    }}

    // Event Listeners for UI
    document.getElementById('partitionSelect').addEventListener('change', (e) => {{
      switchPartition(e.target.value);
    }});
    document.getElementById('btnUMAP').addEventListener('click', () => switchProjection('umap'));
    document.getElementById('btnPCA').addEventListener('click', () => switchProjection('pca'));
    document.getElementById('btnHulls').addEventListener('click', (e) => {{
      showHulls = !showHulls;
      e.target.classList.toggle('active', showHulls);
      updateCentroidsAndHulls();
    }});
    document.getElementById('resetMute').addEventListener('click', () => {{
      mutedClusters.clear();
      updateColors();
      renderLegend();
    }});

    // Raycaster for Hover & Selection
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 4.5;
    const mouse = new THREE.Vector2(-999, -999);
    let hoveredIndex = -1;

    window.addEventListener('mousemove', (e) => {{
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    }});

    function showInspector(idx) {{
      const t = THEMES[idx];
      const card = document.getElementById('inspectorCard');
      const assign = t.p_assignments[activePartitionKey];

      document.getElementById('cardTag').innerText = `THEME #${{t.id}} • ${{assign.cluster_label.toUpperCase()}}`;
      document.getElementById('cardTitle').innerText = t.title;
      document.getElementById('cardDesc').innerText = t.description || 'No description extracted.';
      document.getElementById('cardReason').innerText = t.reasoning ? `Reasoning: ${{t.reasoning}}` : 'No reasoning recorded.';

      const compContainer = document.getElementById('cardComparisons');
      compContainer.innerHTML = Object.keys(PARTITIONS).map(pk => {{
        const p = PARTITIONS[pk];
        const a = t.p_assignments[pk];
        return `
          <div class="comp-row">
            <span class="comp-label">${{p.name.split('(')[0].trim()}}:</span>
            <span class="comp-val">${{a.cluster_label}} (${{a.similarity}})</span>
          </div>
        `;
      }}).join('');

      card.style.display = 'block';
    }}

    window.addEventListener('click', (e) => {{
      if (hoveredIndex !== -1) {{
        showInspector(hoveredIndex);
      }}
    }});

    // Search Autocomplete
    const searchInput = document.getElementById('searchInput');
    const searchResults = document.getElementById('searchResults');

    searchInput.addEventListener('input', (e) => {{
      const q = e.target.value.toLowerCase().trim();
      if (!q || q.length < 2) {{
        searchResults.style.display = 'none';
        return;
      }}
      const matches = THEMES.filter(t => 
        t.title.toLowerCase().includes(q) || 
        t.description.toLowerCase().includes(q)
      ).slice(0, 10);

      if (matches.length > 0) {{
        searchResults.innerHTML = matches.map(m => `
          <div class="search-item" data-id="${{m.id}}">
            <strong>${{m.title}}</strong>
            <div style="font-size: 10px; color: #64748b; margin-top: 2px;">${{m.description.substring(0, 60)}}...</div>
          </div>
        `).join('');
        searchResults.style.display = 'block';

        searchResults.querySelectorAll('.search-item').forEach(el => {{
          el.addEventListener('click', () => {{
            const id = parseInt(el.getAttribute('data-id'));
            focusTheme(id);
            searchResults.style.display = 'none';
          }});
        }});
      }} else {{
        searchResults.style.display = 'none';
      }}
    }});

    document.addEventListener('click', (e) => {{
      if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {{
        searchResults.style.display = 'none';
      }}
    }});

    function focusTheme(idx) {{
      const x = currentPositions[idx * 3];
      const y = currentPositions[idx * 3 + 1];
      const z = currentPositions[idx * 3 + 2];

      showInspector(idx);

      const startCam = camera.position.clone();
      const targetCam = new THREE.Vector3(x, y + 25, z + 70);
      const targetLook = new THREE.Vector3(x, y, z);

      let alpha = 0;
      function anim() {{
        alpha += 0.04;
        camera.position.lerpVectors(startCam, targetCam, alpha);
        controls.target.lerp(targetLook, alpha);
        if (alpha < 1) requestAnimationFrame(anim);
      }}
      anim();
    }}

    // Animation Loop (60 FPS)
    let clock = new THREE.Clock();

    function animate() {{
      requestAnimationFrame(animate);

      let posNeedsUpdate = false;
      const positions = pointGeo.attributes.position.array;
      for (let i = 0; i < THEMES.length * 3; i++) {{
        const diff = targetPositions[i] - currentPositions[i];
        if (Math.abs(diff) > 0.05) {{
          currentPositions[i] += diff * 0.08;
          positions[i] = currentPositions[i];
          posNeedsUpdate = true;
        }}
      }}
      if (posNeedsUpdate) {{
        pointGeo.attributes.position.needsUpdate = true;
      }}

      const time = clock.getElapsedTime();
      centroidGroup.children.forEach((child, i) => {{
        if (child.material.wireframe) {{
          child.rotation.y = time * 0.4 + i;
          child.rotation.x = time * 0.2;
        }}
      }});

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObject(pointCloudMesh);
      if (intersects.length > 0) {{
        const idx = intersects[0].index;
        if (idx !== hoveredIndex) {{
          hoveredIndex = idx;
          document.body.style.cursor = 'pointer';
        }}
      }} else {{
        if (hoveredIndex !== -1) {{
          hoveredIndex = -1;
          document.body.style.cursor = 'default';
        }}
      }}

      controls.update();
      renderer.render(scene, camera);
    }}

    // Resize Handler
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    // Initial Setup
    updateColors();
    updateCentroidsAndHulls();
    renderLegend();
    animate();
  </script>
</body>
</html>
"""

    def get_active_partition_key(self) -> str:
        """Returns the partition key corresponding to the configured theme_clustering_mode."""
        if self.clustering_mode == "fixed":
            return "fixed_ward"
        return "dynamic_leiden"

    def get_active_partition(self, partitions: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the partition data dictionary for the active clustering mode."""
        key = self.get_active_partition_key()
        return partitions.get(key, partitions.get("dynamic_leiden", list(partitions.values())[0]))

    def get_active_exemplars_payload(self, partitions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts each cluster's top prototypical exemplar themes (ranked by distance to centroid)
        formatted cleanly for the downstream LLM master theme synthesis prompt.
        """
        active_part = self.get_active_partition(partitions)
        clusters = active_part.get("clusters", {})
        if isinstance(clusters, dict):
            cluster_items = [(c_id, c_data) for c_id, c_data in sorted(clusters.items())]
        else:
            cluster_items = [(c_data.get("cluster_id", idx), c_data) for idx, c_data in enumerate(clusters)]

        exemplar_groups = []
        for c_id, c_data in cluster_items:
            all_th = c_data.get("all_themes", [])
            top_themes = all_th[:self.exemplar_count]
            t_count = c_data.get("theme_count", c_data.get("size", len(all_th)))
            exemplar_groups.append({
                "cluster_id": c_id,
                "cluster_label": c_data.get("label", f"Cluster #{c_id + 1}"),
                "theme_count": t_count,
                "percentage": c_data.get("percentage", 0.0),
                "exemplars": [
                    {
                        "rank": t.get("rank_in_cluster", r_idx + 1),
                        "title": t.get("title", ""),
                        "description": t.get("description", ""),
                        "reasoning": t.get("reasoning", ""),
                        "similarity_to_centroid": t.get("similarity_to_centroid", 0.0)
                    }
                    for r_idx, t in enumerate(top_themes)
                ]
            })
        return exemplar_groups

    def get_map_payload(
        self,
        max_themes_per_cluster: Optional[int] = None,
        payload_format: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts member themes for each cluster in the active partition formatted for
        the Stage 1 Map phase, sorted in descending order of geometric proximity to centroid.
        Respects max_themes_per_cluster cap and payload_format field length options.
        """
        active_part = self.get_active_partition(self.partitions)
        clusters = active_part.get("clusters", {})
        if isinstance(clusters, dict):
            cluster_items = [(c_id, c_data) for c_id, c_data in sorted(clusters.items())]
        else:
            cluster_items = [(c_data.get("cluster_id", idx), c_data) for idx, c_data in enumerate(clusters)]

        max_cap = max_themes_per_cluster or self.map_max_themes
        p_format = payload_format or self.map_payload_format

        map_payload = []
        for c_id, c_data in cluster_items:
            all_th = c_data.get("all_themes", [])
            selected_th = all_th[:max_cap]

            formatted_lines = []
            for r_idx, t in enumerate(selected_th):
                rank = t.get("rank_in_cluster", r_idx + 1)
                sim = t.get("similarity_to_centroid", 0.0)
                title = str(t.get("title", "")).strip()
                desc = str(t.get("description", "")).strip()
                reason = str(t.get("reasoning", "")).strip()

                if p_format == "title":
                    line = f"  {rank}. {title}"
                elif p_format == "title_desc_reasoning":
                    line = f"  {rank}. {title}: {desc}"
                    if reason:
                        line += f" (Reasoning: {reason})"
                else:  # default "title_desc"
                    line = f"  {rank}. {title}: {desc}"

                formatted_lines.append(line)

            map_payload.append({
                "cluster_id": c_id,
                "cluster_label": c_data.get("label", f"Cluster #{c_id + 1}"),
                "theme_count": len(all_th),
                "displayed_count": len(selected_th),
                "percentage": c_data.get("percentage", 0.0),
                "theme_lines": "\n".join(formatted_lines)
            })

        return map_payload

    def get_reduce_anchors(
        self,
        top_n: Optional[int] = None
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extracts the Top N closest themes to each cluster centroid to serve as
        empirical ground-truth context in the Stage 2 Reduce prompt.
        """
        active_part = self.get_active_partition(self.partitions)
        clusters = active_part.get("clusters", {})
        if isinstance(clusters, dict):
            cluster_items = [(c_id, c_data) for c_id, c_data in sorted(clusters.items())]
        else:
            cluster_items = [(c_data.get("cluster_id", idx), c_data) for idx, c_data in enumerate(clusters)]

        n_anchors = top_n or self.reduce_anchor_count
        anchors = {}

        for c_id, c_data in cluster_items:
            all_th = c_data.get("all_themes", [])
            top_themes = all_th[:n_anchors]
            anchors[c_id] = [
                {
                    "rank": t.get("rank_in_cluster", r_idx + 1),
                    "title": t.get("title", ""),
                    "description": t.get("description", "")
                }
                for r_idx, t in enumerate(top_themes)
            ]

        return anchors

    def cluster_themes(self) -> Dict[str, Any]:
        """
        Executes vector embeddings, graph community detection / hierarchical clustering,
        3D spatial projections, and extracts centroid exemplars for LLM synthesis.
        Does NOT build the visualizer yet, allowing the LLM to synthesize master theme names first.
        """
        print("=" * 65)
        print("🚀 SemanticPrism: Dynamic Theme Clustering Engine")
        print("=" * 65)
        self.raw_themes = self.load_data()
        self.embeddings = self.get_or_compute_embeddings(self.raw_themes)
        self.partitions = self.compute_partitions(self.raw_themes, self.embeddings)
        self.coords_dict = self.compute_3d_projections(self.embeddings)
        self.active_exemplars = self.get_active_exemplars_payload(self.partitions)

        return {
            "partitions": self.partitions,
            "active_exemplars": self.active_exemplars,
            "coords_dict": self.coords_dict,
            "raw_themes": self.raw_themes
        }

    def build_visualizations(
        self,
        master_themes: Optional[List[str]] = None,
        master_domain: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Builds the standalone 3D WebGL visualizer and clustering summary.
        If master_themes are provided (or exist in master_themes.json), applies them
        to the active partition selected in the config so the visualizer displays the
        authoritative LLM-synthesized ontology.
        """
        if not self.partitions or self.coords_dict is None or not self.raw_themes:
            self.cluster_themes()

        # Check if master themes can be auto-loaded from disk if not explicitly passed
        if master_themes is None and self.master_themes_path.exists():
            try:
                with open(self.master_themes_path, "r", encoding="utf-8") as f:
                    mt_data = json.load(f)
                master_themes = mt_data.get("master_themes", [])
                if not master_domain:
                    master_domain = mt_data.get("master_domain")
                print(f"-> Auto-loaded {len(master_themes)} master themes from: {self.master_themes_path}")
            except Exception as e:
                print(f"   [Warning] Could not load existing master themes from {self.master_themes_path}: {e}")

        active_key = self.get_active_partition_key()
        active_part = self.partitions.get(active_key)

        if active_part and master_themes:
            print(f"-> Applying {len(master_themes)} LLM Master Theme names to active partition '{active_key}'...")
            domain_label = f" — Domain: {master_domain}" if master_domain else ""
            if active_key == "dynamic_leiden":
                active_part["name"] = f"Dynamic Topological Leiden (Discovered {len(active_part['clusters'])} Master Themes{domain_label})"
            else:
                active_part["name"] = f"Fixed Variance-Minimized (Ward K={self.fixed_k}{domain_label})"

            if master_domain:
                active_part["master_domain"] = master_domain

            import re
            cluster_id_to_master = {}
            for idx, (c_id, c_data) in enumerate(active_part["clusters"].items()):
                if idx < len(master_themes):
                    raw_val = master_themes[idx]
                    cleaned = re.sub(r"^(?:Cluster\s*\d+[\s:.-]*|\d+[\s:.-]+)", "", str(raw_val)).strip()
                    master_name = cleaned if cleaned else str(raw_val).strip()

                    new_label = f"Master Theme #{idx + 1}: {master_name}"
                    c_data["label"] = new_label
                    c_data["master_theme_name"] = master_name
                    cluster_id_to_master[c_id] = new_label
                else:
                    cluster_id_to_master[c_id] = c_data.get("label", f"Cluster #{c_id + 1}")

            # Update assignment records for all themes in active partition
            for assign in active_part["assignments"]:
                cid = assign.get("cluster_id")
                if cid in cluster_id_to_master:
                    assign["cluster_label"] = cluster_id_to_master[cid]

        # Annotate inactive partition for clear UI differentiation
        for p_key, p_val in self.partitions.items():
            if p_key != active_key and "[Empirical Baseline]" not in p_val["name"]:
                p_val["name"] = f"{p_val['name']} [Empirical Baseline]"

        summary_path = self.export_clustering_summary(self.partitions, self.raw_themes, master_domain=master_domain)
        html_path = self.generate_3d_html(self.raw_themes, self.partitions, self.coords_dict, master_domain=master_domain)

        print("\n" + "=" * 65)
        print("🎉 THEME CLUSTERING & 3D VISUALIZATION ARTIFACTS BUILT!")
        print("=" * 65)
        print(f"1. Interactive 3D Visualizer: file://{html_path.resolve()}")
        print(f"2. Full Groupings JSON:       file://{summary_path.resolve()}")
        print("=" * 65)

        return {
            "summary_path": summary_path,
            "html_path": html_path
        }

    def run(
        self,
        master_themes: Optional[List[str]] = None,
        master_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        cluster_res = self.cluster_themes()
        vis_res = self.build_visualizations(master_themes=master_themes, master_domain=master_domain)
        return {
            **cluster_res,
            **vis_res
        }


# Backwards compatibility alias
ThemeClusteringResearchEngine = ThemeClusteringEngine


if __name__ == "__main__":
    engine = ThemeClusteringEngine()
    engine.run()
