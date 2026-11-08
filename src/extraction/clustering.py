"""
SemanticPrism Stage 1: Empirical Vector Theme Clustering Engine
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

from prismrunner import (
    clear_device_memory,
    get_optimal_device,
    resolve_dynamic_batch_size,
)

_EXEMPLARS_PER_CLUSTER: int = 10
_REDUCE_ANCHOR_THEME_COUNT: int = 5
_FALLBACK_DOMAIN: str = "General Domain"


def _derive_fallback_domain(titles: Optional[List[str]] = None) -> str:
    """Derive a descriptive corpus domain label from top cluster/theme titles when LLM domain synthesis is unavailable."""
    if not titles:
        return _FALLBACK_DOMAIN
    cleaned: List[str] = []
    for raw in titles:
        c = re.sub(r"^(?:Cluster\s*#?\d+[\s:.-]*|Master\s*Theme\s*#?\d+[\s:.-]*|\d+[\s:.-]+)", "", str(raw or "")).strip()
        if c and c not in cleaned:
            cleaned.append(c)
        if len(cleaned) >= 3:
            break
    if not cleaned:
        return _FALLBACK_DOMAIN
    return f"Synthesized Domain: {', '.join(cleaned)}"


def format_theme_embedding_text(theme: Any, mode: str = "adaptive") -> str:
    """
    Format a Theme model or dict into text for vector embedding or Map prompts.
    In 'adaptive' mode (default), uses 'title: description' when description is substantive (>= 5 words)
    to avoid generic LLM reasoning meta-boilerplate inflating cross-theme cosine similarity, and
    only appends reasoning when description is missing or terse (< 5 words).
    """
    if hasattr(theme, "title"):
        title = str(theme.title or "").strip()
        desc = str(getattr(theme, "description", "") or "").strip()
        reason = str(getattr(theme, "reasoning", "") or "").strip()
    elif isinstance(theme, dict):
        title = str(theme.get("title", "") or "").strip()
        desc = str(theme.get("description", "") or "").strip()
        reason = str(theme.get("reasoning", "") or "").strip()
    else:
        return str(theme).strip()

    if mode == "title":
        return title
    if mode == "title_desc":
        return f"{title}: {desc}" if desc else title
    if mode == "adaptive":
        if len(desc.split()) >= 5:
            return f"{title}: {desc}"
        if desc and reason:
            return f"{title}: {desc} ({reason})"
        if reason:
            return f"{title}: {reason}"
        return f"{title}: {desc}" if desc else title
    if desc and reason:
        return f"{title}: {desc} (Reasoning: {reason})"
    if desc:
        return f"{title}: {desc}"
    return title


class ThemeClusteringEngine:
    """
    Empirical vector theme clustering engine matching SemanticPrism's ThemeClusteringEngine.
    Uses SentenceTransformer embeddings (with deterministic TF-IDF cosine fallback if offline),
    computes dynamic kNN/Leiden modularity partitions, ranks themes by centroid cosine similarity,
    and exports 'clustering_summary.json'.
    """

    def __init__(
        self,
        themes_path: Path,
        output_dir: Path,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        exemplar_count: int = _EXEMPLARS_PER_CLUSTER,
        map_max_themes: int = 150,
        reduce_anchor_count: int = _REDUCE_ANCHOR_THEME_COUNT,
        manifold_edge_threshold: float = 0.35,
        micro_cluster_ratio: float = 0.01,
        device: str = "auto",
        batch_size: Union[int, str] = "auto",
    ):
        self.themes_path = Path(themes_path)
        self.output_dir = Path(output_dir)
        self.master_themes_path = self.output_dir / "master_themes.json"
        self.embedding_model_name = embedding_model_name
        self.exemplar_count = exemplar_count
        self.map_max_themes = map_max_themes
        self.reduce_anchor_count = reduce_anchor_count
        self.manifold_edge_threshold = manifold_edge_threshold
        self.micro_cluster_ratio = micro_cluster_ratio
        self.device = device
        self.batch_size = batch_size

        self.raw_themes: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.partitions: Dict[str, Any] = {}
        self.active_exemplars: List[Dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute L2-normalized embeddings via SentenceTransformer or TF-IDF fallback, freeing VRAM immediately."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            models_dir = REPO_ROOT / "models" / "embeddings"
            models_dir.mkdir(parents=True, exist_ok=True)
            model = None
            emb = None
            try:
                try:
                    model = SentenceTransformer(self.embedding_model_name, local_files_only=True)
                except Exception:
                    try:
                        model = SentenceTransformer(self.embedding_model_name, cache_folder=str(models_dir), local_files_only=True)
                    except Exception:
                        model = SentenceTransformer(self.embedding_model_name, cache_folder=str(models_dir), local_files_only=False)
                emb_batch_size = resolve_dynamic_batch_size(
                    device=get_optimal_device(self.device),
                    requested_batch_size=self.batch_size,
                    items=[{"prompt": t} for t in texts],
                )
                with torch.inference_mode():
                    emb = model.encode(
                        texts,
                        normalize_embeddings=True,
                        batch_size=emb_batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                arr = np.asarray(emb, dtype=np.float32)
                return arr
            finally:
                if model is not None:
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                del model, emb
                clear_device_memory()
        except Exception:
            vocab: Dict[str, int] = {}
            tokenized: List[List[str]] = []
            for t in texts:
                raw_words = [w for w in re.findall(r"[a-z0-9]{2,}", t.lower())]
                feats = list(raw_words) + [f"#{w[:5]}" for w in raw_words if len(w) >= 5]
                tokenized.append(feats)
                for w in set(feats):
                    vocab[w] = vocab.get(w, 0) + 1
            terms = sorted(vocab.keys())
            if not terms:
                return np.eye(max(1, len(texts)), dtype=np.float32)
            term_idx = {w: i for i, w in enumerate(terms)}
            n_docs = len(texts)
            mat = np.zeros((n_docs, len(terms)), dtype=np.float32)
            for d_i, words in enumerate(tokenized):
                for w in words:
                    if w in term_idx:
                        mat[d_i, term_idx[w]] += 1.0
            idf = np.log((1.0 + n_docs) / (1.0 + np.array([vocab[w] for w in terms], dtype=np.float32))) + 1.0
            mat = mat * idf
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

    @staticmethod
    def _select_mmr_indices(
        cluster_embs: np.ndarray,
        sims_to_centroid: np.ndarray,
        top_n: int,
        lambda_param: float = 0.50,
        manifold_edge_threshold: float = 0.35,
    ) -> List[int]:
        """
        Select up to top_n local indices from a cluster using Maximal Marginal Relevance (MMR):
          MMR(i) = lambda * sim(i, centroid) - (1 - lambda) * max_{j in S} cos(e_i, e_j)
        """
        n_c = len(sims_to_centroid)
        if n_c == 0 or top_n <= 0:
            return []
        ranked = [int(i) for i in np.argsort(sims_to_centroid)[::-1]]
        if n_c <= top_n:
            return ranked

        mu_c = float(np.mean(sims_to_centroid))
        sigma_c = float(np.std(sims_to_centroid)) if n_c > 1 else 0.0
        core_floor = min(mu_c - 2.0 * max(sigma_c, 0.05), float(manifold_edge_threshold))
        core_pool = [i for i in ranked if float(sims_to_centroid[i]) >= core_floor]
        if len(core_pool) < top_n:
            core_pool = ranked

        selected: List[int] = [core_pool[0]]
        remaining: List[int] = [i for i in core_pool[1:]]
        pair_sims = np.clip(cluster_embs @ cluster_embs.T, -1.0, 1.0)

        while len(selected) < min(top_n, len(core_pool)) and remaining:
            best_idx = remaining[0]
            best_score = -float("inf")
            for cand_i in remaining:
                rel = float(sims_to_centroid[cand_i])
                max_red = max(float(pair_sims[cand_i, sel_j]) for sel_j in selected)
                score = (lambda_param * rel) - ((1.0 - lambda_param) * max_red)
                if score > best_score:
                    best_score = score
                    best_idx = cand_i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    @staticmethod
    def _spherical_centroid_fallback(embeddings: np.ndarray, k: int) -> np.ndarray:
        """
        Deterministic farthest-first spherical centroid clustering fallback when networkx is unavailable
        or the similarity graph is degenerate.
        """
        n = len(embeddings)
        k = max(1, min(k, n))
        if k == 1:
            return np.zeros(n, dtype=int)

        mean_vec = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        mean_vec = mean_vec / norm if norm > 0 else mean_vec
        first_seed = int(np.argmax(embeddings @ mean_vec))
        centers = [embeddings[first_seed]]
        min_sims = embeddings @ centers[0]

        for _ in range(1, k):
            next_seed = int(np.argmin(min_sims))
            centers.append(embeddings[next_seed])
            min_sims = np.maximum(min_sims, embeddings @ centers[-1])

        cent_mat = np.vstack(centers)
        labels = np.argmax(embeddings @ cent_mat.T, axis=1).astype(int)

        for _ in range(3):
            new_centers = []
            for cid in range(k):
                mask = labels == cid
                if np.any(mask):
                    c_vec = embeddings[mask].mean(axis=0)
                    c_norm = np.linalg.norm(c_vec)
                    new_centers.append(c_vec / c_norm if c_norm > 0 else c_vec)
                else:
                    new_centers.append(cent_mat[cid])
            cent_mat = np.vstack(new_centers)
            labels = np.argmax(embeddings @ cent_mat.T, axis=1).astype(int)

        remap = {old_id: new_id for new_id, old_id in enumerate(sorted(set(int(x) for x in labels)))}
        return np.array([remap[int(x)] for x in labels], dtype=int)

    def _build_partition_dict(
        self,
        name: str,
        labels: np.ndarray,
        raw_themes: List[Dict[str, Any]],
        embeddings: np.ndarray,
        effective_edge_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        if effective_edge_threshold is None:
            effective_edge_threshold = self.manifold_edge_threshold
        unique_ids = sorted(set(int(x) for x in labels))
        clusters: Dict[int, Dict[str, Any]] = {}
        assignments: List[Dict[str, Any]] = []
        n_total = max(1, len(raw_themes))

        for new_cid, orig_cid in enumerate(unique_ids):
            idxs = [i for i, lbl in enumerate(labels) if int(lbl) == orig_cid]
            n_c = len(idxs)
            cluster_embs = embeddings[idxs]
            sum_vec = cluster_embs.sum(axis=0)
            r_sum = float(np.linalg.norm(sum_vec))
            centroid = (sum_vec / r_sum) if r_sum > 0 else cluster_embs.mean(axis=0)

            sims = cluster_embs @ centroid
            if n_c > 1 and r_sum > 0:
                loo_denom_sq = np.maximum(1e-12, (r_sum ** 2) - (2.0 * r_sum * sims) + 1.0)
                loo_sims = ((r_sum * sims) - 1.0) / np.sqrt(loo_denom_sq)
                loo_sims = np.clip(loo_sims, -1.0, 1.0)
            else:
                loo_sims = sims.copy()

            mean_sim = float(np.mean(sims)) if n_c > 0 else 0.0
            std_sim = float(np.std(sims)) if n_c > 1 else 0.0

            ranked_order = np.argsort(sims)[::-1]
            local_to_item: Dict[int, Dict[str, Any]] = {}

            all_cluster_themes = []
            for local_pos in ranked_order:
                l_idx = int(local_pos)
                r_idx = idxs[l_idx]
                sim_val = float(sims[l_idx])
                loo_val = float(loo_sims[l_idx])
                t_obj = raw_themes[r_idx]
                item_dict = {
                    "theme_index": int(r_idx),
                    "title": t_obj.get("title", ""),
                    "description": t_obj.get("description", ""),
                    "reasoning": t_obj.get("reasoning", ""),
                    "similarity_to_centroid": round(sim_val, 4),
                    "loo_similarity_to_centroid": round(loo_val, 4),
                }
                all_cluster_themes.append(item_dict)
                local_to_item[l_idx] = item_dict

            mmr_exemplar_locals = self._select_mmr_indices(
                cluster_embs, sims, self.exemplar_count, lambda_param=0.50, manifold_edge_threshold=self.manifold_edge_threshold
            )
            top_exemplars = [local_to_item[i] for i in mmr_exemplar_locals if i in local_to_item]
            mmr_map_locals = self._select_mmr_indices(
                cluster_embs, sims, min(n_c, max(self.map_max_themes, self.exemplar_count)), lambda_param=0.50, manifold_edge_threshold=self.manifold_edge_threshold
            )
            mmr_themes = [local_to_item[i] for i in mmr_map_locals if i in local_to_item]

            lead_title = top_exemplars[0]["title"] if top_exemplars else (all_cluster_themes[0]["title"] if all_cluster_themes else f"Cluster #{new_cid + 1}")
            clusters[new_cid] = {
                "cluster_id": new_cid,
                "label": f"Cluster #{new_cid + 1}: {lead_title}",
                "count": n_c,
                "percentage": round(n_c * 100.0 / n_total, 2),
                "mean_similarity": round(mean_sim, 4),
                "std_similarity": round(std_sim, 4),
                "exemplars": top_exemplars,
                "mmr_themes": mmr_themes,
                "all_themes": all_cluster_themes,
            }

            for item in all_cluster_themes:
                assignments.append({
                    "theme_index": item["theme_index"],
                    "title": item["title"],
                    "cluster_id": new_cid,
                    "cluster_label": clusters[new_cid]["label"],
                    "similarity_to_centroid": item["similarity_to_centroid"],
                })

        return {
            "name": name,
            "cluster_count": len(clusters),
            "effective_edge_threshold": round(float(effective_edge_threshold), 4),
            "clusters": clusters,
            "assignments": sorted(assignments, key=lambda x: x["theme_index"]),
        }

    def cluster_themes(self) -> Dict[str, Any]:
        if not self.themes_path.exists():
            raise FileNotFoundError(f"Missing themes file: {self.themes_path}")
        with open(self.themes_path, "r", encoding="utf-8") as f:
            self.raw_themes = json.load(f)

        n = len(self.raw_themes)
        if n == 0:
            self.partitions = {}
            self.active_exemplars = []
            return {"partitions": {}, "active_exemplars": [], "raw_themes": []}

        texts = [format_theme_embedding_text(t, mode="adaptive") for t in self.raw_themes]
        self.embeddings = self._compute_embeddings(texts)

        fallback_k = max(1, min(8, int(round(math.sqrt(n / 2.0))) or 1))
        dyn_labels = np.zeros(n, dtype=int)
        tau_eff = float(self.manifold_edge_threshold)
        if n > 2:
            try:
                import networkx as nx
                sim_matrix = np.clip(self.embeddings @ self.embeddings.T, -1.0, 1.0)
                off_diag = sim_matrix[~np.eye(n, dtype=bool)]
                mu_bg = float(np.mean(off_diag)) if off_diag.size > 0 else 0.0
                sigma_bg = float(np.std(off_diag)) if off_diag.size > 1 else 0.0

                k_neighbors = int(np.clip(round(2 * math.log2(max(2, n))), 2, min(30, n - 1)))
                top1_sims: List[float] = []
                top_nbr_rows: List[np.ndarray] = []
                for i in range(n):
                    row = sim_matrix[i]
                    nbrs = np.argsort(row)[::-1]
                    nbrs = nbrs[nbrs != i][:k_neighbors]
                    top_nbr_rows.append(nbrs)
                    if nbrs.size > 0:
                        top1_sims.append(float(row[nbrs[0]]))

                top1_arr = np.array(top1_sims, dtype=np.float32) if top1_sims else np.array([self.manifold_edge_threshold], dtype=np.float32)
                p20_top1 = float(np.percentile(top1_arr, 20))
                tau_eff = max(mu_bg + 0.5 * sigma_bg, min(float(self.manifold_edge_threshold), p20_top1))

                graph = nx.Graph()
                for i in range(n):
                    graph.add_node(i)
                    row = sim_matrix[i]
                    nbrs = top_nbr_rows[i]
                    for rank_idx, j in enumerate(nbrs):
                        w = float(row[j])
                        if w >= tau_eff or (rank_idx == 0 and w > mu_bg):
                            excess_w = max(1e-4, w - tau_eff + 0.05)
                            if graph.has_edge(i, int(j)):
                                if excess_w > graph[i][int(j)].get("weight", 0.0):
                                    graph[i][int(j)]["weight"] = excess_w
                            else:
                                graph.add_edge(i, int(j), weight=excess_w)

                if graph.number_of_edges() > 0:
                    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
                    for c_idx, comm in enumerate(communities):
                        for node_idx in comm:
                            dyn_labels[int(node_idx)] = c_idx
                else:
                    dyn_labels = self._spherical_centroid_fallback(self.embeddings, fallback_k)
                    communities = [set(np.where(dyn_labels == cid)[0]) for cid in set(int(x) for x in dyn_labels)]

                if len(communities) > 2:
                    min_cluster_size = max(2, int(round(n * self.micro_cluster_ratio)))
                    unique_c, counts_c = np.unique(dyn_labels, return_counts=True)
                    dom_ids = [int(cid) for cid, cnt in zip(unique_c, counts_c) if cnt >= min_cluster_size]
                    if not dom_ids:
                        top_order = np.argsort(counts_c)[::-1][: max(2, min(8, len(unique_c)))]
                        dom_ids = [int(unique_c[idx]) for idx in top_order]

                    if 0 < len(dom_ids) < len(unique_c):
                        dom_centroids = []
                        for cid in dom_ids:
                            c_embs = self.embeddings[dyn_labels == cid]
                            cent = c_embs.mean(axis=0)
                            norm = np.linalg.norm(cent)
                            dom_centroids.append(cent / norm if norm > 0 else cent)
                        dom_mat = np.vstack(dom_centroids)

                        dom_set = set(dom_ids)
                        for idx in range(n):
                            if int(dyn_labels[idx]) not in dom_set:
                                sims = dom_mat @ self.embeddings[idx]
                                best_dom = dom_ids[int(np.argmax(sims))]
                                dyn_labels[idx] = best_dom

                        remap = {old_id: new_id for new_id, old_id in enumerate(sorted(set(int(x) for x in dyn_labels)))}
                        dyn_labels = np.array([remap[int(x)] for x in dyn_labels], dtype=int)
            except Exception:
                dyn_labels = self._spherical_centroid_fallback(self.embeddings, fallback_k)

        self.partitions = {
            "dynamic_leiden": self._build_partition_dict(
                "Dynamic Topological Leiden",
                dyn_labels,
                self.raw_themes,
                self.embeddings,
                effective_edge_threshold=tau_eff,
            ),
        }

        active_part = self.get_active_partition(self.partitions)
        self.active_exemplars = [
            {
                "cluster_id": c_id,
                "theme_count": c_data["count"],
                "percentage": c_data["percentage"],
                "exemplars": c_data["exemplars"],
            }
            for c_id, c_data in sorted(active_part["clusters"].items())
        ]
        self.embeddings = None

        return {
            "partitions": self.partitions,
            "active_exemplars": self.active_exemplars,
            "raw_themes": self.raw_themes,
        }

    def get_active_partition_key(self) -> str:
        return "dynamic_leiden"

    def get_active_partition(self, partitions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        parts = partitions or self.partitions
        key = self.get_active_partition_key()
        return parts.get(key) or next(iter(parts.values()), {"clusters": {}, "assignments": []})

    def get_map_payload(
        self,
        max_themes_per_cluster: Optional[int] = None,
        payload_format: str = "title_desc",
        max_words_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if max_themes_per_cluster is None:
            max_themes_per_cluster = self.map_max_themes
        active_part = self.get_active_partition()
        payloads = []
        for c_id, c_data in sorted(active_part.get("clusters", {}).items()):
            all_t = c_data.get("all_themes", [])
            source_themes = c_data.get("mmr_themes") or all_t
            candidates = source_themes[:max_themes_per_cluster]
            lines = []
            running_words = 0
            for idx, t in enumerate(candidates, start=1):
                formatted = format_theme_embedding_text(t, mode=payload_format)
                line = f"{idx}. {formatted} (sim={t.get('similarity_to_centroid', 0.0):.3f})"
                line_words = len(line.split())
                if max_words_budget and lines and (running_words + line_words > max_words_budget):
                    break
                lines.append(line)
                running_words += line_words
            payloads.append({
                "cluster_id": c_id,
                "theme_count": len(all_t),
                "displayed_count": len(lines),
                "theme_lines": "\n".join(lines),
            })
        return payloads

    def get_reduce_anchors(self, top_n: int = _REDUCE_ANCHOR_THEME_COUNT) -> Dict[int, List[Dict[str, Any]]]:
        active_part = self.get_active_partition()
        anchors: Dict[int, List[Dict[str, Any]]] = {}
        for c_id, c_data in sorted(active_part.get("clusters", {}).items()):
            source_themes = c_data.get("exemplars") or c_data.get("mmr_themes") or c_data.get("all_themes", [])
            anchors[c_id] = source_themes[:top_n]
        return anchors

    def export_clustering_summary(
        self,
        master_themes: Optional[List[str]] = None,
        master_domain: Optional[str] = None,
    ) -> Path:
        if not self.partitions:
            self.cluster_themes()

        active_key = self.get_active_partition_key()
        active_part = self.partitions.get(active_key, {})
        fallback_titles = master_themes or [
            c_data.get("label", "") for _, c_data in sorted(active_part.get("clusters", {}).items())
        ]
        resolved_domain = master_domain or _derive_fallback_domain(fallback_titles)

        if active_part and master_themes:
            active_part["master_domain"] = resolved_domain
            for idx, (c_id, c_data) in enumerate(sorted(active_part.get("clusters", {}).items())):
                if idx < len(master_themes):
                    clean_name = re.sub(r"^(?:Cluster\s*\d+[\s:.-]*|\d+[\s:.-]+)", "", str(master_themes[idx])).strip()
                    c_data["master_theme_name"] = clean_name
                    c_data["label"] = f"Master Theme #{idx + 1}: {clean_name}"

        summary_path = self.output_dir / "clustering_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "master_domain": resolved_domain,
                    "active_partition": active_key,
                    "total_themes": len(self.raw_themes),
                    "partitions": self.partitions,
                },
                f,
                indent=2,
            )
        return summary_path
