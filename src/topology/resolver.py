import json
import math
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

class TargetResolver:
    """
    Target qualification and triplet filtering criteria used by Stage 4 for LLM synthesis.
    Provides methods to resolve qualified spoke targets, rank/cap global hubs, aggregate enum nodes,
    and filter/cap triplet payloads based on PageRank and intra-cluster edge priority.
    """
    
    @staticmethod
    def qualify_and_resolve_targets(topology: Dict[str, Any], path_type: str = "community", config: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Resolves qualified community targets (>= min_cluster_size),
        and aggregates low-density cluster nodes, orphans, and global hubs into all_enum_nodes.
        
        Returns:
            (targets, all_enum_nodes)
        """
        min_cluster_size = config.get('synthesis', {}).get('min_cluster_size', 15) if config else 15
        
        # 1. Global hubs are routed directly into Enum synthesis (enums.py)
        raw_hubs = list(topology.get("global_hubs", []))
        
        # 2. Aggregate Orphans + Low-Density Nodes + Global Hubs (Enums)
        orphans = list(topology.get("orphans", []))
        small_cluster_nodes = []
        
        for comm in topology.get("communities", []):
            nodes = comm.get("nodes", [])
            if len(nodes) < min_cluster_size:
                small_cluster_nodes.extend(nodes)

        all_enum_nodes = sorted(list(set(orphans + small_cluster_nodes + raw_hubs)))
        
        # 3. Consolidate Qualified Community Targets (>= min_cluster_size)
        targets = []
        raw_hubs_set = set(raw_hubs)
        for comm in topology.get("communities", []):
            nodes_in_comm = comm.get("nodes", [])
            # Invariant: Strict Hub Quarantine (Hubs never enter community targets)
            clean_nodes = [n for n in nodes_in_comm if n not in raw_hubs_set]
            if len(clean_nodes) >= min_cluster_size:
                targets.append({"type": "community", "id": comm.get("community_id"), "nodes": clean_nodes})

        targets.sort(key=lambda x: len(x["nodes"]), reverse=True)

        # 3.5 Pre-Synthesis Community Target Agglomeration Safety Check
        synth_cfg = config.get('synthesis', {}) if config else {}
        enable_target_agglom = synth_cfg.get('enable_target_agglomeration', True)
        target_agglom_sim = float(synth_cfg.get('target_agglomeration_similarity', 0.85))

        if enable_target_agglom and len(targets) > 1:
            all_target_nodes_pool = set()
            for t in targets:
                all_target_nodes_pool.update(t.get("nodes", []))
            ref_cfg = config.get('refinement', {}) if config else {}
            m_name = ref_cfg.get('embedding_model') or "all-MiniLM-L6-v2"
            c_dir = Path(ref_cfg.get('vector_cache_dir', 'outputs/02_refinement/cache'))
            concept_vecs = TargetResolver._get_or_load_concept_embeddings(list(all_target_nodes_pool), cache_dir=c_dir, model_name=m_name)

            # Check if precomputed centroid diagnostics are available as fallback or fast-path similarity matrix
            diag_sims = {}
            if not concept_vecs:
                diag_file = Path(f"outputs/03_topology/03_paths/{path_type}/cluster_centroid_diagnostics.json")
                if diag_file.exists():
                    try:
                        with open(diag_file, "r", encoding="utf-8") as df:
                            diag_data = json.load(df)
                        for c in diag_data.get("communities", []):
                            cid = c.get("community_id")
                            for nn in c.get("nearest_neighbors", []):
                                tid = nn.get("target_community_id")
                                s_val = float(nn.get("cosine_similarity", 0.0))
                                diag_sims[(min(cid, tid), max(cid, tid))] = max(diag_sims.get((min(cid, tid), max(cid, tid)), 0.0), s_val)
                    except Exception:
                        pass

            merged_targets = list(targets)
            for m in merged_targets:
                if "source_ids" not in m:
                    m["source_ids"] = {m["id"]}

            while True:
                best_pair = None
                best_sim = target_agglom_sim

                if concept_vecs:
                    target_centroids = {}
                    for idx, t in enumerate(merged_targets):
                        vecs = [concept_vecs[n.lower().strip()] for n in t.get("nodes", []) if n.lower().strip() in concept_vecs]
                        if vecs:
                            c_sum = np.sum(vecs, axis=0)
                            c_norm = np.linalg.norm(c_sum)
                            target_centroids[idx] = c_sum / c_norm if c_norm > 0 else c_sum

                    for i in range(len(merged_targets)):
                        for j in range(i + 1, len(merged_targets)):
                            if i in target_centroids and j in target_centroids:
                                sim = float(np.dot(target_centroids[i], target_centroids[j]))
                                if sim > best_sim:
                                    best_sim = sim
                                    best_pair = (i, j, sim)
                elif diag_sims:
                    for i in range(len(merged_targets)):
                        for j in range(i + 1, len(merged_targets)):
                            s_list = [diag_sims.get((min(a, b), max(a, b)), 0.0) for a in merged_targets[i]["source_ids"] for b in merged_targets[j]["source_ids"]]
                            if s_list:
                                sim = sum(s_list) / len(s_list)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_pair = (i, j, sim)

                if not best_pair:
                    break

                i, j, sim_val = best_pair
                merged_targets[i]["source_ids"].update(merged_targets[j]["source_ids"])
                # Directional merge: preserve ID of larger community
                if len(merged_targets[i]["nodes"]) < len(merged_targets[j]["nodes"]):
                    merged_targets[i]["id"] = merged_targets[j]["id"]
                merged_targets[i]["nodes"] = sorted(list(set(merged_targets[i]["nodes"] + merged_targets[j]["nodes"])))
                merged_targets.pop(j)

            targets = merged_targets
            for t in targets:
                if "source_ids" in t:
                    t["merged_from"] = sorted(list(t.pop("source_ids")))
            targets.sort(key=lambda x: len(x["nodes"]), reverse=True)

        # Invariant: Mutually Disjoint Community and Enum Partitioning (no node exists in both)
        all_target_nodes = set()
        for t in targets:
            all_target_nodes.update(t["nodes"])
        clean_enum_nodes = sorted([n for n in all_enum_nodes if n not in all_target_nodes])

        return targets, clean_enum_nodes

    @staticmethod
    def filter_and_cap_triplets(
        target_nodes: set,
        triplets: List[Dict[str, Any]],
        topology: Dict[str, Any] = None,
        max_triplets_cap: int = 1800,
        target_id: Any = "global",
        path_type: str = "community",
        config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Filters triplets incident to target_nodes and prioritizes them using a
        pure Local Community Centrality engine.
        
        Strict Separation & Quarantine Rules:
        1. Pure Intra-Community (Default): Both subject and object MUST be community members.
           Completely excludes boundary triplets, preventing cross-cluster overlap.
        2. Hub Quarantine: Any triplet touching a pruned global hub is strictly excluded.
        """
        from collections import Counter

        target_nodes_norm = {str(n).lower().strip() for n in target_nodes}

        sem_cfg = config.get('topology', {}).get('semantic_clustering', {}) if config else {}
        exclude_hubs = sem_cfg.get('exclude_hub_triplets_from_communities', True)
        strictly_intra = sem_cfg.get('triplet_payload_mode', 'strictly_intra_only') == 'strictly_intra_only'

        # Extract global hubs to enforce absolute quarantine purge
        raw_hubs = set(topology.get("global_hubs", [])) if topology else set()
        raw_hubs_norm = {str(h).lower().strip() for h in raw_hubs}

        # 1. Calculate Local Intra-Community Degree and Edge Observation Frequency
        local_degree = Counter()
        edge_frequencies = Counter()

        for t in triplets:
            subj = str(t.get('subject', '')).lower().strip()
            obj = str(t.get('object', '')).lower().strip()
            pred = str(t.get('predicate', '')).strip()

            if not subj or not obj:
                continue

            # Absolute Quarantine: Discard any triplet touching a pruned global hub
            if exclude_hubs and (subj in raw_hubs_norm or obj in raw_hubs_norm):
                continue

            is_subj_in = subj in target_nodes_norm
            is_obj_in = obj in target_nodes_norm

            if is_subj_in and is_obj_in:
                local_degree[subj] += 1
                local_degree[obj] += 1
                edge_frequencies[(subj, pred, obj)] += 1
            elif not strictly_intra and (is_subj_in or is_obj_in):
                edge_frequencies[(subj, pred, obj)] += 1

        # 2. Score candidate triplets using Local Community Priority
        scored_triplets = []
        for t in triplets:
            subj = str(t.get('subject', '')).lower().strip()
            obj = str(t.get('object', '')).lower().strip()

            if not subj or not obj:
                continue

            # Absolute Quarantine: Discard any triplet touching a pruned global hub
            if exclude_hubs and (subj in raw_hubs_norm or obj in raw_hubs_norm):
                continue

            is_subj_in = subj in target_nodes_norm
            is_obj_in = obj in target_nodes_norm

            if strictly_intra:
                # Pure Intra-Community Mode: Both endpoints must be inside target_nodes
                if not (is_subj_in and is_obj_in):
                    continue
            else:
                if not (is_subj_in or is_obj_in):
                    continue

            pred = str(t.get('predicate', '')).strip()
            freq = edge_frequencies.get((subj, pred, obj), 1)
            freq_boost = 1.0 + 0.1 * min(10, freq - 1)

            if is_subj_in and is_obj_in:
                # Tier 1: Pure Intra-Community (Base 1000.0 guarantees absolute priority)
                local_sum = local_degree[subj] + local_degree[obj]
                rank_score = 1000.0 + (local_sum * freq_boost)
            else:
                # Tier 2: Boundary Context (only used if strictly_intra is false)
                internal_node = subj if is_subj_in else obj
                anchor_degree = local_degree[internal_node]
                rank_score = 1.0 + (anchor_degree * freq_boost)

            t_copy = dict(t)
            t_copy["triplet_rank_score"] = round(float(rank_score), 4)
            scored_triplets.append((rank_score, t_copy))

        # 3. Sort descending by community-centric rank score and slice up to cap
        scored_triplets.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_triplets[:max_triplets_cap]]

    @staticmethod
    def _get_or_load_concept_embeddings(
        nodes: List[str],
        cache_dir: Path = Path("outputs/02_refinement/cache"),
        model_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Loads L2-normalized concept embeddings from refinement caches (subject, object, concept),
        and encodes any missing terms using the active model, saving the unified concept cache.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not model_name:
            # Check if any concept or subject meta.json has the exact model_name recorded
            for meta_f in sorted(cache_dir.glob("*_meta.json")):
                try:
                    with open(meta_f, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                    if "model_name" in m_data and m_data["model_name"]:
                        model_name = m_data["model_name"]
                        break
                except Exception:
                    pass
            if not model_name:
                model_name = "all-MiniLM-L6-v2"

        model_slug = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        concept_npy = cache_dir / f"concept_{model_slug}_embeddings.npy"
        concept_meta = cache_dir / f"concept_{model_slug}_meta.json"

        node_map: Dict[str, np.ndarray] = {}

        # 1. Try unified concept cache first
        if concept_npy.exists() and concept_meta.exists():
            try:
                with open(concept_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                vocab = meta.get("vocab", [])
                arr = np.load(concept_npy)
                if len(vocab) == len(arr):
                    for idx, term in enumerate(vocab):
                        node_map[str(term).lower().strip()] = arr[idx]
            except Exception as e:
                print(f"      -> Warning reading concept cache: {e}")

        # Check which requested nodes are missing
        norm_nodes = [str(n).strip() for n in nodes if str(n).strip()]
        missing = [n for n in norm_nodes if n.lower().strip() not in node_map]

        if not missing:
            return node_map

        # 2. Try loading from subject/object caches
        for prefix in ["subject", "object"]:
            sub_npy = cache_dir / f"{prefix}_{model_slug}_embeddings.npy"
            sub_meta = cache_dir / f"{prefix}_{model_slug}_meta.json"
            if sub_npy.exists() and sub_meta.exists():
                try:
                    with open(sub_meta, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    vocab = meta.get("vocab", [])
                    arr = np.load(sub_npy)
                    if len(vocab) == len(arr):
                        for idx, term in enumerate(vocab):
                            k = str(term).lower().strip()
                            if k not in node_map:
                                node_map[k] = arr[idx]
                except Exception:
                    pass

        missing_still = [n for n in norm_nodes if n.lower().strip() not in node_map]
        if missing_still:
            try:
                from sentence_transformers import SentenceTransformer
                unique_missing = sorted(list(set(missing_still)))
                try:
                    model = SentenceTransformer(model_name, local_files_only=True)
                except Exception:
                    model = SentenceTransformer(model_name)
                new_vecs = model.encode(unique_missing, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
                for idx, term in enumerate(unique_missing):
                    node_map[term.lower().strip()] = new_vecs[idx]

                # Update unified cache
                all_terms = list(node_map.keys())
                all_arr = np.array([node_map[t] for t in all_terms])
                np.save(concept_npy, all_arr)
                with open(concept_meta, "w", encoding="utf-8") as f:
                    json.dump({"model_name": model_name, "dimensions": int(all_arr.shape[1]), "vocab": all_terms}, f, indent=2)
                print(f"      -> Unified concept cache saved: {concept_npy.name} ({len(all_terms)} concepts)")
            except Exception as e:
                print(f"      -> Warning: SentenceTransformer encoding error: {e}")

        return node_map

    @staticmethod
    def compute_architectural_tiers(
        target_nodes: set,
        triplets: List[Dict[str, Any]],
        config: Dict[str, Any],
        concept_embeddings: Dict[str, np.ndarray],
        community_id: Any = None,
        community_label: str = "Community"
    ) -> Dict[str, Any]:
        """
        Computes hybrid geometric-topological architectural tiers for a community target:
          - Thematic Prototypicality (P_u): Cosine similarity to degree-weighted Fréchet mean centroid.
          - Intra-Community Degree Centrality (D_u): Normalized local degree.
          - Contextual Fidelity (Gamma_u): Average edge fidelity to source sentences.
          - Composite Architectural Score (S_u): S_u = (max(0, P_u))^w_p * (D_u)^w_d * (Gamma_u)^w_f.
        Partitions entities into:
          - Tier 1: Thematic Nucleus (Top 15-20%, min 2, max 8)
          - Tier 2: Relational Mechanics (Next ~50%, internal degree > 1)
          - Tier 3: Contextual Modifiers & Noise (Bottom ~30%, degree-1 leaves, or low fidelity)
        """
        tier_cfg = config.get('topology', {}).get('architectural_tiers', {}) if 'topology' in config else config.get('architectural_tiers', {})
        weights = tier_cfg.get('weights', {})
        w_proto = float(weights.get('prototypicality', 0.40))
        w_cent = float(weights.get('centrality', 0.40))
        w_fid = float(weights.get('fidelity', 0.20))

        p_t1 = float(tier_cfg.get('tier_1_percentile', 0.80))
        min_t1 = int(tier_cfg.get('tier_1_min_entities', 2))
        max_t1 = int(tier_cfg.get('tier_1_max_entities', 8))
        p_t2 = float(tier_cfg.get('tier_2_percentile', 0.30))
        min_fid_t1 = float(tier_cfg.get('min_fidelity_tier_1', 0.10))
        boundary_discount = float(tier_cfg.get('boundary_discount', 0.5))
        fallback_proto = float(tier_cfg.get('fallback_prototypicality', 0.30))
        fallback_fid = float(tier_cfg.get('fallback_fidelity', 0.28))

        target_nodes_norm = {str(n).lower().strip(): str(n) for n in target_nodes}
        node_internal_deg = Counter()
        node_boundary_deg = Counter()
        node_fidelities = defaultdict(list)
        node_quotes = defaultdict(list)

        for t in triplets:
            s_raw = str(t.get('subject', '')).strip()
            o_raw = str(t.get('object', '')).strip()
            s_k = s_raw.lower()
            o_k = o_raw.lower()
            fid = float(t.get('contextual_fidelity', 0.30))
            quote = str(t.get('source_quote', '')).strip()

            is_s_in = s_k in target_nodes_norm
            is_o_in = o_k in target_nodes_norm

            if is_s_in and is_o_in:
                node_internal_deg[s_k] += 1
                node_internal_deg[o_k] += 1
            elif is_s_in:
                node_boundary_deg[s_k] += 1
            elif is_o_in:
                node_boundary_deg[o_k] += 1

            if is_s_in:
                node_fidelities[s_k].append(fid)
                if quote:
                    node_quotes[s_k].append((fid, quote))
            if is_o_in:
                node_fidelities[o_k].append(fid)
                if quote:
                    node_quotes[o_k].append((fid, quote))

        # 1. Compute Community Centroid Vector
        vec_dim = 384
        for vec in concept_embeddings.values():
            vec_dim = len(vec)
            break

        weighted_vecs = []
        weight_sum = 0.0
        for n_k in target_nodes_norm.keys():
            vec = concept_embeddings.get(n_k)
            if vec is not None:
                deg = node_internal_deg[n_k]
                w = 1.0 + math.log(1.0 + deg)
                weighted_vecs.append(vec * w)
                weight_sum += w

        if weighted_vecs and weight_sum > 0:
            c_sum = np.sum(weighted_vecs, axis=0)
            c_norm = np.linalg.norm(c_sum)
            centroid = c_sum / c_norm if c_norm > 0 else c_sum
        else:
            centroid = np.zeros(vec_dim)

        # 2. Score Each Entity
        scored_entities = []
        max_effective_deg = 1.0
        effective_degrees = {}
        for n_k in target_nodes_norm.keys():
            deg_int = node_internal_deg[n_k]
            deg_bound = node_boundary_deg[n_k]
            eff_deg = deg_int + boundary_discount * deg_bound
            effective_degrees[n_k] = eff_deg
            if eff_deg > max_effective_deg:
                max_effective_deg = eff_deg

        for n_k, orig_name in target_nodes_norm.items():
            vec = concept_embeddings.get(n_k)
            if vec is not None and np.linalg.norm(centroid) > 0:
                proto = float(np.dot(vec, centroid))
            else:
                proto = fallback_proto

            eff_deg = effective_degrees[n_k]
            cent = eff_deg / max_effective_deg

            fids = node_fidelities.get(n_k, [])
            fid = float(np.mean(fids)) if fids else fallback_fid

            s_proto = max(0.001, proto) ** w_proto
            s_cent = cent ** w_cent
            s_fid = max(0.001, fid) ** w_fid
            composite_s = float(s_proto * s_cent * s_fid)

            # Top exemplar quote
            quotes = node_quotes.get(n_k, [])
            if quotes:
                quotes.sort(key=lambda x: x[0], reverse=True)
                top_quote = quotes[0][1]
            else:
                top_quote = ""

            scored_entities.append({
                "entity": orig_name,
                "composite_score": round(composite_s, 4),
                "prototypicality": round(proto, 4),
                "centrality": round(cent, 4),
                "fidelity": round(fid, 4),
                "internal_degree": int(node_internal_deg[n_k]),
                "boundary_degree": int(node_boundary_deg[n_k]),
                "exemplar_quote": top_quote[:240]
            })

        scored_entities.sort(key=lambda x: x["composite_score"], reverse=True)

        # 3. Dynamic Percentile Partitioning
        s_values = np.array([x["composite_score"] for x in scored_entities])
        p80_val = float(np.percentile(s_values, p_t1 * 100)) if len(s_values) > 0 else 0.5
        p30_val = float(np.percentile(s_values, p_t2 * 100)) if len(s_values) > 0 else 0.2

        t1_candidates = []
        demoted_from_t1 = []
        remaining = []

        for item in scored_entities:
            if item["composite_score"] >= p80_val:
                if item["fidelity"] >= min_fid_t1:
                    t1_candidates.append(item)
                else:
                    item_copy = dict(item)
                    item_copy["demoted_due_to_low_fidelity"] = True
                    demoted_from_t1.append(item_copy)
            else:
                remaining.append(item)

        # Enforce min_t1 and max_t1 bounds
        effective_max_t1 = max(min_t1, max_t1)
        if len(t1_candidates) < min_t1 and scored_entities:
            tier_1 = scored_entities[:min(min_t1, len(scored_entities))]
            t1_names = {x["entity"] for x in tier_1}
            pool_for_t2 = [x for x in scored_entities if x["entity"] not in t1_names]
        elif len(t1_candidates) > effective_max_t1:
            tier_1 = t1_candidates[:effective_max_t1]
            t1_names = {x["entity"] for x in tier_1}
            pool_for_t2 = [x for x in (t1_candidates[effective_max_t1:] + remaining) if x["entity"] not in t1_names]
        else:
            tier_1 = t1_candidates
            t1_names = {x["entity"] for x in tier_1}
            pool_for_t2 = [x for x in remaining if x["entity"] not in t1_names]

        # Tier 2: Above p30 and internal degree > 0
        tier_2 = []
        tier_3_pool = list(demoted_from_t1)

        for item in pool_for_t2:
            if item["composite_score"] >= p30_val and (item["internal_degree"] > 0 or item["boundary_degree"] > 1):
                tier_2.append(item)
            else:
                tier_3_pool.append(item)

        tier_3 = tier_3_pool
        tier_3.sort(key=lambda x: x["composite_score"], reverse=True)

        return {
            "community_id": community_id,
            "community_label": community_label,
            "total_nodes": len(target_nodes),
            "tier_counts": {
                "tier_1_nucleus": len(tier_1),
                "tier_2_mechanics": len(tier_2),
                "tier_3_modifiers": len(tier_3)
            },
            "tier_1_nucleus": tier_1,
            "tier_2_mechanics": tier_2,
            "tier_3_modifiers": tier_3
        }

    @staticmethod
    def export_resolved_targets_json(
        topology: Dict[str, Any],
        path_type: str = "community",
        refined_triplets: List[Dict[str, Any]] = None,
        config: Dict[str, Any] = None,
        output_dir: Path = Path("outputs/03_topology/03_paths/community")
    ) -> Path:
        """
        Executes target qualification and triplet PageRank capping, and exports
        resolved_<path_type>_targets.json to output_dir with explicit rank scores.
        Also computes and exports geometric-topological architectural tiers.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        synth_cfg = config.get('synthesis', {}) if config else {}
        min_cluster_size = synth_cfg.get('min_cluster_size', 15)
        max_triplets_cap = synth_cfg.get('max_triplets_per_target', 1800)
        
        # 1. Resolve target qualification & enum nodes
        targets, all_enum_nodes = TargetResolver.qualify_and_resolve_targets(topology, path_type, config)

        tier_cfg = config.get('topology', {}).get('architectural_tiers', {}) if 'topology' in config else config.get('architectural_tiers', {})
        tier_enabled = tier_cfg.get("enabled", True)

        # 2. Pre-load / cache unified concept embeddings for all target nodes (if tiers enabled)
        concept_embeddings = {}
        if tier_enabled:
            all_target_nodes = set()
            for target in targets:
                all_target_nodes.update(target.get("nodes", []))
            ref_cfg = config.get('refinement', {}) if 'refinement' in config else config
            model_name = ref_cfg.get('embedding_model') or config.get('embedding_model')
            cache_dir = Path(ref_cfg.get('vector_cache_dir', 'outputs/02_refinement/cache'))
            concept_embeddings = TargetResolver._get_or_load_concept_embeddings(
                list(all_target_nodes),
                cache_dir=cache_dir,
                model_name=model_name
            )

        # 3. Build target payload records with sorted & capped triplets and architectural tiers
        target_payloads = []
        all_tiers_records = []
        csv_rows = []

        for i, target in enumerate(targets, start=1):
            target_type = target["type"]
            c_id = target["id"]
            target_nodes = set(target["nodes"])
            comm_label = f"Community #{c_id}"
            
            # Filter, score, and cap triplets with Asymmetric Synergy Engine
            capped_triplets = TargetResolver.filter_and_cap_triplets(
                target_nodes, refined_triplets, topology, max_triplets_cap, target_id=c_id, path_type=path_type, config=config
            )

            if tier_enabled:
                # Compute Geometric-Topological Architectural Tiers
                tiers_data = TargetResolver.compute_architectural_tiers(
                    target_nodes=target_nodes,
                    triplets=capped_triplets,
                    config=config,
                    concept_embeddings=concept_embeddings,
                    community_id=c_id,
                    community_label=comm_label
                )
                all_tiers_records.append(tiers_data)

                # Prepare CSV rows for human spreadsheet evaluation
                for rank, item in enumerate(tiers_data.get("tier_1_nucleus", []), start=1):
                    csv_rows.append({
                        "Community ID": f"Community #{c_id}",
                        "Community Label": comm_label,
                        "Tier": "Tier 1: Nucleus",
                        "Rank in Tier": rank,
                        "Entity Name": item["entity"],
                        "Composite Score": item["composite_score"],
                        "Prototypicality": item["prototypicality"],
                        "Centrality": item["centrality"],
                        "Fidelity": item["fidelity"],
                        "Internal Degree": item["internal_degree"],
                        "Boundary Degree": item["boundary_degree"],
                        "Exemplar Quote": item["exemplar_quote"]
                    })
                for rank, item in enumerate(tiers_data.get("tier_2_mechanics", []), start=1):
                    csv_rows.append({
                        "Community ID": f"Community #{c_id}",
                        "Community Label": comm_label,
                        "Tier": "Tier 2: Mechanics",
                        "Rank in Tier": rank,
                        "Entity Name": item["entity"],
                        "Composite Score": item["composite_score"],
                        "Prototypicality": item["prototypicality"],
                        "Centrality": item["centrality"],
                        "Fidelity": item["fidelity"],
                        "Internal Degree": item["internal_degree"],
                        "Boundary Degree": item["boundary_degree"],
                        "Exemplar Quote": item["exemplar_quote"]
                    })
                for rank, item in enumerate(tiers_data.get("tier_3_modifiers", []), start=1):
                    csv_rows.append({
                        "Community ID": f"Community #{c_id}",
                        "Community Label": comm_label,
                        "Tier": "Tier 3: Modifiers",
                        "Rank in Tier": rank,
                        "Entity Name": item["entity"],
                        "Composite Score": item["composite_score"],
                        "Prototypicality": item["prototypicality"],
                        "Centrality": item["centrality"],
                        "Fidelity": item["fidelity"],
                        "Internal Degree": item["internal_degree"],
                        "Boundary Degree": item["boundary_degree"],
                        "Exemplar Quote": item["exemplar_quote"]
                    })
            else:
                tiers_data = {
                    "community_id": c_id,
                    "community_label": comm_label,
                    "target_node_count": len(target_nodes),
                    "tier_1_nucleus": [],
                    "tier_2_mechanics": [],
                    "tier_3_modifiers": []
                }

            filename_suffix = f"community_{c_id}"
            output_module_name = f"{i:02d}_{filename_suffix}.py"
            rank_score = float(len(target_nodes))
            
            payload_item = {
                "target_index": i,
                "target_type": target_type,
                "target_id": c_id,
                "merged_from": target.get("merged_from", [c_id]),
                "output_module_name": output_module_name,
                "rank_score": rank_score,
                "nodes_count": len(target_nodes),
                "nodes": sorted(list(target_nodes)),
                "triplets_count": len(capped_triplets),
                "triplets": capped_triplets,
                "architectural_tiers": tiers_data
            }
            target_payloads.append(payload_item)
            
        # 4. Construct master audit record
        audit_payload = {
            "path_type": path_type,
            "min_cluster_size": min_cluster_size,
            "max_triplets_per_target": max_triplets_cap,
            "global_hubs": list(topology.get("global_hubs", [])),
            "enum_nodes_count": len(all_enum_nodes),
            "enum_nodes": all_enum_nodes,
            "targets_count": len(target_payloads),
            "targets": target_payloads
        }
        
        target_filename = "resolved_community_targets.json"
            
        target_file = output_dir / target_filename
        with open(target_file, "w") as f:
            json.dump(audit_payload, f, indent=2)
            
        print(f"   [OK] Resolved Targets Saved: {target_file}")

        # 5. Export Tier JSON, CSV, and Diagnostics for Human Evaluation
        if path_type == "community" and tier_enabled:
            tier_json_file = output_dir / "community_architectural_tiers.json"
            with open(tier_json_file, "w", encoding="utf-8") as f:
                json.dump(all_tiers_records, f, indent=2)
            print(f"   [OK] Community Architectural Tiers JSON Saved: {tier_json_file}")

            if tier_cfg.get("export_tier_csv", True) and csv_rows:
                tier_csv_file = output_dir / "community_architectural_tiers.csv"
                fieldnames = [
                    "Community ID", "Community Label", "Tier", "Rank in Tier", "Entity Name",
                    "Composite Score", "Prototypicality", "Centrality", "Fidelity",
                    "Internal Degree", "Boundary Degree", "Exemplar Quote"
                ]
                with open(tier_csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_rows)
                print(f"   [OK] Community Architectural Tiers CSV Saved: {tier_csv_file} ({len(csv_rows)} rows)")

            if tier_cfg.get("export_diagnostics", True):
                diag_file = output_dir / "tier_partitioning_diagnostics.json"
                t1_scores = [r["Composite Score"] for r in csv_rows if r["Tier"] == "Tier 1: Nucleus"]
                t2_scores = [r["Composite Score"] for r in csv_rows if r["Tier"] == "Tier 2: Mechanics"]
                t3_scores = [r["Composite Score"] for r in csv_rows if r["Tier"] == "Tier 3: Modifiers"]
                diag = {
                    "total_communities": len(all_tiers_records),
                    "total_partitioned_entities": len(csv_rows),
                    "tier_counts": {
                        "tier_1_nucleus": len(t1_scores),
                        "tier_2_mechanics": len(t2_scores),
                        "tier_3_modifiers": len(t3_scores)
                    },
                    "score_distributions": {
                        "tier_1": {
                            "mean": round(float(np.mean(t1_scores)), 4) if t1_scores else 0.0,
                            "median": round(float(np.median(t1_scores)), 4) if t1_scores else 0.0,
                            "min": round(float(np.min(t1_scores)), 4) if t1_scores else 0.0,
                            "max": round(float(np.max(t1_scores)), 4) if t1_scores else 0.0
                        },
                        "tier_2": {
                            "mean": round(float(np.mean(t2_scores)), 4) if t2_scores else 0.0,
                            "median": round(float(np.median(t2_scores)), 4) if t2_scores else 0.0,
                            "min": round(float(np.min(t2_scores)), 4) if t2_scores else 0.0,
                            "max": round(float(np.max(t2_scores)), 4) if t2_scores else 0.0
                        },
                        "tier_3": {
                            "mean": round(float(np.mean(t3_scores)), 4) if t3_scores else 0.0,
                            "median": round(float(np.median(t3_scores)), 4) if t3_scores else 0.0,
                            "min": round(float(np.min(t3_scores)), 4) if t3_scores else 0.0,
                            "max": round(float(np.max(t3_scores)), 4) if t3_scores else 0.0
                        }
                    },
                    "separation_margin": round(
                        float(np.mean(t1_scores) - np.mean(t3_scores)), 4
                    ) if t1_scores and t3_scores else 0.0
                }
                with open(diag_file, "w", encoding="utf-8") as f:
                    json.dump(diag, f, indent=2)
                print(f"   [OK] Tier Partitioning Diagnostics Saved: {diag_file}")

        return target_file
