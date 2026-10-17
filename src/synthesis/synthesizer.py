import json
import re
import subprocess
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.agents.synthesis_agents import (
    OrphanContext,
    orphan_agent, leiden_schema_agent,
    consolidation_agent, comprehensive_ontology_agent,
    schema_reformat_agent,
    last_llm_responses
)
from src.synthesis import prompts as synth_prompts
from src.topology.resolver import TargetResolver


def cluster_enum_candidates(
    enum_nodes: List[str],
    concept_embeddings: Dict[str, np.ndarray],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Groups orphan and low-density nodes into high-cohesion semantic clusters
    using pre-computed concept vectors. Discards isolated terms and weak clusters.
    """
    enum_cfg = config.get('synthesis', {}).get('enum_clustering', {})
    min_cluster_size = enum_cfg.get('min_cluster_size', 2)
    dist_threshold = enum_cfg.get('distance_threshold', 0.48)
    min_cohesion = enum_cfg.get('min_cohesion_score', 0.55)

    valid_terms = []
    vector_list = []
    for term in enum_nodes:
        clean_key = str(term).lower().strip()
        if clean_key in concept_embeddings:
            valid_terms.append(str(term).strip())
            vector_list.append(concept_embeddings[clean_key])

    if len(valid_terms) < min_cluster_size:
        return []

    X = np.vstack(vector_list)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(X)

    clusters_by_id = defaultdict(lambda: {"terms": [], "vectors": []})
    for term, vec, lbl in zip(valid_terms, X, labels):
        clusters_by_id[lbl]["terms"].append(term)
        clusters_by_id[lbl]["vectors"].append(vec)

    strong_clusters = []
    for c_id, data in clusters_by_id.items():
        terms = data["terms"]
        if len(terms) < min_cluster_size:
            continue

        vecs = np.array(data["vectors"])
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        sims = np.dot(vecs, centroid)
        avg_cohesion = float(np.mean(sims))

        if avg_cohesion >= min_cohesion:
            prototypical_idx = int(np.argmax(sims))
            strong_clusters.append({
                "cluster_id": int(c_id),
                "suggested_category": terms[prototypical_idx],
                "cohesion_score": round(avg_cohesion, 3),
                "member_count": len(terms),
                "terms": terms
            })

    strong_clusters.sort(key=lambda c: (c["member_count"], c["cohesion_score"]), reverse=True)
    return strong_clusters


def clean_python_code(code: str) -> str:
    """Removes markdown formatting, invalid characters, and escapes that LLMs often inject."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    # Fix over-escaped quotes caused by LLM JSON serialization 
    code = code.replace('\\"', '"')
    
    # Replace literal \n escape sequences that some LLMs emit outside raw strings
    code = code.replace('\\n', '\n')
    
    # Specific fix to prevent un-escaping the one known problematic Enum string
    code = code.replace('"the "firstthen" principle"', '\'the "firstthen" principle\'')
    
    # Sanitize invalid variable/enum identifiers with commas or spaces before assignment '='
    def fix_line(line: str) -> str:
        m = re.match(r'^(\s+)([a-zA-Z0-9_, ]+?)(\s*=\s*.*)$', line)
        if m:
            indent, var, rest = m.groups()
            stripped_var = var.strip()
            if (',' in stripped_var or ' ' in stripped_var) and not stripped_var.startswith(('def ', 'class ', 'return ', 'if ', 'elif ')):
                clean_var = re.sub(r'[, ]+', '_', stripped_var)
                return f"{indent}{clean_var}{rest}"
        return line

    code = '\n'.join(fix_line(line_str) for line_str in code.splitlines())
    return code.strip()

class SynthesisPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def execute(self, comm_targets: Dict[str, Any] = None, refined_triplets: List[Dict[str, Any]] = None, original_triplets: List[Dict[str, Any]] = None, taxonomic_map: Dict[str, str] = None, master_themes: List[str] = None):
        print("[Synthesis] Starting Stage 4 Pipeline...")
        if comm_targets:
            print("   ==> Executing Community Path Schema Synthesis...")
            self.execute_path_synthesis(comm_targets, "community", refined_triplets, original_triplets, master_themes)
        else:
            print("   [Warning] Community resolved targets file not found. Skipping synthesis.")

        print("[Synthesis] Stage 4 Pipeline complete.")

    def execute_path_synthesis(self, targets_data: Dict[str, Any], path_type: str, refined_triplets: List[dict], original_triplets: List[dict], master_themes: List[str]):
        path_label = "Community Workflow" if path_type == "community" else f"Path: {path_type.title()}"
        print(f"   -> Starting Synthesis Pass for [{path_label}]...")

        # Read config options
        schema_pass_mode = self.config.get('synthesis', {}).get('schema_pass_mode', 'both')
        min_cluster_size = self.config.get('synthesis', {}).get('min_cluster_size', 32)
        run_norm = schema_pass_mode in ("both", "normalized")
        run_raw = schema_pass_mode in ("both", "raw")
        print(f"      -> Schema Pass Mode: '{schema_pass_mode}' (Normalized: {run_norm}, Raw: {run_raw})")
        print(f"      -> Entity Threshold (min_cluster_size): {min_cluster_size}")

        # Setup logging
        outputs_dir = self.config.get('directories', {}).get('outputs', 'outputs')
        log_dir = Path(outputs_dir) / "04_synthesis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"synthesis_{path_type}_errors.log"
        
        def log_error(msg: str):
            print(msg)
            with open(log_file, "a") as lf:
                lf.write(f"[{datetime.now().isoformat()}] {msg}\n")
                
        def log_detailed_synthesis_error(error: Exception, responses: list, pass_name: str, target_id: Any, i: int):
            from pydantic import ValidationError
            
            error_msg = f"[Synthesis {path_type}] Error generating {pass_name} schema {i} ({target_id}): {error}"
            log_error(error_msg)
            
            def log_detail(msg: str):
                with open(log_file, "a") as lf:
                    lf.write(f"[{datetime.now().isoformat()}] {msg}\n")
            
            cause = getattr(error, '__cause__', None)
            if isinstance(cause, ValidationError):
                log_detail("      -> Pydantic Validation Error Details:")
                try:
                    for err in cause.errors():
                        loc = " -> ".join(str(x) for x in err.get("loc", []))
                        log_detail(f"         Location: {loc}")
                        log_detail(f"         Type:     {err.get('type')}")
                        log_detail(f"         Message:  {err.get('msg')}")
                        log_detail(f"         Input:    {err.get('input')}")
                except Exception:
                    log_detail(f"         {cause}")
            elif cause:
                log_detail(f"      -> Underlying Cause: {cause}")
                
            if responses:
                log_detail(f"      -> Captured LLM Outputs (Attempts: {len(responses)}):")
                for attempt_idx, resp in enumerate(responses, start=1):
                    log_detail(f"         [Attempt {attempt_idx} Output Preview]:")
                    for part in resp:
                        part_str = str(part)
                        if len(part_str) > 1000:
                            part_str = part_str[:1000] + "\n... [truncated]"
                        indented = "\n".join(f"            {line}" for line in part_str.splitlines())
                        log_detail(indented)
                        
        def extract_malformed_text(messages, exception: Exception) -> str:
            from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
            for msg in reversed(messages):
                if isinstance(msg, ModelResponse):
                    tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
                    if tool_calls:
                        args = tool_calls[0].args
                        return json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
                    text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
                    if text_parts:
                        content = text_parts[0].content
                        if "```" in content:
                            lines = content.splitlines()
                            cleaned = [line for line in lines if not line.strip().startswith("```")]
                            content = "\n".join(cleaned)
                        return content.strip()
            if exception:
                return str(exception)
            return ""
        
        # Path-isolated Output directories
        output_base_dir = Path("outputs/04_synthesis/schemas") / path_type
        norm_dir = output_base_dir / "normalized"
        raw_dir = output_base_dir / "raw"
        
        if output_base_dir.exists():
            shutil.rmtree(output_base_dir)
            
        output_base_dir.mkdir(parents=True, exist_ok=True)
        if run_norm:
            norm_dir.mkdir(parents=True, exist_ok=True)
        if run_raw:
            raw_dir.mkdir(parents=True, exist_ok=True)

        # Payload dumping directory
        save_llm_payloads = self.config.get('synthesis', {}).get('save_llm_payloads', False)
        default_payload_dir = Path(self.config.get('directories', {}).get('outputs', 'outputs')) / 'payload_text' / 'stage_04_synthesis'
        payload_base_dir = Path(self.config.get('synthesis', {}).get('payload_output_dir', default_payload_dir)) / path_type
        if save_llm_payloads:
            if payload_base_dir.exists():
                shutil.rmtree(payload_base_dir)
            payload_base_dir.mkdir(parents=True, exist_ok=True)

        def save_payload_file(filename: str, payload_str: str):
            if save_llm_payloads:
                try:
                    with open(payload_base_dir / filename, "w", encoding="utf-8") as pf:
                        pf.write(payload_str)
                except Exception as err:
                    log_error(f"[Synthesis {path_type}] Error saving payload {filename}: {err}")

        # Ingest pre-resolved Targets and Enum Nodes directly from target JSON payload
        targets = targets_data.get("targets", [])
        all_enum_nodes = targets_data.get("enum_nodes", [])
        active_agent = leiden_schema_agent
        
        # Phase 1: Semantic Enum Clustering & Synthesis
        print(f"   -> [{path_label}] Phase 1: Semantic Enum Clustering & Synthesis...")
        print(f"      -> Total Raw Enum Candidates: {len(all_enum_nodes)}")
        global_enums_code = ""

        enum_clustering_cfg = self.config.get('synthesis', {}).get('enum_clustering', {})
        clustering_enabled = enum_clustering_cfg.get('enabled', True)

        if all_enum_nodes:
            if clustering_enabled:
                ref_cfg = self.config.get('refinement', {})
                model_name = ref_cfg.get('embedding_model', 'all-MiniLM-L6-v2')
                cache_dir = Path(ref_cfg.get('vector_cache_dir', 'outputs/02_refinement/cache'))

                concept_embeddings = TargetResolver._get_or_load_concept_embeddings(
                    all_enum_nodes,
                    cache_dir=cache_dir,
                    model_name=model_name
                )

                qualified_clusters = cluster_enum_candidates(all_enum_nodes, concept_embeddings, self.config)
                retained_terms = sum(c["member_count"] for c in qualified_clusters)
                discarded_terms = len(all_enum_nodes) - retained_terms

                print(f"      -> Qualified Strong Clusters: {len(qualified_clusters)} ({retained_terms} concepts)")
                print(f"      -> Filtered Outliers / Noise Discarded: {discarded_terms} concepts")

                # Save cluster diagnostics for transparency
                clusters_audit_file = log_dir / f"enum_clusters_{path_type}.json"
                try:
                    with open(clusters_audit_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "total_raw_nodes": len(all_enum_nodes),
                            "retained_terms_count": retained_terms,
                            "discarded_noise_count": discarded_terms,
                            "clusters_count": len(qualified_clusters),
                            "clusters": qualified_clusters
                        }, f, indent=2)
                except Exception as err:
                    log_error(f"[Synthesis {path_type}] Error saving enum clusters audit: {err}")

                if qualified_clusters:
                    max_clusters_per_batch = enum_clustering_cfg.get('max_clusters_per_batch', 25)
                    selected_clusters = qualified_clusters[:max_clusters_per_batch]
                    if len(qualified_clusters) > max_clusters_per_batch:
                        print(f"      -> Selected Top {max_clusters_per_batch} clusters for primary enum synthesis")

                    payload_data = {
                        "instructions": "Synthesize standardized Python Enum and Literal classes for each strongly grouped semantic cluster.",
                        "candidate_clusters": [
                            {
                                "cluster_id": c["cluster_id"],
                                "category_hint": c["suggested_category"],
                                "cohesion_score": c["cohesion_score"],
                                "terms": c["terms"]
                            }
                            for c in selected_clusters
                        ]
                    }
                    payload = json.dumps(payload_data, indent=2)
                else:
                    payload = ""
            else:
                # Legacy unclustered flat list
                payload = json.dumps(all_enum_nodes)

            if payload:
                orphan_ctx = OrphanContext(master_themes=master_themes)
                save_payload_file("phase1_enums_payload.txt", f"=== SYSTEM PROMPT ===\n{synth_prompts.ORPHAN_SYNTHESIS_SYSTEM_PROMPT}\n\n=== SYSTEM CONTEXT ===\nMaster Themes Context: {json.dumps(master_themes)}\n\n=== USER PAYLOAD ===\n{payload}")
                last_llm_responses.set([])
                from pydantic_ai import capture_run_messages
                with capture_run_messages() as messages:
                    try:
                        result = orphan_agent.run_sync(payload, deps=orphan_ctx)
                        global_enums_code = clean_python_code(result.output.source_code)
                    except Exception as e:
                        malformed_text = extract_malformed_text(messages, e)
                        print(f"      [Warning] Initial enum generation failed for {path_type}. Running Schema Reformat Agent...")
                        reformat_prompt = (
                            f"Original Enum Generation Task payload was:\n{payload}\n\n"
                            f"The malformed output from the failed attempts was:\n{malformed_text}\n\n"
                            f"The parsing validation error that occurred was:\n{str(e)}\n\n"
                            f"Please correct and repair this Python code so it strictly maps to the schemas.GeneratedModule structure. "
                            f"Set module_name to 'enums' and source_code to pure Python Enum definitions."
                        )
                        try:
                            ref_res = schema_reformat_agent.run_sync(reformat_prompt)
                            global_enums_code = clean_python_code(ref_res.output.source_code)
                            print(f"      -> Reformat successful for enums ({path_type})!")
                        except Exception as ref_err:
                            log_error(f"[Synthesis {path_type}] Error generating enums: {ref_err}")
                            global_enums_code = "# Error generating enums"
            else:
                print("      -> No candidate terms passed cohesion thresholds. Skipping enum generation.")
                global_enums_code = "# No strongly grouped enums qualified"

            if run_norm and global_enums_code and not global_enums_code.startswith("# Error"):
                with open(norm_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)
            if run_raw and global_enums_code and not global_enums_code.startswith("# Error"):
                with open(raw_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)

        # Phase 2: Schema Generation
        print(f"   -> [{path_label}] Phase 2: Schema Generation...")
        print(f"      -> Target Composition: {len(targets)} qualified schema targets (global hubs routed to enums.py)")

        use_async = self.config.get('pipeline', {}).get('use_async', False)
        max_async = self.config.get('synthesis', {}).get('max_async_calls', 1)
        payload_mode = self.config.get('synthesis', {}).get('payload_values', 'raw_text')
        max_sentences_per_target = self.config.get('synthesis', {}).get('max_sentences_per_target', 60)
        top_domain_terms_count = self.config.get('synthesis', {}).get('top_domain_terms_count', 12)
        min_sentence_length = self.config.get('synthesis', {}).get('min_sentence_length', 15)

        # Precompute global target counts to score community-specificity of terms (C-TF-IDF)
        global_target_counts = Counter()
        for t in targets:
            for n in t.get("nodes", []):
                global_target_counts[str(n).lower().strip()] += 1

        default_exclusions = [
            'person', 'name', 'support', 'organization', 'thing', 'things', 'away', '-',
            'other', 'none', 'user', 'people', 'system', 'systems', 'someone', 'something',
            'anyone', 'anything', 'everyone', 'everything', 'item', 'items', 'concept', 'concepts',
            'state', 'states', 'data', 'info', 'information', 'value', 'values', 'lot', 'lots',
            'bit', 'bits', 'day', 'days', 'time', 'times', 'year', 'years', 'week', 'weeks',
            'part', 'parts', 'case', 'cases', 'type', 'types', 'category', 'categories', 'cat'
        ]
        cfg_exclusions = self.config.get('synthesis', {}).get('generic_exclusions', default_exclusions)
        GENERIC_EXCLUSIONS = {str(x).lower().strip() for x in cfg_exclusions}

        def extract_top_domain_terms(target_nodes: List[str], triplets: List[dict], top_k: int = 12) -> List[str]:
            target_nodes_norm = {str(n).lower().strip(): str(n) for n in target_nodes}
            local_deg = Counter()
            for tr in triplets:
                s = str(tr.get('subject', '')).lower().strip()
                o = str(tr.get('object', '')).lower().strip()
                if s in target_nodes_norm:
                    local_deg[s] += 1
                if o in target_nodes_norm:
                    local_deg[o] += 1

            scored = []
            for nk, orig in target_nodes_norm.items():
                if nk in GENERIC_EXCLUSIONS or len(nk) <= 2 or nk.isdigit() or not any(c.isalnum() for c in nk):
                    continue
                n_targets = global_target_counts.get(nk, 1)
                deg = local_deg.get(nk, 0)
                multi_word_boost = 1.2 if ' ' in nk else 1.0
                score = ((deg + 1.0) / (n_targets ** 1.5)) * multi_word_boost
                scored.append((score, deg, orig))

            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Deduplicate near-duplicate stems (e.g. singular/plural)
            seen_stems = set()
            cleaned_terms = []
            for _, _, orig in scored:
                base = orig.lower().strip()
                stem = base[:-1] if base.endswith('s') else base
                if stem in seen_stems or base in seen_stems:
                    continue
                seen_stems.add(stem)
                seen_stems.add(base)
                cleaned_terms.append(orig)
                if len(cleaned_terms) >= top_k:
                    break
            return cleaned_terms

        def extract_thematic_observations(triplets: List[dict], target_nodes: List[str], max_sentences: int = 60, mode: str = "raw_text") -> Dict[str, List[str]]:
            target_nodes_norm = {str(n).lower().strip() for n in target_nodes}
            quote_info = {} # quote -> {'theme': str, 'fidelities': list, 'svos': list}

            for tr in triplets:
                subj = str(tr.get('subject', '')).strip()
                pred = str(tr.get('predicate', '')).strip()
                obj = str(tr.get('object', '')).strip()
                svo = f"{subj} {pred} {obj}." if (subj and pred and obj) else ""
                
                quote = str(tr.get('source_quote', '')).strip()
                if not quote or len(quote) < min_sentence_length:
                    if mode in ["triples", "both"] and svo:
                        quote = svo
                    else:
                        continue

                theme = tr.get('theme_association', 'General Domain Observations') or 'General Domain Observations'
                fid = float(tr.get('contextual_fidelity', 0.50))

                if quote not in quote_info:
                    quote_info[quote] = {'theme': theme, 'fidelities': [fid], 'svos': [svo] if svo else []}
                else:
                    quote_info[quote]['fidelities'].append(fid)
                    if svo and svo not in quote_info[quote]['svos']:
                        quote_info[quote]['svos'].append(svo)

            # Score each sentence
            scored_quotes = []
            for q, info in quote_info.items():
                avg_fid = sum(info['fidelities']) / len(info['fidelities'])
                q_lower = q.lower()
                node_overlap = sum(1 for n in target_nodes_norm if n in q_lower)
                sentence_score = avg_fid * (1.0 + 0.25 * min(4, node_overlap))
                
                # Format text item based on mode
                if mode == "triples":
                    text_item = " ".join(info['svos']) if info['svos'] else q
                elif mode == "both":
                    svo_summary = f" [Facts: {'; '.join(info['svos'][:2])}]" if info['svos'] else ""
                    text_item = f"{q}{svo_summary}"
                else: # raw_text
                    text_item = q
                    
                scored_quotes.append((sentence_score, text_item, info['theme']))

            scored_quotes.sort(key=lambda x: x[0], reverse=True)
            selected = scored_quotes[:max_sentences]

            theme_groups = defaultdict(list)
            for _, text_item, theme in selected:
                theme_groups[theme].append(text_item)

            return dict(theme_groups)

        def format_phase2_payload(target_id: Any, target_nodes: List[str], subset: List[dict], mode: str) -> str:
            top_terms = extract_top_domain_terms(target_nodes, subset, top_k=top_domain_terms_count)
            observations = extract_thematic_observations(subset, target_nodes, max_sentences=max_sentences_per_target, mode=mode)

            payload_dict = {
                "community_id": target_id,
                "top_domain_terms": top_terms,
                "thematic_observations": observations
            }
            return json.dumps(payload_dict, indent=2)

        if use_async:
            print(f"      -> Executing Phase 2 concurrently with max_async_calls: {max_async}")
            
            async def run_single_pass(subset, is_normalized, i, target_type, target_id, sem, target_nodes):
                async with sem:
                    pass_name = "normalized" if is_normalized else "raw"
                    target_dir = norm_dir if is_normalized else raw_dir
                    print(f"         -> API call for Target {i}/{len(targets)} ({target_type} {target_id}) - {pass_name} (Size: {len(target_nodes)} nodes, {len(subset)} triplets)...")
                    user_payload = format_phase2_payload(target_id, list(target_nodes), subset, payload_mode)
                    active_sys_prompt = synth_prompts.LEIDEN_SCHEMA_SYNTHESIS_PROMPT
                    save_payload_file(f"phase2_target_{i:02d}_{pass_name}_{target_id}_payload.txt", f"=== SYSTEM PROMPT ===\n{active_sys_prompt}\n\n=== USER PAYLOAD ===\n{user_payload}")
                    last_llm_responses.set([])
                    from pydantic_ai import capture_run_messages
                    with capture_run_messages() as messages:
                        try:
                            res = await active_agent.run(user_payload)
                            filename = f"{i:02d}_{res.output.module_name}.py"
                            with open(target_dir / filename, "w") as f:
                                f.write(clean_python_code(res.output.source_code))
                        except Exception as e:
                            # Reformat attempt (3rd attempt after 2 failures)
                            malformed_text = extract_malformed_text(messages, e)
                            print(f"            [Warning] Initial {pass_name} extraction failed for Target {i} after 2 tries. Running Schema Reformat Agent...")
                            
                            reformat_prompt = (
                                f"Original Schema Generation Task context was:\n{user_payload}\n\n"
                                f"The malformed Python output from the failed attempts was:\n{malformed_text}\n\n"
                                f"The parsing validation error that occurred was:\n{str(e)}\n\n"
                                f"Please correct and repair this Python code so it strictly maps to the schemas.GeneratedModule structure. "
                                f"The output must consist strictly of the Pydantic GeneratedModule schema (with 'module_name' and 'source_code') "
                                f"and must contain no conversational text, notes, markdown code blocks, or other extraneous words. Focus strictly on fixing the validation error."
                            )
                            try:
                                res = await schema_reformat_agent.run(reformat_prompt)
                                filename = f"{i:02d}_{res.output.module_name}.py"
                                with open(target_dir / filename, "w") as f:
                                    f.write(clean_python_code(res.output.source_code))
                                print(f"            -> Reformat successful for Target {i}!")
                            except Exception as reformat_err:
                                log_detailed_synthesis_error(reformat_err, last_llm_responses.get(), pass_name, f"{target_type} {target_id}", i)

            async def run_phase2_async():
                sem = asyncio.Semaphore(max_async)
                tasks = []
                
                for i, target in enumerate(targets, start=1):
                    target_nodes = set(target["nodes"])
                    target_type = target.get("target_type", target.get("type", "spoke"))
                    target_id = target.get("target_id", target.get("id"))
                    
                    if run_norm:
                        norm_subset = target.get("triplets", [])
                        if norm_subset:
                            tasks.append(run_single_pass(norm_subset, True, i, target_type, target_id, sem, target_nodes))
                    
                    if run_raw:
                        raw_subset = list(target.get("triplets", []))
                        if original_triplets:
                            extra_raw = [t for t in original_triplets if str(t.get('subject','')).lower().strip() in target_nodes or str(t.get('object','')).lower().strip() in target_nodes]
                            existing_quotes = {tr.get('source_quote') for tr in raw_subset if tr.get('source_quote')}
                            for extra in extra_raw:
                                q = extra.get('source_quote')
                                if q and q not in existing_quotes:
                                    raw_subset.append(extra)
                                    existing_quotes.add(q)
                        if raw_subset:
                            tasks.append(run_single_pass(raw_subset, False, i, target_type, target_id, sem, target_nodes))
                
                if tasks:
                    await asyncio.gather(*tasks)

            asyncio.run(run_phase2_async())
        else:
            def run_single_pass_sync(subset, is_normalized, i, target_type, target_id, target_nodes):
                pass_name = "normalized" if is_normalized else "raw"
                target_dir = norm_dir if is_normalized else raw_dir
                print(f"         -> API call for Target {i}/{len(targets)} ({target_type} {target_id}) - {pass_name} (Size: {len(target_nodes)} nodes, {len(subset)} triplets)...")
                user_payload = format_phase2_payload(target_id, list(target_nodes), subset, payload_mode)
                active_sys_prompt = synth_prompts.LEIDEN_SCHEMA_SYNTHESIS_PROMPT
                save_payload_file(f"phase2_target_{i:02d}_{pass_name}_{target_id}_payload.txt", f"=== SYSTEM PROMPT ===\n{active_sys_prompt}\n\n=== USER PAYLOAD ===\n{user_payload}")
                last_llm_responses.set([])
                from pydantic_ai import capture_run_messages
                with capture_run_messages() as messages:
                    try:
                        res = active_agent.run_sync(user_payload)
                        filename = f"{i:02d}_{res.output.module_name}.py"
                        with open(target_dir / filename, "w") as f:
                            f.write(clean_python_code(res.output.source_code))
                    except Exception as e:
                        # Reformat attempt (3rd attempt after 2 failures)
                        malformed_text = extract_malformed_text(messages, e)
                        print(f"            [Warning] Initial {pass_name} extraction failed for Target {i} after 2 tries. Running Schema Reformat Agent...")
                        
                        reformat_prompt = (
                            f"Original Schema Generation Task context was:\n{user_payload}\n\n"
                            f"The malformed Python output from the failed attempts was:\n{malformed_text}\n\n"
                            f"The parsing validation error that occurred was:\n{str(e)}\n\n"
                            f"Please correct and repair this Python code so it strictly maps to the schemas.GeneratedModule structure. "
                            f"The output must consist strictly of the Pydantic GeneratedModule schema (with 'module_name' and 'source_code') "
                            f"and must contain no conversational text, notes, markdown code blocks, or other extraneous words. Focus strictly on fixing the validation error."
                        )
                        try:
                            res = schema_reformat_agent.run_sync(reformat_prompt)
                            filename = f"{i:02d}_{res.output.module_name}.py"
                            with open(target_dir / filename, "w") as f:
                                f.write(clean_python_code(res.output.source_code))
                            print(f"            -> Reformat successful for Target {i}!")
                        except Exception as reformat_err:
                            log_detailed_synthesis_error(reformat_err, last_llm_responses.get(), pass_name, f"{target_type} {target_id}", i)

            for i, target in enumerate(targets, start=1):
                target_nodes = set(target["nodes"])
                target_type = target.get("target_type", target.get("type", "spoke"))
                target_id = target.get("target_id", target.get("id"))
                
                if run_norm:
                    norm_subset = target.get("triplets", [])
                    if norm_subset:
                        run_single_pass_sync(norm_subset, True, i, target_type, target_id, target_nodes)
                        
                if run_raw:
                    raw_subset = list(target.get("triplets", []))
                    if original_triplets:
                        extra_raw = [t for t in original_triplets if str(t.get('subject','')).lower().strip() in target_nodes or str(t.get('object','')).lower().strip() in target_nodes]
                        existing_quotes = {tr.get('source_quote') for tr in raw_subset if tr.get('source_quote')}
                        for extra in extra_raw:
                            q = extra.get('source_quote')
                            if q and q not in existing_quotes:
                                raw_subset.append(extra)
                                existing_quotes.add(q)
                    if raw_subset:
                        run_single_pass_sync(raw_subset, False, i, target_type, target_id, target_nodes)

        # Phase 3: Global Consolidation
        print(f"   -> [{path_label}] Phase 3: Consolidation...")
        
        from src.utils.token_helper import estimate_tokens
        import re

        def strip_comments_and_docstrings(code: str) -> str:
            code = re.sub(r'#.*', '', code)
            code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
            code = re.sub(r'\n\s*\n', '\n', code)
            return code.strip()

        synthesis_cap = self.config.get('synthesis', {}).get('context_window_cap', 16384)
        sys_tokens = estimate_tokens(synth_prompts.CONSOLIDATION_SYSTEM_PROMPT)
        max_payload_tokens = max(500, synthesis_cap - sys_tokens - 1000)

        async def run_consolidation_agent_async(payload: str) -> str:
            result = await consolidation_agent.run(payload)
            return clean_python_code(result.output.source_code)

        def run_consolidation_agent_sync(payload: str) -> str:
            result = consolidation_agent.run_sync(payload)
            return clean_python_code(result.output.source_code)

        async def process_consolidation_async(files_list: List[dict], schema_dir: Path) -> str:
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            save_payload_file(f"phase3_consolidation_{schema_dir.name}_payload.txt", f"=== SYSTEM PROMPT ===\n{synth_prompts.CONSOLIDATION_SYSTEM_PROMPT}\n\n=== USER PAYLOAD ===\n{payload}")
            if estimate_tokens(payload) <= max_payload_tokens:
                return await run_consolidation_agent_async(payload)
                
            for f in files_list:
                f["content"] = strip_comments_and_docstrings(f["content"])
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            if estimate_tokens(payload) <= max_payload_tokens:
                return await run_consolidation_agent_async(payload)
                
            sub_batches, current_batch = [], []
            for f in files_list:
                test_batch = current_batch + [f]
                test_payload = "\n".join([f"--- File: {x['name']} ---\n{x['content']}\n" for x in test_batch])
                if estimate_tokens(test_payload) > max_payload_tokens and current_batch:
                    sub_batches.append(current_batch)
                    current_batch = [f]
                else:
                    current_batch.append(f)
            if current_batch:
                sub_batches.append(current_batch)
                
            intermediate_results = []
            for idx, batch in enumerate(sub_batches, start=1):
                res = await process_consolidation_async(batch, schema_dir)
                intermediate_results.append({"name": f"intermediate_{idx}.py", "content": res})
                
            return await process_consolidation_async(intermediate_results, schema_dir)

        def process_consolidation_sync(files_list: List[dict], schema_dir: Path) -> str:
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            save_payload_file(f"phase3_consolidation_{schema_dir.name}_payload.txt", f"=== SYSTEM PROMPT ===\n{synth_prompts.CONSOLIDATION_SYSTEM_PROMPT}\n\n=== USER PAYLOAD ===\n{payload}")
            if estimate_tokens(payload) <= max_payload_tokens:
                return run_consolidation_agent_sync(payload)
                
            for f in files_list:
                f["content"] = strip_comments_and_docstrings(f["content"])
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            if estimate_tokens(payload) <= max_payload_tokens:
                return run_consolidation_agent_sync(payload)
                
            sub_batches, current_batch = [], []
            for f in files_list:
                test_batch = current_batch + [f]
                test_payload = "\n".join([f"--- File: {x['name']} ---\n{x['content']}\n" for x in test_batch])
                if estimate_tokens(test_payload) > max_payload_tokens and current_batch:
                    sub_batches.append(current_batch)
                    current_batch = [f]
                else:
                    current_batch.append(f)
            if current_batch:
                sub_batches.append(current_batch)
                
            intermediate_results = []
            for idx, batch in enumerate(sub_batches, start=1):
                res = process_consolidation_sync(batch, schema_dir)
                intermediate_results.append({"name": f"intermediate_{idx}.py", "content": res})
                
            return process_consolidation_sync(intermediate_results, schema_dir)

        def consolidate_schemas(schema_dir: Path):
            py_files = sorted([f for f in schema_dir.glob("*.py") if f.name not in ("master_ontology.py", "enums.py", "__init__.py")])
            if not py_files:
                return
                
            files_list = []
            for pf in py_files:
                with open(pf, "r") as f:
                    files_list.append({"name": pf.name, "content": f.read()})
                    
            if use_async:
                try:
                    master_code = asyncio.run(process_consolidation_async(files_list, schema_dir))
                    with open(schema_dir / "master_ontology.py", "w") as f:
                        f.write(master_code)
                except Exception as e:
                    log_error(f"[Synthesis {path_type}] Error generating master_ontology for {schema_dir.name}: {e}")
            else:
                try:
                    master_code = process_consolidation_sync(files_list, schema_dir)
                    with open(schema_dir / "master_ontology.py", "w") as f:
                        f.write(master_code)
                except Exception as e:
                    log_error(f"[Synthesis {path_type}] Error generating master_ontology for {schema_dir.name}: {e}")

        if run_norm:
            consolidate_schemas(norm_dir)
        if run_raw:
            consolidate_schemas(raw_dir)

        # Phase 4: Path Comprehensive Ontology
        print(f"   -> [{path_label}] Phase 4: Master Ontology Synthesis...")
        last_llm_responses.set([])
        from pydantic_ai import capture_run_messages
        with capture_run_messages() as messages:
            try:
                norm_master = open(norm_dir / "master_ontology.py").read() if run_norm and (norm_dir / "master_ontology.py").exists() else ""
                raw_master = open(raw_dir / "master_ontology.py").read() if run_raw and (raw_dir / "master_ontology.py").exists() else ""
                
                enums_code = ""
                if run_norm and (norm_dir / "enums.py").exists():
                    enums_code = open(norm_dir / "enums.py").read()
                elif run_raw and (raw_dir / "enums.py").exists():
                    enums_code = open(raw_dir / "enums.py").read()
                    
                final_payload = f"--- ENUMS ---\n{enums_code}\n\n--- RAW MASTER ONTOLOGY ---\n{raw_master}\n\n--- NORMALIZED MASTER ONTOLOGY ---\n{norm_master}"
                save_payload_file("phase4_comprehensive_ontology_payload.txt", f"=== SYSTEM PROMPT ===\n{synth_prompts.FINAL_ONTOLOGY_SYSTEM_PROMPT}\n\n=== USER PAYLOAD ===\n{final_payload}")
                final_res = comprehensive_ontology_agent.run_sync(final_payload)
                final_code = clean_python_code(final_res.output.source_code)
            except Exception as e:
                malformed_text = extract_malformed_text(messages, e)
                print(f"      [Warning] Initial comprehensive ontology synthesis failed for {path_type}. Running Schema Reformat Agent...")
                reformat_prompt = (
                    f"Original Comprehensive Ontology Task payload was:\n{final_payload[:3000]}\n\n"
                    f"The malformed output from the failed attempts was:\n{malformed_text}\n\n"
                    f"The parsing validation error that occurred was:\n{str(e)}\n\n"
                    f"Please correct and repair this Python code so it strictly maps to the schemas.GeneratedModule structure. "
                    f"Set module_name to 'comprehensive_ontology' and source_code to the consolidated Python ontology code."
                )
                try:
                    ref_res = schema_reformat_agent.run_sync(reformat_prompt)
                    final_code = clean_python_code(ref_res.output.source_code)
                    print(f"      -> Reformat successful for comprehensive_ontology ({path_type})!")
                except Exception as ref_err:
                    log_error(f"[Synthesis {path_type}] Error generating comprehensive_ontology: {ref_err}")
                    final_code = "# Error generating comprehensive ontology"

        with open(output_base_dir / "comprehensive_ontology.py", "w") as f:
            f.write(final_code)

        # Finalization
        if run_norm:
            with open(norm_dir / "__init__.py", "w") as f:
                f.write("# Normalized Schemas")
            try:
                subprocess.run(["ruff", "format", str(norm_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on norm: {e}")
                
        if run_raw:
            with open(raw_dir / "__init__.py", "w") as f:
                f.write("# Raw Schemas")
            try:
                subprocess.run(["ruff", "format", str(raw_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on raw: {e}")

        try:
            subprocess.run(["ruff", "format", str(output_base_dir / "comprehensive_ontology.py")], capture_output=True, text=True, check=False)
        except Exception as e:
            log_error(f"[Synthesis {path_type}] Error running ruff on comprehensive: {e}")

        print(f"   [OK] [{path_label}] Synthesis complete: {output_base_dir}")
