import json
import os
import subprocess
import shutil
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

from src.config import settings
from src.agents.synthesis_agents import (
    OrphanContext, SynthesisContext,
    orphan_agent, leiden_schema_agent, node2vec_schema_agent,
    consolidation_agent, comprehensive_ontology_agent,
    last_llm_responses
)

def clean_python_code(code: str) -> str:
    """Removes markdown formatting and escapes that LLMs often inject."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    # Fix over-escaped quotes caused by LLM JSON serialization 
    code = code.replace('\\"', '"')
    
    # Specific fix to prevent un-escaping the one known problematic Enum string
    code = code.replace('"the "firstthen" principle"', '\'the "firstthen" principle\'')
    
    return code.strip()

class SynthesisPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def execute(self, comm_topology: Dict[str, Any] = None, emb_topology: Dict[str, Any] = None, refined_triplets: List[Dict[str, Any]] = None, original_triplets: List[Dict[str, Any]] = None, taxonomic_map: Dict[str, str] = None, master_themes: List[str] = None):
        print("[Synthesis] Starting Stage 4 Pipeline...")
        synth_mode = self.config.get('synthesis', {}).get('execution_mode', 'community')
        print(f"   -> Synthesis Execution Mode: '{synth_mode}'")
        
        if synth_mode in ("both", "community"):
            if comm_topology:
                print("   ==> Executing Path 1: Community Path Synthesis...")
                self.execute_path_synthesis(comm_topology, "community", refined_triplets, original_triplets, master_themes)
            else:
                print("   [Warning] Community topology partition not found. Skipping Path 1 synthesis.")

        if synth_mode in ("both", "embedding"):
            if emb_topology:
                print("   ==> Executing Path 2: Embedding Path Synthesis...")
                self.execute_path_synthesis(emb_topology, "embedding", refined_triplets, original_triplets, master_themes)
            else:
                print("   [Warning] Embedding topology partition not found. Skipping Path 2 synthesis.")

        print("[Synthesis] Stage 4 Pipeline complete.")

    def execute_path_synthesis(self, topology: Dict[str, Any], path_type: str, refined_triplets: List[dict], original_triplets: List[dict], master_themes: List[str]):
        path_label = "Path 1: Community Workflow" if path_type == "community" else "Path 2: Embedding Categorical"
        print(f"   -> Starting Synthesis Pass for [{path_label}]...")

        # Read config options
        schema_pass_mode = self.config.get('synthesis', {}).get('schema_pass_mode', 'both')
        min_cluster_size = self.config.get('synthesis', {}).get('min_cluster_size', 5)
        run_norm = schema_pass_mode in ("both", "normalized")
        run_raw = schema_pass_mode in ("both", "raw")
        print(f"      -> Schema Pass Mode: '{schema_pass_mode}' (Normalized: {run_norm}, Raw: {run_raw})")
        print(f"      -> Entity Threshold (min_cluster_size): {min_cluster_size}")

        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"synthesis_{path_type}_errors.log"
        
        def log_error(msg: str):
            print(msg)
            with open(log_file, "a") as lf:
                lf.write(f"[{datetime.now().isoformat()}] {msg}\n")
        
        # Path-isolated Output directories
        output_base_dir = Path("outputs/schemas") / path_type
        norm_dir = output_base_dir / "normalized"
        raw_dir = output_base_dir / "raw"
        
        if output_base_dir.exists():
            shutil.rmtree(output_base_dir)
            
        output_base_dir.mkdir(parents=True, exist_ok=True)
        if run_norm:
            norm_dir.mkdir(parents=True, exist_ok=True)
        if run_raw:
            raw_dir.mkdir(parents=True, exist_ok=True)

        # Resolve Hubs and Filter Count based on Centrality Metrics
        raw_hubs = topology.get("global_hubs", [])
        max_hub_targets = self.config.get('synthesis', {}).get('max_hub_targets', 10)
        metrics = topology.get("node_metrics", {})
        
        def get_hub_sort_key(node_id):
            node_m = metrics.get(node_id, {})
            return (node_m.get("betweenness_centrality", 0.0), node_m.get("degree_centrality", 0.0))
            
        sorted_hubs = sorted(raw_hubs, key=get_hub_sort_key, reverse=True)
        hubs = sorted_hubs[:max_hub_targets]
        discarded_hubs = sorted_hubs[max_hub_targets:]

        # Phase 1: Orphan & Low-Density Node Aggregation (Enums)
        print(f"   -> [{path_label}] Phase 1: Orphan & Low-Density Node Aggregation (Enums)...")
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
        print(f"      -> Total Enum Nodes: {len(all_enum_nodes)} ({len(orphans)} orphans + {len(small_cluster_nodes)} from low-density clusters + {len(discarded_hubs)} discarded hubs < {min_cluster_size} nodes)")
        global_enums_code = ""
        
        if all_enum_nodes:
            orphan_ctx = OrphanContext(master_themes=master_themes)
            payload = json.dumps(all_enum_nodes)
            last_llm_responses.set([])
            try:
                result = orphan_agent.run_sync(payload, deps=orphan_ctx)
                global_enums_code = clean_python_code(result.output.source_code)
                
                if run_norm:
                    with open(norm_dir / "enums.py", "w") as f:
                        f.write(global_enums_code)
                if run_raw:
                    with open(raw_dir / "enums.py", "w") as f:
                        f.write(global_enums_code)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error generating enums: {e}")
                attempts = last_llm_responses.get()
                if attempts:
                    log_error(f"[Synthesis {path_type}] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")
                global_enums_code = "# Error generating enums"

        # Phase 2: Schema Generation
        print(f"   -> [{path_label}] Phase 2: Schema Generation...")
        
        # Consolidate Targets (>= min_cluster_size)
        targets = []
        
        if path_type == "embedding":
            active_agent = node2vec_schema_agent
            for cluster in topology.get("structural_clusters", []):
                nodes_in_cluster = cluster.get("nodes", [])
                if len(nodes_in_cluster) >= min_cluster_size:
                    targets.append({"type": "structural_cluster", "id": cluster.get("cluster_id"), "nodes": nodes_in_cluster})
            print(f"      -> Qualified Structural Clusters (>= {min_cluster_size} nodes): {len(targets)} targets")
        else:
            active_agent = leiden_schema_agent
            for comm in topology.get("communities", []):
                nodes_in_comm = comm.get("nodes", [])
                if len(nodes_in_comm) >= min_cluster_size:
                    targets.append({"type": "community", "id": comm.get("community_id"), "nodes": nodes_in_comm})
            print(f"      -> Qualified Communities (>= {min_cluster_size} nodes): {len(targets)} targets")
            
        targets.sort(key=lambda x: len(x["nodes"]), reverse=True)
        spoke_count = len(targets)
        
        for hub in hubs:
            targets.append({"type": "hub", "id": hub, "nodes": [hub]})
            
        print(f"      -> Target Composition: {len(targets)} total targets ({spoke_count} spokes + {len(hubs)} global hubs)")
            
        inheritance_map = topology.get("theme_inheritance", [])
        synth_ctx = SynthesisContext(
            global_enums=global_enums_code,
            theme_inheritance=json.dumps(inheritance_map, indent=2)
        )

        use_async = self.config.get('pipeline', {}).get('use_async', False)
        max_async = self.config.get('synthesis', {}).get('max_async_calls', 1)
        synthesis_cap = self.config.get('synthesis', {}).get('context_window_cap', 16384)

        if use_async:
            print(f"      -> Executing Phase 2 concurrently with max_async_calls: {max_async}")
            
            async def run_single_pass(subset, is_normalized, i, target_type, target_id, sem, node_count):
                async with sem:
                    pass_name = "normalized" if is_normalized else "raw"
                    target_dir = norm_dir if is_normalized else raw_dir
                    print(f"         -> API call for Target {i}/{len(targets)} ({target_type} {target_id}) - {pass_name} (Size: {node_count} nodes, {len(subset)} triplets)...")
                    last_llm_responses.set([])
                    try:
                        res = await active_agent.run(json.dumps(subset), deps=synth_ctx)
                        filename = f"{i:02d}_{res.output.module_name}.py"
                        with open(target_dir / filename, "w") as f:
                            f.write(clean_python_code(res.output.source_code))
                    except Exception as e:
                        log_error(f"[Synthesis {path_type}] Error generating {pass_name} schema {i} ({target_type} {target_id}): {e}")

            async def run_phase2_async():
                sem = asyncio.Semaphore(max_async)
                tasks = []
                max_triplets_cap = self.config.get('synthesis', {}).get('max_triplets_per_target', 1000)
                
                for i, target in enumerate(targets, start=1):
                    target_nodes = set(target["nodes"])
                    
                    # Find all triplet indices associated with this target's nodes
                    matching_indices = []
                    for idx, refined_t in enumerate(refined_triplets):
                        subj = str(refined_t.get('subject', '')).lower().strip()
                        obj = str(refined_t.get('object', '')).lower().strip()
                        if subj in target_nodes or obj in target_nodes:
                            matching_indices.append(idx)
                            
                    # Prune matching indices if they exceed the cap based on centrality PageRank and intra-community status
                    if len(matching_indices) > max_triplets_cap:
                        metrics = topology.get("node_metrics", {})
                        scored_indices = []
                        for idx in matching_indices:
                            refined_t = refined_triplets[idx]
                            subj = str(refined_t.get('subject', '')).lower().strip()
                            obj = str(refined_t.get('object', '')).lower().strip()
                            
                            subj_pr = metrics.get(subj, {}).get("pagerank", 0.0)
                            obj_pr = metrics.get(obj, {}).get("pagerank", 0.0)
                            base_score = subj_pr + obj_pr
                            
                            # Prioritize intra-community connections
                            if (subj in target_nodes) and (obj in target_nodes):
                                base_score += 1.0
                                
                            scored_indices.append((base_score, idx))
                            
                        scored_indices.sort(key=lambda x: x[0], reverse=True)
                        matching_indices = [idx for _, idx in scored_indices[:max_triplets_cap]]
                    
                    if run_norm:
                        norm_subset = [refined_triplets[idx] for idx in matching_indices]
                        if norm_subset:
                            tasks.append(run_single_pass(norm_subset, True, i, target["type"], target["id"], sem, len(target_nodes)))
                    
                    if run_raw:
                        raw_subset = [original_triplets[idx] for idx in matching_indices if idx < len(original_triplets)]
                        if raw_subset:
                            tasks.append(run_single_pass(raw_subset, False, i, target["type"], target["id"], sem, len(target_nodes)))
                
                if tasks:
                    await asyncio.gather(*tasks)

            asyncio.run(run_phase2_async())
        else:
            max_triplets_cap = self.config.get('synthesis', {}).get('max_triplets_per_target', 1000)
            for i, target in enumerate(targets, start=1):
                target_nodes = set(target["nodes"])
                node_count = len(target_nodes)
                
                # Find all triplet indices associated with this target's nodes
                matching_indices = []
                for idx, refined_t in enumerate(refined_triplets):
                    subj = str(refined_t.get('subject', '')).lower().strip()
                    obj = str(refined_t.get('object', '')).lower().strip()
                    if subj in target_nodes or obj in target_nodes:
                        matching_indices.append(idx)
                        
                # Prune matching indices if they exceed the cap based on centrality PageRank and intra-community status
                if len(matching_indices) > max_triplets_cap:
                    metrics = topology.get("node_metrics", {})
                    scored_indices = []
                    for idx in matching_indices:
                        refined_t = refined_triplets[idx]
                        subj = str(refined_t.get('subject', '')).lower().strip()
                        obj = str(refined_t.get('object', '')).lower().strip()
                        
                        subj_pr = metrics.get(subj, {}).get("pagerank", 0.0)
                        obj_pr = metrics.get(obj, {}).get("pagerank", 0.0)
                        base_score = subj_pr + obj_pr
                        
                        # Prioritize intra-community connections
                        if (subj in target_nodes) and (obj in target_nodes):
                            base_score += 1.0
                            
                        scored_indices.append((base_score, idx))
                        
                    scored_indices.sort(key=lambda x: x[0], reverse=True)
                    matching_indices = [idx for _, idx in scored_indices[:max_triplets_cap]]
                
                if run_norm:
                    norm_subset = [refined_triplets[idx] for idx in matching_indices]
                    if norm_subset:
                        print(f"         -> API call for Target {i}/{len(targets)} ({target['type']} {target['id']}) - normalized (Size: {node_count} nodes, {len(norm_subset)} triplets)...")
                        last_llm_responses.set([])
                        try:
                            norm_res = active_agent.run_sync(json.dumps(norm_subset), deps=synth_ctx)
                            filename = f"{i:02d}_{norm_res.output.module_name}.py"
                            with open(norm_dir / filename, "w") as f:
                                f.write(clean_python_code(norm_res.output.source_code))
                        except Exception as e:
                            log_error(f"[Synthesis {path_type}] Error generating normalized schema {i} ({target['type']} {target['id']}): {e}")

                if run_raw:
                    raw_subset = [original_triplets[idx] for idx in matching_indices if idx < len(original_triplets)]
                    if raw_subset:
                        print(f"         -> API call for Target {i}/{len(targets)} ({target['type']} {target['id']}) - raw (Size: {node_count} nodes, {len(raw_subset)} triplets)...")
                        last_llm_responses.set([])
                        try:
                            raw_res = active_agent.run_sync(json.dumps(raw_subset), deps=synth_ctx)
                            filename = f"{i:02d}_{raw_res.output.module_name}.py"
                            with open(raw_dir / filename, "w") as f:
                                f.write(clean_python_code(raw_res.output.source_code))
                        except Exception as e:
                            log_error(f"[Synthesis {path_type}] Error generating raw schema {i} ({target['type']} {target['id']}): {e}")

        # Phase 3: Global Consolidation
        print(f"   -> [{path_label}] Phase 3: Consolidation...")
        
        from src.utils.token_helper import estimate_tokens
        from src.synthesis import prompts as synth_prompts
        import re

        def strip_comments_and_docstrings(code: str) -> str:
            code = re.sub(r'#.*', '', code)
            code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
            code = re.sub(r'\n\s*\n', '\n', code)
            return code.strip()

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
        try:
            norm_master = open(norm_dir / "master_ontology.py").read() if run_norm and (norm_dir / "master_ontology.py").exists() else ""
            raw_master = open(raw_dir / "master_ontology.py").read() if run_raw and (raw_dir / "master_ontology.py").exists() else ""
            
            enums_code = ""
            if run_norm and (norm_dir / "enums.py").exists():
                enums_code = open(norm_dir / "enums.py").read()
            elif run_raw and (raw_dir / "enums.py").exists():
                enums_code = open(raw_dir / "enums.py").read()
                
            final_payload = f"--- ENUMS ---\n{enums_code}\n\n--- RAW MASTER ONTOLOGY ---\n{raw_master}\n\n--- NORMALIZED MASTER ONTOLOGY ---\n{norm_master}"
            final_res = comprehensive_ontology_agent.run_sync(final_payload)
            final_code = clean_python_code(final_res.output.source_code)
            
            with open(output_base_dir / "comprehensive_ontology.py", "w") as f:
                f.write(final_code)
                
        except Exception as e:
            log_error(f"[Synthesis {path_type}] Error generating comprehensive_ontology: {e}")

        # Finalization
        if run_norm:
            with open(norm_dir / "__init__.py", "w") as f: f.write("# Normalized Schemas")
            try:
                subprocess.run(["ruff", "format", str(norm_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on norm: {e}")
                
        if run_raw:
            with open(raw_dir / "__init__.py", "w") as f: f.write("# Raw Schemas")
            try:
                subprocess.run(["ruff", "format", str(raw_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on raw: {e}")

        try:
            subprocess.run(["ruff", "format", str(output_base_dir / "comprehensive_ontology.py")], capture_output=True, text=True, check=False)
        except Exception as e:
            log_error(f"[Synthesis {path_type}] Error running ruff on comprehensive: {e}")

        print(f"   [OK] [{path_label}] Synthesis complete: {output_base_dir}")
