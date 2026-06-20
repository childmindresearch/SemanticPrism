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

    def execute(self, topology: Dict[str, Any], refined_triplets: List[Dict[str, Any]], original_triplets: List[Dict[str, Any]], taxonomic_map: Dict[str, str], master_themes: List[str]):
        print("[Synthesis] Starting Stage 4 Pipeline...")
        
        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "synthesis_errors.log"
        
        def log_error(msg: str):
            print(msg)
            with open(log_file, "a") as lf:
                lf.write(f"[{datetime.now().isoformat()}] {msg}\n")
        
        # Output directories
        norm_dir = Path("outputs/schemas/normalized")
        raw_dir = Path("outputs/schemas/raw")
        
        # Clear existing directories to prevent overlap from previous runs
        if norm_dir.exists():
            shutil.rmtree(norm_dir)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            
        norm_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Orphan Aggregation (Enums)
        print("   -> Running Phase 1: Orphan Aggregation (Enums)...")
        orphans = topology.get("orphans", [])
        global_enums_code = ""
        
        if orphans:
            orphan_ctx = OrphanContext(master_themes=master_themes)
            payload = json.dumps(orphans)
            last_llm_responses.set([])
            try:
                result = orphan_agent.run_sync(payload, deps=orphan_ctx)
                global_enums_code = clean_python_code(result.output.source_code)
                
                with open(norm_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)
                with open(raw_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)
            except Exception as e:
                log_error(f"[Synthesis] Error generating enums: {e}")
                attempts = last_llm_responses.get()
                if attempts:
                    log_error(f"[Synthesis] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")
                global_enums_code = "# Error generating enums"

        # Phase 2: Dual-Pass Schema Generation
        print("   -> Running Phase 2: Dual-Pass Schema Generation...")
        
        # Consolidate Targets
        hubs = topology.get("global_hubs", [])
        
        targets = []
        clustering_strategy = self.config.get('synthesis', {}).get('clustering_strategy', 'leiden')
        min_cluster_size = self.config.get('synthesis', {}).get('min_cluster_size', 3)
        
        if clustering_strategy == 'node2vec':
            active_agent = node2vec_schema_agent
            for cluster in topology.get("structural_clusters", []):
                nodes_in_cluster = cluster.get("nodes", [])
                if len(nodes_in_cluster) >= min_cluster_size:
                    targets.append({"type": "structural_cluster", "id": cluster.get("cluster_id"), "nodes": nodes_in_cluster})
            print(f"   -> Using Node2Vec Structural Clusters ({len(targets)} targets)")
        else:
            active_agent = leiden_schema_agent
            for comm in topology.get("communities", []):
                nodes_in_comm = comm.get("nodes", [])
                if len(nodes_in_comm) >= min_cluster_size:
                    targets.append({"type": "community", "id": comm.get("community_id"), "nodes": nodes_in_comm})
            print(f"   -> Using Leiden Communities ({len(targets)} targets)")
            
        # Sort targets (Descending size)
        targets.sort(key=lambda x: len(x["nodes"]), reverse=True)
        
        for hub in hubs:
            targets.append({"type": "hub", "id": hub, "nodes": [hub]})
            
        inheritance_map = topology.get("theme_inheritance", [])
        synth_ctx = SynthesisContext(
            global_enums=global_enums_code,
            theme_inheritance=json.dumps(inheritance_map, indent=2)
        )

        use_async = self.config.get('pipeline', {}).get('use_async', False)
        max_async = self.config.get('synthesis', {}).get('max_async_calls', 1)
        synthesis_cap = self.config.get('synthesis', {}).get('context_window_cap', 16384)

        if use_async:
            print(f"   -> Executing Phase 2 concurrently with max_async_calls: {max_async}")
            
            async def run_single_pass(subset, is_normalized, i, target_type, target_id, sem):
                async with sem:
                    pass_name = "normalized" if is_normalized else "raw"
                    target_dir = norm_dir if is_normalized else raw_dir
                    print(f"      -> Running API call for Target {i}/{len(targets)} ({target_type} {target_id}) - {pass_name}...")
                    last_llm_responses.set([])
                    try:
                        res = await active_agent.run(json.dumps(subset), deps=synth_ctx)
                        filename = f"{i:02d}_{res.output.module_name}.py"
                        with open(target_dir / filename, "w") as f:
                            f.write(clean_python_code(res.output.source_code))
                    except Exception as e:
                        log_error(f"[Synthesis] Error generating {pass_name} schema {i} ({target_type} {target_id}): {e}")
                        attempts = last_llm_responses.get()
                        if attempts:
                            log_error(f"[Synthesis] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")

            async def run_phase2_async():
                sem = asyncio.Semaphore(max_async)
                tasks = []
                for i, target in enumerate(targets, start=1):
                    target_nodes = set(target["nodes"])
                    
                    # Pass A: Normalized
                    norm_subset = []
                    for t in refined_triplets:
                        subj = str(t.get('subject', '')).lower().strip()
                        obj = str(t.get('object', '')).lower().strip()
                        if subj in target_nodes or obj in target_nodes:
                            norm_subset.append(t)
                    
                    if norm_subset:
                        tasks.append(run_single_pass(norm_subset, True, i, target["type"], target["id"], sem))
                    
                    # Pass B: Raw Unwound
                    raw_subset = []
                    for idx, raw_t in enumerate(original_triplets):
                        if idx < len(refined_triplets):
                            refined_t = refined_triplets[idx]
                            subj = str(refined_t.get('subject', '')).lower().strip()
                            obj = str(refined_t.get('object', '')).lower().strip()
                            if subj in target_nodes or obj in target_nodes:
                                raw_subset.append(raw_t)
                    
                    if raw_subset:
                        tasks.append(run_single_pass(raw_subset, False, i, target["type"], target["id"], sem))
                
                if tasks:
                    await asyncio.gather(*tasks)

            asyncio.run(run_phase2_async())
        else:
            for i, target in enumerate(targets, start=1):
                target_nodes = set(target["nodes"])
                print(f"      -> Processing Target {i}/{len(targets)} ({target['type']} {target['id']})")
                
                # Pass A: Normalized
                norm_subset = []
                for t in refined_triplets:
                    subj = str(t.get('subject', '')).lower().strip()
                    obj = str(t.get('object', '')).lower().strip()
                    if subj in target_nodes or obj in target_nodes:
                        norm_subset.append(t)
                
                if norm_subset:
                    last_llm_responses.set([])
                    try:
                        norm_res = active_agent.run_sync(json.dumps(norm_subset), deps=synth_ctx)
                        filename = f"{i:02d}_{norm_res.output.module_name}.py"
                        with open(norm_dir / filename, "w") as f:
                            f.write(clean_python_code(norm_res.output.source_code))
                    except Exception as e:
                        log_error(f"[Synthesis] Error generating normalized schema {i} ({target['type']} {target['id']}): {e}")
                        attempts = last_llm_responses.get()
                        if attempts:
                            log_error(f"[Synthesis] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")

                # Pass B: Raw Unwound
                raw_subset = []
                for idx, raw_t in enumerate(original_triplets):
                    if idx < len(refined_triplets):
                        refined_t = refined_triplets[idx]
                        subj = str(refined_t.get('subject', '')).lower().strip()
                        obj = str(refined_t.get('object', '')).lower().strip()
                        if subj in target_nodes or obj in target_nodes:
                            raw_subset.append(raw_t)
                
                if raw_subset:
                    last_llm_responses.set([])
                    try:
                        raw_res = active_agent.run_sync(json.dumps(raw_subset), deps=synth_ctx)
                        filename = f"{i:02d}_{raw_res.output.module_name}.py"
                        with open(raw_dir / filename, "w") as f:
                            f.write(clean_python_code(raw_res.output.source_code))
                    except Exception as e:
                        log_error(f"[Synthesis] Error generating raw schema {i} ({target['type']} {target['id']}): {e}")
                        attempts = last_llm_responses.get()
                        if attempts:
                            log_error(f"[Synthesis] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")

        # Phase 3: Global Consolidation
        print("   -> Running Phase 3: Global Consolidation...")
        
        from src.utils.token_helper import estimate_tokens
        from src.synthesis import prompts as synth_prompts
        import re

        def strip_comments_and_docstrings(code: str) -> str:
            # Remove single line comments
            code = re.sub(r'#.*', '', code)
            # Remove docstrings
            code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
            # Remove double empty lines
            code = re.sub(r'\n\s*\n', '\n', code)
            return code.strip()

        sys_tokens = estimate_tokens(synth_prompts.CONSOLIDATION_SYSTEM_PROMPT)
        max_payload_tokens = synthesis_cap - sys_tokens - 1000  # 1000 output buffer
        if max_payload_tokens < 500:
            max_payload_tokens = 500

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
                
            # Try stripping comments
            for f in files_list:
                f["content"] = strip_comments_and_docstrings(f["content"])
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            if estimate_tokens(payload) <= max_payload_tokens:
                print(f"[Synthesis] Stripped comments/docstrings to fit context window in {schema_dir.name}.")
                return await run_consolidation_agent_async(payload)
                
            # Hierarchical split
            print(f"[Synthesis] Payload exceeds context cap in {schema_dir.name}. Splitting consolidation into sub-batches.")
            sub_batches = []
            current_batch = []
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
                
            # Try stripping comments
            for f in files_list:
                f["content"] = strip_comments_and_docstrings(f["content"])
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            if estimate_tokens(payload) <= max_payload_tokens:
                print(f"[Synthesis] Stripped comments/docstrings to fit context window in {schema_dir.name}.")
                return run_consolidation_agent_sync(payload)
                
            # Hierarchical split
            print(f"[Synthesis] Payload exceeds context cap in {schema_dir.name}. Splitting consolidation into sub-batches.")
            sub_batches = []
            current_batch = []
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

        if use_async:
            async def consolidate_schemas_async(schema_dir: Path) -> None:
                files_list = []
                for filepath in schema_dir.glob("*.py"):
                    if filepath.name not in ("enums.py", "__init__.py", "master_ontology.py"):
                        try:
                            with open(filepath, "r") as f:
                                files_list.append({"name": filepath.name, "content": f.read()})
                        except Exception as e:
                            log_error(f"[Synthesis] Error reading {filepath.name} for consolidation: {e}")
                
                if files_list:
                    print(f"      -> Consolidating {len(files_list)} files in {schema_dir.name}...")
                    last_llm_responses.set([])
                    try:
                        master_code = await process_consolidation_async(files_list, schema_dir)
                        with open(schema_dir / "master_ontology.py", "w") as f:
                            f.write(master_code)
                    except Exception as e:
                        log_error(f"[Synthesis] Error generating master_ontology for {schema_dir.name}: {e}")
                        attempts = last_llm_responses.get()
                        if attempts:
                            log_error(f"[Synthesis] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")

            async def run_consolidation_async():
                await asyncio.gather(
                    consolidate_schemas_async(norm_dir),
                    consolidate_schemas_async(raw_dir)
                )

            asyncio.run(run_consolidation_async())
        else:
            def consolidate_schemas(schema_dir: Path) -> None:
                files_list = []
                for filepath in schema_dir.glob("*.py"):
                    if filepath.name not in ("enums.py", "__init__.py", "master_ontology.py"):
                        try:
                            with open(filepath, "r") as f:
                                files_list.append({"name": filepath.name, "content": f.read()})
                        except Exception as e:
                            log_error(f"[Synthesis] Error reading {filepath.name} for consolidation: {e}")
                
                if files_list:
                    print(f"      -> Consolidating {len(files_list)} files in {schema_dir.name}...")
                    last_llm_responses.set([])
                    try:
                        master_code = process_consolidation_sync(files_list, schema_dir)
                        with open(schema_dir / "master_ontology.py", "w") as f:
                            f.write(master_code)
                    except Exception as e:
                        log_error(f"[Synthesis] Error generating master_ontology for {schema_dir.name}: {e}")
                        attempts = last_llm_responses.get()
                        if attempts:
                            log_error(f"[Synthesis] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")

            consolidate_schemas(norm_dir)
            consolidate_schemas(raw_dir)

        # Phase 4: Final Comprehensive Ontology
        print("   -> Running Phase 4: Final Comprehensive Ontology Generation...")
        last_llm_responses.set([])
        try:
            with open(norm_dir / "master_ontology.py", "r") as f:
                norm_master = f.read()
            with open(raw_dir / "master_ontology.py", "r") as f:
                raw_master = f.read()
            with open(norm_dir / "enums.py", "r") as f:
                enums_code = f.read()
                
            final_payload = f"--- ENUMS ---\n{enums_code}\n\n--- RAW MASTER ONTOLOGY ---\n{raw_master}\n\n--- NORMALIZED MASTER ONTOLOGY ---\n{norm_master}"
            
            final_res = comprehensive_ontology_agent.run_sync(final_payload)
            final_code = clean_python_code(final_res.output.source_code)
            
            # Save to root schemas directory
            root_schemas_dir = Path("outputs/schemas")
            with open(root_schemas_dir / "comprehensive_ontology.py", "w") as f:
                f.write(final_code)
                
        except Exception as e:
            log_error(f"[Synthesis] Error generating comprehensive ontology: {e}")
            attempts = last_llm_responses.get()
            if attempts:
                log_error(f"[Synthesis] Attempted extract from LLM:\n{json.dumps(attempts, indent=2)}")

        # Finalization
        print("   -> Finalizing output modules...")
        with open(norm_dir / "__init__.py", "w") as f:
            f.write("# Normalized Schemas")
        with open(raw_dir / "__init__.py", "w") as f:
            f.write("# Raw Schemas")
            
        # Run ruff formatter
        try:
            res_norm = subprocess.run(["ruff", "format", str(norm_dir)], capture_output=True, text=True, check=False)
            if res_norm.returncode != 0:
                log_error(f"[Synthesis] Ruff formatting failed on normalized schemas:\n{res_norm.stderr or res_norm.stdout}")
                
            res_raw = subprocess.run(["ruff", "format", str(raw_dir)], capture_output=True, text=True, check=False)
            if res_raw.returncode != 0:
                log_error(f"[Synthesis] Ruff formatting failed on raw schemas:\n{res_raw.stderr or res_raw.stdout}")
                
            res_comp = subprocess.run(["ruff", "format", str(Path("outputs/schemas/comprehensive_ontology.py"))], capture_output=True, text=True, check=False)
            if res_comp.returncode != 0:
                log_error(f"[Synthesis] Ruff formatting failed on comprehensive ontology:\n{res_comp.stderr or res_comp.stdout}")
        except Exception as e:
            log_error(f"[Synthesis] Critical Error running ruff: {e}")

        print("[Synthesis] Stage 4 Pipeline complete.")
