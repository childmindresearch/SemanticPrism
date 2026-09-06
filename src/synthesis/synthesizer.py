import json
import subprocess
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.config import settings
from src.agents.synthesis_agents import (
    OrphanContext, SynthesisContext,
    orphan_agent, leiden_schema_agent, node2vec_schema_agent,
    consolidation_agent, comprehensive_ontology_agent,
    schema_reformat_agent,
    last_llm_responses
)
from src.topology.resolver import TargetResolver

def prune_unused_enums(code: str) -> str:
    """
    Parses generated Python code via AST, discovers Enum class definitions,
    finds all identifier references in BaseModel field annotations,
    and removes Enum definitions whose names are never referenced by any BaseModel.
    """
    import ast

    if not code or not code.strip():
        return code

    try:
        tree = ast.parse(code)
    except Exception:
        return code

    enum_class_names = set()
    model_field_annotation_refs = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            
            if any("Enum" in b for b in bases):
                enum_class_names.add(node.name)
            else:
                for item in ast.walk(node):
                    if isinstance(item, ast.Name):
                        model_field_annotation_refs.add(item.id)

    unused_enums = enum_class_names - model_field_annotation_refs
    if not unused_enums:
        return code

    tree.body = [node for node in tree.body if not (isinstance(node, ast.ClassDef) and node.name in unused_enums)]
    try:
        return ast.unparse(tree)
    except Exception:
        return code

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

    def execute(self, comm_targets: Dict[str, Any] = None, emb_targets: Dict[str, Any] = None, refined_triplets: List[Dict[str, Any]] = None, original_triplets: List[Dict[str, Any]] = None, taxonomic_map: Dict[str, str] = None, master_themes: List[str] = None):
        print("[Synthesis] Starting Stage 4 Pipeline...")
        synth_mode = self.config.get('synthesis', {}).get('execution_mode', 'community')
        print(f"   -> Synthesis Execution Mode: '{synth_mode}'")
        
        if synth_mode in ("both", "community"):
            if comm_targets:
                print("   ==> Executing Path 1: Community Path Synthesis...")
                self.execute_path_synthesis(comm_targets, "community", refined_triplets, original_triplets, master_themes)
            else:
                print("   [Warning] Community resolved targets file not found. Skipping Path 1 synthesis.")

        if synth_mode in ("both", "embedding"):
            if emb_targets:
                print("   ==> Executing Path 2: Embedding Path Synthesis...")
                self.execute_path_synthesis(emb_targets, "embedding", refined_triplets, original_triplets, master_themes)
            else:
                print("   [Warning] Embedding resolved targets file not found. Skipping Path 2 synthesis.")

        print("[Synthesis] Stage 4 Pipeline complete.")

    def execute_path_synthesis(self, targets_data: Dict[str, Any], path_type: str, refined_triplets: List[dict], original_triplets: List[dict], master_themes: List[str]):
        path_label = "Path 1: Community Workflow" if path_type == "community" else "Path 2: Embedding Categorical"
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
        log_dir = Path(outputs_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"synthesis_{path_type}_errors.log"
        
        def log_error(msg: str):
            print(msg)
            with open(log_file, "a") as lf:
                lf.write(f"[{datetime.now().isoformat()}] {msg}\n")
                
        def log_detailed_synthesis_error(error: Exception, responses: list, pass_name: str, target_id: Any, i: int):
            from pydantic import ValidationError
            from pydantic_ai.exceptions import UnexpectedModelBehavior
            
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
                            cleaned = [l for l in lines if not l.strip().startswith("```")]
                            content = "\n".join(cleaned)
                        return content.strip()
            if exception:
                return str(exception)
            return ""
        
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

        # Ingest pre-resolved Targets and Enum Nodes directly from target JSON payload
        targets = targets_data.get("targets", [])
        all_enum_nodes = targets_data.get("enum_nodes", [])
        active_agent = node2vec_schema_agent if path_type == "embedding" else leiden_schema_agent
        
        # Phase 1: Orphan & Low-Density Node Aggregation (Enums)
        print(f"   -> [{path_label}] Phase 1: Orphan & Low-Density Node Aggregation (Enums)...")
        print(f"      -> Total Enum Nodes: {len(all_enum_nodes)}")
        global_enums_code = ""
        
        if all_enum_nodes:
            orphan_ctx = OrphanContext(master_themes=master_themes)
            payload = json.dumps(all_enum_nodes)
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

            if run_norm and global_enums_code and not global_enums_code.startswith("# Error"):
                with open(norm_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)
            if run_raw and global_enums_code and not global_enums_code.startswith("# Error"):
                with open(raw_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)

        # Phase 2: Schema Generation
        print(f"   -> [{path_label}] Phase 2: Schema Generation...")
        spoke_count = sum(1 for t in targets if t.get("target_type") != "hub" and t.get("type") != "hub")
        hub_count = sum(1 for t in targets if t.get("target_type") == "hub" or t.get("type") == "hub")
        print(f"      -> Target Composition: {len(targets)} total targets ({spoke_count} spokes + {hub_count} global hubs)")
            
        synth_ctx = SynthesisContext(
            global_enums=global_enums_code
        )

        use_async = self.config.get('pipeline', {}).get('use_async', False)
        max_async = self.config.get('synthesis', {}).get('max_async_calls', 1)

        if use_async:
            print(f"      -> Executing Phase 2 concurrently with max_async_calls: {max_async}")
            
            async def run_single_pass(subset, is_normalized, i, target_type, target_id, sem, node_count):
                async with sem:
                    pass_name = "normalized" if is_normalized else "raw"
                    target_dir = norm_dir if is_normalized else raw_dir
                    print(f"         -> API call for Target {i}/{len(targets)} ({target_type} {target_id}) - {pass_name} (Size: {node_count} nodes, {len(subset)} triplets)...")
                    last_llm_responses.set([])
                    from pydantic_ai import capture_run_messages
                    with capture_run_messages() as messages:
                        try:
                            res = await active_agent.run(json.dumps(subset), deps=synth_ctx)
                            filename = f"{i:02d}_{res.output.module_name}.py"
                            with open(target_dir / filename, "w") as f:
                                f.write(clean_python_code(res.output.source_code))
                        except Exception as e:
                            # Reformat attempt (3rd attempt after 2 failures)
                            malformed_text = extract_malformed_text(messages, e)
                            print(f"            [Warning] Initial {pass_name} extraction failed for Target {i} after 2 tries. Running Schema Reformat Agent...")
                            
                            reformat_prompt = (
                                f"Original Schema Generation Task context was:\n{json.dumps(subset)}\n\n"
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
                            tasks.append(run_single_pass(norm_subset, True, i, target_type, target_id, sem, len(target_nodes)))
                    
                    if run_raw:
                        raw_subset = [t for t in original_triplets if str(t.get('subject','')).lower().strip() in target_nodes or str(t.get('object','')).lower().strip() in target_nodes]
                        if raw_subset:
                            tasks.append(run_single_pass(raw_subset, False, i, target_type, target_id, sem, len(target_nodes)))
                
                if tasks:
                    await asyncio.gather(*tasks)

            asyncio.run(run_phase2_async())
        else:
            def run_single_pass_sync(subset, is_normalized, i, target_type, target_id, node_count):
                pass_name = "normalized" if is_normalized else "raw"
                target_dir = norm_dir if is_normalized else raw_dir
                print(f"         -> API call for Target {i}/{len(targets)} ({target_type} {target_id}) - {pass_name} (Size: {node_count} nodes, {len(subset)} triplets)...")
                last_llm_responses.set([])
                from pydantic_ai import capture_run_messages
                with capture_run_messages() as messages:
                    try:
                        res = active_agent.run_sync(json.dumps(subset), deps=synth_ctx)
                        filename = f"{i:02d}_{res.output.module_name}.py"
                        with open(target_dir / filename, "w") as f:
                            f.write(clean_python_code(res.output.source_code))
                    except Exception as e:
                        # Reformat attempt (3rd attempt after 2 failures)
                        malformed_text = extract_malformed_text(messages, e)
                        print(f"            [Warning] Initial {pass_name} extraction failed for Target {i} after 2 tries. Running Schema Reformat Agent...")
                        
                        reformat_prompt = (
                            f"Original Schema Generation Task context was:\n{json.dumps(subset)}\n\n"
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
                node_count = len(target_nodes)
                
                if run_norm:
                    norm_subset = target.get("triplets", [])
                    if norm_subset:
                        run_single_pass_sync(norm_subset, True, i, target_type, target_id, node_count)
                        
                if run_raw:
                    raw_subset = [t for t in original_triplets if str(t.get('subject','')).lower().strip() in target_nodes or str(t.get('object','')).lower().strip() in target_nodes]
                    if raw_subset:
                        run_single_pass_sync(raw_subset, False, i, target_type, target_id, node_count)

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

        pruned_final_code = prune_unused_enums(final_code)
        with open(output_base_dir / "comprehensive_ontology.py", "w") as f:
            f.write(pruned_final_code)

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
