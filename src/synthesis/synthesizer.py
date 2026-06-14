import json
import os
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel

from src.synthesis import schemas
from src.synthesis import prompts
from src.config import settings

# Determine AI Model Provider Details
provider = settings['llm']['provider']
model_name = settings['llm']['model_name']
base_url = settings['llm'].get('base_url')

if provider == 'ollama':
    from pydantic_ai.providers.ollama import OllamaProvider
    custom_provider = OllamaProvider(base_url=base_url)
    pydantic_model = OllamaModel(model_name, provider=custom_provider)
else:
    api_key = settings['llm'].get('api_key', '')
    if api_key:
        if provider == 'google':
            os.environ['GOOGLE_API_KEY'] = api_key
        # Add other providers here if necessary in the future
    pydantic_model = f"{provider}:{model_name}"

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

@dataclass
class OrphanContext:
    master_themes: List[str]

@dataclass
class SynthesisContext:
    global_enums: str
    theme_inheritance: str

# Agent 1: Orphan Enum Agent
orphan_agent = Agent(
    model=pydantic_model,
    deps_type=OrphanContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.ORPHAN_SYNTHESIS_SYSTEM_PROMPT,
    retries=3
)

@orphan_agent.system_prompt
def add_orphan_context(ctx: RunContext[OrphanContext]) -> str:
    return f"\nMaster Themes Context: {json.dumps(ctx.deps.master_themes)}"

# Agent 2a: Leiden Ontology Schema Agent
leiden_schema_agent = Agent(
    model=pydantic_model,
    deps_type=SynthesisContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.LEIDEN_SCHEMA_SYNTHESIS_PROMPT,
    retries=3
)

@leiden_schema_agent.system_prompt
def add_leiden_schema_context(ctx: RunContext[SynthesisContext]) -> str:
    return f"\nTheme Inheritance Mapping:\n{ctx.deps.theme_inheritance}\n\nGlobal Enums Available:\n{ctx.deps.global_enums}"

# Agent 2b: Node2Vec Ontology Schema Agent
node2vec_schema_agent = Agent(
    model=pydantic_model,
    deps_type=SynthesisContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.NODE2VEC_SCHEMA_SYNTHESIS_PROMPT,
    retries=3
)

@node2vec_schema_agent.system_prompt
def add_node2vec_schema_context(ctx: RunContext[SynthesisContext]) -> str:
    return f"\nTheme Inheritance Mapping:\n{ctx.deps.theme_inheritance}\n\nGlobal Enums Available:\n{ctx.deps.global_enums}"

# Agent 3: Consolidation Agent
consolidation_agent = Agent(
    model=pydantic_model,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.CONSOLIDATION_SYSTEM_PROMPT,
    retries=3
)

# Agent 4: Comprehensive Ontology Agent
comprehensive_ontology_agent = Agent(
    model=pydantic_model,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.FINAL_ONTOLOGY_SYSTEM_PROMPT,
    retries=3
)

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
            try:
                result = orphan_agent.run_sync(payload, deps=orphan_ctx)
                global_enums_code = clean_python_code(result.output.source_code)
                
                with open(norm_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)
                with open(raw_dir / "enums.py", "w") as f:
                    f.write(global_enums_code)
            except Exception as e:
                log_error(f"[Synthesis] Error generating enums: {e}")
                global_enums_code = "# Error generating enums"

        # Phase 2: Dual-Pass Schema Generation
        print("   -> Running Phase 2: Dual-Pass Schema Generation...")
        
        # Consolidate Targets
        hubs = topology.get("global_hubs", [])
        
        targets = []
        clustering_strategy = self.config.get('synthesis', {}).get('clustering_strategy', 'leiden')
        
        if clustering_strategy == 'node2vec':
            active_agent = node2vec_schema_agent
            for cluster in topology.get("structural_clusters", []):
                targets.append({"type": "structural_cluster", "id": cluster.get("cluster_id"), "nodes": cluster.get("nodes", [])})
            print(f"   -> Using Node2Vec Structural Clusters ({len(targets)} targets)")
        else:
            active_agent = leiden_schema_agent
            for comm in topology.get("communities", []):
                targets.append({"type": "community", "id": comm.get("community_id"), "nodes": comm.get("nodes", [])})
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
                try:
                    norm_res = active_agent.run_sync(json.dumps(norm_subset), deps=synth_ctx)
                    filename = f"{i:02d}_{norm_res.output.module_name}.py"
                    with open(norm_dir / filename, "w") as f:
                        f.write(clean_python_code(norm_res.output.source_code))
                except Exception as e:
                    log_error(f"[Synthesis] Error generating normalized schema {i}: {e}")

            # Pass B: Raw Unwound
            raw_subset = []
            for idx, raw_t in enumerate(original_triplets):
                # We can guarantee a 1:1 index match between raw and refined triples.
                # If the refined version of this triple belongs to the community, we pull the raw version.
                if idx < len(refined_triplets):
                    refined_t = refined_triplets[idx]
                    subj = str(refined_t.get('subject', '')).lower().strip()
                    obj = str(refined_t.get('object', '')).lower().strip()
                    if subj in target_nodes or obj in target_nodes:
                        raw_subset.append(raw_t)
            
            if raw_subset:
                try:
                    raw_res = active_agent.run_sync(json.dumps(raw_subset), deps=synth_ctx)
                    filename = f"{i:02d}_{raw_res.output.module_name}.py"
                    with open(raw_dir / filename, "w") as f:
                        f.write(clean_python_code(raw_res.output.source_code))
                except Exception as e:
                    log_error(f"[Synthesis] Error generating raw schema {i}: {e}")

        # Phase 3: Global Consolidation
        print("   -> Running Phase 3: Global Consolidation...")
        
        def consolidate_schemas(schema_dir: Path) -> None:
            # Read all generated schema files except enums.py and __init__.py
            schema_contents = []
            for filepath in schema_dir.glob("*.py"):
                if filepath.name not in ("enums.py", "__init__.py", "master_ontology.py"):
                    try:
                        with open(filepath, "r") as f:
                            content = f.read()
                            schema_contents.append(f"--- File: {filepath.name} ---\n{content}\n")
                    except Exception as e:
                        log_error(f"[Synthesis] Error reading {filepath.name} for consolidation: {e}")
            
            if schema_contents:
                payload = "\n".join(schema_contents)
                print(f"      -> Consolidating {len(schema_contents)} files in {schema_dir.name}...")
                try:
                    result = consolidation_agent.run_sync(payload)
                    master_code = clean_python_code(result.output.source_code)
                    with open(schema_dir / "master_ontology.py", "w") as f:
                        f.write(master_code)
                except Exception as e:
                    log_error(f"[Synthesis] Error generating master_ontology for {schema_dir.name}: {e}")

        consolidate_schemas(norm_dir)
        consolidate_schemas(raw_dir)

        # Phase 4: Final Comprehensive Ontology
        print("   -> Running Phase 4: Final Comprehensive Ontology Generation...")
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
