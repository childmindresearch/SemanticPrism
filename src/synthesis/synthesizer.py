"""
SemanticPrism: Master Synthesis Output Generation
The final stochastic phase transforming graph arrays into verified Pydantic schema maps safely.
"""

import yaml
import json
import asyncio
import os
import re
from typing import Dict, Any, List

import src.synthesis.prompts as prompts
from src.synthesis.schemas import GeneratedSchema
from src.core.logger import get_logger
from src.llm.llm_client import SemanticLLMClient

logger = get_logger("SynthesisEngine")

def _to_snake_case(text: str) -> str:
    """Converts text to a clean snake_case string for filenames."""
    s = text.lower().strip()
    s = re.sub(r'[^a-z0-9_]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

class SynthesisEngine:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the SynthesisEngine by loading configuration, setting up the output directory, and initializing the LLM client.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.critical(f"Config mapping aborted: {e}")
            raise e
            
        self.use_async = self.config.get('pipeline', {}).get('use_async', False)
        self.max_concurrent = self.config.get('pipeline', {}).get('max_concurrent_llm_calls', 3)
        self.output_dir = "outputs"
        self.schemas_dir = self.config.get('output', {}).get('schemas_dir', 'outputs/schemas')
        
        logger.info("Initializing Master LLM Factory for semantic synthesis mappings natively.")
        self.llm = SemanticLLMClient(config_path)
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Synthesis Engine Initialized natively.")

    async def generate_schemas(self, hierarchy_payload: Dict[str, Any], master_domain: str, theme_inheritance_map: Dict[str, List[str]] = None, strategy: str = "standard") -> Dict[str, Any]:
        """
        Iterates over mapped communities and uses the LLM to generate Pydantic schemas and Python code.
        Supports standard and hub_and_spoke strategies.
        """
        logger.info(f"Synthesizing schemas dynamically using strategy: {strategy}")
        sem = asyncio.Semaphore(self.max_concurrent)
        
        communities = hierarchy_payload.get("communities", hierarchy_payload) if isinstance(hierarchy_payload, dict) else hierarchy_payload
        master_hub = hierarchy_payload.get("master_hub", {}) if strategy == "hub_and_spoke" else {}
        orphans = hierarchy_payload.get("orphans", []) if strategy == "hub_and_spoke" else []
        
        hub_node_name = master_hub.get("node") if master_hub else None

        async def process_master(hub_data):
            if not hub_data or not hub_data.get("node"): return None
            node = hub_data.get("node")
            edge_strings = [f"{e['source']} -[{e['data'].get('predicate', 'links')}]-> {e['target']}" for e in hub_data.get("relationships", [])]
            context = {"hub_node": node, "domain": master_domain, "relationships": edge_strings}
            
            user_msg = prompts.PYDANTIC_CODE_GEN_USER_PROMPT.format(
                inheritance_guidelines="You are the master root node. Provide a foundational interface.",
                community_graph_json=json.dumps(context, indent=2)
            )
            return await self.llm.safe_api_call_async(prompts.HUB_NODE_SYSTEM_PROMPT, user_msg, GeneratedSchema)

        async def process_orphans(orphans_list):
            if not orphans_list: return None
            all_nodes = set()
            all_edges = []
            for comm in orphans_list:
                all_nodes.update(comm.get("nodes", []))
                for e in comm.get("edges", []):
                    all_edges.append(f"{e['source']} -[{e['data'].get('predicate', 'links')}]-> {e['target']}")
            context = {"domain": master_domain, "nodes": list(all_nodes), "isolated_facts": all_edges}
            
            user_msg = prompts.PYDANTIC_CODE_GEN_USER_PROMPT.format(
                inheritance_guidelines="No inheritance. Purely isolated constants.",
                community_graph_json=json.dumps(context, indent=2)
            )
            from src.synthesis.schemas import OrphanEnumSchema
            return await self.llm.safe_api_call_async(prompts.ORPHAN_ENUMS_SYSTEM_PROMPT, user_msg, OrphanEnumSchema)

        async def process_community(comm_key: str, comm_data: Dict[str, Any]):
            edge_strings = [f"{e['source']} -[{e['data'].get('predicate', 'links')}]-> {e['target']}" for e in comm_data.get("edges", [])]
            context = {"community_id": comm_key, "domain": master_domain, "nodes": comm_data.get("nodes", []), "relationships": edge_strings}
            
            inheritance_guidelines = "No strict inheritance detected. Default to BaseModel."
            if strategy == "hub_and_spoke" and hub_node_name:
                inheritance_guidelines = f"This community is part of a hub-and-spoke model. The master hub is '{hub_node_name}'. Ensure your concrete models contain a composition field referencing the master hub."
            elif theme_inheritance_map:
                inheritance_guidelines = f"Global Theme Inheritance Map (Subclass -> Parent): {json.dumps(theme_inheritance_map)}\nIf this community represents a subclassed theme, generate Protocol interfaces to implement the inheritance dynamically."
            
            user_msg = prompts.PYDANTIC_CODE_GEN_USER_PROMPT.format(
                inheritance_guidelines=inheritance_guidelines,
                community_graph_json=json.dumps(context, indent=2)
            )
            
            async with sem:
                res = await self.llm.safe_api_call_async(prompts.PYDANTIC_CODE_GEN_SYSTEM_PROMPT, user_msg, GeneratedSchema)
            return comm_key, res

        final_results = {"strategy": strategy, "communities": {}, "master": None, "orphans": None}

        # Process everything
        if self.use_async:
            tasks = [process_community(k, v) for k, v in communities.items()]
            if strategy == "hub_and_spoke":
                master_task = asyncio.create_task(process_master(master_hub))
                orphan_task = asyncio.create_task(process_orphans(orphans))
                
            resolved_schemas = await asyncio.gather(*tasks)
            
            if strategy == "hub_and_spoke":
                final_results["master"] = await master_task
                final_results["orphans"] = await orphan_task
        else:
            resolved_schemas = []
            for k, v in communities.items():
                async with sem:
                    resolved_schemas.append(await process_community(k, v))
            if strategy == "hub_and_spoke":
                final_results["master"] = await process_master(master_hub)
                final_results["orphans"] = await process_orphans(orphans)
                
        for key, schema in resolved_schemas:
            if schema is not None:
                final_results["communities"][key] = schema
                
        return final_results

    def build_global_context(self, synthesis_results: Dict[str, Any]) -> str:
        """
        Aggregates generated schemas into a dictionary, saves them as a JSON file, and writes associated Pydantic Python schemas to an executable Python file. Returns the JSON file path.
        """
        logger.info("Executing Cross-Community Output Parsing safely.")
        output_payload = {}
        python_blocks = []
        
        strategy = synthesis_results.get("strategy", "standard")
        communities = synthesis_results.get("communities", {})
        master_schema = synthesis_results.get("master")
        orphan_schema = synthesis_results.get("orphans")
        
        if strategy == "hub_and_spoke":
            if master_schema:
                output_payload["master_hub"] = master_schema.model_dump()
                if master_schema.protocols_code:
                    python_blocks.append(f"### GLOBAL INTERFACES (MASTER HUB) ###\n{master_schema.protocols_code}\n\n")
                if master_schema.concrete_models_code:
                    python_blocks.append(f"### GLOBAL CONTEXT (MASTER HUB) ###\n{master_schema.concrete_models_code}\n\n")
                    
        for k, schema in communities.items():
            output_payload[k] = schema.model_dump()
            if schema.protocols_code:
                python_blocks.append(f"### Protocols/Interfaces for Community: {schema.title} ###\n{schema.protocols_code}\n\n")
            if schema.concrete_models_code:
                python_blocks.append(f"### Concrete Models for Community: {schema.title} ###\n{schema.concrete_models_code}\n\n")
                
        if strategy == "hub_and_spoke" and orphan_schema:
            output_payload["orphans"] = orphan_schema.model_dump()
            if orphan_schema.enums_code:
                python_blocks.append(f"### GLOBAL ENUMS & CONSTANTS (ORPHANS) ###\n{orphan_schema.enums_code}\n\n")
            
        file_path = os.path.join(self.output_dir, "semantic_prism_master_graph.json")
        py_file_path = os.path.join(self.output_dir, "semantic_models.py")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_payload, f, indent=4)
            
            if python_blocks:
                with open(py_file_path, 'w', encoding='utf-8') as f:
                    f.write('from pydantic import BaseModel, Field\nfrom typing import List, Optional, Protocol, Any, Literal\nimport abc\nfrom enum import Enum\n\n')
                    f.write("\n".join(python_blocks))
                logger.info(f"Pydantic Python Schemas brilliantly dynamically structurally successfully mapped to {py_file_path}")
                
            logger.info(f"Master Synthesis exported flawlessly to {file_path}")
            
            if self.schemas_dir:
                os.makedirs(self.schemas_dir, exist_ok=True)
                
                # Write Master Schema if exists
                if strategy == "hub_and_spoke" and master_schema:
                    master_path = os.path.join(self.schemas_dir, "master_context.py")
                    with open(master_path, 'w', encoding='utf-8') as f:
                        f.write('from pydantic import BaseModel, Field\nfrom typing import List, Optional, Protocol, Any\nimport abc\n\n')
                        if master_schema.protocols_code: f.write(master_schema.protocols_code + "\n\n")
                        if master_schema.concrete_models_code: f.write(master_schema.concrete_models_code + "\n")
                        
                # Write Orphans if exist
                if strategy == "hub_and_spoke" and orphan_schema:
                    orphan_path = os.path.join(self.schemas_dir, "global_enums.py")
                    with open(orphan_path, 'w', encoding='utf-8') as f:
                        f.write('from enum import Enum\nfrom typing import Literal\n\n')
                        if orphan_schema.enums_code: f.write(orphan_schema.enums_code + "\n")
                
                # Write individual community schemas
                for k, schema in communities.items():
                    individual_blocks = []
                    if schema.protocols_code:
                        individual_blocks.append(schema.protocols_code)
                    if schema.concrete_models_code:
                        individual_blocks.append(schema.concrete_models_code)
                        
                    if individual_blocks:
                        filename = f"{_to_snake_case(schema.title)}.py"
                        individual_path = os.path.join(self.schemas_dir, filename)
                        try:
                            with open(individual_path, 'w', encoding='utf-8') as f:
                                f.write('from pydantic import BaseModel, Field\nfrom typing import List, Optional, Protocol, Any\nimport abc\n')
                                if strategy == "hub_and_spoke":
                                    f.write('from .master_context import *\n')
                                f.write('\n')
                                f.write("\n\n".join(individual_blocks))
                                f.write("\n")
                            logger.info(f"Successfully wrote standalone community schema file to: {individual_path}")
                        except Exception as e_ind:
                            logger.error(f"Failed to write standalone community schema file {filename}: {e_ind}")
                            
        except Exception as e:
            logger.error(f"Export mapping natively failed: {e}")
            
        return file_path
