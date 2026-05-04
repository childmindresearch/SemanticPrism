"""
SemanticPrism: Master Synthesis Output Generation
The final stochastic phase transforming graph arrays into verified Pydantic schema maps safely.
"""

import yaml
import json
import asyncio
import os
from typing import Dict, Any, List

import src.synthesis.prompts as prompts
from src.synthesis.schemas import GeneratedSchema
from src.core.logger import get_logger
from src.llm.llm_client import SemanticLLMClient

logger = get_logger("SynthesisEngine")

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
        
        logger.info("Initializing Master LLM Factory for semantic synthesis mappings natively.")
        self.llm = SemanticLLMClient(config_path)
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Synthesis Engine Initialized natively.")

    async def generate_schemas(self, hierarchy: Dict[str, Dict[str, Any]], master_domain: str, theme_inheritance_map: Dict[str, List[str]] = None) -> Dict[str, GeneratedSchema]:
        """
        Iterates over mapped communities and uses the LLM to generate Pydantic schemas and Python code from the graph structure.
        """
        logger.info(f"Synthesizing {len(hierarchy)} mapped communities dynamically.")
        sem = asyncio.Semaphore(self.max_concurrent)
        results = {}
        
        async def process_community(comm_key: str, comm_data: Dict[str, Any]):
            # Stringify graph arrays for LLM digestion securely
            edge_strings = [f"{e['source']} -[{e['data'].get('predicate', 'links')}]-> {e['target']}" for e in comm_data.get("edges", [])]
            
            context = {
                "community_id": comm_key,
                "domain": master_domain,
                "nodes": comm_data.get("nodes", []),
                "relationships": edge_strings
            }
            
            inheritance_guidelines = "No strict inheritance detected. Default to BaseModel."
            if theme_inheritance_map:
                inheritance_guidelines = f"Global Theme Inheritance Map (Subclass -> Parent): {json.dumps(theme_inheritance_map)}\nIf this community represents a subclassed theme, generate Protocol interfaces to implement the inheritance dynamically."
            
            user_msg = prompts.PYDANTIC_CODE_GEN_USER_PROMPT.format(
                inheritance_guidelines=inheritance_guidelines,
                community_graph_json=json.dumps(context, indent=2)
            )
            
            if self.use_async:
                async with sem:
                    res = await self.llm.safe_api_call_async(
                        prompts.PYDANTIC_CODE_GEN_SYSTEM_PROMPT,
                        user_msg,
                        GeneratedSchema
                    )
            else:
                res = self.llm.safe_api_call_sync(
                    prompts.PYDANTIC_CODE_GEN_SYSTEM_PROMPT,
                    user_msg,
                    GeneratedSchema
                )
            
            return comm_key, res
                
        if self.use_async:
            tasks = [process_community(k, v) for k, v in hierarchy.items()]
            resolved_schemas = await asyncio.gather(*tasks)
        else:
            resolved_schemas = []
            for k, v in hierarchy.items():
                resolved_schemas.append(await process_community(k, v))
        
        for key, schema in resolved_schemas:
            if schema is not None:
                results[key] = schema
                
        return results

    def build_global_context(self, resolved_schemas: Dict[str, GeneratedSchema]) -> str:
        """
        Aggregates generated schemas into a dictionary, saves them as a JSON file, and writes associated Pydantic Python schemas to an executable Python file. Returns the JSON file path.
        """
        logger.info("Executing Cross-Community Output Parsing safely.")
        output_payload = {}
        python_blocks = []
        
        for k, schema in resolved_schemas.items():
            output_payload[k] = schema.model_dump()
            if schema.protocols_code:
                python_blocks.append(f"### Protocols/Interfaces for Community: {schema.title} ###\n{schema.protocols_code}\n\n")
            if schema.concrete_models_code:
                # Provide functional demarcations per generated mathematical community natively
                python_blocks.append(f"### Concrete Models for Community: {schema.title} ###\n{schema.concrete_models_code}\n\n")
            
        file_path = os.path.join(self.output_dir, "semantic_prism_master_graph.json")
        py_file_path = os.path.join(self.output_dir, "semantic_models.py")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_payload, f, indent=4)
            
            if python_blocks:
                with open(py_file_path, 'w', encoding='utf-8') as f:
                    f.write('from pydantic import BaseModel, Field\nfrom typing import List, Optional, Protocol, Any\nimport abc\n\n')
                    f.write("\n".join(python_blocks))
                logger.info(f"Pydantic Python Schemas brilliantly dynamically structurally successfully mapped to {py_file_path}")
                
            logger.info(f"Master Synthesis exported flawlessly to {file_path}")
        except Exception as e:
            logger.error(f"Export mapping natively failed: {e}")
            
        return file_path
