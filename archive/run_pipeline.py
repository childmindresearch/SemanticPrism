"""
SemanticPrism: Global Execution Runner
Automates the isolation and discovery of raw text strings physically located 
in the inputs directory and dispatches them organically through the full SemanticPrism architecture.
"""

import os
import glob
import asyncio
from src.orchestrator.pipeline import SemanticPrismOrchestrator
from src.core.logger import get_logger

logger = get_logger("SemanticRunner")

async def execute():
    logger.info("Initializing Autorecovery SemanticPrism Runner natively.")
    target_dir = os.path.join("inputs", "testdocs")
    
    files = glob.glob(os.path.join(target_dir, "*.txt")) + glob.glob(os.path.join(target_dir, "*.md"))
    if not files:
        logger.error(f"No textual configurations located inside '{target_dir}'. Create a mock text file gracefully to proceed inherently.")
        return
        
    logger.info(f"Targets Acquired safely: {len(files)} documents.")
    
    raw_texts = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_texts.append(f.read())
        except Exception as e:
            logger.error(f"File loading bypassed strictly: {e}")
            
    if not raw_texts:
        logger.error("No valid text blocks successfully loaded.")
        return
        
    logger.info("Instantiating formal pipeline structural execution cleanly.")
    orchestrator = SemanticPrismOrchestrator("config.yaml")
    out_payload = await orchestrator.execute_knowledge_pipeline(raw_texts)
    
    logger.info(f"Runner completed cleanly natively flawlessly securely. Synthesis Object mapping saved realistically correctly to: {out_payload}")

if __name__ == "__main__":
    asyncio.run(execute())
