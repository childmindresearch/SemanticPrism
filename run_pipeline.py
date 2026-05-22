import os
import glob
import asyncio
from src.orchestrator.pipeline import SemanticPrismOrchestrator

async def main():
    print("✅ Core imports loaded successfully.")
    
    target_dir = "inputs/testdocs"
    files = glob.glob(os.path.join(target_dir, "*.txt")) + glob.glob(os.path.join(target_dir, "*.md"))

    raw_documents = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            raw_documents.append({"filename": path, "text": f.read()})
            
    print(f"Ingested {len(raw_documents)} logic documents natively.")

    if not raw_documents:
        print("⚠️ No documents discovered in inputs/reports directory. Exiting.")
        return

    # Instantiate the Master Pipeline Orchestrator
    orchestrator = SemanticPrismOrchestrator("config.yaml")
    print("✅ Master Pipeline Orchestrator initialized.")

    # Execute the structured knowledge pipeline
    file_path = await orchestrator.execute_knowledge_pipeline(raw_documents)
    
    if file_path:
        print(f"🎉 Pipeline Complete! Final logical schema outputs synthesized securely to: {file_path}")
    else:
        print("⚠️ Pipeline ended early or failed to synthesize schemas.")

if __name__ == "__main__":
    asyncio.run(main())
