import json
from pathlib import Path
from src.config import settings
from src.synthesis.synthesizer import SynthesisPipeline

def load_json(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Required file {filepath} not found.")
        return None

def main():
    print("=== SemanticPrism Stage 4: Synthesis Pipeline ===")
    
    config = settings

    # Check synthesis execution mode
    synth_mode = config.get('synthesis', {}).get('execution_mode', 'community')
    print(f"   -> Synthesis Execution Mode: '{synth_mode}'")

    comm_targets = None
    emb_targets = None

    if synth_mode in ("both", "community"):
        comm_targets = load_json("outputs/03_topology/community/resolved_community_targets.json")

    if synth_mode in ("both", "embedding"):
        emb_targets = load_json("outputs/03_topology/embedding/resolved_embedded_targets.json")

    refined_triplets = load_json("outputs/02_refinement/refined_triplets.json")
    original_triplets = load_json("outputs/01_extraction/original_triplets.json")
    taxonomic_map = load_json("outputs/02_refinement/taxonomic_map.json")
    master_themes_raw = load_json("outputs/01_extraction/master_themes.json")
    
    # Check if files exist
    if not (comm_targets or emb_targets) or not all([refined_triplets, original_triplets, taxonomic_map, master_themes_raw]):
        print("Error: Cannot run Stage 4. Missing required preceding stage outputs.")
        return

    master_themes = []
    if isinstance(master_themes_raw, dict):
        master_themes = master_themes_raw.get("master_themes", [])
    elif isinstance(master_themes_raw, list):
        master_themes = master_themes_raw

    pipeline = SynthesisPipeline(config)
    pipeline.execute(comm_targets=comm_targets, emb_targets=emb_targets, refined_triplets=refined_triplets, original_triplets=original_triplets, taxonomic_map=taxonomic_map, master_themes=master_themes)
    
    print("=== Stage 4 Synthesis Completed Successfully ===")

if __name__ == "__main__":
    main()
