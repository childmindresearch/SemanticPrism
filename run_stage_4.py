import json
import yaml
from pathlib import Path
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
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Stage 1-3 Outputs
    topology = load_json("outputs/03_topology/topology_partitions.json")
    refined_triplets = load_json("outputs/02_refinement/refined_triplets.json")
    original_triplets = load_json("outputs/01_extraction/original_triplets.json")
    taxonomic_map = load_json("outputs/02_refinement/taxonomic_map.json")
    master_themes_raw = load_json("outputs/01_extraction/master_themes.json")
    
    # Check if files exist
    if not all([topology, refined_triplets, original_triplets, taxonomic_map, master_themes_raw]):
        print("Cannot run Stage 4. Missing preceding stage outputs.")
        return

    # Assuming master_themes.json is a dict like {"master_domain": "...", "master_themes": ["...", "..."]}
    # or just a list. Handle gracefully.
    master_themes = []
    if isinstance(master_themes_raw, dict):
        master_themes = master_themes_raw.get("master_themes", [])
    elif isinstance(master_themes_raw, list):
        master_themes = master_themes_raw

    pipeline = SynthesisPipeline(config)
    pipeline.execute(topology, refined_triplets, original_triplets, taxonomic_map, master_themes)
    
    print("=== Stage 4 Synthesis Completed Successfully ===")

if __name__ == "__main__":
    main()
