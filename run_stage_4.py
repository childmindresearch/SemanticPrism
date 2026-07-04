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

    # Check synthesis execution mode
    synth_mode = config.get('synthesis', {}).get('execution_mode', 'community')
    print(f"   -> Synthesis Execution Mode: '{synth_mode}'")

    comm_topology = None
    emb_topology = None

    if synth_mode in ("both", "community"):
        comm_topology = load_json("outputs/03_topology/community/topology_partitions.json")
        if not comm_topology:
            print("   [Fallback] Checking legacy topology_partitions.json path...")
            comm_topology = load_json("outputs/03_topology/topology_partitions.json")

    if synth_mode in ("both", "embedding"):
        emb_topology = load_json("outputs/03_topology/embedding/topology_partitions.json")
        if not emb_topology and not comm_topology:
            print("   [Fallback] Checking legacy topology_partitions.json path...")
            emb_topology = load_json("outputs/03_topology/topology_partitions.json")

    refined_triplets = load_json("outputs/02_refinement/refined_triplets.json")
    original_triplets = load_json("outputs/01_extraction/original_triplets.json")
    taxonomic_map = load_json("outputs/02_refinement/taxonomic_map.json")
    master_themes_raw = load_json("outputs/01_extraction/master_themes.json")
    
    # Check if files exist
    if not (comm_topology or emb_topology) or not all([refined_triplets, original_triplets, taxonomic_map, master_themes_raw]):
        print("Error: Cannot run Stage 4. Missing required preceding stage outputs.")
        return

    master_themes = []
    if isinstance(master_themes_raw, dict):
        master_themes = master_themes_raw.get("master_themes", [])
    elif isinstance(master_themes_raw, list):
        master_themes = master_themes_raw

    pipeline = SynthesisPipeline(config)
    pipeline.execute(comm_topology=comm_topology, emb_topology=emb_topology, refined_triplets=refined_triplets, original_triplets=original_triplets, taxonomic_map=taxonomic_map, master_themes=master_themes)
    
    print("=== Stage 4 Synthesis Completed Successfully ===")

if __name__ == "__main__":
    main()
