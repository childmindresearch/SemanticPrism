# SemanticPrism

SemanticPrism is an automated pipeline that transforms unstructured domain knowledge (such as clinical reports or technical documentation) into structured, deployable Python ontologies (Pydantic models).

Instead of generating schemas in a single LLM prompt—which often leads to hallucinations and overlapping concepts—SemanticPrism extracts atomic facts, groups them with vector clustering, maps them into mathematical graph networks, and uses topological boundaries to synthesize clean, modular data models.

---

## Pipeline Overview

```mermaid
graph TD
    Inputs["Source Documents<br/>(Text / Parquet)"] --> S1["Stage 1: Extraction<br/><i>Theme Discovery & Triplet Extraction</i>"]
    S1 -->|"Semantic Triplets"| S2["Stage 2: Refinement<br/><i>Vector Clustering & Taxonomic Lifting</i>"]
    S2 -->|"Refined Triplets"| S3["Stage 3: Graph Topology<br/><i>Community & Structural Partitioning</i>"]
    S3 -->|"Graph Partitions"| S4["Stage 4: Synthesis<br/><i>Pydantic Schema & Enum Generation</i>"]
    S4 --> Outputs["Python Ontologies<br/>(Pydantic Models)"]
```

---

## Quickstart

### Prerequisites
- **Python**: 3.10+
- **Local LLM**: [Ollama](https://ollama.com/) (e.g., `qwen2.5:7b`)

### Installation
```bash
git clone https://github.com/childmindresearch/SemanticPrism.git
cd SemanticPrism
pip install -r requirements.txt
```

### Running the Pipeline

Execute the full end-to-end pipeline:
```bash
python3 run_pipeline.py
```

Or run individual stages independently:
- **Stage 1 (Extraction)**: `python3 run_stage_1_extraction.py`
- **Stage 2 (Refinement)**: `python3 run_stage_2_refinement.py`
- **Stage 3 (Topology)**: `python3 run_stage_3_topology.py`
- **Stage 4 (Synthesis)**: `python3 run_stage_4_synthesis.py`

---

## Pipeline Stages

### Stage 1: Extraction
Ingests plain text files or Parquet datasets, identifies overarching domain themes, and extracts structured Subject-Predicate-Object (S-P-O) semantic triplets anchored to verbatim source quotes.

### Stage 2: Refinement
Encodes extracted entities and relations into vector embeddings, groups them into semantic clusters, and optionally maps terms into standardized taxonomic concepts using LLM hypernym resolution.

### Stage 3: Graph Topology
Builds a directed network graph from the refined triplets and partitions the network along two complementary paths:
- **Community Workflow Path**: Uses the Leiden algorithm to identify modular event sequences and functional subgraphs.
- **Embedding Categorical Path**: Uses Node2Vec graph embeddings and K-Means to identify structural categories.

### Stage 4: Synthesis
Translates graph partitions and low-density nodes into deployable Python code:
- Generates Python `Enum` classes for orphan and boundary entities (`enums.py`).
- Synthesizes Pydantic `BaseModel` classes for qualified clusters.
- Consolidates schemas and formats the final `comprehensive_ontology.py` using Ruff.

---

## Configuration (`configs/`)

Pipeline behavior is configured via modular YAML files:

- **[configs/llm.yaml](configs/llm.yaml)**: Local Ollama model name, API endpoints, temperature, and context window limits.
- **[configs/io.yaml](configs/io.yaml)**: Ingestion source (`directory` vs. `parquet`), input/output paths, and resume modes (`skip` vs. `overwrite`).
- **[configs/refinement.yaml](configs/refinement.yaml)**: Embedding models, taxonomic lifting toggles, target theme count guidance, and clustering thresholds.
- **[configs/topology.yaml](configs/topology.yaml)**: Execution mode (`community`, `embedding`, `both`), Leiden modularity, and Node2Vec parameters.
- **[configs/synthesis.yaml](configs/synthesis.yaml)**: Minimum cluster sizes, hub target caps, and payload options.

---

## Output Structure (`outputs/`)

```text
outputs/
├── 01_extraction/          # Raw triplets, discovered themes, and master theme mappings
├── 02_refinement/          # Refined triplets, vector cluster mappings, and taxonomic maps
├── 03_topology/            # Community and embedding graph partitions
├── 04_synthesis/           # Generated Pydantic schemas, enums, and comprehensive ontologies
├── payload_text/           # Centralized record of prompts and payloads sent to the LLM
└── visuals/                # Interactive PyVis HTML network graphs
```
