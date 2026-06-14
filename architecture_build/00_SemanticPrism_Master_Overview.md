# SemanticPrism: Master Project Overview

## 1. Project Purpose
**SemanticPrism** is an advanced, agentic NLP pipeline designed to ingest entirely unstructured text corpuses and autonomously synthesize a rigorous, mathematically-backed **Pydantic Ontology Schema**. 

Unlike standard RAG or generative summarization pipelines, SemanticPrism seeks to fundamentally understand the architecture of a text. It extracts logical concepts, clusters them mathematically, visualizes their relationships via topological graphs, and then literally writes decoupled, Object-Oriented Python code (`BaseModel`s and `Enum`s).

**The Ultimate Goal:** The generated python schemas are engineered to be so accurate, descriptive, and robust that a completely separate "Data Extraction LLM" can immediately use those generated classes to rip structured data out of new documents without hallucinating or crashing.

## 2. Core Engineering Philosophies
1. **Sequential & Modular:** The project is broken into 4 perfectly isolated stages. Each stage must be able to run independently via a standalone script (e.g., `run_stage_1.py`), ingesting the JSON outputs of the previous stage. 
2. **Strictly Schema-Driven (Pydantic AI):** Gone are the days of parsing stringified JSON from an LLM. All LLM interactions in this project use **Pydantic AI Agents**. The LLM must conform to predefined data models natively.
3. **Parameter-Driven:** There are zero hardcoded thresholds, model names, or file paths in the source code. A global `config.yaml` file acts as the ultimate source of truth.
4. **Math over Guesswork:** We do not rely on an LLM to guess what topics are related. We use Vector Embeddings, Cosine Similarity, PageRank Centrality, and Leiden Modularity to mathematically prove relationships before the LLM ever writes a line of code.

## 3. The Pipeline Architecture

The build is separated into four distinct `.md` guides. You must execute and verify them sequentially.

### Stage 1: Extraction (`Stage1_Extraction_Build_Guide.md`)
*   **The Goal:** Rip the raw concepts out of the text.
*   **The Process:** Natively reads a batch of `.txt` documents from a config-defined directory. Chunks the text and uses Pydantic AI to extract generalized `Themes` and granular Subject-Verb-Object (SVO) `Triplets`.
*   **Key Mechanic:** Prioritizes data preservation. The extracted items are saved in a pristine, untouched `original_triplets.json` state.

### Stage 2: Refinement (`Stage2_Refinement_Build_Guide.md`)
*   **The Goal:** Normalize and consolidate synonymous concepts.
*   **The Process:** Cleans the text strings natively, then uses Vector Embeddings to cluster similar triples. Uses a geometric centroid to propose the best term.
*   **Key Mechanic:** "Taxonomic Lifting". If the mathematical cluster is confusing, an LLM selects the best "Hypernym" to represent the cluster. It saves a 1:1 `taxonomic_map.json` forward dictionary, then mutates and saves the SVOs to `refined_triplets.json`.

### Stage 3: Topology (`Stage3_Topology_Build_Guide.md`)
*   **The Goal:** Pure mathematics. Map the topological structure of the data.
*   **The Process:** Builds a NetworkX directed graph. Uses PageRank to mathematically locate massive "Death Star" Global Hubs. Prunes out isolated 1-degree "Orphans". Uses Leiden Modularity to partition the remaining graph into distinct semantic Communities.
*   **Key Mechanic:** Calculates `ThemeInheritance` using native Python set-intersection overlaps (e.g., "Do 80% of the concepts in B exist in A? Then B subclasses A"). Outputs interactive PyVis HTML graphs and a `topology_partitions.json` file.

### Stage 4: Synthesis (`Stage4_Synthesis_Build_Guide.md`)
*   **The Goal:** Generate the final Python codebase.
*   **The Process:** Runs a **Dual-Pass Generation Loop**. First, it processes the communities using the normalized hypernyms. Second, it uses the 1:1 `taxonomic_map.json` to filter the original data and pass the raw SVO text to the Schema Agent.
*   **Key Mechanic:** Outputs side-by-side schemas into `schemas/normalized/` and `schemas/raw/` for comparative analysis. Uses mathematical `ThemeInheritance` to natively write Object-Oriented subclassing (`class SpecificCommunity(GeneralHub):`).

## 4. The Pydantic Synthesis Skill (`Pydantic_Synthesis_Skill.md`)
Because the final output of this entire pipeline is *Python Code intended for another LLM to use*, that code must be bulletproof. This document provides the LLM with strict rules for code generation:
*   **Optional by Default:** Fields must be `Optional[str] = Field(default=None)` so missing data doesn't crash future extraction passes.
*   **Descriptive Fields:** Every field requires a rich `description="..."` parameter so future LLMs know what the field actually means.
*   **The "Escape Hatch" Principle:** Forces the LLM to inject `OTHER` values into Enums, paired with `_raw` string fields and generic `unmapped_attributes` dicts so unpredictable data has a place to land.

---
**Instructions for the Build LLM:**
Do not hallucinate steps ahead. Read the `Global_Architecture_Setup.md`, then proceed directly to `Stage 1` and build it. Stop and ask for verification before moving on.
