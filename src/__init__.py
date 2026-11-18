"""
SemanticPrism Core Library
==========================
Modular implementation of the 5-stage SemanticPrism pipeline:
  - src.extraction : Stage 1 Relational Extraction (Theme Discovery, Clustering & SVO Triple Parsing)
  - src.refinement : Stage 2 Hypernym Lifting (Two-Tier Vector Clustering, Taxonomic Lifting & Grounding)
  - src.topology   : Stage 3 Community Partitioning (Hub Decoupling, Leiden Modularity & Tier Stratification)
  - src.synthesis  : Stage 4 Schema Synthesis (AST-Verified Pydantic BaseModel & Enum Compilation)
  - src.mapping    : Stage 5 Tabular Projection (Schema-Constrained Ingestion & Relational/Wide Export)
"""
