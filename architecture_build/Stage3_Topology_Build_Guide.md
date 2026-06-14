# SemanticPrism Stage 3: Topology Pipeline Build Specification

This document provides explicit instructions for building the Stage 3 Topology phase. This stage uses pure mathematics and graph theory (no LLMs) to convert the refined triplets into a partitioned structural map, explicitly preparing the data for a "Hub-and-Spoke" code synthesis architecture.

## 1. Architectural Layout
Construct the module using cleanly separated files inside `src/topology/`:
1.  `schemas.py`: Contains strictly Pydantic data models for structural outputs.
2.  `graph_builder.py`: Contains the NetworkX logic, Leiden clustering, Spectral math, simplified Hypergraph set-logic, and the PyVis visualization generation.

---

## 2. Schemas (`src/topology/schemas.py`)
Implement the following Pydantic models to strictly enforce the output structure of the math pipeline.

### Core Models:
1.  **`NodeMetrics`**: 
    *   Fields: `node_id` (str), `degree_centrality` (float), `pagerank` (float), `is_hub` (bool), `is_orphan` (bool).
2.  **`CommunityPartition`**: 
    *   Fields: `community_id` (int), `nodes` (List[str]).
3.  **`ThemeInheritance`**:
    *   Fields: `parent_theme` (str), `child_theme` (str), `overlap_score` (float).
4.  **`TopologyResult`**: 
    *   Fields: `global_hubs` (List[str]), `communities` (List[CommunityPartition]), `orphans` (List[str]), `node_metrics` (Dict[str, NodeMetrics]), `theme_inheritance` (List[ThemeInheritance]).

---

## 3. Master Execution Pipeline (`graph_builder.py`)

Create a master class `TopologyPipeline` to execute the sequence logically. It must accept the global `config` mapping.

### Step 3.1: NetworkX Graph Construction
1.  **Ingestion:** Load the `refined_triplets.json` from Stage 2.
2.  **Directed Graph:** Initialize a `networkx.DiGraph()`. Add an edge for every triple: `graph.add_edge(subject, object, predicate=predicate)`. If an edge already exists, increment a `weight` attribute to represent structural frequency.
3.  **Undirected Graph:** Create a parallel undirected copy `networkx.Graph(directed_graph)` for use in community detection algorithms that do not natively support directed edges.

### Step 3.2: Spectral Hub & Orphan Identification
1.  **Calculate Centralities:** Run `nx.pagerank()` and `nx.degree_centrality()` on the directed graph.
2.  **Identify Super Hubs:** Define "Global Hubs" (the "Death Star" nodes) by taking the top N% of nodes based on PageRank (where N is derived from `config['topology']['spectral_variance_retention']`). These represent global protocols or base interfaces.
3.  **Identify Orphans:** Identify nodes with a degree of 1 (or isolated). These represent loose enums, literal strings, or orphan properties.
4.  **Populate Metrics:** Map these calculations back into a dictionary of `NodeMetrics` objects.

### Step 3.3: Leiden Modularity (Community Detection)
*Context: If hubs are left in the graph, they pull disparate communities together. If orphans are left, they create thousands of micro-clusters.*
1.  **Graph Pruning:** Create a "Sub-Graph" by stripping out all identified `global_hubs` and `orphans` from the undirected graph.
2.  **Leiden Clustering:** Run the Leiden Modularity algorithm on the pruned sub-graph using the `leiden_resolution` defined in `config.yaml`.
3.  **Format Partitions:** Convert the algorithm output into a list of `CommunityPartition` objects.

### Step 3.4: Hypergraph Theme Inheritance (Set-Based)
*Replaces legacy matrix Laplacian math with highly efficient native Python sets to calculate semantic inheritance.*
1.  **Map Entities to Themes:** Group all nodes (entities) associated with each Theme. Store as `theme_sets = {"ThemeA": set(nodes), "ThemeB": set(nodes)}`.
2.  **Calculate Overlap:** For every pair of themes (A, B), calculate the percentage of entities in B that also exist in A using a set intersection: `overlap_score = len(theme_sets[A].intersection(theme_sets[B])) / len(theme_sets[B])`.
3.  **Enforce Threshold:** If `overlap_score >= config['topology']['inheritance_overlap_threshold']`, create a `ThemeInheritance` record indicating that B logically inherits from A.
4.  **Append to Output:** Attach the resulting `ThemeInheritance` relationships to the `TopologyResult`.

### Step 3.5: Data Persistence
1.  **Assemble Final Payload:** Instantiate the master `TopologyResult` object using the hubs, orphans, communities, metrics, and theme inheritance logic generated above.
2.  **Save State:** Serialize the `TopologyResult` to JSON and save it to `outputs/03_topology/topology_partitions.json`. 

### Step 3.6: PyVis Native Visualizations
*Store all interactive HTML graphs exclusively in the global `outputs/visuals/` directory.*
1.  **Standard Topology Visual:** Iterate through the `TopologyResult`. Color `global_hubs` Red (large scale), `orphans` Gray (small scale), and `communities` distinct categorical colors. Add original edges. Save HTML to `outputs/visuals/interactive_topology_graph.html`.
2.  **Hypergraph Visual:** Create a secondary `pyvis.network.Network` to visualize the N-ary inheritance structure. Add Themes as central parent nodes (Colored Blue, extremely large scale). Add their associated entities as child nodes connected to their respective Themes. Highlight the `ThemeInheritance` relationships with heavy directed arrows between the Theme nodes. Save HTML to `outputs/visuals/interactive_hypergraph.html`.

---

## 6. Isolated Execution (`run_stage_3.py`)
To ensure strict modularity, this stage must be executable in complete isolation. Create a standalone `run_stage_3.py` execution script that initializes the pipeline, loads the `refined_triplets.json` from Stage 2, executes the `TopologyPipeline`, and cleanly saves its partitions and interactive HTML graphs. This guarantees that the stage can be tested, debugged, and run entirely independently.
