#!/usr/bin/env python
# coding: utf-8

# # SemanticPrism Diagnostic Workflow
# Step-by-step interactive workflow breaking down the pipeline execution.
# Now mathematically 1:1 functionally identical to the `pipeline.py` JSON storage footprint!
# 

# In[1]:


import os
import json
import glob
import time
from datetime import datetime
import logging
import IPython.display
import nest_asyncio
nest_asyncio.apply()

from typing import Any
from src.core.logger import save_execution_log
from src.extraction.extractor import ExtractionPipeline
from src.extraction.normalize_text import execute_normalization_phase
from src.embedding.embedding import EmbeddingPipeline
from src.nlp.hypernyms import HypernymPipeline
from src.nlp.nlp_mapping import NamingResolutionPipeline
from src.topology.graph_builder import TopologyEngine
from src.synthesis.synthesizer import SynthesisEngine
from src.helpers.visualizer import SemanticVisualizer

def _save_state(data: Any, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        default_encoder = lambda x: list(x) if isinstance(x, set) else str(x)
        if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'model_dump'):
            json.dump([item.model_dump(mode='json') for item in data], f, indent=4, default=default_encoder)
        elif hasattr(data, 'model_dump'):
            json.dump(data.model_dump(mode='json'), f, indent=4, default=default_encoder)
        else:
            json.dump(data, f, indent=4, default=default_encoder)

# Dynamically locate text files
target_dir = "inputs/testdocs"
files = glob.glob(os.path.join(target_dir, "*.txt")) + glob.glob(os.path.join(target_dir, "*.md"))
raw_texts = []
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        raw_texts.append(f.read())
print(f"Successfully loaded {len(raw_texts)} documents natively from {target_dir}.")
if not raw_texts:
    print("WARNING: Create physical text files manually inside inputs/testdocs gracefully.")


# ### 1. Extraction Pipeline
# 

# In[2]:


extractor = ExtractionPipeline("config.yaml")

start_time = time.time()
start_datetime = datetime.now()
workflow_errors = []
master_context = None

def _dump_current_log():
    all_errors = workflow_errors.copy()
    if hasattr(extractor, 'llm'): all_errors.extend(extractor.llm.error_history)
    if 'hypernyms' in globals() and hasattr(hypernyms, 'llm'): all_errors.extend(hypernyms.llm.error_history)
    if 'synthesizer' in globals() and hasattr(synthesizer, 'llm'): all_errors.extend(synthesizer.llm.error_history)
    
    all_ctxs = []
    if hasattr(extractor, 'llm'): all_ctxs.extend(extractor.llm.context_history)
    if 'hypernyms' in globals() and hasattr(hypernyms, 'llm'): all_ctxs.extend(hypernyms.llm.context_history)
    if 'synthesizer' in globals() and hasattr(synthesizer, 'llm'): all_ctxs.extend(synthesizer.llm.context_history)

    distilled_t_count = len(master_context.master_themes) if master_context and hasattr(master_context, 'master_themes') else 0
    
    metrics = {
        "start_datetime": start_datetime,
        "duration": time.time() - start_time,
        "use_async": getattr(extractor, 'use_async', False),
        "model_name": extractor.config.get('llm', {}).get('model_name', 'Unknown'),
        "connection_protocol": extractor.config.get('llm', {}).get('connection_protocol', 'Unknown'),
        "doc_count": len(raw_texts),
        "doc_lengths": [len(doc) for doc in raw_texts],
        "all_ctxs": all_ctxs,
        "all_themes_count": len(all_themes) if 'all_themes' in globals() else 0,
        "distilled_t_count": distilled_t_count,
        "raw_triples_count": len(raw_triples) if 'raw_triples' in globals() else 0,
        "orig_subjs": len(original_subjs) if 'original_subjs' in globals() else 0,
        "orig_preds": len(original_preds) if 'original_preds' in globals() else 0,
        "orig_objs": len(original_objs) if 'original_objs' in globals() else 0,
        "norm_subjs": len(norm_subjs) if 'norm_subjs' in globals() else 0,
        "norm_preds": len(norm_preds) if 'norm_preds' in globals() else 0,
        "norm_objs": len(norm_objs) if 'norm_objs' in globals() else 0,
        "all_errors": all_errors
    }
    workflow_logger = logging.getLogger("DiagnosticWorkflow")
    save_execution_log(metrics, workflow_logger)

# Baseline trackings 
original_subjs = set()
original_preds = set()
original_objs = set()
norm_subjs = set()
norm_preds = set()
norm_objs = set()

all_themes = []
for idx, text in enumerate(raw_texts):
    print(f"Processing themes for document {idx + 1}/{len(raw_texts)}")
    themes = await extractor.discover_themes(text)
    all_themes.extend(themes)
_save_state(all_themes, "outputs/01_extraction/original_themes.json")

weighted_string = extractor.weight_themes(all_themes)
master_context = await extractor.consolidate_themes(weighted_string)
_save_state(master_context, "outputs/01_extraction/distilled_themes.json")
_dump_current_log()

master_domain = master_context.master_domain if master_context else "General"
raw_triples = []
for idx, text in enumerate(raw_texts):
    print(f"Processing triples for document {idx + 1}/{len(raw_texts)}")
    trips = await extractor.extract_triples(text, master_context)
    raw_triples.extend(trips)
_save_state(raw_triples, "outputs/01_extraction/original_triplets.json")

print(f"Logically Extracted Triples seamlessly across all documents: {len(raw_triples)}")

visualizer = SemanticVisualizer()
visualizer.visualize_triples(raw_triples, "outputs/01_extraction/01_raw_triples_graph.html", "Phase 1: Raw Extractions")
IPython.display.display(IPython.display.IFrame("outputs/01_extraction/01_raw_triples_graph.html", width="100%", height="600px"))

original_subjs = {t.subject for t in raw_triples}
original_preds = {t.predicate for t in raw_triples}
original_objs = {t.object for t in raw_triples}

normalized_triples, norm_subjs, norm_preds, norm_objs = await execute_normalization_phase(
    extractor,
    raw_triples,
    master_domain,
    _save_state
)
_dump_current_log()

# ### 2. Embedding Compression
# 

# In[4]:


embedder = EmbeddingPipeline("config.yaml")

proposed_clusters = embedder.process_triples(normalized_triples)
_save_state(proposed_clusters, "outputs/02_embedding/clusters_identified.json")
_dump_current_log()

print(f"Mathematical Compression Output: {len(proposed_clusters)} distinct physical clusters grouped by Euclidean cosine thresholds.")


# ### 3. Taxonomic Hypernym Lifting (Hybrid)
# 

# In[5]:


hypernyms = HypernymPipeline("config.yaml")

verified_clusters = await hypernyms.validate_context_vectors(proposed_clusters, master_domain)
_save_state(verified_clusters, "outputs/03_hypernym_lifting/verified_clusters.json")

hypernym_mapping = await hypernyms.taxonomic_lift(verified_clusters, master_domain)
_save_state(hypernym_mapping, "outputs/03_hypernym_lifting/hypernym_mapping.json")
_dump_current_log()

print("Hybrid LLM Verification and taxonomic mapping extracted explicitly.")


# ### 4. Taxonomic Naming Resolution
# 

# In[6]:


mapper = NamingResolutionPipeline()

mapped_triples = mapper.resolve_names(normalized_triples, hypernym_mapping)
_save_state(mapped_triples, "outputs/04_mapping/mapped_triplets.json")
_dump_current_log()

visualizer.visualize_triples(mapped_triples, "outputs/04_mapping/02_resolved_triples_graph.html", "Phase 4: Abstracted Taxonomy")
IPython.display.display(IPython.display.IFrame("outputs/04_mapping/02_resolved_triples_graph.html", width="100%", height="600px"))


# ### 5. NetworkX Topological Community Graphing
# 

# In[7]:


topology = TopologyEngine()

graph = topology.build_graph(mapped_triples)
partition = topology.detect_communities(graph)
hierarchy = topology.extract_hierarchy(graph, partition)

_save_state(partition, "outputs/05_topology/modularity_partition.json")
_save_state(hierarchy, "outputs/05_topology/extracted_hierarchy.json")
_dump_current_log()

visualizer.visualize_topology(graph, partition, "outputs/05_topology/03_topology_communities_graph.html", "Phase 5: Modularity Topology")
IPython.display.display(IPython.display.IFrame("outputs/05_topology/03_topology_communities_graph.html", width="100%", height="600px"))


# ### 6. Semantic JSON Structure Synthesis
# 

# In[ ]:


synthesizer = SynthesisEngine("config.yaml")
resolved_schemas = await synthesizer.generate_schemas(hierarchy, master_domain)
file_path = synthesizer.build_global_context(resolved_schemas)
_dump_current_log()

print(f"Final output synthesized seamlessly to: {file_path}")

# ### 7. Logging & Diagnostic Preservation

# In[8]:

_dump_current_log()

# In[ ]:





# In[ ]:





# In[ ]:




