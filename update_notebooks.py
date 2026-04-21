import json
import os

NOTEBOOKS = ["diagnostic_workflow.ipynb", "SemanticPrism_Demo.ipynb"]

DUMP_LOG_CODE = [
    "master_context = None\n",
    "\n",
    "def _dump_current_log():\n",
    "    all_errors = workflow_errors.copy()\n",
    "    if hasattr(extractor, 'llm'): all_errors.extend(extractor.llm.error_history)\n",
    "    if 'hypernyms' in globals() and hasattr(hypernyms, 'llm'): all_errors.extend(hypernyms.llm.error_history)\n",
    "    if 'synthesizer' in globals() and hasattr(synthesizer, 'llm'): all_errors.extend(synthesizer.llm.error_history)\n",
    "    \n",
    "    all_ctxs = []\n",
    "    if hasattr(extractor, 'llm'): all_ctxs.extend(extractor.llm.context_history)\n",
    "    if 'hypernyms' in globals() and hasattr(hypernyms, 'llm'): all_ctxs.extend(hypernyms.llm.context_history)\n",
    "    if 'synthesizer' in globals() and hasattr(synthesizer, 'llm'): all_ctxs.extend(synthesizer.llm.context_history)\n",
    "\n",
    "    distilled_t_count = len(master_context.master_themes) if master_context and hasattr(master_context, 'master_themes') else 0\n",
    "    \n",
    "    metrics = {\n",
    "        \"start_datetime\": start_datetime,\n",
    "        \"duration\": time.time() - start_time,\n",
    "        \"use_async\": getattr(extractor, 'use_async', False),\n",
    "        \"model_name\": extractor.config.get('llm', {}).get('model_name', 'Unknown'),\n",
    "        \"connection_protocol\": extractor.config.get('llm', {}).get('connection_protocol', 'Unknown'),\n",
    "        \"doc_count\": len(raw_texts),\n",
    "        \"doc_lengths\": [len(doc) for doc in raw_texts],\n",
    "        \"all_ctxs\": all_ctxs,\n",
    "        \"all_themes_count\": len(all_themes) if 'all_themes' in globals() else 0,\n",
    "        \"distilled_t_count\": distilled_t_count,\n",
    "        \"raw_triples_count\": len(raw_triples) if 'raw_triples' in globals() else 0,\n",
    "        \"orig_subjs\": len(original_subjs) if 'original_subjs' in globals() else 0,\n",
    "        \"orig_preds\": len(original_preds) if 'original_preds' in globals() else 0,\n",
    "        \"orig_objs\": len(original_objs) if 'original_objs' in globals() else 0,\n",
    "        \"norm_subjs\": len(norm_subjs) if 'norm_subjs' in globals() else 0,\n",
    "        \"norm_preds\": len(norm_preds) if 'norm_preds' in globals() else 0,\n",
    "        \"norm_objs\": len(norm_objs) if 'norm_objs' in globals() else 0,\n",
    "        \"all_errors\": all_errors\n",
    "    }\n",
    "    workflow_logger = logging.getLogger(\"DiagnosticWorkflow\")\n",
    "    save_execution_log(metrics, workflow_logger)\n"
]

for nb_name in NOTEBOOKS:
    if not os.path.exists(nb_name): continue
    
    with open(nb_name, "r") as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        if cell["cell_type"] != "code": continue
        
        src = cell["source"]
        
        # Inject the closure right after workflow_errors = []
        for i, line in enumerate(src):
            if line.strip() == "workflow_errors = []":
                # Check if already injected
                if i+1 < len(src) and "master_context = None" in src[i+1]:
                    continue
                # Inject
                src = src[:i+1] + ["\n"] + DUMP_LOG_CODE + src[i+1:]
                cell["source"] = src
                break
                
        # Inject _dump_current_log() after _save_state occurrences
        src = cell["source"]
        new_src = []
        for line in src:
            new_src.append(line)
            if "_save_state" in line and "def _save_state" not in line and "_dump_current_log()" not in "".join(src):
                # only safely inject if it's a standalone call
                if ".json" in line:
                    new_src.append("_dump_current_log()\n")
        cell["source"] = new_src
        
        # Remove the final block gathering metrics since it's now handled by the closure
        src = cell["source"]
        if "metrics = {" in "".join(src) and "all_errors =" in "".join(src):
            if "def _dump_current_log" not in "".join(src):
                # This is the final cell! Replace it with just _dump_current_log()
                cell["source"] = ["_dump_current_log()\n"]
                
    with open(nb_name, "w") as f:
        json.dump(nb, f, indent=1)
        
print("Updated notebooks successfully.")
