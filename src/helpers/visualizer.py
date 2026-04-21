"""
SemanticPrism: Semantic Visualizer Node
Provides offline pure python visual abstraction plotting utilizing PyVis safely securely.
"""

import networkx as nx
from pyvis.network import Network
from typing import List
import os
from src.extraction.schemas import RawTriple
from src.core.logger import get_logger

logger = get_logger("Visualizer")

class SemanticVisualizer:
    def __init__(self):
        logger.info("Initializing Static Network Visualizer efficiently rationally.")
        self.colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#9B59B6", "#1ABC9C", "#E67E22", "#E74C3C", "#8E44AD", "#3498DB"]

    def _create_base_network(self, title: str) -> Network:
        """
        Synthesizes a standardized robustly styled interactive HTML block rationally.
        Uses dark mode intuitively explicitly cleanly realistically natively.
        """
        net = Network(notebook=False, directed=True, bgcolor="#222222", font_color="white", width="100%", height="800px", heading=title)
        net.repulsion(node_distance=200, spring_length=250)
        return net
        
    def visualize_triples(self, triples: List[RawTriple], filepath: str, title: str = "Extracted Graph Topology"):
        """
        Translates raw decoupled extracted triples cleanly onto visual matrices elegantly organically.
        """
        logger.info(f"Generating abstract visual organically seamlessly correctly optimally for {len(triples)} logical facts securely.")
        net = self._create_base_network(title)
        
        for t in triples:
            if t.subject and t.object:
                # Bypass duplicate identical nodes conditionally elegantly functionally stringently dynamically legitimately
                if str(t.subject) not in net.get_nodes():
                    net.add_node(str(t.subject), label=str(t.subject), shape="dot", size=20, color="#1ABC9C")
                if str(t.object) not in net.get_nodes():
                    net.add_node(str(t.object), label=str(t.object), shape="dot", size=20, color="#3498DB")
                    
                label_txt = str(t.predicate)
                net.add_edge(str(t.subject), str(t.object), title=label_txt, label=label_txt, color="#BDC3C7")
                
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            net.write_html(filepath)
            logger.info(f"Visual flawlessly written to: {filepath}")
        except Exception as e:
            logger.warning(f"Visualization extraction manually disrupted natively: {e}")

    def visualize_topology(self, graph: nx.DiGraph, partition: dict, filepath: str, title: str = "Semantic Modularity Isolation"):
        """
        Generates unified community detection visuals structurally correctly dependably securely authentically.
        """
        logger.info(f"Generating geometric topology abstraction logically natively safely.")
        net = self._create_base_network(title)
        
        for node in graph.nodes():
            comm = partition.get(node, 0)
            node_clr = self.colors[comm % len(self.colors)]
            
            node_weight = graph.degree(node) * 5
            node_size = min(max(node_weight, 15), 50)
            
            net.add_node(str(node), label=str(node), title=f"Community {comm}", color=node_clr, shape="dot", size=node_size)
            
        for u, v, data in graph.edges(data=True):
            preds = list(data.get("predicates", []))
            lbl = " | ".join(preds) if preds else data.get("predicate", "associates")
            net.add_edge(str(u), str(v), title=lbl, label=lbl, color="#95A5A6")
            
        try:
            net.write_html(filepath)
            logger.info(f"Structural Geometric Community visually exported to: {filepath}")
        except Exception as e:
            logger.warning(f"Geometric visual extraction failed: {e}")
