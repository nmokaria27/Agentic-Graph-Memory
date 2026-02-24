"""
Knowledge Graph Visualization Module.

Provides multiple visualization options:
- Interactive HTML (pyvis)
- Static plots (networkx + matplotlib)
- Hierarchical layouts
- Force-directed graphs
"""

from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx and matplotlib not available. Install with: pip install networkx matplotlib")

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Warning: pyvis not available. Install with: pip install pyvis")


class KGVisualizer:
    """Visualize knowledge graphs with multiple backends."""
    
    def __init__(self, kg_data: Optional[Dict] = None, kg_file: Optional[str] = None):
        """
        Initialize visualizer with KG data or file.
        
        Args:
            kg_data: Dict containing 'entities' and 'triples'
            kg_file: Path to JSON file with KG export
        """
        if kg_file:
            with open(kg_file, 'r') as f:
                data = json.load(f)
                self.kg_data = data.get('knowledge_graph', data)
        elif kg_data:
            self.kg_data = kg_data
        else:
            self.kg_data = {'entities': [], 'triples': []}
        
        self.entities = self.kg_data.get('entities', [])
        self.triples = self.kg_data.get('triples', [])
    
    def visualize_interactive(
        self,
        output_file: str = "kg_visualization.html",
        height: str = "800px",
        width: str = "100%",
        notebook: bool = False,
    ) -> Optional[str]:
        """
        Create interactive HTML visualization using pyvis.
        
        Args:
            output_file: Path to save HTML file
            height: Height of visualization
            width: Width of visualization
            notebook: Whether running in Jupyter notebook
            
        Returns:
            Path to HTML file or None if pyvis not available
        """
        if not PYVIS_AVAILABLE:
            print("pyvis not available. Install with: pip install pyvis")
            return None
        
        net = Network(
            height=height,
            width=width,
            notebook=notebook,
            directed=True,
            bgcolor="#222222",
            font_color="white",
        )
        
        # Configure physics
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 200,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          }
        }
        """)
        
        # Color mapping for entity types
        entity_types = set(e.get('type', 'UNKNOWN') for e in self.entities)
        colors = self._generate_colors(len(entity_types))
        type_to_color = dict(zip(entity_types, colors))
        
        # Add nodes
        entity_id_map = {}
        for entity in self.entities:
            entity_id = entity.get('id', 'unknown')
            labels = entity.get('labels', [entity_id])
            label = labels[0] if labels else entity_id
            entity_type = entity.get('type', 'UNKNOWN')
            
            # Shorten long labels
            display_label = label if len(label) <= 40 else label[:37] + "..."
            
            net.add_node(
                entity_id,
                label=display_label,
                title=f"<b>{label}</b><br>Type: {entity_type}<br>ID: {entity_id}",
                color=type_to_color[entity_type],
                size=20,
            )
            entity_id_map[entity_id] = label
        
        # Add edges
        for triple in self.triples:
            subject = triple.get('subject', triple.get('head'))
            relation = triple.get('relation', triple.get('predicate'))
            obj = triple.get('object', triple.get('tail'))
            confidence = triple.get('confidence', 0.5)
            
            if subject and relation and obj:
                net.add_edge(
                    subject,
                    obj,
                    label=relation,
                    title=f"{relation} (conf: {confidence:.2f})",
                    width=confidence * 3,
                    color="#888888",
                )
        
        # Save and return
        net.save_graph(output_file)
        print(f"Interactive visualization saved to: {output_file}")
        return output_file
    
    def visualize_static(
        self,
        output_file: str = "kg_static.png",
        layout: str = "spring",
        figsize: Tuple[int, int] = (16, 12),
        node_size: int = 3000,
        font_size: int = 10,
    ) -> Optional[str]:
        """
        Create static visualization using networkx and matplotlib.
        
        Args:
            output_file: Path to save image
            layout: Layout algorithm (spring, circular, kamada_kawai, shell)
            figsize: Figure size (width, height)
            node_size: Size of nodes
            font_size: Font size for labels
            
        Returns:
            Path to image file or None if networkx not available
        """
        if not NETWORKX_AVAILABLE:
            print("networkx/matplotlib not available. Install with: pip install networkx matplotlib")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        entity_types = {}
        for entity in self.entities:
            entity_id = entity.get('id', 'unknown')
            labels = entity.get('labels', [entity_id])
            label = labels[0] if labels else entity_id
            entity_type = entity.get('type', 'UNKNOWN')
            
            # Shorten label
            display_label = label if len(label) <= 30 else label[:27] + "..."
            
            G.add_node(entity_id, label=display_label, type=entity_type)
            entity_types[entity_id] = entity_type
        
        # Add edges
        edge_labels = {}
        for triple in self.triples:
            subject = triple.get('subject', triple.get('head'))
            relation = triple.get('relation', triple.get('predicate'))
            obj = triple.get('object', triple.get('tail'))
            confidence = triple.get('confidence', 0.5)
            
            if subject and relation and obj:
                G.add_edge(subject, obj, relation=relation, confidence=confidence)
                edge_labels[(subject, obj)] = relation
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        ax.axis('off')
        
        # Color nodes by type
        unique_types = list(set(entity_types.values()))
        colors = self._generate_colors(len(unique_types))
        type_to_color = dict(zip(unique_types, colors))
        node_colors = [type_to_color[entity_types[node]] for node in G.nodes()]
        
        # Draw graph
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_size,
            alpha=0.9,
            ax=ax,
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#666666',
            arrows=True,
            arrowsize=20,
            width=2,
            alpha=0.6,
            ax=ax,
            connectionstyle="arc3,rad=0.1",
        )
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=font_size,
            font_color='white',
            font_weight='bold',
            ax=ax,
        )
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=font_size - 2,
            font_color='#aaaaaa',
            ax=ax,
        )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=type_to_color[t], edgecolor='white', label=t)
            for t in unique_types
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=font_size,
            facecolor='#2e2e2e',
            edgecolor='white',
            labelcolor='white',
        )
        
        plt.title(
            f"Knowledge Graph: {len(G.nodes())} entities, {len(G.edges())} relations",
            color='white',
            fontsize=14,
            pad=20,
        )
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
        print(f"Static visualization saved to: {output_file}")
        plt.close()
        
        return output_file
    
    def print_stats(self):
        """Print KG statistics."""
        print(f"\n{'='*60}")
        print(f"KNOWLEDGE GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"Entities: {len(self.entities)}")
        print(f"Triples: {len(self.triples)}")
        
        # Entity types
        entity_types = {}
        for e in self.entities:
            t = e.get('type', 'UNKNOWN')
            entity_types[t] = entity_types.get(t, 0) + 1
        
        print(f"\nEntity Types:")
        for t, count in sorted(entity_types.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")
        
        # Relation types
        relation_types = {}
        for r in self.triples:
            rel = r.get('relation', r.get('predicate', 'UNKNOWN'))
            relation_types[rel] = relation_types.get(rel, 0) + 1
        
        print(f"\nRelation Types:")
        for r, count in sorted(relation_types.items(), key=lambda x: -x[1]):
            print(f"  {r}: {count}")
        
        print(f"{'='*60}\n")
    
    @staticmethod
    def _generate_colors(n: int) -> List[str]:
        """Generate n visually distinct colors."""
        if n == 0:
            return []
        
        # Use HSV color space for good distribution
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255),
            )
            colors.append(hex_color)
        return colors


def visualize_kg_from_file(
    kg_file: str,
    output_html: str = "kg_interactive.html",
    output_png: str = "kg_static.png",
    create_interactive: bool = True,
    create_static: bool = True,
):
    """
    Convenience function to visualize KG from export file.
    
    Args:
        kg_file: Path to kg_export.json
        output_html: Path for interactive HTML
        output_png: Path for static PNG
        create_interactive: Whether to create interactive viz
        create_static: Whether to create static viz
    """
    viz = KGVisualizer(kg_file=kg_file)
    viz.print_stats()
    
    if create_interactive:
        viz.visualize_interactive(output_html)
    
    if create_static:
        viz.visualize_static(output_png, layout="spring")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        kg_file = sys.argv[1]
    else:
        kg_file = "kg_export.json"
    
    if Path(kg_file).exists():
        print(f"Visualizing: {kg_file}")
        visualize_kg_from_file(kg_file)
    else:
        print(f"File not found: {kg_file}")
        print("Usage: python kg_visualizer.py <kg_export.json>")
