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

        # Build label→ID lookup for resolving triple endpoints
        self._label_to_id: Dict[str, str] = {}
        self._entity_id_set: set = set()
        for e in self.entities:
            eid = e.get('id', 'unknown')
            self._entity_id_set.add(eid)
            self._label_to_id[eid.lower()] = eid
            for lbl in e.get('labels', []):
                self._label_to_id[lbl.lower()] = eid
            if e.get('text'):
                self._label_to_id[e['text'].lower()] = eid

    def _resolve_endpoint(self, name: str) -> Optional[str]:
        """Resolve a triple endpoint (subject/object text) to an entity ID."""
        if not name:
            return None
        if name in self._entity_id_set:
            return name
        return self._label_to_id.get(name.lower())
    
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
        
        # Add edges (resolve subject/object text → entity ID)
        existing_nodes = set(net.get_nodes())
        for triple in self.triples:
            subject = triple.get('subject', triple.get('head'))
            relation = triple.get('relation', triple.get('predicate'))
            obj = triple.get('object', triple.get('tail'))
            confidence = triple.get('confidence', 0.5)

            if subject and relation and obj and subject != obj:
                s_id = self._resolve_endpoint(subject)
                o_id = self._resolve_endpoint(obj)
                if s_id and o_id and s_id in existing_nodes and o_id in existing_nodes and s_id != o_id:
                    net.add_edge(
                        s_id,
                        o_id,
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
        
        # Add edges (resolve subject/object text → entity ID)
        edge_labels = {}
        for triple in self.triples:
            subject = triple.get('subject', triple.get('head'))
            relation = triple.get('relation', triple.get('predicate'))
            obj = triple.get('object', triple.get('tail'))
            confidence = triple.get('confidence', 0.5)

            if subject and relation and obj and subject != obj:
                s_id = self._resolve_endpoint(subject)
                o_id = self._resolve_endpoint(obj)
                if s_id and o_id and s_id in entity_types and o_id in entity_types and s_id != o_id:
                    G.add_edge(s_id, o_id, relation=relation, confidence=confidence)
                    edge_labels[(s_id, o_id)] = relation
        
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


    def visualize_interactive_enhanced(
        self,
        output_file: str = "kg_explorer.html",
        qa_results: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a self-contained interactive HTML visualization with vis.js
        and a QA panel that highlights evidence triples when clicked.

        Args:
            output_file: Path to save the HTML file
            qa_results: List of QA result dicts from QAOrchestrator

        Returns:
            Path to generated HTML file
        """
        # Build entity type color map
        entity_types = sorted(set(e.get('type', 'UNKNOWN') for e in self.entities))
        colors = self._generate_colors(len(entity_types))
        type_to_color = dict(zip(entity_types, colors))

        # Build nodes JSON
        nodes = []
        for entity in self.entities:
            eid = entity.get('id', 'unknown')
            labels = entity.get('labels', [eid])
            label = labels[0] if labels else eid
            entity_type = entity.get('type', 'UNKNOWN')
            display_label = label if len(label) <= 40 else label[:37] + "..."
            nodes.append({
                "id": eid,
                "label": display_label,
                "title": f"<b>{label}</b><br>Type: {entity_type}<br>ID: {eid}",
                "color": type_to_color.get(entity_type, "#888888"),
                "size": 20,
                "font": {"color": "white", "size": 12},
            })

        # Build edges JSON — resolve text labels to entity IDs
        edges = []
        connected_ids = set()
        for i, triple in enumerate(self.triples):
            subject = triple.get('subject', triple.get('head'))
            relation = triple.get('relation', triple.get('predicate'))
            obj = triple.get('object', triple.get('tail'))
            confidence = triple.get('confidence', 0.5)
            if subject and relation and obj and subject != obj:
                s_id = self._resolve_endpoint(subject)
                o_id = self._resolve_endpoint(obj)
                if s_id and o_id and s_id != o_id:
                    connected_ids.add(s_id)
                    connected_ids.add(o_id)
                    edges.append({
                        "id": f"edge_{i}",
                        "from": s_id,
                        "to": o_id,
                        "label": relation,
                        "title": f"{relation} (conf: {confidence:.2f})",
                        "width": max(1, confidence * 3),
                        "color": {"color": "#888888", "highlight": "#00ff88"},
                        "font": {"color": "#cccccc", "size": 10, "strokeWidth": 0},
                        "arrows": "to",
                    })

        # Stats
        pct_connected = (len(connected_ids) / len(self.entities) * 100) if self.entities else 0

        # Legend HTML
        legend_items = "".join(
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{c};margin-right:6px;"></span>'
            f'<span style="color:#ccc;font-size:11px;">{t}</span></div>'
            for t, c in type_to_color.items()
        )

        # QA panel HTML — collect evidence from domain_responses[].evidence
        qa_cards_html = ""
        qa_data_js = "[]"
        if qa_results:
            qa_items = []
            for qi, qr in enumerate(qa_results):
                question = qr.get("question", f"Question {qi+1}")
                answer = qr.get("final_answer", "No answer")
                confidence = qr.get("overall_confidence", 0)
                coverage = qr.get("overall_coverage", 0)
                domain_responses = qr.get("domain_responses", [])

                # Collect evidence triples from domain responses
                all_evidence = []
                for dr in domain_responses:
                    for ev in dr.get("evidence", []):
                        ev_str = ev if isinstance(ev, str) else str(ev)
                        if ev_str not in all_evidence:
                            all_evidence.append(ev_str)

                # Domain badges + details
                domain_badges = ""
                domain_details = ""
                for dr in domain_responses:
                    did = dr.get("domain_id", "?")
                    dconf = dr.get("confidence", 0)
                    domain_badges += f'<span class="badge domain-badge">{did}</span>'
                    topics_used = dr.get("topics_used", [])
                    topics_str = ", ".join(topics_used) if topics_used else "all"
                    domain_details += (
                        f'<div class="domain-detail">'
                        f'<b>{did}</b> (conf: {dconf:.2f}) &mdash; topics: {topics_str}'
                        f'</div>'
                    )

                # Evidence list HTML
                evidence_html = ""
                for ev in all_evidence:
                    evidence_html += f'<li class="evidence-item">{ev}</li>'

                qa_items.append({
                    "question": question,
                    "evidence_triples": all_evidence,
                })

                conf_color = "#4caf50" if confidence >= 0.7 else "#ff9800" if confidence >= 0.4 else "#f44336"
                cov_color = "#4caf50" if coverage >= 0.7 else "#ff9800" if coverage >= 0.4 else "#f44336"

                qa_cards_html += f'''
                <div class="qa-card" data-qi="{qi}" onclick="toggleQA({qi})">
                  <div class="qa-question">Q{qi+1}: {question}</div>
                  <div class="qa-badges">
                    {domain_badges}
                    <span class="badge" style="background:{conf_color}">conf: {confidence:.2f}</span>
                    <span class="badge" style="background:{cov_color}">cov: {coverage:.2f}</span>
                  </div>
                  <div class="qa-details" id="qa-details-{qi}" style="display:none;">
                    <div class="qa-answer">{answer}</div>
                    <div class="qa-section-label">Evidence Triples ({len(all_evidence)}):</div>
                    <ul class="evidence-list">{evidence_html if evidence_html else "<li style='color:#666'>No evidence triples</li>"}</ul>
                    <div class="qa-section-label">Domain Experts:</div>
                    {domain_details}
                  </div>
                </div>'''

            qa_data_js = json.dumps(qa_items)

        # Build label→ID lookup for JS-side matching
        js_label_map = json.dumps(self._label_to_id)

        nodes_js = json.dumps(nodes)
        edges_js = json.dumps(edges)

        html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Knowledge Graph Explorer</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; height: 100vh; overflow: hidden; }}
.container {{ display: flex; height: 100vh; }}
.graph-panel {{ flex: 0 0 65%; position: relative; border-right: 2px solid #333; }}
#graph {{ width: 100%; height: 100%; }}
.qa-panel {{ flex: 0 0 35%; overflow-y: auto; padding: 16px; background: #16213e; }}
.stats-overlay {{ position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.75); padding: 10px 14px; border-radius: 8px; font-size: 12px; z-index: 10; }}
.stats-overlay div {{ margin: 2px 0; }}
.legend-overlay {{ position: absolute; bottom: 12px; left: 12px; background: rgba(0,0,0,0.75); padding: 10px 14px; border-radius: 8px; z-index: 10; }}
.qa-panel h2 {{ margin-bottom: 12px; color: #00ff88; font-size: 16px; }}
.qa-card {{ background: #1a1a3e; border: 1px solid #333; border-radius: 8px; padding: 12px; margin-bottom: 10px; cursor: pointer; transition: border-color 0.2s; }}
.qa-card:hover {{ border-color: #00ff88; }}
.qa-card.active {{ border-color: #00ff88; box-shadow: 0 0 8px rgba(0,255,136,0.3); }}
.qa-question {{ font-weight: 600; font-size: 13px; margin-bottom: 6px; }}
.qa-badges {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; color: white; background: #555; }}
.domain-badge {{ background: #0a3d62; }}
.qa-details {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #444; font-size: 12px; }}
.qa-answer {{ color: #b0e0ff; margin-bottom: 8px; line-height: 1.5; max-height: 200px; overflow-y: auto; }}
.qa-section-label {{ font-weight: 600; color: #888; margin: 8px 0 4px; font-size: 11px; text-transform: uppercase; }}
.evidence-list {{ list-style: none; padding: 0; }}
.evidence-item {{ padding: 3px 6px; margin: 2px 0; border-radius: 4px; font-size: 11px; font-family: monospace; background: #222; }}
.evidence-item.matched {{ background: rgba(0,255,136,0.15); border-left: 3px solid #00ff88; }}
.evidence-item.unmatched {{ background: rgba(255,68,68,0.1); border-left: 3px solid #f44336; }}
.domain-detail {{ font-size: 11px; color: #aaa; margin: 2px 0; padding: 2px 6px; }}
.no-qa {{ color: #666; font-style: italic; padding: 20px; text-align: center; }}
.ask-box {{ margin-bottom: 14px; }}
.ask-box textarea {{ width: 100%; background: #1a1a3e; border: 1px solid #444; border-radius: 6px; color: #eee; padding: 8px; font-size: 13px; resize: vertical; min-height: 40px; font-family: inherit; }}
.ask-box textarea:focus {{ outline: none; border-color: #00ff88; }}
.ask-box button {{ margin-top: 6px; background: #00ff88; color: #111; border: none; border-radius: 6px; padding: 6px 16px; font-weight: 600; font-size: 12px; cursor: pointer; }}
.ask-box button:hover {{ background: #00cc6e; }}
.ask-box button:disabled {{ background: #555; color: #999; cursor: not-allowed; }}
.ask-box .ask-status {{ font-size: 11px; color: #888; margin-top: 4px; }}
</style>
</head>
<body>
<div class="container">
  <div class="graph-panel">
    <div class="stats-overlay">
      <div><b>Entities:</b> {len(self.entities)}</div>
      <div><b>Triples:</b> {len(edges)}</div>
      <div><b>Connected:</b> {len(connected_ids)}/{len(self.entities)} ({pct_connected:.0f}%)</div>
    </div>
    <div class="legend-overlay">{legend_items}</div>
    <div id="graph"></div>
  </div>
  <div class="qa-panel">
    <h2>KG Evidence Explorer</h2>
    <div class="ask-box">
      <textarea id="user-question" placeholder="Type your own question..."></textarea>
      <button id="ask-btn" onclick="askQuestion()">Ask (requires backend)</button>
      <div class="ask-status" id="ask-status">Tip: run the QA server to enable live questions</div>
    </div>
    <div id="qa-cards">
    {qa_cards_html if qa_results else "<div class='no-qa'>No QA results loaded</div>"}
    </div>
  </div>
</div>
<script>
var nodesData = new vis.DataSet({nodes_js});
var edgesData = new vis.DataSet({edges_js});
var labelToId = {js_label_map};
var container = document.getElementById("graph");
var data = {{ nodes: nodesData, edges: edgesData }};
var options = {{
  physics: {{
    solver: "forceAtlas2Based",
    forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01, springLength: 200, springConstant: 0.08 }},
    maxVelocity: 50, timestep: 0.35,
    stabilization: {{ iterations: 150 }}
  }},
  interaction: {{ hover: true, tooltipDelay: 100 }},
  edges: {{ smooth: {{ type: "continuous" }} }}
}};
var network = new vis.Network(container, data, options);

var qaData = {qa_data_js};
var activeQA = -1;
var originalEdges = {{}};
var originalNodes = {{}};

// Save original styles
edgesData.forEach(function(e) {{ originalEdges[e.id] = {{ color: JSON.parse(JSON.stringify(e.color || {{}})), width: e.width || 1 }}; }});
nodesData.forEach(function(n) {{ originalNodes[n.id] = {{ color: n.color, borderWidth: n.borderWidth || 1 }}; }});

function resolveId(name) {{
  if (!name) return null;
  // Direct node check
  if (nodesData.get(name)) return name;
  // Label lookup
  var lower = name.toLowerCase();
  if (labelToId[lower]) return labelToId[lower];
  return null;
}}

function parseEvidence(evStr) {{
  // Format 1: "(entity) -[RELATION]-> (entity) (conf=X.XX)"
  var re1 = /\\(([^)]+)\\)\\s*-\\[([^\\]]+)\\]->\\s*\\(([^)]+)\\)/;
  var m = evStr.match(re1);
  if (m) return {{ subj: m[1].trim(), rel: m[2].trim(), obj: m[3].trim() }};
  // Format 2: "entity -[RELATION]-> entity" (no parens)
  var re2 = /^([^-\\[]+?)\\s*-\\[([^\\]]+)\\]->\\s*(.+?)(?:\\s*\\(conf.*)?$/;
  m = evStr.match(re2);
  if (m) return {{ subj: m[1].trim(), rel: m[2].trim(), obj: m[3].trim() }};
  return null;
}}

function findEdge(parsed) {{
  if (!parsed) return null;
  var sId = resolveId(parsed.subj);
  var oId = resolveId(parsed.obj);
  var found = null;
  edgesData.forEach(function(e) {{
    if (found) return;
    var relMatch = (e.label || "").toLowerCase() === parsed.rel.toLowerCase();
    // Match by resolved IDs
    if (sId && oId && e.from === sId && e.to === oId && relMatch) {{
      found = e.id; return;
    }}
    // Relaxed: match endpoints without relation type
    if (sId && oId && e.from === sId && e.to === oId) {{
      found = e.id; return;
    }}
    // Fallback: match by display label
    var fromNode = nodesData.get(e.from);
    var toNode = nodesData.get(e.to);
    if (fromNode && toNode) {{
      var fl = (fromNode.label || "").toLowerCase();
      var tl = (toNode.label || "").toLowerCase();
      if (fl === parsed.subj.toLowerCase() && tl === parsed.obj.toLowerCase()) {{
        found = e.id;
      }}
    }}
  }});
  return found;
}}

function resetGraph() {{
  edgesData.forEach(function(e) {{
    var orig = originalEdges[e.id] || {{}};
    edgesData.update({{ id: e.id, color: orig.color || {{ color: "#888888" }}, width: orig.width || 1, shadow: false }});
  }});
  nodesData.forEach(function(n) {{
    var orig = originalNodes[n.id] || {{}};
    nodesData.update({{ id: n.id, color: orig.color, borderWidth: 1, shadow: false }});
  }});
}}

function highlightEvidence(qi) {{
  // Dim everything to gray
  edgesData.forEach(function(e) {{
    edgesData.update({{ id: e.id, color: {{ color: "rgba(80,80,80,0.12)", highlight: "rgba(80,80,80,0.12)" }}, width: 0.5, shadow: false }});
  }});
  nodesData.forEach(function(n) {{
    nodesData.update({{ id: n.id, color: {{ background: "#333", border: "#444" }}, borderWidth: 1, shadow: false }});
  }});

  var qa = qaData[qi];
  if (!qa) return;
  var evidenceItems = document.querySelectorAll("#qa-details-" + qi + " .evidence-item");

  qa.evidence_triples.forEach(function(ev, idx) {{
    var parsed = parseEvidence(ev);
    var edgeId = findEdge(parsed);
    var item = evidenceItems[idx];
    if (edgeId) {{
      edgesData.update({{
        id: edgeId,
        color: {{ color: "#00ff88", highlight: "#00ff88" }},
        width: 4,
        shadow: {{ enabled: true, color: "#00ff88", size: 12 }}
      }});
      var edge = edgesData.get(edgeId);
      if (edge) {{
        var origFrom = originalNodes[edge.from] || {{}};
        var origTo = originalNodes[edge.to] || {{}};
        nodesData.update({{ id: edge.from, color: origFrom.color, borderWidth: 3, shadow: {{ enabled: true, color: "#00ff88", size: 10 }} }});
        nodesData.update({{ id: edge.to, color: origTo.color, borderWidth: 3, shadow: {{ enabled: true, color: "#00ff88", size: 10 }} }});
      }}
      if (item) item.className = "evidence-item matched";
    }} else {{
      if (item) item.className = "evidence-item unmatched";
    }}
  }});
}}

function toggleQA(qi) {{
  var cards = document.querySelectorAll(".qa-card");
  if (activeQA === qi) {{
    activeQA = -1;
    cards[qi].classList.remove("active");
    document.getElementById("qa-details-" + qi).style.display = "none";
    resetGraph();
    return;
  }}
  if (activeQA >= 0 && activeQA < cards.length) {{
    cards[activeQA].classList.remove("active");
    document.getElementById("qa-details-" + activeQA).style.display = "none";
  }}
  activeQA = qi;
  cards[qi].classList.add("active");
  document.getElementById("qa-details-" + qi).style.display = "block";
  highlightEvidence(qi);
}}

// === Ask your own question (requires QA server at localhost:5050) ===
function askQuestion() {{
  var q = document.getElementById("user-question").value.trim();
  if (!q) return;
  var btn = document.getElementById("ask-btn");
  var status = document.getElementById("ask-status");
  btn.disabled = true;
  status.textContent = "Asking...";

  fetch("http://localhost:5050/qa", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify({{ question: q }})
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(result) {{
    // Add to qaData
    var allEvidence = [];
    (result.domain_responses || []).forEach(function(dr) {{
      (dr.evidence || []).forEach(function(ev) {{
        if (allEvidence.indexOf(ev) === -1) allEvidence.push(ev);
      }});
    }});
    var qi = qaData.length;
    qaData.push({{ question: q, evidence_triples: allEvidence }});

    // Build card HTML
    var conf = result.overall_confidence || 0;
    var cov = result.overall_coverage || 0;
    var confCol = conf >= 0.7 ? "#4caf50" : conf >= 0.4 ? "#ff9800" : "#f44336";
    var covCol = cov >= 0.7 ? "#4caf50" : cov >= 0.4 ? "#ff9800" : "#f44336";
    var domBadges = (result.domain_responses || []).map(function(d) {{
      return '<span class="badge domain-badge">' + (d.domain_id || "?") + '</span>';
    }}).join("");
    var evHtml = allEvidence.map(function(e) {{
      return '<li class="evidence-item">' + e + '</li>';
    }}).join("") || "<li style='color:#666'>No evidence triples</li>";
    var domDetails = (result.domain_responses || []).map(function(d) {{
      return '<div class="domain-detail"><b>' + (d.domain_id||"?") + '</b> (conf: ' + (d.confidence||0).toFixed(2) + ') &mdash; topics: ' + (d.topics_used||[]).join(", ") + '</div>';
    }}).join("");

    var card = document.createElement("div");
    card.className = "qa-card";
    card.setAttribute("data-qi", qi);
    card.onclick = function() {{ toggleQA(qi); }};
    card.innerHTML = '<div class="qa-question">Q' + (qi+1) + ': ' + q + '</div>' +
      '<div class="qa-badges">' + domBadges +
      '<span class="badge" style="background:' + confCol + '">conf: ' + conf.toFixed(2) + '</span>' +
      '<span class="badge" style="background:' + covCol + '">cov: ' + cov.toFixed(2) + '</span></div>' +
      '<div class="qa-details" id="qa-details-' + qi + '" style="display:none;">' +
      '<div class="qa-answer">' + (result.final_answer || "No answer") + '</div>' +
      '<div class="qa-section-label">Evidence Triples (' + allEvidence.length + '):</div>' +
      '<ul class="evidence-list">' + evHtml + '</ul>' +
      '<div class="qa-section-label">Domain Experts:</div>' + domDetails + '</div>';

    document.getElementById("qa-cards").appendChild(card);
    status.textContent = "Answer added below!";
    btn.disabled = false;
    document.getElementById("user-question").value = "";
    // Auto-expand it
    toggleQA(qi);
  }})
  .catch(function(err) {{
    status.textContent = "Server not running. Start: python3 qa_server.py";
    btn.disabled = false;
  }});
}}

// Allow Enter key to submit
document.getElementById("user-question").addEventListener("keydown", function(e) {{
  if (e.key === "Enter" && !e.shiftKey) {{
    e.preventDefault();
    askQuestion();
  }}
}});
</script>
</body>
</html>'''

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Enhanced interactive visualization saved to: {output_file}")
        return output_file


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
