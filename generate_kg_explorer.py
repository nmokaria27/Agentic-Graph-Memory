"""
Generate kg_explorer.html with current KG data and QA results.
This creates the sophisticated explorer UI with interactive graph + QA panel.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import colorsys


def generate_colors(n: int) -> List[str]:
    """Generate n visually distinct colors."""
    if n == 0:
        return []
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


def load_kg(kg_file: str) -> Dict[str, Any]:
    """Load KG from JSON export."""
    with open(kg_file, 'r') as f:
        data = json.load(f)
    return data.get('knowledge_graph', data)


def load_qa_results(qa_file: str) -> List[Dict[str, Any]]:
    """Load QA results if available."""
    if not Path(qa_file).exists():
        return []
    with open(qa_file, 'r') as f:
        return json.load(f)


def prepare_nodes_and_edges(kg_data: Dict[str, Any]) -> tuple:
    """Prepare node and edge data for vis.js."""
    entities = kg_data.get('entities', [])
    triples = kg_data.get('triples', [])
    
    # Map entity types to colors
    entity_types = list(set(e.get('type', 'UNKNOWN') for e in entities))
    colors = generate_colors(len(entity_types))
    type_to_color = dict(zip(entity_types, colors))
    
    # Build nodes
    nodes = []
    label_to_id = {}
    
    for entity in entities:
        eid = entity.get('id', 'unknown')
        labels = entity.get('labels', [eid])
        label = labels[0] if labels else eid
        etype = entity.get('type', 'UNKNOWN')
        
        # Shorten long labels
        display_label = label if len(label) <= 40 else label[:37] + "..."
        
        node = {
            "id": eid,
            "label": display_label,
            "title": f"<b>{label}</b><br>Type: {etype}<br>ID: {eid}",
            "color": type_to_color[etype],
            "size": 20,
            "font": {"color": "white", "size": 12}
        }
        nodes.append(node)
        
        # Build label mapping
        label_to_id[eid.lower()] = eid
        label_to_id[label.lower()] = eid
    
    # Build edges
    edges = []
    for idx, triple in enumerate(triples):
        subj = triple.get('subject', triple.get('head'))
        rel = triple.get('relation', triple.get('predicate', 'UNKNOWN'))
        obj = triple.get('object', triple.get('tail'))
        conf = triple.get('confidence', 0.5)
        
        if subj and obj:
            edge = {
                "id": f"edge_{idx}",
                "from": subj,
                "to": obj,
                "label": rel,
                "title": f"{rel} (conf: {conf:.2f})",
                "width": conf * 3,
                "color": {"color": "#888888", "highlight": "#00ff88"},
                "font": {"color": "#cccccc", "size": 10, "strokeWidth": 0},
                "arrows": "to"
            }
            edges.append(edge)
    
    return nodes, edges, label_to_id, type_to_color


def build_legend_html(type_to_color: Dict[str, str]) -> str:
    """Build legend overlay HTML."""
    legend_items = []
    for etype, color in sorted(type_to_color.items())[:15]:  # Limit to top 15
        display_name = etype if len(etype) <= 30 else etype[:27] + "..."
        legend_items.append(
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:50%;background:{color};margin-right:6px;"></span>'
            f'<span style="color:#ccc;font-size:11px;">{display_name}</span></div>'
        )
    return ''.join(legend_items)


def build_qa_cards_html(qa_results: List[Dict[str, Any]]) -> str:
    """Build QA cards HTML."""
    if not qa_results:
        return '<div class="no-qa">No QA results available. Run run_domain_qa.py to generate questions.</div>'
    
    cards = []
    for idx, qa in enumerate(qa_results):
        question = qa.get('question', 'Unknown question')
        answer = qa.get('final_answer', 'No answer available')
        confidence = qa.get('overall_confidence', 0.0)
        coverage = qa.get('overall_coverage', 0.0)
        
        # Domain badges
        domains = qa.get('domain_contributions', {})
        domain_badges = ''.join(
            f'<span class="badge domain-badge">{d}</span>'
            for d in list(domains.keys())[:3]
        )
        
        # Confidence/coverage badges
        conf_color = "#4caf50" if confidence >= 0.7 else "#ff9800"
        cov_color = "#4caf50" if coverage >= 0.7 else "#ff9800"
        
        # Evidence triples
        evidence = qa.get('evidence_triples', [])
        evidence_html = ''.join(
            f'<li class="evidence-item">{ev}</li>'
            for ev in evidence[:10]  # Limit to 10
        )
        
        # Domain details
        domain_details = ''.join(
            f'<div class="domain-detail"><b>{d}</b> (conf: {info.get("confidence", 0):.2f}) &mdash; '
            f'topics: {", ".join(info.get("topics_used", []))}</div>'
            for d, info in list(domains.items())[:3]
        )
        
        card = f'''
                <div class="qa-card" data-qi="{idx}" onclick="toggleQA({idx})">
                  <div class="qa-question">Q{idx+1}: {question}</div>
                  <div class="qa-badges">
                    {domain_badges}
                    <span class="badge" style="background:{conf_color}">conf: {confidence:.2f}</span>
                    <span class="badge" style="background:{cov_color}">cov: {coverage:.2f}</span>
                  </div>
                  <div class="qa-details" id="qa-details-{idx}" style="display:none;">
                    <div class="qa-answer">{answer}</div>
                    <div class="qa-section-label">Evidence Triples ({len(evidence)}):</div>
                    <ul class="evidence-list">{evidence_html}</ul>
                    <div class="qa-section-label">Domain Experts:</div>
                    {domain_details}
                  </div>
                </div>'''
        cards.append(card)
    
    return '\n'.join(cards)


def generate_html(kg_file: str, qa_file: str, output_file: str):
    """Generate complete kg_explorer.html."""
    
    print(f"Loading KG from {kg_file}...")
    kg_data = load_kg(kg_file)
    
    print(f"Loading QA results from {qa_file}...")
    qa_results = load_qa_results(qa_file)
    
    print("Preparing graph data...")
    nodes, edges, label_to_id, type_to_color = prepare_nodes_and_edges(kg_data)
    
    num_entities = len(nodes)
    num_triples = len(edges)
    connected = sum(1 for n in nodes if any(e['from'] == n['id'] or e['to'] == n['id'] for e in edges))
    
    print(f"Graph stats: {num_entities} entities, {num_triples} triples, {connected} connected")
    
    legend_html = build_legend_html(type_to_color)
    qa_cards_html = build_qa_cards_html(qa_results)
    
    # Extract QA data for JavaScript
    qa_data_js = []
    for qa in qa_results:
        qa_data_js.append({
            'question': qa.get('question', ''),
            'evidence_triples': qa.get('evidence_triples', [])
        })
    
    html_template = f'''<!DOCTYPE html>
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
.legend-overlay {{ position: absolute; bottom: 12px; left: 12px; background: rgba(0,0,0,0.75); padding: 10px 14px; border-radius: 8px; z-index: 10; max-height: 400px; overflow-y: auto; }}
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
      <div><b>Entities:</b> {num_entities}</div>
      <div><b>Triples:</b> {num_triples}</div>
      <div><b>Connected:</b> {connected}/{num_entities} ({100*connected//num_entities if num_entities else 0}%)</div>
    </div>
    <div class="legend-overlay">{legend_html}</div>
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
    {qa_cards_html}
    </div>
  </div>
</div>
<script>
var nodesData = new vis.DataSet({json.dumps(nodes)});
var edgesData = new vis.DataSet({json.dumps(edges)});
var labelToId = {json.dumps(label_to_id)};
var container = document.getElementById("graph");
var data = {{ nodes: nodesData, edges: edgesData }};
var options = {{
  nodes: {{ font: {{ color: "white" }} }},
  edges: {{ smooth: {{ type: "continuous" }} }}
}};
var network = new vis.Network(container, data, options);

var qaData = {json.dumps(qa_data_js)};
var activeQA = -1;
var originalEdges = {{}};
var originalNodes = {{}};

edgesData.forEach(function(e) {{ originalEdges[e.id] = {{ color: JSON.parse(JSON.stringify(e.color || {{}})), width: e.width || 1 }}; }});
nodesData.forEach(function(n) {{ originalNodes[n.id] = {{ color: n.color, borderWidth: n.borderWidth || 1 }}; }});

function resolveId(name) {{
  if (!name) return null;
  if (nodesData.get(name)) return name;
  var lower = name.toLowerCase();
  if (labelToId[lower]) return labelToId[lower];
  return null;
}}

function parseEvidence(evStr) {{
  var re1 = /\\(([^)]+)\\)\\s*-\\[([^\\]]+)\\]->\\s*\\(([^)]+)\\)/;
  var m = evStr.match(re1);
  if (m) return {{ subj: m[1].trim(), rel: m[2].trim(), obj: m[3].trim() }};
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
    if (sId && oId && e.from === sId && e.to === oId && relMatch) {{
      found = e.id; return;
    }}
    if (sId && oId && e.from === sId && e.to === oId) {{
      found = e.id; return;
    }}
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
      var nodes = [edge.from, edge.to];
      nodes.forEach(function(nId) {{
        nodesData.update({{
          id: nId,
          borderWidth: 4,
          color: {{ border: "#00ff88" }},
          shadow: {{ enabled: true, color: "#00ff88", size: 8 }}
        }});
      }});
      if (item) item.classList.add("matched");
    }} else {{
      if (item) item.classList.add("unmatched");
    }}
  }});
}}

function toggleQA(qi) {{
  var card = document.querySelector('[data-qi="' + qi + '"]');
  var details = document.getElementById("qa-details-" + qi);
  
  if (activeQA === qi) {{
    card.classList.remove("active");
    details.style.display = "none";
    resetGraph();
    activeQA = -1;
  }} else {{
    document.querySelectorAll(".qa-card").forEach(function(c) {{ c.classList.remove("active"); }});
    document.querySelectorAll(".qa-details").forEach(function(d) {{ d.style.display = "none"; }});
    card.classList.add("active");
    details.style.display = "block";
    resetGraph();
    highlightEvidence(qi);
    activeQA = qi;
  }}
}}

async function askQuestion() {{
  var question = document.getElementById("user-question").value.trim();
  if (!question) return;
  
  var btn = document.getElementById("ask-btn");
  var status = document.getElementById("ask-status");
  btn.disabled = true;
  status.textContent = "Asking...";
  
  try {{
    var response = await fetch("http://localhost:5050/qa", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ question: question }})
    }});
    
    if (!response.ok) throw new Error("Server error");
    
    var result = await response.json();
    status.textContent = "Answer received!";
    
    var newCard = document.createElement("div");
    newCard.className = "qa-card";
    newCard.innerHTML = '<div class="qa-question">Your Q: ' + question + '</div>' +
      '<div class="qa-answer" style="display:block; margin-top:8px;">' + result.final_answer + '</div>';
    document.getElementById("qa-cards").insertBefore(newCard, document.getElementById("qa-cards").firstChild);
    
    setTimeout(function() {{ status.textContent = "Tip: run the QA server to enable live questions"; }}, 3000);
  }} catch (e) {{
    status.textContent = "Error: Is QA server running? (python qa_server.py)";
  }} finally {{
    btn.disabled = false;
  }}
}}
</script>
</body>
</html>'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"\n✓ Generated {output_file}")
    print(f"  - {num_entities} entities")
    print(f"  - {num_triples} triples")
    print(f"  - {len(qa_results)} QA examples")


if __name__ == "__main__":
    import sys
    
    kg_file = sys.argv[1] if len(sys.argv) > 1 else "kg_export.json"
    qa_file = sys.argv[2] if len(sys.argv) > 2 else "qa_results.json"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "kg_explorer.html"
    
    if not Path(kg_file).exists():
        print(f"Error: {kg_file} not found")
        sys.exit(1)
    
    generate_html(kg_file, qa_file, output_file)
    print(f"\nOpen {output_file} in your browser to explore the knowledge graph!")
