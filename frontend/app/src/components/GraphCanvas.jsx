import { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';

const EDGE_COLOR    = '#7c83e8';
const EDGE_HL_COLOR = '#1de9b6';
const TEAL          = '#1de9b6';
const NODE_MIN_R    = 6;
const NODE_MAX_R    = 22;

function nodeRadius(degree, maxDegree) {
  if (!maxDegree) return NODE_MIN_R;
  return NODE_MIN_R + (degree / maxDegree) * (NODE_MAX_R - NODE_MIN_R);
}

export default function GraphCanvas({
  entities, triples, entityTypes, mode,
  highlightedNodes, highlightedEdges,
  selectedNodeId, onNodeClick, onNodeHover,
  hoveredChipNodeId, tweaks,
  nodeColorById,
}) {
  const svgRef       = useRef(null);
  const simRef       = useRef(null);
  const nodeGrpRef   = useRef(null);
  const linkGrpRef   = useRef(null);
  const degreesRef   = useRef({});
  const maxDegreeRef = useRef(1);
  const zoomBehavRef = useRef(null);

  const [tooltip, setTooltip]       = useState(null);
  const [detailNode, setDetailNode] = useState(null);

  const onNodeClickRef = useRef(onNodeClick);
  const onNodeHoverRef = useRef(onNodeHover);
  useEffect(() => { onNodeClickRef.current = onNodeClick; }, [onNodeClick]);
  useEffect(() => { onNodeHoverRef.current = onNodeHover; }, [onNodeHover]);
  const setTooltipRef = useRef(setTooltip);
  const setDetailRef  = useRef(setDetailNode);

  const neighborMap = useMemo(() => {
    const map = {};
    triples.forEach(t => {
      if (!map[t.subject]) map[t.subject] = [];
      if (!map[t.object])  map[t.object]  = [];
      map[t.subject].push({ dir: 'out', relation: t.relation, id: t.object });
      map[t.object].push({ dir: 'in',  relation: t.relation, id: t.subject });
    });
    return map;
  }, [triples]);

  // ── Main D3 setup ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current || !entities.length) return;

    const svg    = d3.select(svgRef.current);
    const width  = svgRef.current.clientWidth  || 900;
    const height = svgRef.current.clientHeight || 700;

    svg.selectAll('*').remove();

    // Arrowhead markers
    const defs = svg.append('defs');
    const addMarker = (id, color, opacity) => {
      defs.append('marker')
        .attr('id', id).attr('viewBox', '0 0 10 7')
        .attr('refX', 9).attr('refY', 3.5)
        .attr('markerWidth', 6).attr('markerHeight', 5)
        .attr('orient', 'auto')
        .append('polygon')
        .attr('points', '0 0, 10 3.5, 0 7')
        .attr('fill', color).attr('fill-opacity', opacity);
    };
    addMarker('arrow-normal', EDGE_COLOR, 0.65);
    addMarker('arrow-hl', EDGE_HL_COLOR, 1);

    // Build data
    const entityMap = {};
    entities.forEach(e => { entityMap[e.id] = e; });

    const nodes = entities.map(e => ({ ...e }));
    const links = triples
      .filter(t => entityMap[t.subject] && entityMap[t.object] && t.subject !== t.object)
      .map(t => ({ ...t, source: t.subject, target: t.object }));

    const degrees = {};
    nodes.forEach(n => { degrees[n.id] = 0; });
    links.forEach(l => {
      degrees[l.subject] = (degrees[l.subject] || 0) + 1;
      degrees[l.object]  = (degrees[l.object]  || 0) + 1;
    });
    const maxDegree = Math.max(...Object.values(degrees), 1);
    degreesRef.current   = degrees;
    maxDegreeRef.current = maxDegree;

    // Zoomable group
    const g = svg.append('g').attr('class', 'graph-root');
    const zoom = d3.zoom()
      .scaleExtent([0.08, 4])
      .on('zoom', ev => g.attr('transform', ev.transform));
    svg.call(zoom).on('dblclick.zoom', null);
    zoomBehavRef.current = zoom;

    svg.on('click', () => {
      setDetailRef.current(null);
      onNodeClickRef.current(null);
    });

    // Edges
    const linkGroup = g.append('g').attr('class', 'links');
    const link = linkGroup.selectAll('line')
      .data(links).join('line')
      .attr('class', 'edge-link')
      .attr('stroke', EDGE_COLOR)
      .attr('stroke-width', d => Math.max(0.6, (d.confidence || 0.7) * 1.4))
      .attr('stroke-opacity', tweaks?.edgeOpacity ?? 0.45)
      .attr('marker-end', 'url(#arrow-normal)')
      .on('mouseover', function (ev, d) {
        const rect = svgRef.current.getBoundingClientRect();
        setTooltipRef.current({ kind: 'edge', d, x: ev.clientX - rect.left, y: ev.clientY - rect.top });
        d3.select(this).attr('stroke', TEAL).attr('stroke-opacity', 1);
      })
      .on('mousemove', function (ev) {
        const rect = svgRef.current.getBoundingClientRect();
        setTooltipRef.current(p => p ? { ...p, x: ev.clientX - rect.left, y: ev.clientY - rect.top } : null);
      })
      .on('mouseout', function () {
        setTooltipRef.current(null);
        d3.select(this).attr('stroke', EDGE_COLOR).attr('stroke-opacity', tweaks?.edgeOpacity ?? 0.45);
      });

    linkGrpRef.current = link;

    // Nodes
    const drag = d3.drag()
      .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end',   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; });

    const nodeGroup = g.append('g').attr('class', 'nodes')
      .selectAll('g').data(nodes).join('g')
      .attr('class', 'node-group').attr('cursor', 'pointer')
      .call(drag)
      .on('mouseover', function (ev, d) {
        const r = nodeRadius(degrees[d.id] || 0, maxDegree);
        d3.select(this).select('.node-circle').transition().duration(120).attr('r', r * 1.25);
        const rect = svgRef.current.getBoundingClientRect();
        setTooltipRef.current({ kind: 'node', d, degree: degrees[d.id] || 0, x: ev.clientX - rect.left, y: ev.clientY - rect.top });
        onNodeHoverRef.current(d.id);
      })
      .on('mousemove', function (ev) {
        const rect = svgRef.current.getBoundingClientRect();
        setTooltipRef.current(p => p ? { ...p, x: ev.clientX - rect.left, y: ev.clientY - rect.top } : null);
      })
      .on('mouseout', function (ev, d) {
        const r = nodeRadius(degrees[d.id] || 0, maxDegree);
        d3.select(this).select('.node-circle').transition().duration(120).attr('r', r);
        setTooltipRef.current(null);
        onNodeHoverRef.current(null);
      })
      .on('click', function (ev, d) {
        ev.stopPropagation();
        const neighbors = neighborMap[d.id] || [];
        setDetailRef.current(prev => prev?.id === d.id ? null : { ...d, degree: degrees[d.id] || 0, neighbors });
        onNodeClickRef.current(d.id);
      });

    // Pulse ring
    nodeGroup.append('circle')
      .attr('class', 'node-pulse')
      .attr('r', d => nodeRadius(degrees[d.id] || 0, maxDegree) + 5)
      .attr('fill', 'none').attr('stroke', TEAL).attr('stroke-width', 2).attr('opacity', 0);

    // Main circle
    nodeGroup.append('circle')
      .attr('class', 'node-circle')
      .attr('r', d => nodeRadius(degrees[d.id] || 0, maxDegree))
      .attr('fill', d => (nodeColorById && nodeColorById[d.id])
        || (entityTypes[d.type] || {}).color
        || '#4a5568')
      .attr('stroke', '#0f1117').attr('stroke-width', 2);

    // Labels for high-degree nodes
    nodeGroup.each(function (d) {
      if ((degrees[d.id] || 0) >= 3 || tweaks?.showAllLabels) {
        const label = (d.labels[0] || d.id);
        const short = label.length > 18 ? label.slice(0, 16) + '\u2026' : label;
        d3.select(this).append('text')
          .attr('class', 'node-label')
          .attr('dy', nodeRadius(degrees[d.id] || 0, maxDegree) + 11)
          .attr('text-anchor', 'middle')
          .attr('font-size', 9)
          .attr('fill', '#6a7585')
          .attr('font-family', "'JetBrains Mono', monospace")
          .attr('pointer-events', 'none')
          .text(short);
      }
    });

    nodeGrpRef.current = nodeGroup;

    // Simulation
    const strength = tweaks?.forceStrength ?? -300;
    const sim = d3.forceSimulation(nodes)
      .force('link',      d3.forceLink(links).id(d => d.id).distance(130).strength(0.7))
      .force('charge',    d3.forceManyBody().strength(strength))
      .force('center',    d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => nodeRadius(degrees[d.id] || 0, maxDegree) + 6));
    simRef.current = sim;

    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const r = nodeRadius(degrees[d.target.id] || 0, maxDegree) + 11;
          return d.target.x - (dx / dist) * r;
        })
        .attr('y2', d => {
          const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const r = nodeRadius(degrees[d.target.id] || 0, maxDegree) + 11;
          return d.target.y - (dy / dist) * r;
        });

      nodeGroup.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => sim.stop();
  }, [entities, triples, entityTypes, nodeColorById]); // eslint-disable-line

  // ── Highlight effect (QA mode) ────────────────────────────────────────────
  useEffect(() => {
    const ng = nodeGrpRef.current;
    const lg = linkGrpRef.current;
    if (!ng || !lg) return;

    if (mode === 'qa' && highlightedNodes && highlightedNodes.size > 0) {
      ng.transition().duration(400)
        .attr('opacity', d => highlightedNodes.has(d.id) ? 1 : 0.1);
      lg.transition().duration(400)
        .attr('stroke-opacity', d => highlightedEdges?.has(d.id) ? 0.9 : 0.04)
        .attr('stroke', d => highlightedEdges?.has(d.id) ? EDGE_HL_COLOR : EDGE_COLOR)
        .attr('marker-end', d => highlightedEdges?.has(d.id) ? 'url(#arrow-hl)' : 'url(#arrow-normal)');
    } else {
      ng.transition().duration(400).attr('opacity', 1);
      lg.transition().duration(400)
        .attr('stroke-opacity', 0.45)
        .attr('stroke', EDGE_COLOR)
        .attr('marker-end', 'url(#arrow-normal)');
    }
  }, [mode, highlightedNodes, highlightedEdges]);

  // ── Selected node effect ──────────────────────────────────────────────────
  useEffect(() => {
    const ng = nodeGrpRef.current;
    if (!ng) return;
    ng.select('.node-circle')
      .attr('stroke', d => d.id === selectedNodeId ? TEAL : '#0f1117')
      .attr('stroke-width', d => d.id === selectedNodeId ? 3 : 2);
  }, [selectedNodeId]);

  // ── Chip hover pulse ──────────────────────────────────────────────────────
  useEffect(() => {
    const ng = nodeGrpRef.current;
    if (!ng) return;
    ng.selectAll('.node-pulse').attr('opacity', 0);
    if (!hoveredChipNodeId) return;
    const target = ng.filter(d => d.id === hoveredChipNodeId);
    const r = nodeRadius(degreesRef.current[hoveredChipNodeId] || 0, maxDegreeRef.current);
    target.select('.node-pulse')
      .attr('r', r + 4).attr('opacity', 0.9)
      .transition().duration(600)
      .attr('r', r + 20).attr('opacity', 0)
      .on('end', function () { d3.select(this).attr('r', r + 4); });
  }, [hoveredChipNodeId]);

  // ── Zoom controls ─────────────────────────────────────────────────────────
  const handleZoomIn    = () => d3.select(svgRef.current).transition().call(zoomBehavRef.current.scaleBy, 1.4);
  const handleZoomOut   = () => d3.select(svgRef.current).transition().call(zoomBehavRef.current.scaleBy, 0.72);
  const handleFitScreen = () => d3.select(svgRef.current).transition().duration(500).call(zoomBehavRef.current.transform, d3.zoomIdentity.translate(20, 20).scale(0.9));
  const handleReset     = () => { if (simRef.current) simRef.current.alpha(0.6).restart(); };

  const canvasBtnStyle = {
    width: 32, height: 32, border: '1px solid rgba(255,255,255,0.1)',
    background: 'rgba(24,28,38,0.92)', color: '#8892a4', borderRadius: 4,
    cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 16, transition: 'color 0.15s, border-color 0.15s',
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0f1117', overflow: 'hidden' }}>
      <svg ref={svgRef} style={{ width: '100%', height: '100%', display: 'block' }} />

      {/* Canvas controls */}
      <div style={{ position: 'absolute', top: 16, right: 16, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {[
          { icon: '+', title: 'Zoom in',       action: handleZoomIn },
          { icon: '\u2212', title: 'Zoom out', action: handleZoomOut },
          { icon: '\u2291', title: 'Fit to screen', action: handleFitScreen },
          { icon: '\u21BA', title: 'Reset layout',  action: handleReset },
        ].map(btn => (
          <button key={btn.title} title={btn.title} onClick={btn.action} style={canvasBtnStyle}
            onMouseEnter={e => { e.currentTarget.style.color = TEAL; e.currentTarget.style.borderColor = 'rgba(29,233,182,0.4)'; }}
            onMouseLeave={e => { e.currentTarget.style.color = '#8892a4'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; }}>
            {btn.icon}
          </button>
        ))}
      </div>

      {/* Stats overlay */}
      <div style={{ position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)', display: 'flex', gap: 16, pointerEvents: 'none' }}>
        {[
          { label: 'entities', value: entities.length },
          { label: 'triples',  value: triples.length },
          { label: 'connected', value: `${Math.round((new Set([...triples.map(t=>t.subject),...triples.map(t=>t.object)]).size / Math.max(entities.length,1))*100)}%` },
        ].map(s => (
          <div key={s.label} style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: '#3d4555', background: 'rgba(15,17,23,0.7)', padding: '3px 8px', borderRadius: 3, border: '1px solid rgba(255,255,255,0.04)' }}>
            <span style={{ color: TEAL }}>{s.value}</span> {s.label}
          </div>
        ))}
      </div>

      {/* Entity type legend */}
      <div style={{ position: 'absolute', bottom: detailNode ? 176 : 16, left: 16, display: 'flex', flexDirection: 'column', gap: 4, transition: 'bottom 0.25s ease' }}>
        {Object.entries(entityTypes).map(([type, { color }]) => (
          <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: '#5a6375', fontFamily: "'JetBrains Mono', monospace" }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
            {type}
          </div>
        ))}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div style={{ position: 'absolute', left: tooltip.x + 14, top: tooltip.y - 12, zIndex: 200, pointerEvents: 'none' }}>
          <div style={{ background: '#1e2334', border: '1px solid rgba(124,131,232,0.3)', borderRadius: 4, padding: '10px 14px', minWidth: 200, maxWidth: 260, boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }}>
            {tooltip.kind === 'node' && (() => {
              const { d, degree } = tooltip;
              const typeInfo = entityTypes[d.type] || {};
              return (
                <>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: typeInfo.color || EDGE_COLOR, marginBottom: 5, letterSpacing: '0.05em' }}>{d.type}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', marginBottom: 8, lineHeight: 1.3 }}>{d.labels[0] || d.id}</div>
                  <div style={{ display: 'flex', gap: 16, fontSize: 11, marginBottom: 6 }}>
                    <span style={{ color: '#6a7585' }}>conf <span style={{ color: TEAL }}>{(d.metadata?.confidence || 0).toFixed(2)}</span></span>
                    <span style={{ color: '#6a7585' }}>deg <span style={{ color: '#e2e8f0' }}>{degree}</span></span>
                  </div>
                  <div style={{ fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace" }}>{d.metadata?.source}</div>
                </>
              );
            })()}
            {tooltip.kind === 'edge' && (() => {
              const { d } = tooltip;
              const srcLabel = d.source.labels?.[0] || d.source.id || d.source;
              const tgtLabel = d.target.labels?.[0] || d.target.id || d.target;
              return (
                <>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: EDGE_COLOR, marginBottom: 5, letterSpacing: '0.05em' }}>{d.relation}</div>
                  <div style={{ fontSize: 11, color: '#8892a4', marginBottom: 6 }}>
                    <span style={{ color: '#c8d3e0' }}>{typeof srcLabel === 'string' ? srcLabel : ''}</span>
                    <span style={{ color: '#3d4555' }}> &rarr; </span>
                    <span style={{ color: '#c8d3e0' }}>{typeof tgtLabel === 'string' ? tgtLabel : ''}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#6a7585' }}>conf <span style={{ color: TEAL }}>{(d.confidence || 0).toFixed(2)}</span></div>
                </>
              );
            })()}
          </div>
        </div>
      )}

      {/* Detail panel */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        background: '#181c26', borderTop: '1px solid rgba(124,131,232,0.18)',
        padding: '14px 20px', transform: detailNode ? 'translateY(0)' : 'translateY(100%)',
        transition: 'transform 0.22s ease', height: 160, overflow: 'hidden',
        display: 'flex', gap: 32, alignItems: 'flex-start',
      }}>
        {detailNode && (() => {
          const typeInfo = entityTypes[detailNode.type] || {};
          const neighbors = detailNode.neighbors || [];
          return (
            <>
              <div style={{ minWidth: 220 }}>
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: typeInfo.color || EDGE_COLOR, marginBottom: 5, letterSpacing: '0.06em' }}>{detailNode.type}</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0', marginBottom: 6 }}>{detailNode.labels[0] || detailNode.id}</div>
                {detailNode.labels.length > 1 && (
                  <div style={{ fontSize: 11, color: '#5a6375', marginBottom: 8 }}>{detailNode.labels.slice(1).join(', ')}</div>
                )}
                <div style={{ display: 'flex', gap: 20, fontSize: 11, marginBottom: 6 }}>
                  <span style={{ color: '#6a7585' }}>id <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#8892a4' }}>{detailNode.id}</span></span>
                  <span style={{ color: '#6a7585' }}>conf <span style={{ color: TEAL }}>{(detailNode.metadata?.confidence || 0).toFixed(2)}</span></span>
                  <span style={{ color: '#6a7585' }}>degree <span style={{ color: '#e2e8f0' }}>{detailNode.degree}</span></span>
                </div>
                <div style={{ fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace" }}>{detailNode.metadata?.source}</div>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 11, color: '#4a5568', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Description</div>
                <div style={{ fontSize: 12, color: '#8892a4', lineHeight: 1.5, marginBottom: 8 }}>{detailNode.metadata?.description}</div>
              </div>
              {neighbors.length > 0 && (
                <div style={{ minWidth: 240, maxWidth: 280 }}>
                  <div style={{ fontSize: 11, color: '#4a5568', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Connections</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 96, overflowY: 'auto' }}>
                    {neighbors.slice(0, 6).map((n, i) => (
                      <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 11 }}>
                        <span style={{ color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", fontSize: 9 }}>{n.dir === 'out' ? '\u2192' : '\u2190'}</span>
                        <span style={{ color: EDGE_COLOR, fontFamily: "'JetBrains Mono', monospace", fontSize: 9, flexShrink: 0 }}>{n.relation}</span>
                        <span style={{ color: '#8892a4', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{n.id}</span>
                      </div>
                    ))}
                    {neighbors.length > 6 && <div style={{ fontSize: 10, color: '#3d4555' }}>+{neighbors.length - 6} more</div>}
                  </div>
                </div>
              )}
              <button onClick={() => { setDetailNode(null); onNodeClick(null); }}
                style={{ position: 'absolute', top: 12, right: 16, background: 'none', border: 'none', color: '#3d4555', cursor: 'pointer', fontSize: 16, padding: 4 }}>
                &#x2715;
              </button>
            </>
          );
        })()}
      </div>
    </div>
  );
}
