import { useState, useEffect, useMemo, useCallback } from 'react';
import GraphCanvas from './components/GraphCanvas.jsx';
import Sidebar from './components/Sidebar.jsx';
import QAPanel from './components/QAPanel.jsx';
import UploadPanel from './components/UploadPanel.jsx';
import { fetchHealth, fetchKGData, submitQuestion } from './api.js';
import {
  MOCK_ENTITY_TYPES, MOCK_ENTITIES, MOCK_TRIPLES,
  MOCK_RELATION_TYPES,
} from './mockData.js';

function finalizeTurn(setQaHistory, turnId, fullAnswer, referencedNodes, referencedEdges, domainResponses, model) {
  setQaHistory(prev => prev.map(t =>
    t.id === turnId ? {
      ...t, status: 'done', answer: fullAnswer || '',
      referencedNodes: referencedNodes || [],
      referencedEdges: referencedEdges || [],
      nodeCount: (referencedNodes || []).length,
      domainResponses: domainResponses || [],
      model: model || t.model,
    } : t
  ));
}

function failTurn(setQaHistory, turnId, errorMessage) {
  setQaHistory(prev => prev.map(t =>
    t.id === turnId ? { ...t, status: 'error', error: errorMessage } : t
  ));
}

// ── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  // ── Data state ────────────────────────────────────────────────────────────
  const [entityTypes, setEntityTypes]     = useState(MOCK_ENTITY_TYPES);
  const [entities, setEntities]           = useState(MOCK_ENTITIES);
  const [triples, setTriples]             = useState(MOCK_TRIPLES);
  const [relationTypes, setRelationTypes] = useState(MOCK_RELATION_TYPES);
  const [orgChart, setOrgChart]           = useState(null);
  const [reloadKey, setReloadKey]         = useState(0);

  // ── Mode ──────────────────────────────────────────────────────────────────
  const [mode, setMode]               = useState('explore');
  const [selectedModel, setModel]     = useState('gemma4:31b');
  const [serverConnected, setServer]  = useState(false);
  const [backendStatus, setBackendStatus] = useState('loading'); // 'loading' | 'online' | 'offline' | 'no-kg'
  const [qaReady, setQaReady]         = useState(false);

  // ── Filters ───────────────────────────────────────────────────────────────
  const [filters, setFilters] = useState(() => ({
    entityTypes:         Object.fromEntries(Object.keys(MOCK_ENTITY_TYPES).map(k => [k, true])),
    relationTypes:       Object.fromEntries(MOCK_RELATION_TYPES.map(r => [r, true])),
    confidenceThreshold: 0,
    searchQuery:         '',
  }));

  // ── QA state ──────────────────────────────────────────────────────────────
  const [qaHistory,    setQaHistory]    = useState([]);
  const [selectedQaId, setSelectedQaId] = useState(null);
  const [hoveredQaId,  setHoveredQaId]  = useState(null);

  // ── Graph interaction ─────────────────────────────────────────────────────
  const [selectedNodeId,    setSelectedNodeId]    = useState(null);
  const [hoveredChipNodeId, setHoveredChipNodeId] = useState(null);

  const tweaks = { forceStrength: -300, edgeOpacity: 0.45, showAllLabels: false };

  const DOMAIN_PALETTE = [
    '#7c83e8', '#1de9b6', '#f59e0b', '#ef4444', '#22c55e',
    '#06b6d4', '#a78bfa', '#ec4899', '#84cc16', '#fb923c',
  ];

  const nodeColorById = useMemo(() => {
    if (!orgChart || !orgChart.domains?.length) return null;
    const domainColor = {};
    orgChart.domains.forEach((d, i) => {
      domainColor[d.domain_id] = DOMAIN_PALETTE[i % DOMAIN_PALETTE.length];
    });
    const map = {};
    Object.entries(orgChart.assignments || {}).forEach(([eid, did]) => {
      if (domainColor[did]) map[eid] = domainColor[did];
    });
    return Object.keys(map).length ? map : null;
  }, [orgChart]);

  // ── Fetch data from server ────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        const health = await fetchHealth();
        if (cancelled) return;

        if (health.status === 'ok') {
          setServer(true);
          setQaReady(!!health.qa_ready);

          if (health.kg_loaded) {
            const data = await fetchKGData();
            if (cancelled) return;

            setEntities(data.entities);
            setTriples(data.triples);
            setEntityTypes(data.entity_types);
            setRelationTypes(data.relation_types);
            setOrgChart(data.org_chart ?? null);

            setFilters({
              entityTypes: Object.fromEntries(Object.keys(data.entity_types).map(k => [k, true])),
              relationTypes: Object.fromEntries(data.relation_types.map(r => [r, true])),
              confidenceThreshold: 0,
              searchQuery: '',
            });
            setBackendStatus('online');
          } else {
            setBackendStatus('no-kg');
          }
        } else {
          setBackendStatus('offline');
        }
      } catch {
        if (!cancelled) {
          setServer(false);
          setBackendStatus('offline');
        }
      }
    }

    loadData();
    return () => { cancelled = true; };
  }, [reloadKey]);

  // ── Derived: filtered entities / triples ──────────────────────────────────
  const filteredEntities = useMemo(() => {
    const q = filters.searchQuery.toLowerCase();
    return entities.filter(e => {
      if (filters.entityTypes[e.type] === false) return false;
      if (q && !e.labels.some(l => l.toLowerCase().includes(q)) && !e.id.includes(q)) return false;
      return true;
    });
  }, [entities, filters]);

  const filteredTriples = useMemo(() => {
    const ids = new Set(filteredEntities.map(e => e.id));
    return triples.filter(t =>
      ids.has(t.subject) && ids.has(t.object) &&
      filters.relationTypes[t.relation] !== false &&
      (t.confidence == null || t.confidence >= filters.confidenceThreshold)
    );
  }, [triples, filteredEntities, filters]);

  // ── Derived: highlighted nodes/edges for graph ────────────────────────────
  const { highlightedNodes, highlightedEdges } = useMemo(() => {
    const qaId = hoveredQaId || selectedQaId;
    if (!qaId || mode !== 'qa') return { highlightedNodes: new Set(), highlightedEdges: new Set() };
    const turn = qaHistory.find(t => t.id === qaId);
    if (!turn || turn.status === 'loading') return { highlightedNodes: new Set(), highlightedEdges: new Set() };
    const hnodes = new Set(turn.referencedNodes || []);
    // Use explicit edge IDs if available, otherwise derive from triples whose both endpoints are highlighted
    const hedges = (turn.referencedEdges && turn.referencedEdges.length > 0)
      ? new Set(turn.referencedEdges)
      : new Set(triples.filter(t => hnodes.has(t.subject) && hnodes.has(t.object)).map(t => t.id));
    return { highlightedNodes: hnodes, highlightedEdges: hedges };
  }, [hoveredQaId, selectedQaId, qaHistory, mode, triples]);

  // ── QA submit ─────────────────────────────────────────────────────────────
  const handleSubmitQuestion = useCallback(async (question) => {
    const turnId = `qa_${Date.now()}`;
    const newTurn = {
      id: turnId, question, status: 'loading', streamedAnswer: '',
      answer: '', referencedNodes: [], referencedEdges: [],
      nodeCount: 0, model: selectedModel,
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
    };
    setQaHistory(prev => [...prev, newTurn]);
    setSelectedQaId(turnId);
    if (mode !== 'qa') setMode('qa');

    try {
      const data = await submitQuestion(question, selectedModel);
      finalizeTurn(setQaHistory, turnId,
        data.answer || '',
        data.referenced_nodes || [],
        data.referenced_edges || [],
        data.domain_responses || [],
        data.model);
    } catch (err) {
      const msg = err?.name === 'AbortError' || err?.name === 'TimeoutError'
        ? 'Request timed out. The QA pipeline (3–15 LLM calls) may be slow on a cold Ollama or large model. Try again, or run a smaller model.'
        : (err?.message || String(err) || 'Unknown error');
      failTurn(setQaHistory, turnId, msg);
    }
  }, [selectedModel, mode]);

  const clearChats = useCallback(() => {
    setQaHistory([]);
    setSelectedQaId(null);
    setHoveredQaId(null);
  }, []);

  // Sample QAs are no longer auto-loaded — they were mock data and misled users
  // into thinking the backend had answered. Empty history shows suggested
  // question prompts (in QAPanel) which the user clicks to fire real questions.

  // ── Layout ────────────────────────────────────────────────────────────────
  const qaMode = mode === 'qa';

  const banner = (() => {
    if (backendStatus === 'loading' || backendStatus === 'online') return null;
    const text = backendStatus === 'offline'
      ? 'Backend offline — showing demo data. Start scripts/api_server.py to load the real KG.'
      : 'Backend online but no KG loaded — showing demo data. Run the pipeline first or check governed_kg_export.json.';
    return (
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 500,
        padding: '7px 14px', textAlign: 'center',
        background: 'rgba(239,68,68,0.12)', borderBottom: '1px solid rgba(239,68,68,0.3)',
        color: '#ffb4b4', fontSize: 11,
        fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.04em',
      }}>{text}</div>
    );
  })();

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: '#0f1117' }}>
      {banner}

      {/* Left sidebar */}
      <Sidebar
        mode={mode}
        onModeChange={setMode}
        filters={filters}
        onFiltersChange={setFilters}
        entityTypes={entityTypes}
        relationTypes={relationTypes}
        qaHistory={qaHistory}
        selectedQaId={selectedQaId}
        onQaSelect={id => setSelectedQaId(prev => prev === id ? null : id)}
        onQaHover={setHoveredQaId}
        serverConnected={serverConnected}
      />

      {/* Center: graph canvas */}
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <GraphCanvas
          entities={filteredEntities}
          triples={filteredTriples}
          entityTypes={entityTypes}
          mode={mode}
          highlightedNodes={highlightedNodes}
          highlightedEdges={highlightedEdges}
          selectedNodeId={selectedNodeId}
          onNodeClick={id => setSelectedNodeId(prev => prev === id ? null : id)}
          onNodeHover={() => {}}
          hoveredChipNodeId={hoveredChipNodeId}
          tweaks={tweaks}
          nodeColorById={nodeColorById}
        />

        {/* QA dim overlay */}
        {qaMode && (
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            background: 'rgba(15,17,23,0.18)', transition: 'opacity 0.4s',
            borderRight: '1px solid rgba(255,255,255,0.05)',
          }} />
        )}

        <UploadPanel onIngestComplete={() => setReloadKey(k => k + 1)} />

        {/* Mode indicator badge */}
        <div style={{
          position: 'absolute', top: 16, left: 16,
          padding: '4px 10px', borderRadius: 3, fontSize: 10,
          fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
          background: qaMode ? 'rgba(124,131,232,0.15)' : 'rgba(29,233,182,0.12)',
          color: qaMode ? '#7c83e8' : '#1de9b6',
          border: `1px solid ${qaMode ? 'rgba(124,131,232,0.25)' : 'rgba(29,233,182,0.2)'}`,
          pointerEvents: 'none',
        }}>
          {qaMode ? 'QA \u2014 EVIDENCE VIEW' : 'EXPLORE'}
        </div>
      </div>

      {/* Right QA panel (slides in) */}
      <div style={{
        width: qaMode ? 380 : 0,
        overflow: 'hidden',
        transition: 'width 0.28s cubic-bezier(0.4,0,0.2,1)',
        flexShrink: 0,
      }}>
        <div style={{ width: 380, height: '100%' }}>
          <QAPanel
            qaHistory={qaHistory}
            onSubmitQuestion={handleSubmitQuestion}
            onChipHover={setHoveredChipNodeId}
            selectedModel={selectedModel}
            onModelChange={setModel}
            activeQaId={selectedQaId}
            entities={entities}
            entityTypes={entityTypes}
            onClearChats={clearChats}
            qaReady={qaReady}
          />
        </div>
      </div>
    </div>
  );
}
