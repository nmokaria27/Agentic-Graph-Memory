import { useState, useEffect, useMemo, useCallback } from 'react';
import GraphCanvas from './components/GraphCanvas.jsx';
import Sidebar from './components/Sidebar.jsx';
import QAPanel from './components/QAPanel.jsx';
import UploadPanel from './components/UploadPanel.jsx';
import GovernancePanel from './components/GovernancePanel.jsx';
import StatusBar from './components/StatusBar.jsx';
import {
  fetchHealth, fetchKGData, submitQuestion, submitQuestionStream,
  fetchEvalRuns, fetchEvalGraph,
} from './api.js';
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

  // ── Graph source: live governed KG, or a cached eval run (read-only) ──────
  // { kind: 'live' } | { kind: 'eval', dir, strategy, doc }
  const [graphSource, setGraphSource] = useState({ kind: 'live' });
  const [evalRuns, setEvalRuns]       = useState([]);
  const [evalError, setEvalError]     = useState('');

  // ── Mode ──────────────────────────────────────────────────────────────────
  const [mode, setMode]               = useState('explore');
  const [selectedModel, setModel]     = useState('gemma4:31b');
  const [serverConnected, setServer]  = useState(false);
  const [backendStatus, setBackendStatus] = useState('loading'); // 'loading' | 'online' | 'offline' | 'no-kg'
  const [qaReady, setQaReady]         = useState(false);

  // ── Governance ──────────────────────────────────────────────────────────
  const [selectedDomainId, setSelectedDomainId] = useState(null);
  const [auditTriple, setAuditTriple]           = useState(null);

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

  // ── Discover available eval runs (once) ───────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetchEvalRuns()
      .then(data => { if (!cancelled) setEvalRuns(data.runs || []); })
      .catch(() => { /* eval endpoint optional — ignore if unreachable */ });
    return () => { cancelled = true; };
  }, []);

  // ── Fetch graph data from server (live KG, or a selected eval run) ────────
  useEffect(() => {
    let cancelled = false;

    function applyGraphData(data) {
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
    }

    async function loadLive() {
      try {
        const health = await fetchHealth();
        if (cancelled) return;

        if (health.status === 'ok') {
          setServer(true);
          setQaReady(!!health.qa_ready);

          if (health.kg_loaded) {
            const data = await fetchKGData();
            if (cancelled) return;
            applyGraphData(data);
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

    async function loadEval({ dir, strategy, doc }) {
      setEvalError('');
      try {
        const health = await fetchHealth();
        if (cancelled) return;
        setServer(health.status === 'ok');
        setQaReady(!!health.qa_ready);
        if (health.status !== 'ok') { setBackendStatus('offline'); return; }

        const data = await fetchEvalGraph(dir, strategy, doc);
        if (cancelled) return;
        applyGraphData(data);
        setBackendStatus('online');
      } catch (err) {
        if (!cancelled) {
          setBackendStatus('offline');
          setEvalError(err?.message || 'Failed to load eval run');
        }
      }
    }

    if (graphSource.kind === 'eval') loadEval(graphSource);
    else loadLive();

    return () => { cancelled = true; };
  }, [reloadKey, graphSource]);

  // Graph size for the currently displayed source (live or eval) — shown in
  // the sidebar footer and status bar.
  const kgStats = useMemo(() => ({ entities: entities.length, triples: triples.length }), [entities, triples]);

  // ── Derived: filtered entities / triples ──────────────────────────────────
  // Cap rendered nodes for performance on very large KGs; highest-degree
  // entities are kept so the visible graph stays informative.
  const MAX_RENDER_NODES = 1500;

  const degreeById = useMemo(() => {
    const deg = {};
    triples.forEach(t => {
      deg[t.subject] = (deg[t.subject] || 0) + 1;
      deg[t.object]  = (deg[t.object]  || 0) + 1;
    });
    return deg;
  }, [triples]);

  const { filteredEntities, nodeCapApplied } = useMemo(() => {
    const q = filters.searchQuery.toLowerCase();
    let list = entities.filter(e => {
      if (filters.entityTypes[e.type] === false) return false;
      if (q && !e.labels.some(l => l.toLowerCase().includes(q)) && !e.id.includes(q)) return false;
      return true;
    });
    let capped = false;
    if (list.length > MAX_RENDER_NODES) {
      capped = true;
      list = [...list]
        .sort((a, b) => (degreeById[b.id] || 0) - (degreeById[a.id] || 0))
        .slice(0, MAX_RENDER_NODES);
    }
    return { filteredEntities: list, nodeCapApplied: capped };
  }, [entities, filters, degreeById]);

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
    const empty = { highlightedNodes: new Set(), highlightedEdges: new Set() };

    if (mode === 'governance') {
      if (auditTriple) {
        const gnodes = new Set([auditTriple.subject, auditTriple.object].filter(Boolean));
        const gedges = new Set(triples
          .filter(t => t.subject === auditTriple.subject && t.relation === auditTriple.relation && t.object === auditTriple.object)
          .map(t => t.id));
        return { highlightedNodes: gnodes, highlightedEdges: gedges };
      }
      if (selectedDomainId && orgChart?.domains) {
        const domain = orgChart.domains.find(d => d.domain_id === selectedDomainId);
        if (!domain) return empty;
        const gnodes = new Set(domain.entity_ids || []);
        const gedges = new Set(triples.filter(t => gnodes.has(t.subject) || gnodes.has(t.object)).map(t => t.id));
        return { highlightedNodes: gnodes, highlightedEdges: gedges };
      }
      return empty;
    }

    const qaId = hoveredQaId || selectedQaId;
    if (!qaId || mode !== 'qa') return empty;
    const turn = qaHistory.find(t => t.id === qaId);
    if (!turn || turn.status === 'loading') return empty;
    const hnodes = new Set(turn.referencedNodes || []);
    // Use explicit edge IDs if available, otherwise derive from triples whose both endpoints are highlighted
    const hedges = (turn.referencedEdges && turn.referencedEdges.length > 0)
      ? new Set(turn.referencedEdges)
      : new Set(triples.filter(t => hnodes.has(t.subject) && hnodes.has(t.object)).map(t => t.id));
    return { highlightedNodes: hnodes, highlightedEdges: hedges };
  }, [hoveredQaId, selectedQaId, qaHistory, mode, triples, selectedDomainId, auditTriple, orgChart]);

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

    const onProgress = (msg) => {
      setQaHistory(prev => prev.map(t =>
        t.id === turnId ? { ...t, stage: msg.stage, stageInfo: msg.info || {} } : t
      ));
    };

    try {
      let data;
      try {
        data = await submitQuestionStream(question, selectedModel, onProgress);
      } catch (streamErr) {
        // Older backends without /qa/stream — fall back to the blocking endpoint.
        if (String(streamErr?.message || '').includes('HTTP 404')) {
          data = await submitQuestion(question, selectedModel);
        } else {
          throw streamErr;
        }
      }
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

  const handlePipelineComplete = useCallback(() => setReloadKey(k => k + 1), []);

  // ── Layout ────────────────────────────────────────────────────────────────
  const handleModeChange = useCallback((m) => {
    setMode(m);
    if (m !== 'governance') {
      setSelectedDomainId(null);
      setAuditTriple(null);
    }
  }, []);

  const qaMode = mode === 'qa';
  const govMode = mode === 'governance';
  const panelOpen = qaMode || govMode;

  const banner = (() => {
    if (backendStatus === 'loading' || backendStatus === 'online') return null;
    const text = graphSource.kind === 'eval' && evalError
      ? `Failed to load eval run: ${evalError}`
      : backendStatus === 'offline'
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
        onModeChange={handleModeChange}
        graphSource={graphSource}
        onGraphSourceChange={setGraphSource}
        evalRuns={evalRuns}
        filters={filters}
        onFiltersChange={setFilters}
        entityTypes={entityTypes}
        relationTypes={relationTypes}
        qaHistory={qaHistory}
        selectedQaId={selectedQaId}
        onQaSelect={id => setSelectedQaId(prev => prev === id ? null : id)}
        onQaHover={setHoveredQaId}
        serverConnected={serverConnected}
        kgStats={kgStats}
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

        {/* QA / governance dim overlay */}
        {panelOpen && (
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            background: 'rgba(15,17,23,0.18)', transition: 'opacity 0.4s',
            borderRight: '1px solid rgba(255,255,255,0.05)',
          }} />
        )}

        <UploadPanel onIngestComplete={handlePipelineComplete} />

        <StatusBar
          serverConnected={serverConnected}
          kgStats={kgStats}
          onPipelineComplete={handlePipelineComplete}
        />

        {/* Mode indicator badge */}
        <div style={{
          position: 'absolute', top: 16, left: 16,
          padding: '4px 10px', borderRadius: 3, fontSize: 10,
          fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.08em',
          background: qaMode ? 'rgba(124,131,232,0.15)' : govMode ? 'rgba(245,158,11,0.12)' : 'rgba(29,233,182,0.12)',
          color: qaMode ? '#7c83e8' : govMode ? '#f59e0b' : '#1de9b6',
          border: `1px solid ${qaMode ? 'rgba(124,131,232,0.25)' : govMode ? 'rgba(245,158,11,0.25)' : 'rgba(29,233,182,0.2)'}`,
          pointerEvents: 'none',
        }}>
          {qaMode ? 'QA \u2014 EVIDENCE VIEW' : govMode ? 'GOVERNANCE' : 'EXPLORE'}
        </div>

        {/* Node cap notice */}
        {nodeCapApplied && (
          <div style={{
            position: 'absolute', bottom: 16, right: 16,
            padding: '4px 10px', borderRadius: 3, fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace",
            background: 'rgba(245,158,11,0.1)', color: '#f59e0b',
            border: '1px solid rgba(245,158,11,0.25)', pointerEvents: 'none',
          }}>
            showing top {MAX_RENDER_NODES} nodes by degree {'\u2014'} use filters to narrow
          </div>
        )}
      </div>

      {/* Right panel (slides in): QA or Governance */}
      <div style={{
        width: panelOpen ? 380 : 0,
        overflow: 'hidden',
        transition: 'width 0.28s cubic-bezier(0.4,0,0.2,1)',
        flexShrink: 0,
      }}>
        <div style={{ width: 380, height: '100%' }}>
          {govMode ? (
            <GovernancePanel
              orgChart={orgChart}
              selectedDomainId={selectedDomainId}
              onDomainSelect={id => { setSelectedDomainId(id); setAuditTriple(null); }}
              onAuditSelect={t => { setAuditTriple(t); setSelectedDomainId(null); }}
              onKGChanged={() => setReloadKey(k => k + 1)}
            />
          ) : (
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
          )}
        </div>
      </div>
    </div>
  );
}
