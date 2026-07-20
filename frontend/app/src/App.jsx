import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import GraphCanvas from './components/GraphCanvas.jsx';
import Sidebar from './components/Sidebar.jsx';
import QAPanel from './components/QAPanel.jsx';
import UploadPanel from './components/UploadPanel.jsx';
import GovernancePanel from './components/GovernancePanel.jsx';
import StatusBar from './components/StatusBar.jsx';
import Toasts from './components/Toasts.jsx';
import {
  fetchHealth, fetchKGData, fetchEvalRuns, fetchEvalGraph,
  submitQuestion, submitQuestionStream,
} from './api.js';
import {
  MOCK_ENTITY_TYPES, MOCK_ENTITIES, MOCK_TRIPLES,
  MOCK_RELATION_TYPES,
} from './mockData.js';

const QA_STORAGE_KEY = 'makg_qa_history';
const DOMAIN_PALETTE = [
  '#7c83e8', '#1de9b6', '#f59e0b', '#ef4444', '#22c55e',
  '#06b6d4', '#a78bfa', '#ec4899', '#84cc16', '#fb923c',
];

function loadStoredQaHistory() {
  try {
    const raw = JSON.parse(localStorage.getItem(QA_STORAGE_KEY) || '[]');
    if (!Array.isArray(raw)) return [];
    return raw
      .filter(t => t && (t.status === 'done' || t.status === 'error' || t.status === 'cancelled'))
      .slice(-50);
  } catch {
    return [];
  }
}

// Preserve the user's current selections when a (re)loaded graph brings a new
// type/relation catalog: known keys keep their state, new keys default to on.
function mergeFilters(prev, entityTypes, relationTypes) {
  return {
    entityTypes: Object.fromEntries(Object.keys(entityTypes).map(k => [k, prev.entityTypes[k] !== false])),
    relationTypes: Object.fromEntries(relationTypes.map(r => [r, prev.relationTypes[r] !== false])),
    confidenceThreshold: prev.confidenceThreshold,
    searchQuery: prev.searchQuery,
  };
}

function finalizeTurn(setQaHistory, turnId, data) {
  setQaHistory(prev => prev.map(t =>
    t.id === turnId ? {
      ...t,
      status: 'done',
      answer: data.answer || '',
      referencedNodes: data.referenced_nodes || [],
      referencedEdges: data.referenced_edges || [],
      nodeCount: (data.referenced_nodes || []).length,
      domainResponses: data.domain_responses || [],
      model: data.model || t.model,
      durationMs: t.startedAt ? Date.now() - t.startedAt : null,
    } : t
  ));
}

function endTurn(setQaHistory, turnId, status, errorMessage) {
  setQaHistory(prev => prev.map(t =>
    t.id === turnId
      ? { ...t, status, error: errorMessage, durationMs: t.startedAt ? Date.now() - t.startedAt : null }
      : t
  ));
}

// ── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  // ── Data state ────────────────────────────────────────────────────────────
  const [entityTypes, setEntityTypes]     = useState({});
  const [entities, setEntities]           = useState([]);
  const [triples, setTriples]             = useState([]);
  const [relationTypes, setRelationTypes] = useState([]);
  const [orgChart, setOrgChart]           = useState(null);
  const [reloadKey, setReloadKey]         = useState(0);

  // ── Graph source (live KG vs cached eval runs) ────────────────────────────
  const [graphSource, setGraphSource] = useState({ kind: 'live' });
  const [evalRuns, setEvalRuns]       = useState([]);

  // ── Mode / connectivity ───────────────────────────────────────────────────
  const [mode, setMode]               = useState('explore');
  const [selectedModel, setModel]     = useState('');
  const [serverConnected, setServer]  = useState(false);
  const [backendStatus, setBackendStatus] = useState('loading'); // loading | online | offline | no-kg
  const [qaReady, setQaReady]         = useState(false);
  const [kgStats, setKgStats]         = useState(null); // server-side totals from /health

  // ── Governance ────────────────────────────────────────────────────────────
  const [selectedDomainId, setSelectedDomainId] = useState(null);
  const [auditTriple, setAuditTriple]           = useState(null);

  // ── Filters ───────────────────────────────────────────────────────────────
  const [filters, setFilters] = useState({
    entityTypes: {}, relationTypes: {}, confidenceThreshold: 0, searchQuery: '',
  });
  const [debouncedQuery, setDebouncedQuery] = useState('');
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(filters.searchQuery), 250);
    return () => clearTimeout(t);
  }, [filters.searchQuery]);

  // ── Focus mode / fly-to ───────────────────────────────────────────────────
  const [focus, setFocus]   = useState(null); // { nodeId, hops }
  const [flyTo, setFlyTo]   = useState(null); // { id, n }

  // ── QA state (persisted) ──────────────────────────────────────────────────
  const [qaHistory,    setQaHistory]    = useState(loadStoredQaHistory);
  const [selectedQaId, setSelectedQaId] = useState(null);
  const [hoveredQaId,  setHoveredQaId]  = useState(null);
  const qaControllersRef = useRef({});

  useEffect(() => {
    try {
      const save = qaHistory.filter(t => t.status !== 'loading').slice(-50);
      localStorage.setItem(QA_STORAGE_KEY, JSON.stringify(save));
    } catch { /* storage full/blocked — persistence is best-effort */ }
  }, [qaHistory]);

  // ── Graph interaction ─────────────────────────────────────────────────────
  const [selectedNodeId,    setSelectedNodeId]    = useState(null);
  const [hoveredChipNodeId, setHoveredChipNodeId] = useState(null);

  // ── Toasts ────────────────────────────────────────────────────────────────
  const [toasts, setToasts] = useState([]);
  const pushToast = useCallback((text, kind = 'info') => {
    const id = `t_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    setToasts(prev => [...prev.slice(-3), { id, text, kind }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4500);
  }, []);
  const dismissToast = useCallback((id) => setToasts(prev => prev.filter(t => t.id !== id)), []);

  const tweaks = { forceStrength: -300, edgeOpacity: 0.45, showAllLabels: false };

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

  // ── Health polling (drives connectivity + reconnect refetch) ──────────────
  const prevOnlineRef = useRef(null);
  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function tick() {
      let online = false;
      try {
        const h = await fetchHealth();
        if (cancelled) return;
        online = h.status === 'ok';
        setQaReady(!!h.qa_ready);
        setKgStats(h.kg_stats ?? null);
      } catch { /* offline */ }
      if (cancelled) return;
      setServer(online);
      if (online && prevOnlineRef.current === false) {
        pushToast('Backend reconnected — refreshing data', 'success');
        setReloadKey(k => k + 1);
      }
      prevOnlineRef.current = online;
      timer = setTimeout(tick, 10_000);
    }

    tick();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [pushToast]);

  // ── Eval run discovery (once per reconnect cycle) ─────────────────────────
  useEffect(() => {
    if (!serverConnected) return undefined;
    let cancelled = false;
    fetchEvalRuns()
      .then(d => { if (!cancelled) setEvalRuns(d.runs || []); })
      .catch(() => { /* eval browsing is optional */ });
    return () => { cancelled = true; };
  }, [serverConnected]);

  // ── Graph data loading (live KG or eval run) ──────────────────────────────
  useEffect(() => {
    let cancelled = false;

    function applyGraphData(data) {
      if (cancelled) return;
      setEntities(data.entities);
      setTriples(data.triples);
      setEntityTypes(data.entity_types);
      setRelationTypes(data.relation_types);
      setOrgChart(data.org_chart ?? null);
      setFilters(prev => mergeFilters(prev, data.entity_types, data.relation_types));
      setFocus(null);
      setBackendStatus('online');
    }

    function applyMockData() {
      if (cancelled) return;
      setEntities(MOCK_ENTITIES);
      setTriples(MOCK_TRIPLES);
      setEntityTypes(MOCK_ENTITY_TYPES);
      setRelationTypes(MOCK_RELATION_TYPES);
      setOrgChart(null);
      setFilters(prev => mergeFilters(prev, MOCK_ENTITY_TYPES, MOCK_RELATION_TYPES));
    }

    async function load() {
      try {
        if (graphSource.kind === 'eval') {
          const data = await fetchEvalGraph(graphSource.dir, graphSource.strategy, graphSource.doc || 'all');
          applyGraphData(data);
          return;
        }
        const health = await fetchHealth();
        if (cancelled) return;
        if (health.status === 'ok' && health.kg_loaded) {
          applyGraphData(await fetchKGData());
        } else if (health.status === 'ok') {
          setBackendStatus('no-kg');
          applyMockData();
        } else {
          setBackendStatus('offline');
          applyMockData();
        }
      } catch (err) {
        if (cancelled) return;
        if (graphSource.kind === 'eval') {
          pushToast(`Eval run failed to load: ${err?.message || err}`, 'error');
          setBackendStatus('online'); // backend is up; just this run failed
        } else {
          setBackendStatus('offline');
          applyMockData();
        }
      }
    }

    load();
    return () => { cancelled = true; };
  }, [reloadKey, graphSource, pushToast]);

  // ── Derived: filtered entities / triples ──────────────────────────────────
  const MAX_RENDER_NODES = 1500;

  const degreeById = useMemo(() => {
    const deg = {};
    triples.forEach(t => {
      deg[t.subject] = (deg[t.subject] || 0) + 1;
      deg[t.object]  = (deg[t.object]  || 0) + 1;
    });
    return deg;
  }, [triples]);

  // Focus mode: BFS neighborhood around the focused node.
  const focusSet = useMemo(() => {
    if (!focus?.nodeId) return null;
    const adj = {};
    triples.forEach(t => {
      (adj[t.subject] = adj[t.subject] || []).push(t.object);
      (adj[t.object]  = adj[t.object]  || []).push(t.subject);
    });
    let frontier = [focus.nodeId];
    const seen = new Set(frontier);
    for (let h = 0; h < (focus.hops || 1); h += 1) {
      const next = [];
      for (const id of frontier) {
        for (const nb of adj[id] || []) {
          if (!seen.has(nb)) { seen.add(nb); next.push(nb); }
        }
      }
      frontier = next;
    }
    return seen;
  }, [focus, triples]);

  const { filteredEntities, nodeCapApplied } = useMemo(() => {
    const q = debouncedQuery.toLowerCase();
    let list = entities.filter(e => {
      if (focusSet && !focusSet.has(e.id)) return false;
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
  }, [entities, filters.entityTypes, debouncedQuery, degreeById, focusSet]);

  const filteredTriples = useMemo(() => {
    const ids = new Set(filteredEntities.map(e => e.id));
    return triples.filter(t =>
      ids.has(t.subject) && ids.has(t.object) &&
      filters.relationTypes[t.relation] !== false &&
      (t.confidence == null || t.confidence >= filters.confidenceThreshold)
    );
  }, [triples, filteredEntities, filters.relationTypes, filters.confidenceThreshold]);

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
    const hedges = (turn.referencedEdges && turn.referencedEdges.length > 0)
      ? new Set(turn.referencedEdges)
      : new Set(triples.filter(t => hnodes.has(t.subject) && hnodes.has(t.object)).map(t => t.id));
    return { highlightedNodes: hnodes, highlightedEdges: hedges };
  }, [hoveredQaId, selectedQaId, qaHistory, mode, triples, selectedDomainId, auditTriple, orgChart]);

  // ── QA submit / cancel ────────────────────────────────────────────────────
  const handleSubmitQuestion = useCallback(async (question) => {
    const turnId = `qa_${Date.now()}`;
    const controller = new AbortController();
    qaControllersRef.current[turnId] = controller;

    const newTurn = {
      id: turnId, question, status: 'loading',
      answer: '', referencedNodes: [], referencedEdges: [],
      nodeCount: 0, model: selectedModel, startedAt: Date.now(),
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
    };
    setQaHistory(prev => [...prev, newTurn]);
    setSelectedQaId(turnId);
    setMode('qa');

    const onProgress = (msg) => {
      setQaHistory(prev => prev.map(t =>
        t.id === turnId ? { ...t, stage: msg.stage, stageInfo: msg.info || {} } : t
      ));
    };

    try {
      let data;
      try {
        data = await submitQuestionStream(question, selectedModel, onProgress, { signal: controller.signal });
      } catch (streamErr) {
        // Older backends without /qa/stream — fall back to the blocking endpoint.
        if (String(streamErr?.message || '').includes('HTTP 404')) {
          data = await submitQuestion(question, selectedModel, { signal: controller.signal });
        } else {
          throw streamErr;
        }
      }
      finalizeTurn(setQaHistory, turnId, data);
    } catch (err) {
      if (controller.userCancelled) {
        endTurn(setQaHistory, turnId, 'cancelled');
        pushToast('Question cancelled (the server may still finish it in the background)', 'info');
      } else {
        const msg = err?.name === 'AbortError' || err?.name === 'TimeoutError'
          ? 'Request timed out. The QA pipeline (3–15 LLM calls) may be slow on a cold or large model. Try again, or pick a faster model.'
          : (err?.message || String(err) || 'Unknown error');
        endTurn(setQaHistory, turnId, 'error', msg);
        pushToast('QA request failed', 'error');
      }
    } finally {
      delete qaControllersRef.current[turnId];
    }
  }, [selectedModel, pushToast]);

  const handleCancelQa = useCallback((turnId) => {
    const c = qaControllersRef.current[turnId];
    if (c) {
      c.userCancelled = true;
      c.abort();
    }
  }, []);

  const clearChats = useCallback(() => {
    setQaHistory([]);
    setSelectedQaId(null);
    setHoveredQaId(null);
    try { localStorage.removeItem(QA_STORAGE_KEY); } catch { /* ignore */ }
  }, []);

  // ── Focus mode / fly-to handlers ──────────────────────────────────────────
  const entityLabel = useCallback((id) => {
    const e = entities.find(x => x.id === id);
    return e?.labels?.[0] || id;
  }, [entities]);

  const handleNodeFocus = useCallback((id) => {
    setFocus({ nodeId: id, hops: 1 });
  }, []);

  const handleSearchEnter = useCallback(() => {
    const q = filters.searchQuery.trim().toLowerCase();
    if (!q) return;
    const hit = filteredEntities.find(e => e.labels?.some(l => l.toLowerCase().includes(q)))
      || filteredEntities.find(e => e.id.includes(q));
    if (hit) {
      setSelectedNodeId(hit.id);
      setFlyTo(f => ({ id: hit.id, n: (f?.n || 0) + 1 }));
    } else {
      pushToast('No visible entity matches that search', 'info');
    }
  }, [filters.searchQuery, filteredEntities, pushToast]);

  // ── Keyboard shortcuts: "/" focuses search, Esc clears focus/selection ────
  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target?.tagName || '';
      if (e.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(tag)) {
        e.preventDefault();
        document.getElementById('entity-search')?.focus();
      } else if (e.key === 'Escape') {
        setFocus(null);
        setSelectedNodeId(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // ── Layout ────────────────────────────────────────────────────────────────
  const handleModeChange = useCallback((m) => {
    setMode(m);
    if (m !== 'governance') {
      setSelectedDomainId(null);
      setAuditTriple(null);
    }
  }, []);

  const handlePipelineComplete = useCallback(() => {
    pushToast('Pipeline finished — graph refreshed', 'success');
    setReloadKey(k => k + 1);
  }, [pushToast]);

  const handleExported = useCallback((what) => pushToast(`${what} exported`, 'success'), [pushToast]);

  const qaMode = mode === 'qa';
  const govMode = mode === 'governance';
  const panelOpen = qaMode || govMode;
  const isLoadingData = backendStatus === 'loading';

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
        onModeChange={handleModeChange}
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
        graphSource={graphSource}
        onGraphSourceChange={setGraphSource}
        evalRuns={evalRuns}
        onSearchEnter={handleSearchEnter}
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
          onNodeFocus={handleNodeFocus}
          hoveredChipNodeId={hoveredChipNodeId}
          tweaks={tweaks}
          nodeColorById={nodeColorById}
          flyTo={flyTo}
          loading={isLoadingData}
          onExported={handleExported}
        />

        <StatusBar
          serverConnected={serverConnected}
          kgStats={kgStats}
          onPipelineComplete={handlePipelineComplete}
        />

        {/* QA / governance dim overlay */}
        {panelOpen && (
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
          background: qaMode ? 'rgba(124,131,232,0.15)' : govMode ? 'rgba(245,158,11,0.12)' : 'rgba(29,233,182,0.12)',
          color: qaMode ? '#7c83e8' : govMode ? '#f59e0b' : '#1de9b6',
          border: `1px solid ${qaMode ? 'rgba(124,131,232,0.25)' : govMode ? 'rgba(245,158,11,0.25)' : 'rgba(29,233,182,0.2)'}`,
          pointerEvents: 'none',
        }}>
          {qaMode ? 'QA — EVIDENCE VIEW' : govMode ? 'GOVERNANCE' : 'EXPLORE'}
        </div>

        {/* Focus mode chip */}
        {focus && (
          <div style={{
            position: 'absolute', top: 48, left: 16, zIndex: 220,
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '4px 10px', borderRadius: 3, fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace",
            background: 'rgba(29,233,182,0.1)', color: '#1de9b6',
            border: '1px solid rgba(29,233,182,0.3)',
          }}>
            FOCUS: {entityLabel(focus.nodeId)}
            <button
              onClick={() => setFocus(f => ({ ...f, hops: f.hops === 1 ? 2 : 1 }))}
              title="Toggle neighborhood depth"
              style={{
                background: 'rgba(255,255,255,0.06)', border: 'none', borderRadius: 2,
                color: '#1de9b6', fontSize: 9, cursor: 'pointer', padding: '1px 6px',
                fontFamily: "'JetBrains Mono', monospace",
              }}>
              {focus.hops}-hop
            </button>
            <button
              onClick={() => setFocus(null)}
              title="Clear focus (Esc)"
              style={{
                background: 'none', border: 'none', color: '#1de9b6',
                fontSize: 11, cursor: 'pointer', padding: 0, lineHeight: 1,
              }}>
              ✕
            </button>
          </div>
        )}

        {/* Node cap notice */}
        {nodeCapApplied && (
          <div style={{
            position: 'absolute', bottom: 16, right: 16,
            padding: '4px 10px', borderRadius: 3, fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace",
            background: 'rgba(245,158,11,0.1)', color: '#f59e0b',
            border: '1px solid rgba(245,158,11,0.25)', pointerEvents: 'none',
          }}>
            showing top {MAX_RENDER_NODES} nodes by degree — use filters or double-click a node to focus
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
              onCancelQa={handleCancelQa}
              qaReady={qaReady}
              serverConnected={serverConnected}
              kgStats={kgStats}
            />
          )}
        </div>
      </div>

      <Toasts toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
