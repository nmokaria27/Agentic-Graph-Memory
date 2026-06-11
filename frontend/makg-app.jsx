// makg-app.jsx — Main application shell, state orchestration, layout

const { useState, useEffect, useRef, useMemo, useCallback } = React;
const { ENTITY_TYPES, ENTITIES, TRIPLES, SAMPLE_QA, RELATION_TYPES } = window.MAKG_DATA;

const API_BASE_URL = 'http://localhost:8000';

// ── Streaming simulation ────────────────────────────────────────────────────
function streamAnswer(setQaHistory, turnId, fullAnswer, referencedNodes, referencedEdges) {
  let charIndex = 0;
  const interval = setInterval(() => {
    const chunk = Math.floor(Math.random() * 9) + 4;
    charIndex = Math.min(charIndex + chunk, fullAnswer.length);
    const slice = fullAnswer.slice(0, charIndex);
    setQaHistory(prev => prev.map(t =>
      t.id === turnId ? { ...t, status: 'streaming', streamedAnswer: slice } : t
    ));
    if (charIndex >= fullAnswer.length) {
      clearInterval(interval);
      setTimeout(() => {
        setQaHistory(prev => prev.map(t =>
          t.id === turnId ? {
            ...t, status: 'done', answer: fullAnswer,
            referencedNodes, referencedEdges,
            nodeCount: referencedNodes.size,
          } : t
        ));
      }, 250);
    }
  }, 22);
}

// Pick the best-matching sample QA by keyword overlap
function bestMockMatch(question) {
  const qWords = new Set(question.toLowerCase().split(/\W+/).filter(w => w.length > 3));
  let best = SAMPLE_QA[0], bestScore = 0;
  SAMPLE_QA.forEach(qa => {
    const qaWords = qa.question.toLowerCase().split(/\W+/).filter(w => w.length > 3);
    const score = qaWords.filter(w => qWords.has(w)).length;
    if (score > bestScore) { bestScore = score; best = qa; }
  });
  return best;
}

// ── Main App ────────────────────────────────────────────────────────────────
function App() {
  // ── mode ──────────────────────────────────────────────────────────────────
  const [mode, setMode]               = useState('explore');
  const [selectedModel, setModel]     = useState('GPT-4o');
  const [serverConnected, setServer]  = useState(false);

  // ── filters ───────────────────────────────────────────────────────────────
  const [filters, setFilters] = useState(() => ({
    entityTypes:         Object.fromEntries(Object.keys(ENTITY_TYPES).map(k => [k, true])),
    relationTypes:       Object.fromEntries(RELATION_TYPES.map(r => [r, true])),
    confidenceThreshold: 0,
    searchQuery:         '',
  }));

  // ── QA state ──────────────────────────────────────────────────────────────
  const [qaHistory,    setQaHistory]    = useState([]);
  const [selectedQaId, setSelectedQaId] = useState(null);
  const [hoveredQaId,  setHoveredQaId]  = useState(null);

  // ── graph interaction state ───────────────────────────────────────────────
  const [selectedNodeId,   setSelectedNodeId]   = useState(null);
  const [hoveredChipNodeId, setHoveredChipNodeId] = useState(null);

  // ── tweaks ────────────────────────────────────────────────────────────────
  const { tweaks, setTweak, TweaksPanel: TweaksUI } = typeof useTweaks !== 'undefined'
    ? useTweaks(window.TWEAK_DEFAULTS || {})
    : { tweaks: window.TWEAK_DEFAULTS || {}, setTweak: () => {}, TweaksPanel: null };

  // ── server health check ───────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(1500) })
      .then(r => r.ok && setServer(true))
      .catch(() => setServer(false));
  }, []);

  // ── derived: filtered entities / triples ─────────────────────────────────
  const filteredEntities = useMemo(() => {
    const q = filters.searchQuery.toLowerCase();
    return ENTITIES.filter(e => {
      if (filters.entityTypes[e.type] === false) return false;
      if (q && !e.labels.some(l => l.toLowerCase().includes(q)) && !e.id.includes(q)) return false;
      return true;
    });
  }, [filters]);

  const filteredTriples = useMemo(() => {
    const ids = new Set(filteredEntities.map(e => e.id));
    return TRIPLES.filter(t =>
      ids.has(t.subject) && ids.has(t.object) &&
      filters.relationTypes[t.relation] !== false &&
      (t.confidence == null || t.confidence >= filters.confidenceThreshold)
    );
  }, [filteredEntities, filters]);

  // ── derived: highlighted nodes/edges for graph ───────────────────────────
  const { highlightedNodes, highlightedEdges } = useMemo(() => {
    const qaId = hoveredQaId || selectedQaId;
    if (!qaId || mode !== 'qa') return { highlightedNodes: new Set(), highlightedEdges: new Set() };
    const turn = qaHistory.find(t => t.id === qaId);
    if (!turn || turn.status === 'loading') return { highlightedNodes: new Set(), highlightedEdges: new Set() };
    return {
      highlightedNodes: turn.referencedNodes || new Set(),
      highlightedEdges: turn.referencedEdges || new Set(),
    };
  }, [hoveredQaId, selectedQaId, qaHistory, mode]);

  // ── QA submit ─────────────────────────────────────────────────────────────
  const handleSubmitQuestion = useCallback(async (question) => {
    const turnId = `qa_${Date.now()}`;
    const newTurn = {
      id: turnId, question, status: 'loading', streamedAnswer: '',
      answer: '', referencedNodes: new Set(), referencedEdges: new Set(),
      nodeCount: 0, model: selectedModel,
      timestamp: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }),
    };
    setQaHistory(prev => [...prev, newTurn]);
    setSelectedQaId(turnId);
    if (mode !== 'qa') setMode('qa');

    // 1.2s artificial routing delay
    await new Promise(r => setTimeout(r, 1200));

    try {
      const resp = await fetch(`${API_BASE_URL}/qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, model: selectedModel }),
        signal: AbortSignal.timeout(8000),
      });
      if (!resp.ok) throw new Error('server error');
      const data = await resp.json();
      streamAnswer(setQaHistory, turnId, data.answer || '',
        new Set(data.referenced_nodes || []),
        new Set(data.referenced_edges || []));
    } catch {
      // Fall back to mock
      const mock = bestMockMatch(question);
      streamAnswer(setQaHistory, turnId, mock.answer, mock.referencedNodes, mock.referencedEdges);
    }
  }, [selectedModel, mode]);

  // ── on mode switch to QA: load sample QA turns if empty ──────────────────
  const hasLoadedSamples = useRef(false);
  useEffect(() => {
    if (mode === 'qa' && qaHistory.length === 0 && !hasLoadedSamples.current) {
      hasLoadedSamples.current = true;
      // Stagger sample turns in
      SAMPLE_QA.forEach((qa, i) => {
        setTimeout(() => {
          setQaHistory(prev => [...prev, {
            id: qa.id, question: qa.question, status: 'done',
            streamedAnswer: '', answer: qa.answer,
            referencedNodes: qa.referencedNodes,
            referencedEdges: qa.referencedEdges,
            nodeCount: qa.nodeCount, model: qa.model, timestamp: qa.timestamp,
          }]);
          if (i === 0) setSelectedQaId(qa.id);
        }, i * 80);
      });
    }
  }, [mode]);

  // ── layout ────────────────────────────────────────────────────────────────
  const qaMode = mode === 'qa';

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: '#0f1117' }}>

      {/* ── Left sidebar ─────────────────────────────────────────────────── */}
      <LeftSidebar
        mode={mode}
        onModeChange={setMode}
        filters={filters}
        onFiltersChange={setFilters}
        qaHistory={qaHistory}
        selectedQaId={selectedQaId}
        onQaSelect={id => setSelectedQaId(prev => prev === id ? null : id)}
        onQaHover={setHoveredQaId}
        serverConnected={serverConnected}
      />

      {/* ── Center: graph canvas ──────────────────────────────────────────── */}
      <div style={{ flex: 1, position: 'relative', minWidth: 0 }}>
        <GraphCanvas
          entities={filteredEntities}
          triples={filteredTriples}
          mode={mode}
          highlightedNodes={highlightedNodes}
          highlightedEdges={highlightedEdges}
          selectedNodeId={selectedNodeId}
          onNodeClick={id => setSelectedNodeId(prev => prev === id ? null : id)}
          onNodeHover={() => {}}
          hoveredChipNodeId={hoveredChipNodeId}
          tweaks={tweaks}
        />

        {/* QA mode dim overlay (subtle background shift) */}
        {qaMode && (
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            background: 'rgba(15,17,23,0.18)', transition: 'opacity 0.4s',
            borderRight: '1px solid rgba(255,255,255,0.05)',
          }} />
        )}

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
          {qaMode ? 'QA — EVIDENCE VIEW' : 'EXPLORE'}
        </div>
      </div>

      {/* ── Right QA panel (slides in) ────────────────────────────────────── */}
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
          />
        </div>
      </div>

      {/* ── Tweaks panel ──────────────────────────────────────────────────── */}
      {typeof TweaksPanel !== 'undefined' && (
        <TweaksPanel>
          <TweakSection label="Graph">
            <TweakSlider label="Force strength" k="forceStrength" min={-600} max={-80} step={20} value={tweaks.forceStrength} onChange={v => setTweak('forceStrength', v)} />
            <TweakSlider label="Edge opacity" k="edgeOpacity" min={0.1} max={1} step={0.05} value={tweaks.edgeOpacity} onChange={v => setTweak('edgeOpacity', v)} />
            <TweakToggle label="Show all labels" k="showAllLabels" value={tweaks.showAllLabels} onChange={v => setTweak('showAllLabels', v)} />
          </TweakSection>
          <TweakSection label="Appearance">
            <TweakColor label="Accent" k="accentColor" options={['#1de9b6', '#7c83e8', '#f59e0b', '#e879f9']} value={tweaks.accentColor} onChange={v => setTweak('accentColor', v)} />
          </TweakSection>
        </TweaksPanel>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
