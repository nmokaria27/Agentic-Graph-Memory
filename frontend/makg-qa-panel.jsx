// makg-qa-panel.jsx — Right QA panel: input, model selector, streaming feed, chips

const { useState: useStQ, useEffect: useEffQ, useRef: useRefQ, useCallback: useCbQ } = React;
const { ENTITY_TYPES: ET } = window.MAKG_DATA;

const TEAL_Q   = '#1de9b6';
const INDIGO_Q = '#7c83e8';
const MODELS   = ['GPT-4o', 'GPT-5', 'DeepSeek-v3', 'Gemma-3'];

function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong style="color:#c8d3e0">$1</strong>')
    .replace(/\n\n/g, '</p><p style="margin:0 0 10px">')
    .replace(/\n—\s/g, '<br><span style="color:#7c83e8">—</span> ')
    .replace(/\n/g, '<br>');
  return `<p style="margin:0 0 10px">${html}</p>`;
}

function ChipRow({ turn, onChipHover }) {
  const [visibleCount, setVisibleCountQ] = useStQ(0);

  useEffQ(() => {
    if (turn.status !== 'done') { setVisibleCountQ(0); return; }
    const nodes = [...(turn.referencedNodes || new Set())];
    const edges = [...(turn.referencedEdges || new Set())];
    const total = nodes.length + edges.length;
    let i = 0;
    const timer = setInterval(() => {
      i++;
      setVisibleCountQ(i);
      if (i >= total) clearInterval(timer);
    }, 55);
    return () => clearInterval(timer);
  }, [turn.status]);  // eslint-disable-line

  if (turn.status !== 'done') return null;
  const nodeIds = [...(turn.referencedNodes || new Set())];
  const edgeIds = [...(turn.referencedEdges || new Set())];
  const totalChips = nodeIds.length + edgeIds.length;
  if (totalChips === 0) return null;

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#3d4555', marginBottom: 8 }}>
        Referenced nodes &amp; edges
      </div>
      <div style={{ display: 'flex', gap: 5, overflowX: 'auto', paddingBottom: 6, flexWrap: 'wrap' }}>
        {nodeIds.map((nid, idx) => {
          const entity = window.MAKG_DATA.ENTITIES.find(e => e.id === nid);
          const color  = (ET[entity?.type] || {}).color || '#4a5568';
          const label  = entity ? (entity.labels[0] || nid) : nid;
          const visible = idx < visibleCount;
          return (
            <div key={nid}
              onMouseEnter={() => onChipHover(nid)}
              onMouseLeave={() => onChipHover(null)}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 5,
                padding: '3px 8px', borderRadius: 3, cursor: 'default', flexShrink: 0,
                background: `${color}18`, border: `1px solid ${color}44`,
                opacity: visible ? 1 : 0, transform: visible ? 'translateY(0)' : 'translateY(4px)',
                transition: 'opacity 0.25s ease, transform 0.25s ease',
                fontSize: 11, color: '#c8d3e0', fontFamily: "'JetBrains Mono', monospace",
              }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: color, display: 'inline-block', flexShrink: 0 }} />
              <span style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 10 }}>{label}</span>
            </div>
          );
        })}
        {edgeIds.map((eid, idx) => {
          const triple = window.MAKG_DATA.TRIPLES.find(t => t.id === eid);
          const label  = triple ? triple.relation : eid;
          const visible = (nodeIds.length + idx) < visibleCount;
          return (
            <div key={eid}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 5,
                padding: '3px 8px', borderRadius: 3, cursor: 'default', flexShrink: 0,
                background: 'rgba(124,131,232,0.1)', border: '1px solid rgba(124,131,232,0.25)',
                opacity: visible ? 1 : 0, transform: visible ? 'translateY(0)' : 'translateY(4px)',
                transition: 'opacity 0.25s ease, transform 0.25s ease',
              }}>
              <span style={{ color: INDIGO_Q, fontSize: 10 }}>→</span>
              <span style={{ fontSize: 10, color: '#7c83e8', fontFamily: "'JetBrains Mono', monospace", maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function QATurn({ turn, onChipHover }) {
  return (
    <div style={{ marginBottom: 20 }}>
      {/* Question bubble */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 10 }}>
        <div style={{
          maxWidth: '82%', padding: '9px 13px', borderRadius: '4px 4px 2px 4px',
          background: 'rgba(29,233,182,0.1)', border: '1px solid rgba(29,233,182,0.2)',
          fontSize: 13, color: '#c8d3e0', lineHeight: 1.5,
        }}>
          {turn.question}
        </div>
      </div>

      {/* Answer */}
      <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
        <div style={{ maxWidth: '95%', width: '100%' }}>
          <div style={{
            padding: '10px 13px', borderRadius: '2px 4px 4px 4px',
            background: '#1e2334', border: '1px solid rgba(255,255,255,0.07)',
            fontSize: 13, color: '#8892a4', lineHeight: 1.65, minHeight: 48,
          }}>
            {/* Model badge */}
            <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: INDIGO_Q, background: 'rgba(124,131,232,0.12)', padding: '2px 7px', borderRadius: 2 }}>
                {turn.model}
              </span>
              {turn.timestamp && (
                <span style={{ fontSize: 10, color: '#2a3040', fontFamily: "'JetBrains Mono', monospace" }}>{turn.timestamp}</span>
              )}
            </div>

            {turn.status === 'loading' && (
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '8px 0' }}>
                <span style={{ fontSize: 12, color: '#3d4555', fontStyle: 'italic' }}>Routing to domain experts</span>
                {[0,1,2].map(i => (
                  <span key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: TEAL_Q, display: 'inline-block', animation: `dotBounce 1.2s ${i * 0.2}s ease-in-out infinite` }} />
                ))}
              </div>
            )}

            {turn.status === 'streaming' && (
              <div style={{ whiteSpace: 'pre-wrap', color: '#8892a4', lineHeight: 1.65 }}>
                {turn.streamedAnswer}
                <span style={{ animation: 'cursorBlink 0.75s step-end infinite', color: TEAL_Q, marginLeft: 1 }}>█</span>
              </div>
            )}

            {turn.status === 'done' && (
              <div
                style={{ color: '#8892a4', lineHeight: 1.65 }}
                dangerouslySetInnerHTML={{ __html: renderMarkdown(turn.answer) }}
              />
            )}
          </div>

          <ChipRow turn={turn} onChipHover={onChipHover} />
        </div>
      </div>
    </div>
  );
}

function QAPanel({
  qaHistory, onSubmitQuestion, onChipHover,
  selectedModel, onModelChange, activeQaId, scrollToId,
}) {
  const [input, setInput]       = useStQ('');
  const [modelOpen, setModelOpen] = useStQ(false);
  const feedRef   = useRefQ(null);
  const inputRef  = useRefQ(null);

  // Auto-scroll when new turns appear
  useEffQ(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [qaHistory.length, activeQaId]);

  const handleSubmit = useCbQ(() => {
    const q = input.trim();
    if (!q) return;
    setInput('');
    onSubmitQuestion(q);
  }, [input, onSubmitQuestion]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  const suggestions = [
    'What biomarkers are associated with insulin resistance?',
    'How is NAFLD diagnosed and which methods are used?',
    'Which methods are applied to adipose tissue samples?',
    'How does obesity relate to metabolic syndrome?',
  ];

  const panelStyle = {
    width: 380, flexShrink: 0, height: '100%',
    background: '#181c26', borderLeft: '1px solid rgba(255,255,255,0.07)',
    display: 'flex', flexDirection: 'column', overflow: 'hidden',
  };

  return (
    <div style={panelStyle}>
      {/* ── Top: input + model selector ──────────────────────────────────── */}
      <div style={{ padding: '14px 14px 12px', borderBottom: '1px solid rgba(255,255,255,0.07)', flexShrink: 0 }}>
        <div style={{ position: 'relative', marginBottom: 10 }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question grounded in the knowledge graph…"
            rows={3}
            style={{
              width: '100%', background: '#0f1117',
              border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4,
              padding: '9px 44px 9px 12px', color: '#c8d3e0', fontSize: 12,
              fontFamily: "'DM Sans', sans-serif", lineHeight: 1.5, resize: 'none',
              outline: 'none', boxSizing: 'border-box',
              transition: 'border-color 0.15s',
            }}
            onFocus={e => { e.target.style.borderColor = 'rgba(29,233,182,0.4)'; }}
            onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; }}
          />
          <button onClick={handleSubmit}
            disabled={!input.trim()}
            style={{
              position: 'absolute', right: 8, bottom: 8,
              width: 28, height: 28, borderRadius: 4,
              background: input.trim() ? TEAL_Q : 'rgba(255,255,255,0.06)',
              border: 'none', cursor: input.trim() ? 'pointer' : 'default',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.2s',
            }}>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
              <path d="M2 8h12M10 4l4 4-4 4" stroke={input.trim() ? '#0f1117' : '#3d4555'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>

        {/* Model selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, position: 'relative' }}>
          <span style={{ fontSize: 10, color: '#3d4555', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Model</span>
          <button onClick={() => setModelOpen(x => !x)}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '3px 10px', borderRadius: 3, cursor: 'pointer',
              background: 'rgba(124,131,232,0.12)', border: '1px solid rgba(124,131,232,0.25)',
              color: INDIGO_Q, fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
            }}>
            {selectedModel}
            <svg width="8" height="8" viewBox="0 0 10 10" fill="none">
              <path d="M2 4l3 3 3-3" stroke={INDIGO_Q} strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
          {modelOpen && (
            <div style={{
              position: 'absolute', top: 28, left: 56, zIndex: 300,
              background: '#1e2334', border: '1px solid rgba(124,131,232,0.25)',
              borderRadius: 4, overflow: 'hidden', minWidth: 140,
              boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
            }}>
              {MODELS.map(m => (
                <button key={m} onClick={() => { onModelChange(m); setModelOpen(false); }}
                  style={{
                    display: 'block', width: '100%', padding: '8px 14px', textAlign: 'left',
                    background: m === selectedModel ? 'rgba(124,131,232,0.15)' : 'transparent',
                    border: 'none', cursor: 'pointer', color: m === selectedModel ? INDIGO_Q : '#6a7585',
                    fontSize: 12, fontFamily: "'JetBrains Mono', monospace", transition: 'background 0.1s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(124,131,232,0.1)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = m === selectedModel ? 'rgba(124,131,232,0.15)' : 'transparent'; }}>
                  {m}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Feed ─────────────────────────────────────────────────────────── */}
      <div ref={feedRef} style={{ flex: 1, overflowY: 'auto', padding: '16px 14px' }}
        onClick={() => setModelOpen(false)}>

        {qaHistory.length === 0 && (
          <div>
            <div style={{ fontSize: 11, color: '#2a3040', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
              Suggested questions
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {suggestions.map((s, i) => (
                <button key={i} onClick={() => onSubmitQuestion(s)}
                  style={{
                    padding: '9px 12px', background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.08)', borderRadius: 4,
                    color: '#6a7585', fontSize: 12, textAlign: 'left', cursor: 'pointer',
                    lineHeight: 1.4, transition: 'all 0.15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(29,233,182,0.3)'; e.currentTarget.style.color = '#8892a4'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = '#6a7585'; }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {qaHistory.map(turn => (
          <QATurn key={turn.id} turn={turn} onChipHover={onChipHover} />
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { QAPanel });
