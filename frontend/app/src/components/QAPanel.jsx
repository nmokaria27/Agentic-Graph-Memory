import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchModels, fetchKGStats } from '../api.js';

const TEAL   = '#1de9b6';
const INDIGO = '#7c83e8';
const DEFAULT_MODELS = ['gemma4:31b'];

function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong style="color:#c8d3e0">$1</strong>')
    .replace(/\n\n/g, '</p><p style="margin:0 0 10px">')
    .replace(/\n/g, '<br>');
  return `<p style="margin:0 0 10px">${html}</p>`;
}

function ChipRow({ turn, onChipHover, entities, entityTypes }) {
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    if (turn.status !== 'done') { setVisibleCount(0); return; }
    const nodeIds = turn.referencedNodes || [];
    const edgeIds = turn.referencedEdges || [];
    const total = nodeIds.length + edgeIds.length;
    let i = 0;
    const timer = setInterval(() => {
      i++;
      setVisibleCount(i);
      if (i >= total) clearInterval(timer);
    }, 55);
    return () => clearInterval(timer);
  }, [turn.status]); // eslint-disable-line

  if (turn.status !== 'done') return null;
  const nodeIds = turn.referencedNodes || [];
  if (nodeIds.length === 0) return null;

  const entityMap = {};
  entities.forEach(e => { entityMap[e.id] = e; });

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#3d4555', marginBottom: 8 }}>
        Referenced nodes
      </div>
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', paddingBottom: 6 }}>
        {nodeIds.map((nid, idx) => {
          const entity = entityMap[nid];
          const color  = (entityTypes[entity?.type] || {}).color || '#4a5568';
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
      </div>
    </div>
  );
}

function DomainPanel({ domainResponses }) {
  const [expanded, setExpanded] = useState(null);
  if (!domainResponses || domainResponses.length === 0) return null;

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#3d4555', marginBottom: 8 }}>
        Domain responses ({domainResponses.length})
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {domainResponses.map((dr, i) => {
          const conf = dr.confidence || 0;
          const confColor = conf >= 0.7 ? '#22c55e' : conf >= 0.4 ? '#f59e0b' : '#ef4444';
          const isOpen = expanded === i;
          return (
            <div key={i} style={{ background: '#131825', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
              <button
                onClick={() => setExpanded(isOpen ? null : i)}
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '7px 10px',
                  background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left',
                }}>
                <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: INDIGO, background: 'rgba(124,131,232,0.1)', padding: '1px 6px', borderRadius: 2, flexShrink: 0 }}>
                  {dr.domain_id || 'domain'}
                </span>
                <span style={{ flex: 1 }} />
                <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: confColor, background: `${confColor}1a`, padding: '1px 6px', borderRadius: 2 }}>
                  {conf.toFixed(2)}
                </span>
                <span style={{ fontSize: 9, color: '#3d4555', marginLeft: 4 }}>{isOpen ? '▲' : '▼'}</span>
              </button>
              {isOpen && (
                <div style={{ padding: '0 10px 10px' }}>
                  {dr.topics_used && dr.topics_used.length > 0 && (
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 7 }}>
                      {dr.topics_used.map((t, j) => (
                        <span key={j} style={{ fontSize: 9, color: '#4a5568', background: 'rgba(255,255,255,0.04)', padding: '1px 5px', borderRadius: 2, fontFamily: "'JetBrains Mono', monospace" }}>
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                  {dr.evidence && dr.evidence.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      {dr.evidence.map((ev, j) => (
                        <div key={j} style={{ fontSize: 9, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.4, wordBreak: 'break-all' }}>
                          {typeof ev === 'string' ? ev : `${ev.subject} -[${ev.relation}]-> ${ev.object}`}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function QATurn({ turn, onChipHover, entities, entityTypes }) {
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
            {/* Model badge + timestamp */}
            <div style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: INDIGO, background: 'rgba(124,131,232,0.12)', padding: '2px 7px', borderRadius: 2 }}>
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
                  <span key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: TEAL, display: 'inline-block', animation: `dotBounce 1.2s ${i * 0.2}s ease-in-out infinite` }} />
                ))}
              </div>
            )}

            {turn.status === 'streaming' && (
              <div style={{ whiteSpace: 'pre-wrap', color: '#8892a4', lineHeight: 1.65 }}>
                {turn.streamedAnswer}
                <span style={{ animation: 'cursorBlink 0.75s step-end infinite', color: TEAL, marginLeft: 1 }}>&#x2588;</span>
              </div>
            )}

            {turn.status === 'done' && (
              <div
                style={{ color: '#8892a4', lineHeight: 1.65 }}
                dangerouslySetInnerHTML={{ __html: renderMarkdown(turn.answer) }}
              />
            )}

            {turn.status === 'error' && (
              <div style={{
                padding: '8px 10px', borderRadius: 3,
                background: 'rgba(239,68,68,0.08)',
                border: '1px solid rgba(239,68,68,0.25)',
                color: '#ffb4b4', fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.5,
              }}>
                <div style={{ fontSize: 9, letterSpacing: '0.08em', marginBottom: 4, color: '#ef4444' }}>
                  QA REQUEST FAILED
                </div>
                {turn.error || 'Unknown error'}
              </div>
            )}
          </div>

          <ChipRow turn={turn} onChipHover={onChipHover} entities={entities} entityTypes={entityTypes} />
          <DomainPanel domainResponses={turn.domainResponses} />
        </div>
      </div>
    </div>
  );
}

export default function QAPanel({
  qaHistory, onSubmitQuestion, onChipHover,
  selectedModel, onModelChange, activeQaId,
  entities, entityTypes,
  onClearChats, qaReady = true,
}) {
  const [input, setInput]       = useState('');
  const [modelOpen, setModelOpen] = useState(false);
  const [models, setModels]     = useState(DEFAULT_MODELS);
  const [activeModel, setActiveModel] = useState(null);  // model the server is actually using
  const [kgStats, setKgStats]   = useState(null);
  const feedRef  = useRef(null);

  useEffect(() => {
    let cancelled = false;
    fetchModels()
      .then(d => {
        if (cancelled) return;
        if (Array.isArray(d?.models) && d.models.length) setModels(d.models);
        if (d?.default) {
          setActiveModel(d.default);
          if (onModelChange) onModelChange(d.default);
        }
      })
      .catch(() => { /* keep DEFAULT_MODELS */ });
    fetchKGStats()
      .then(d => { if (!cancelled) setKgStats(d?.kg ?? null); })
      .catch(() => { /* ignore */ });
    return () => { cancelled = true; };
  }, []); // eslint-disable-line

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [qaHistory.length, activeQaId]);

  const handleSubmit = useCallback(() => {
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

  return (
    <div style={{
      width: 380, flexShrink: 0, height: '100%',
      background: '#181c26', borderLeft: '1px solid rgba(255,255,255,0.07)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Input + model selector */}
      <div style={{ padding: '14px 14px 12px', borderBottom: '1px solid rgba(255,255,255,0.07)', flexShrink: 0 }}>
        <div style={{ position: 'relative', marginBottom: 10 }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question grounded in the knowledge graph..."
            rows={3}
            style={{
              width: '100%', background: '#0f1117',
              border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4,
              padding: '9px 44px 9px 12px', color: '#c8d3e0', fontSize: 12,
              fontFamily: "'DM Sans', sans-serif", lineHeight: 1.5, resize: 'none',
              outline: 'none', boxSizing: 'border-box', transition: 'border-color 0.15s',
            }}
            onFocus={e => { e.target.style.borderColor = 'rgba(29,233,182,0.4)'; }}
            onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; }}
          />
          <button onClick={handleSubmit}
            disabled={!input.trim() || !qaReady}
            style={{
              position: 'absolute', right: 8, bottom: 8,
              width: 28, height: 28, borderRadius: 4,
              background: (input.trim() && qaReady) ? TEAL : 'rgba(255,255,255,0.06)',
              border: 'none', cursor: (input.trim() && qaReady) ? 'pointer' : 'default',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'background 0.2s',
            }}>
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none">
              <path d="M2 8h12M10 4l4 4-4 4" stroke={(input.trim() && qaReady) ? '#0f1117' : '#3d4555'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>

        {!qaReady && (
          <div style={{
            fontSize: 10, color: '#f59e0b', marginBottom: 8,
            fontFamily: "'JetBrains Mono', monospace",
            padding: '4px 8px', background: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.25)', borderRadius: 3,
          }}>
            QA system not ready. Start backend without --data-only, or wait for org chart to build.
          </div>
        )}

        {/* Model selector + clear chats */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, position: 'relative' }}>
          <span style={{ fontSize: 10, color: '#3d4555', letterSpacing: '0.08em', textTransform: 'uppercase' }}>Model</span>
          <button onClick={() => setModelOpen(x => !x)}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '3px 10px', borderRadius: 3, cursor: 'pointer',
              background: 'rgba(124,131,232,0.12)', border: '1px solid rgba(124,131,232,0.25)',
              color: INDIGO, fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
            }}>
            {selectedModel}
            <svg width="8" height="8" viewBox="0 0 10 10" fill="none">
              <path d="M2 4l3 3 3-3" stroke={INDIGO} strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
          {activeModel && (
            <span title={`Backend currently running ${activeModel}`}
              style={{
                fontSize: 9, padding: '2px 6px', borderRadius: 2,
                background: 'rgba(29,233,182,0.1)',
                border: '1px solid rgba(29,233,182,0.25)',
                color: TEAL, fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: '0.05em',
              }}>
              IN USE: {activeModel}
            </span>
          )}
          <div style={{ flex: 1 }} />
          {qaHistory.length > 0 && onClearChats && (
            <button
              onClick={onClearChats}
              title="Clear all chats"
              style={{
                padding: '3px 9px', borderRadius: 3, cursor: 'pointer',
                background: 'rgba(239,68,68,0.08)',
                border: '1px solid rgba(239,68,68,0.25)',
                color: '#ffb4b4', fontSize: 10,
                fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.05em',
              }}
              onMouseEnter={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.15)'; }}
              onMouseLeave={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.08)'; }}
            >
              CLEAR
            </button>
          )}
          {modelOpen && (
            <div style={{
              position: 'absolute', top: 28, left: 56, zIndex: 300,
              background: '#1e2334', border: '1px solid rgba(124,131,232,0.25)',
              borderRadius: 4, overflow: 'hidden', minWidth: 140,
              boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
            }}>
              {models.map(m => (
                <button key={m} onClick={() => { onModelChange(m); setModelOpen(false); }}
                  style={{
                    display: 'block', width: '100%', padding: '8px 14px', textAlign: 'left',
                    background: m === selectedModel ? 'rgba(124,131,232,0.15)' : 'transparent',
                    border: 'none', cursor: 'pointer', color: m === selectedModel ? INDIGO : '#6a7585',
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

      {/* Feed */}
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
          <QATurn key={turn.id} turn={turn} onChipHover={onChipHover} entities={entities} entityTypes={entityTypes} />
        ))}
      </div>

      {kgStats && (
        <div style={{
          padding: '7px 14px', borderTop: '1px solid rgba(255,255,255,0.07)',
          fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace",
          display: 'flex', justifyContent: 'space-between', flexShrink: 0,
        }}>
          <span>entities: {kgStats.num_entities ?? '?'}</span>
          <span>triples: {kgStats.num_triples ?? '?'}</span>
        </div>
      )}
    </div>
  );
}
