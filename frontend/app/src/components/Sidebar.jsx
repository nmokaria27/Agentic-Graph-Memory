import { useState, useEffect } from 'react';
import { fetchEvalMetrics } from '../api.js';

const TEAL   = '#1de9b6';
const INDIGO = '#7c83e8';

const AMBER = '#f59e0b';

const selectStyle = {
  width: '100%', background: '#0f1117', border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 4, padding: '6px 8px', color: '#c8d3e0', fontSize: 11,
  fontFamily: "'JetBrains Mono', monospace", outline: 'none', cursor: 'pointer',
};

const sourceToKey = (s) => s.kind === 'live' ? 'live' : `eval|${s.dir}|${s.strategy}`;
const keyToSource = (key) => {
  if (key === 'live') return { kind: 'live' };
  const [, dir, strategy] = key.split('|');
  return { kind: 'eval', dir, strategy, doc: 'all' };
};

export default function Sidebar({
  mode, onModeChange,
  filters, onFiltersChange,
  entityTypes, relationTypes,
  qaHistory, selectedQaId, onQaSelect, onQaHover,
  serverConnected,
  kgStats,
  graphSource, onGraphSourceChange, evalRuns,
  onSearchEnter,
}) {
  const [relExpanded, setRelExpanded] = useState(false);
  // Metrics are cached keyed by run so switching runs shows nothing stale
  // (derived below) without a state reset inside the effect.
  const [metricsCache, setMetricsCache] = useState(null); // { key, summary }

  const isEval = graphSource?.kind === 'eval';
  const currentRun = isEval
    ? (evalRuns || []).find(r => r.dir === graphSource.dir && r.strategy === graphSource.strategy)
    : null;
  const metricsKey = isEval ? `${graphSource.dir}|${graphSource.strategy}` : null;

  useEffect(() => {
    if (!metricsKey) return undefined;
    let cancelled = false;
    const [dir, strategy] = metricsKey.split('|');
    fetchEvalMetrics(dir, strategy)
      .then(d => { if (!cancelled) setMetricsCache({ key: metricsKey, summary: d.summary }); })
      .catch(() => { /* metrics are optional */ });
    return () => { cancelled = true; };
  }, [metricsKey]);

  const runMetrics = metricsCache && metricsCache.key === metricsKey ? metricsCache.summary : null;

  const sectionLabel = {
    fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase',
    color: '#3d4555', marginBottom: 8, fontWeight: 600,
  };

  return (
    <div style={{
      width: 240, flexShrink: 0, height: '100%',
      background: '#12151f', borderRight: '1px solid rgba(255,255,255,0.07)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Logo */}
      <div style={{ padding: '20px 18px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.02em', color: '#e2e8f0', lineHeight: 1 }}>
          Ma<span style={{ color: TEAL }}>K</span>G
        </div>
        <div style={{ fontSize: 10, color: '#3d4555', marginTop: 4, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.04em' }}>
          MULTI-AGENT KG SYSTEM
        </div>
      </div>

      {/* Graph source */}
      <div style={{ padding: '12px 14px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ ...sectionLabel, marginBottom: 6 }}>Graph source</div>
        <select
          value={graphSource ? sourceToKey(graphSource) : 'live'}
          onChange={e => onGraphSourceChange && onGraphSourceChange(keyToSource(e.target.value))}
          style={selectStyle}>
          <option value="live">Live KG (governed)</option>
          {(evalRuns || []).map(r => (
            <option key={`${r.dir}|${r.strategy}`} value={`eval|${r.dir}|${r.strategy}`}>
              {r.label}
            </option>
          ))}
        </select>
        {isEval && currentRun && (
          <select
            value={graphSource.doc || 'all'}
            onChange={e => onGraphSourceChange && onGraphSourceChange({ ...graphSource, doc: e.target.value })}
            style={{ ...selectStyle, marginTop: 6 }}>
            <option value="all">All {currentRun.doc_count} docs (merged)</option>
            {currentRun.doc_indices.map(i => (
              <option key={i} value={String(i)}>doc {i}</option>
            ))}
          </select>
        )}
      </div>

      {/* Mode switcher */}
      <div style={{ padding: '12px 14px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ display: 'flex', gap: 0, border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, overflow: 'hidden' }}>
          {[
            { label: 'Explore', value: 'explore', color: TEAL,   bg: 'rgba(29,233,182,0.12)' },
            { label: 'QA',      value: 'qa',      color: INDIGO, bg: 'rgba(124,131,232,0.15)' },
            { label: 'Govern',  value: 'governance', color: AMBER, bg: 'rgba(245,158,11,0.12)' },
          ].map((m, i, arr) => {
            const active = mode === m.value;
            return (
              <button key={m.value} onClick={() => onModeChange(m.value)}
                style={{
                  flex: 1, padding: '7px 0', fontSize: 11, fontWeight: 500,
                  border: 'none', cursor: 'pointer', transition: 'all 0.2s',
                  background: active ? m.bg : 'transparent',
                  color: active ? m.color : '#4a5568',
                  borderRight: i < arr.length - 1 ? '1px solid rgba(255,255,255,0.08)' : 'none',
                }}>
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px 0' }}>

        {mode === 'explore' && (
          <>
            {/* Eval run metrics */}
            {isEval && runMetrics && (
              <div style={{ marginBottom: 16 }}>
                <div style={sectionLabel}>Run metrics</div>
                <div style={{
                  display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 10px',
                  fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
                  background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: 4, padding: '8px 10px', color: '#6a7585',
                }}>
                  <span>docs</span><span style={{ color: '#c8d3e0', textAlign: 'right' }}>{runMetrics.docs}</span>
                  <span>pred / gold ent</span><span style={{ color: '#c8d3e0', textAlign: 'right' }}>{runMetrics.pred_entities}/{runMetrics.gold_entities}</span>
                  <span>pred / gold tri</span><span style={{ color: '#c8d3e0', textAlign: 'right' }}>{runMetrics.pred_triples}/{runMetrics.gold_triples}</span>
                  <span>llm calls</span><span style={{ color: '#c8d3e0', textAlign: 'right' }}>{runMetrics.llm_calls}</span>
                  <span>empty calls</span><span style={{ color: runMetrics.empty_calls ? AMBER : '#c8d3e0', textAlign: 'right' }}>{runMetrics.empty_calls}</span>
                  <span>errors</span><span style={{ color: runMetrics.errors ? '#ef4444' : '#c8d3e0', textAlign: 'right' }}>{runMetrics.errors}</span>
                  <span>mean wall</span><span style={{ color: '#c8d3e0', textAlign: 'right' }}>{runMetrics.mean_wall_s}s</span>
                </div>
                {runMetrics.model && (
                  <div style={{ fontSize: 9, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={runMetrics.model}>
                    model: {runMetrics.model}
                  </div>
                )}
              </div>
            )}

            {/* Search */}
            <div style={{ marginBottom: 16 }}>
              <div style={sectionLabel}>Search entity {'—'} Enter to fly to</div>
              <input
                id="entity-search"
                type="text"
                placeholder="Filter by name..."
                value={filters.searchQuery}
                onChange={e => onFiltersChange(f => ({ ...f, searchQuery: e.target.value }))}
                onKeyDown={e => { if (e.key === 'Enter' && onSearchEnter) onSearchEnter(); }}
                style={{
                  width: '100%', background: '#0f1117', border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 4, padding: '7px 10px', color: '#c8d3e0', fontSize: 12,
                  fontFamily: "'JetBrains Mono', monospace", outline: 'none', boxSizing: 'border-box',
                }}
                onFocus={e => { e.target.style.borderColor = 'rgba(29,233,182,0.4)'; }}
                onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; }}
              />
            </div>

            {/* Confidence threshold */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ ...sectionLabel, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Confidence</span>
                <span style={{ color: TEAL, fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}>
                  &ge;{filters.confidenceThreshold.toFixed(2)}
                </span>
              </div>
              <input type="range" min="0" max="1" step="0.05"
                value={filters.confidenceThreshold}
                onChange={e => onFiltersChange(f => ({ ...f, confidenceThreshold: +e.target.value }))}
                style={{ width: '100%', accentColor: TEAL, cursor: 'pointer' }}
              />
            </div>

            {/* Entity types */}
            <div style={{ marginBottom: 16 }}>
              <div style={sectionLabel}>Entity types</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {Object.entries(entityTypes).map(([type, { color }]) => {
                  const active = filters.entityTypes[type] !== false;
                  return (
                    <button key={type}
                      onClick={() => onFiltersChange(f => ({ ...f, entityTypes: { ...f.entityTypes, [type]: !active } }))}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 8, padding: '5px 8px',
                        background: active ? `${color}18` : 'transparent',
                        border: `1px solid ${active ? color + '55' : 'rgba(255,255,255,0.06)'}`,
                        borderRadius: 3, cursor: 'pointer', transition: 'all 0.15s',
                        textAlign: 'left', width: '100%',
                      }}>
                      <span style={{ width: 7, height: 7, borderRadius: '50%', background: active ? color : '#2a3040', flexShrink: 0, display: 'inline-block', transition: 'background 0.15s' }} />
                      <span style={{ fontSize: 11, color: active ? '#c8d3e0' : '#3d4555', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.02em' }}>{type}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Relation types */}
            <div style={{ marginBottom: 14 }}>
              <button onClick={() => setRelExpanded(x => !x)}
                style={{ ...sectionLabel, display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'none', border: 'none', cursor: 'pointer', width: '100%', padding: 0 }}>
                <span>Relation types</span>
                <span style={{ color: '#3d4555', fontSize: 10 }}>{relExpanded ? '\u25B2' : '\u25BC'}</span>
              </button>
              {relExpanded && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {relationTypes.map(rel => {
                    const active = filters.relationTypes[rel] !== false;
                    return (
                      <button key={rel}
                        onClick={() => onFiltersChange(f => ({ ...f, relationTypes: { ...f.relationTypes, [rel]: !active } }))}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px',
                          background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left',
                        }}>
                        <span style={{ width: 10, height: 1, background: active ? INDIGO : '#2a3040', display: 'inline-block', flexShrink: 0 }} />
                        <span style={{ fontSize: 10, color: active ? '#6a7585' : '#2a3040', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.02em' }}>{rel}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </>
        )}

        {mode === 'governance' && (
          <div style={{ fontSize: 11, color: '#4a5568', lineHeight: 1.6, padding: '8px 2px' }}>
            <div style={sectionLabel}>Governance mode</div>
            Inspect governed domains, the decision audit log, and the pending-review
            queue in the panel on the right.
            <div style={{ marginTop: 10, fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", lineHeight: 1.7 }}>
              · click a domain to highlight its subgraph<br />
              · click an audit entry to locate its triple<br />
              · approve / reject escalated triples
            </div>
          </div>
        )}

        {mode === 'qa' && (
          <>
            <div style={sectionLabel}>Session history</div>
            {qaHistory.length === 0 && (
              <div style={{ fontSize: 12, color: '#2a3040', textAlign: 'center', padding: '24px 0', fontStyle: 'italic' }}>
                No questions yet
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {qaHistory.map((turn, i) => {
                const isActive = selectedQaId === turn.id;
                const isLoading = turn.status === 'loading' || turn.status === 'streaming';
                return (
                  <button key={turn.id}
                    onClick={() => onQaSelect(turn.id)}
                    onMouseEnter={() => onQaHover(turn.id)}
                    onMouseLeave={() => onQaHover(null)}
                    style={{
                      display: 'flex', alignItems: 'flex-start', gap: 8, padding: '8px 10px',
                      background: isActive ? 'rgba(29,233,182,0.08)' : 'rgba(255,255,255,0.02)',
                      border: `1px solid ${isActive ? 'rgba(29,233,182,0.3)' : 'rgba(255,255,255,0.06)'}`,
                      borderRadius: 4, cursor: 'pointer', textAlign: 'left', transition: 'all 0.15s',
                    }}>
                    <span style={{ fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", flexShrink: 0, marginTop: 1 }}>Q{i+1}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 11, color: isActive ? '#c8d3e0' : '#6a7585', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', lineHeight: 1.4 }}>
                        {turn.question}
                      </div>
                      {!isLoading && turn.nodeCount > 0 && (
                        <div style={{ marginTop: 4 }}>
                          <span style={{ fontSize: 9, fontFamily: "'JetBrains Mono', monospace", color: TEAL, background: 'rgba(29,233,182,0.12)', padding: '1px 5px', borderRadius: 2 }}>
                            {turn.nodeCount} nodes
                          </span>
                        </div>
                      )}
                      {isLoading && (
                        <div style={{ marginTop: 4, display: 'flex', gap: 3 }}>
                          {[0,1,2].map(j => (
                            <span key={j} style={{ width: 4, height: 4, borderRadius: '50%', background: TEAL, display: 'inline-block', animation: `dotBounce 1.2s ${j*0.2}s ease-in-out infinite` }} />
                          ))}
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Footer: server status */}
      <div style={{ padding: '12px 16px', borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
          background: serverConnected ? '#22c55e' : '#f59e0b',
          boxShadow: serverConnected ? '0 0 0 2px rgba(34,197,94,0.25)' : '0 0 0 2px rgba(245,158,11,0.25)',
          animation: 'statusPulse 2s ease-in-out infinite',
          display: 'inline-block',
        }} />
        <span style={{ fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.04em' }}>
          {serverConnected ? 'Connected to server' : 'Using mock data'}
        </span>
        {serverConnected && kgStats && (
          <span style={{ marginLeft: 'auto', fontSize: 9, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace" }}>
            {kgStats.entities ?? 0}e · {kgStats.triples ?? 0}t
          </span>
        )}
      </div>
    </div>
  );
}
