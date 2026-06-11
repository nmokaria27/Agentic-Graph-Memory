// makg-sidebar.jsx — Left sidebar: mode switch, filters (Explore), history (QA)

const { useState: useSt } = React;
const { ENTITY_TYPES, RELATION_TYPES } = window.MAKG_DATA;

const TEAL_S  = '#1de9b6';
const INDIGO_S = '#7c83e8';

function LeftSidebar({
  mode, onModeChange,
  filters, onFiltersChange,
  qaHistory, selectedQaId, onQaSelect, onQaHover,
  serverConnected,
}) {
  const [relExpanded, setRelExpanded] = useSt(false);

  const sidebarStyle = {
    width: 240, flexShrink: 0, height: '100%',
    background: '#12151f',
    borderRight: '1px solid rgba(255,255,255,0.07)',
    display: 'flex', flexDirection: 'column',
    overflow: 'hidden',
  };

  const sectionLabel = {
    fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase',
    color: '#3d4555', marginBottom: 8, fontWeight: 600,
  };

  return (
    <div style={sidebarStyle}>
      {/* ── Logo ─────────────────────────────────────────────────────────── */}
      <div style={{ padding: '20px 18px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: '-0.02em', color: '#e2e8f0', lineHeight: 1 }}>
          Ma<span style={{ color: TEAL_S }}>K</span>G
        </div>
        <div style={{ fontSize: 10, color: '#3d4555', marginTop: 4, fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.04em' }}>
          MULTI-AGENT KG SYSTEM
        </div>
      </div>

      {/* ── Mode switcher ────────────────────────────────────────────────── */}
      <div style={{ padding: '12px 14px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ display: 'flex', gap: 0, border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, overflow: 'hidden' }}>
          {['Explore', 'QA'].map(m => {
            const active = mode === m.toLowerCase();
            return (
              <button key={m} onClick={() => onModeChange(m.toLowerCase())}
                style={{
                  flex: 1, padding: '7px 0', fontSize: 12, fontWeight: 500,
                  border: 'none', cursor: 'pointer', transition: 'all 0.2s',
                  background: active ? (m === 'Explore' ? 'rgba(29,233,182,0.12)' : 'rgba(124,131,232,0.15)') : 'transparent',
                  color: active ? (m === 'Explore' ? TEAL_S : INDIGO_S) : '#4a5568',
                  borderRight: m === 'Explore' ? '1px solid rgba(255,255,255,0.08)' : 'none',
                }}>
                {m}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Scrollable body ──────────────────────────────────────────────── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '14px 14px 0' }}>

        {mode === 'explore' && (
          <>
            {/* Search */}
            <div style={{ marginBottom: 16 }}>
              <div style={sectionLabel}>Search entity</div>
              <input
                type="text"
                placeholder="Filter by name…"
                value={filters.searchQuery}
                onChange={e => onFiltersChange(f => ({ ...f, searchQuery: e.target.value }))}
                style={{
                  width: '100%', background: '#0f1117', border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 4, padding: '7px 10px', color: '#c8d3e0', fontSize: 12,
                  fontFamily: "'JetBrains Mono', monospace", outline: 'none',
                }}
                onFocus={e => { e.target.style.borderColor = 'rgba(29,233,182,0.4)'; }}
                onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.1)'; }}
              />
            </div>

            {/* Confidence threshold */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ ...sectionLabel, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Confidence</span>
                <span style={{ color: TEAL_S, fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}>
                  ≥{filters.confidenceThreshold.toFixed(2)}
                </span>
              </div>
              <input type="range" min="0" max="1" step="0.05"
                value={filters.confidenceThreshold}
                onChange={e => onFiltersChange(f => ({ ...f, confidenceThreshold: +e.target.value }))}
                style={{ width: '100%', accentColor: TEAL_S, cursor: 'pointer' }}
              />
            </div>

            {/* Entity types */}
            <div style={{ marginBottom: 16 }}>
              <div style={sectionLabel}>Entity types</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {Object.entries(ENTITY_TYPES).map(([type, { color, label }]) => {
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
                style={{ ...sectionLabel, display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'none', border: 'none', cursor: 'pointer', width: '100%', padding: 0, marginBottom: 8 }}>
                <span style={sectionLabel}>Relation types</span>
                <span style={{ color: '#3d4555', fontSize: 10 }}>{relExpanded ? '▲' : '▼'}</span>
              </button>
              {relExpanded && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {RELATION_TYPES.map(rel => {
                    const active = filters.relationTypes[rel] !== false;
                    return (
                      <button key={rel}
                        onClick={() => onFiltersChange(f => ({ ...f, relationTypes: { ...f.relationTypes, [rel]: !active } }))}
                        style={{
                          display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px',
                          background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left',
                        }}>
                        <span style={{ width: 10, height: 1, background: active ? INDIGO_S : '#2a3040', display: 'inline-block', flexShrink: 0 }} />
                        <span style={{ fontSize: 10, color: active ? '#6a7585' : '#2a3040', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.02em' }}>{rel}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </>
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
                    }}
                    onMouseOver={e => { if (!isActive) { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'; } }}
                    onMouseOut={e => { if (!isActive) { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'; } }}>
                    <span style={{ fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", flexShrink: 0, marginTop: 1 }}>Q{i+1}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 11, color: isActive ? '#c8d3e0' : '#6a7585', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', lineHeight: 1.4 }}>
                        {turn.question}
                      </div>
                      {!isLoading && turn.nodeCount > 0 && (
                        <div style={{ marginTop: 4 }}>
                          <span style={{ fontSize: 9, fontFamily: "'JetBrains Mono', monospace", color: TEAL_S, background: 'rgba(29,233,182,0.12)', padding: '1px 5px', borderRadius: 2 }}>
                            {turn.nodeCount} nodes
                          </span>
                        </div>
                      )}
                      {isLoading && (
                        <div style={{ marginTop: 4, display: 'flex', gap: 3 }}>
                          {[0,1,2].map(j => (
                            <span key={j} style={{ width: 4, height: 4, borderRadius: '50%', background: TEAL_S, display: 'inline-block', animation: `dotBounce 1.2s ${j*0.2}s ease-in-out infinite` }} />
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

      {/* ── Footer: server status ────────────────────────────────────────── */}
      <div style={{
        padding: '12px 16px', borderTop: '1px solid rgba(255,255,255,0.06)',
        display: 'flex', alignItems: 'center', gap: 8,
      }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
          background: serverConnected ? '#22c55e' : '#f59e0b',
          boxShadow: serverConnected ? '0 0 0 2px rgba(34,197,94,0.25)' : '0 0 0 2px rgba(245,158,11,0.25)',
          animation: 'statusPulse 2s ease-in-out infinite',
          display: 'inline-block',
        }} />
        <span style={{ fontSize: 10, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace", letterSpacing: '0.04em' }}>
          {serverConnected ? 'Connected to lab server' : 'Using mock data'}
        </span>
      </div>
    </div>
  );
}

Object.assign(window, { LeftSidebar });
