import { useState, useEffect, useCallback } from 'react';
import { fetchGovernanceAudit, fetchGovernancePending, submitGovernanceReview } from '../api.js';

const TEAL   = '#1de9b6';
const INDIGO = '#7c83e8';

const ACTION_COLORS = {
  approve:      '#22c55e',
  auto_approve: '#4ade80',
  reject:       '#ef4444',
  revise:       '#f59e0b',
  escalate:     '#a78bfa',
};

const MONO = "'JetBrains Mono', monospace";

function ActionBadge({ action }) {
  const color = ACTION_COLORS[action] || '#6a7585';
  return (
    <span style={{
      fontSize: 9, padding: '1px 6px', borderRadius: 2, flexShrink: 0,
      color, background: `${color}1a`, border: `1px solid ${color}40`,
      fontFamily: MONO, letterSpacing: '0.04em',
    }}>{action}</span>
  );
}

function TripleLine({ subject, relation, object }) {
  return (
    <div style={{ fontSize: 10, fontFamily: MONO, lineHeight: 1.5, wordBreak: 'break-word' }}>
      <span style={{ color: '#c8d3e0' }}>{subject}</span>
      <span style={{ color: INDIGO }}> -[{relation}]-&gt; </span>
      <span style={{ color: '#c8d3e0' }}>{object}</span>
    </div>
  );
}

const sectionLabel = {
  fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase',
  color: '#3d4555', marginBottom: 8, fontWeight: 600,
};

export default function GovernancePanel({
  orgChart,
  selectedDomainId, onDomainSelect,
  onAuditSelect,
  onKGChanged,
}) {
  const [tab, setTab] = useState('domains'); // 'domains' | 'pending' | 'audit'
  const [pending, setPending]   = useState([]);
  const [audit, setAudit]       = useState({ entries: [], total: 0, stats: null });
  const [actionFilter, setActionFilter] = useState('');
  const [busyIndex, setBusyIndex] = useState(null);
  const [error, setError] = useState('');
  const [selectedAuditIdx, setSelectedAuditIdx] = useState(null);

  const loadPending = useCallback(() => {
    fetchGovernancePending()
      .then(d => setPending(d.pending || []))
      .catch(e => setError(e.message));
  }, []);

  const loadAudit = useCallback(() => {
    fetchGovernanceAudit({ limit: 100, action: actionFilter || undefined })
      .then(d => setAudit({ entries: d.entries || [], total: d.total || 0, stats: d.stats || null }))
      .catch(e => setError(e.message));
  }, [actionFilter]);

  useEffect(() => { loadPending(); }, [loadPending]);
  useEffect(() => { loadAudit(); }, [loadAudit]);

  const handleReview = async (index, action) => {
    setBusyIndex(index);
    setError('');
    try {
      await submitGovernanceReview(index, action);
      loadPending();
      loadAudit();
      if (action === 'approve' && onKGChanged) onKGChanged();
    } catch (e) {
      setError(e.message || 'review failed');
    } finally {
      setBusyIndex(null);
    }
  };

  const domains = orgChart?.domains || [];
  const decisionCounts = audit.stats?.decision_counts || {};

  return (
    <div style={{
      width: 380, flexShrink: 0, height: '100%',
      background: '#181c26', borderLeft: '1px solid rgba(255,255,255,0.07)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.07)', flexShrink: 0 }}>
        {[
          { id: 'domains', label: `Domains (${domains.length})` },
          { id: 'pending', label: `Pending (${pending.length})` },
          { id: 'audit',   label: `Audit (${audit.total})` },
        ].map(t => {
          const active = tab === t.id;
          return (
            <button key={t.id} onClick={() => setTab(t.id)}
              style={{
                flex: 1, padding: '11px 0', fontSize: 11, fontWeight: 500,
                background: active ? 'rgba(124,131,232,0.1)' : 'transparent',
                border: 'none', borderBottom: `2px solid ${active ? INDIGO : 'transparent'}`,
                color: active ? INDIGO : '#4a5568', cursor: 'pointer', transition: 'all 0.15s',
                fontFamily: MONO, letterSpacing: '0.03em',
              }}>
              {t.label}
            </button>
          );
        })}
      </div>

      {error && (
        <div style={{
          margin: '10px 14px 0', padding: '6px 10px', borderRadius: 3, fontSize: 10,
          background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
          color: '#ffb4b4', fontFamily: MONO,
        }}>{error}</div>
      )}

      <div style={{ flex: 1, overflowY: 'auto', padding: '14px' }}>

        {/* ── Domains tab ── */}
        {tab === 'domains' && (
          <>
            <div style={sectionLabel}>Governed domains — click to highlight subgraph</div>
            {domains.length === 0 && (
              <div style={{ fontSize: 12, color: '#2a3040', textAlign: 'center', padding: '24px 0', fontStyle: 'italic' }}>
                No org chart loaded
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {domains.map(d => {
                const active = selectedDomainId === d.domain_id;
                return (
                  <button key={d.domain_id}
                    onClick={() => onDomainSelect(active ? null : d.domain_id)}
                    style={{
                      textAlign: 'left', padding: '10px 12px', borderRadius: 4, cursor: 'pointer',
                      background: active ? 'rgba(29,233,182,0.08)' : 'rgba(255,255,255,0.02)',
                      border: `1px solid ${active ? 'rgba(29,233,182,0.35)' : 'rgba(255,255,255,0.06)'}`,
                      transition: 'all 0.15s',
                    }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: active ? TEAL : '#c8d3e0' }}>{d.label}</span>
                      <span style={{ fontSize: 9, fontFamily: MONO, color: '#6a7585', background: 'rgba(255,255,255,0.04)', padding: '1px 6px', borderRadius: 2 }}>
                        {d.entity_count} entities
                      </span>
                    </div>
                    {d.owner_label && (
                      <div style={{ fontSize: 10, color: '#5a6375', marginBottom: 4, fontFamily: MONO }}>
                        owner: {d.owner_label}
                      </div>
                    )}
                    {d.description && (
                      <div style={{ fontSize: 10, color: '#4a5568', lineHeight: 1.4, marginBottom: 5 }}>
                        {d.description.length > 110 ? d.description.slice(0, 108) + '…' : d.description}
                      </div>
                    )}
                    {(d.topics || []).length > 0 && (
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {d.topics.slice(0, 5).map(t => (
                          <span key={t.topic_id} style={{ fontSize: 9, color: INDIGO, background: 'rgba(124,131,232,0.1)', padding: '1px 5px', borderRadius: 2, fontFamily: MONO }}>
                            {t.label}
                          </span>
                        ))}
                        {d.topics.length > 5 && <span style={{ fontSize: 9, color: '#3d4555' }}>+{d.topics.length - 5}</span>}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </>
        )}

        {/* ── Pending tab ── */}
        {tab === 'pending' && (
          <>
            <div style={sectionLabel}>Pending review queue</div>
            {pending.length === 0 && (
              <div style={{ fontSize: 12, color: '#2a3040', textAlign: 'center', padding: '24px 0', fontStyle: 'italic' }}>
                No triples awaiting review.
                <div style={{ fontSize: 10, marginTop: 6 }}>
                  (Run the pipeline with strict or triage governance to queue escalations.)
                </div>
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {pending.map(p => (
                <div key={p.index} style={{
                  padding: '10px 12px', borderRadius: 4,
                  background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(167,139,250,0.2)',
                }}>
                  <TripleLine subject={p.subject} relation={p.relation} object={p.object} />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
                    {p.confidence != null && (
                      <span style={{ fontSize: 9, color: '#6a7585', fontFamily: MONO }}>
                        conf {Number(p.confidence).toFixed(2)}
                      </span>
                    )}
                    <div style={{ flex: 1 }} />
                    <button
                      disabled={busyIndex === p.index}
                      onClick={() => handleReview(p.index, 'approve')}
                      style={{
                        padding: '3px 12px', borderRadius: 3, cursor: 'pointer', fontSize: 10,
                        background: 'rgba(34,197,94,0.12)', border: '1px solid rgba(34,197,94,0.35)',
                        color: '#4ade80', fontFamily: MONO,
                        opacity: busyIndex === p.index ? 0.5 : 1,
                      }}>
                      APPROVE
                    </button>
                    <button
                      disabled={busyIndex === p.index}
                      onClick={() => handleReview(p.index, 'reject')}
                      style={{
                        padding: '3px 12px', borderRadius: 3, cursor: 'pointer', fontSize: 10,
                        background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                        color: '#ffb4b4', fontFamily: MONO,
                        opacity: busyIndex === p.index ? 0.5 : 1,
                      }}>
                      REJECT
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* ── Audit tab ── */}
        {tab === 'audit' && (
          <>
            {/* Decision count summary */}
            {Object.keys(decisionCounts).length > 0 && (
              <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginBottom: 12 }}>
                {Object.entries(decisionCounts).map(([a, n]) => (
                  <span key={a} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                    <ActionBadge action={a} />
                    <span style={{ fontSize: 10, color: '#6a7585', fontFamily: MONO }}>{n}</span>
                  </span>
                ))}
              </div>
            )}

            {/* Action filter */}
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 12 }}>
              {['', 'approve', 'auto_approve', 'reject', 'revise', 'escalate'].map(a => {
                const active = actionFilter === a;
                return (
                  <button key={a || 'all'} onClick={() => setActionFilter(a)}
                    style={{
                      padding: '2px 8px', borderRadius: 3, cursor: 'pointer', fontSize: 9,
                      fontFamily: MONO, letterSpacing: '0.04em',
                      background: active ? 'rgba(124,131,232,0.15)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${active ? 'rgba(124,131,232,0.4)' : 'rgba(255,255,255,0.07)'}`,
                      color: active ? INDIGO : '#5a6375',
                    }}>
                    {a || 'all'}
                  </button>
                );
              })}
            </div>

            <div style={sectionLabel}>
              Audit log — newest first ({audit.entries.length} of {audit.total})
            </div>
            {audit.entries.length === 0 && (
              <div style={{ fontSize: 12, color: '#2a3040', textAlign: 'center', padding: '24px 0', fontStyle: 'italic' }}>
                No audit entries
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {audit.entries.map((e, i) => {
                const t = e.triple || {};
                const active = selectedAuditIdx === i;
                return (
                  <button key={i}
                    onClick={() => {
                      const next = active ? null : i;
                      setSelectedAuditIdx(next);
                      if (onAuditSelect) onAuditSelect(next === null ? null : t);
                    }}
                    style={{
                      textAlign: 'left', padding: '8px 10px', borderRadius: 4, cursor: 'pointer',
                      background: active ? 'rgba(29,233,182,0.07)' : 'rgba(255,255,255,0.02)',
                      border: `1px solid ${active ? 'rgba(29,233,182,0.3)' : 'rgba(255,255,255,0.05)'}`,
                      transition: 'all 0.15s',
                    }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                      <ActionBadge action={e.action} />
                      {e.domain_id && (
                        <span style={{ fontSize: 9, color: '#5a6375', fontFamily: MONO, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {e.domain_id}
                        </span>
                      )}
                      <span style={{ flex: 1 }} />
                      {e.committed && (
                        <span style={{ fontSize: 8, color: TEAL, fontFamily: MONO }}>committed</span>
                      )}
                    </div>
                    <TripleLine subject={t.subject || '?'} relation={t.relation || '?'} object={t.object || '?'} />
                    {active && e.rationale && (
                      <div style={{ fontSize: 10, color: '#5a6375', lineHeight: 1.45, marginTop: 6, fontStyle: 'italic' }}>
                        {e.rationale}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
