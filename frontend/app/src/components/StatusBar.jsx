import { useState, useEffect, useRef } from 'react';
import { fetchPipelineStatus } from '../api.js';

const TEAL   = '#1de9b6';
const INDIGO = '#7c83e8';

const mono = { fontFamily: "'JetBrains Mono', monospace" };

/**
 * Always-visible pipeline status strip (top center).
 * Polls /pipeline/status — 2.5s while a job is active, 10s when idle.
 * Shows a progress bar (completed stages / total) for the active job and
 * fires onPipelineComplete when a job finishes so the app can refetch the KG.
 */
export default function StatusBar({ serverConnected, kgStats, onPipelineComplete }) {
  const [activeJob, setActiveJob] = useState(null);
  const [stages, setStages]       = useState([]);
  const prevRunningIds = useRef(new Set());

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function poll() {
      let delay = 10_000;
      try {
        const data = await fetchPipelineStatus();
        if (cancelled) return;
        if (Array.isArray(data.stages) && data.stages.length) setStages(data.stages);
        const jobs = data.jobs || [];
        const active = jobs.find(j => j.status === 'running') || jobs.find(j => j.status === 'queued') || null;
        setActiveJob(active);
        if (active) delay = 2_500;

        // A job we saw running is no longer running -> it completed or failed.
        const runningNow = new Set(jobs.filter(j => j.status === 'running' || j.status === 'queued').map(j => j.job_id));
        for (const id of prevRunningIds.current) {
          if (!runningNow.has(id)) {
            const done = jobs.find(j => j.job_id === id);
            if (done?.status === 'completed' && onPipelineComplete) onPipelineComplete();
          }
        }
        prevRunningIds.current = runningNow;
      } catch {
        if (!cancelled) setActiveJob(null);
      }
      if (!cancelled) timer = setTimeout(poll, delay);
    }

    poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [onPipelineComplete]);

  const cp = activeJob?.checkpoint || {};
  const completed = (cp.completed_stages || []).length;
  const total = stages.length || 11;
  const frac = activeJob ? Math.min(completed / total, 1) : 0;
  const stageLabel = stages.find(s => s.id === cp.current_stage)?.label
    || (cp.current_stage ? `Stage ${cp.current_stage}` : activeJob?.status === 'queued' ? 'Queued' : 'Starting');

  return (
    <div style={{
      position: 'absolute', top: 14, left: '50%', transform: 'translateX(-50%)',
      zIndex: 240, display: 'flex', alignItems: 'center', gap: 10,
      padding: '5px 12px', borderRadius: 16,
      background: 'rgba(24,28,38,0.92)', border: '1px solid rgba(255,255,255,0.08)',
      pointerEvents: 'none', maxWidth: '60%',
    }}>
      <span style={{
        width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
        background: serverConnected ? '#22c55e' : '#f59e0b',
        boxShadow: serverConnected ? '0 0 6px rgba(34,197,94,0.6)' : '0 0 6px rgba(245,158,11,0.6)',
      }} />
      <span style={{ ...mono, fontSize: 10, color: '#5a6375', whiteSpace: 'nowrap' }}>
        {serverConnected
          ? `${kgStats?.entities ?? kgStats?.num_entities ?? '?'}e · ${kgStats?.triples ?? kgStats?.num_triples ?? '?'}t`
          : 'offline'}
      </span>

      {activeJob && (
        <>
          <span style={{ width: 1, height: 14, background: 'rgba(255,255,255,0.1)' }} />
          <span style={{ ...mono, fontSize: 10, color: INDIGO, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 140 }}>
            {activeJob.filename || activeJob.job_id}
          </span>
          <div style={{ width: 90, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)', overflow: 'hidden', flexShrink: 0 }}>
            <div style={{
              width: `${Math.max(frac * 100, 4)}%`, height: '100%', borderRadius: 2,
              background: `linear-gradient(90deg, ${INDIGO}, ${TEAL})`, transition: 'width 0.6s ease',
            }} />
          </div>
          <span style={{ ...mono, fontSize: 10, color: TEAL, whiteSpace: 'nowrap' }}>
            {completed}/{total} · {stageLabel}
          </span>
          {(cp.entity_count || cp.triple_count) ? (
            <span style={{ ...mono, fontSize: 10, color: '#5a6375', whiteSpace: 'nowrap' }}>
              {cp.entity_count || 0}e {cp.triple_count || 0}t
            </span>
          ) : null}
        </>
      )}
    </div>
  );
}
