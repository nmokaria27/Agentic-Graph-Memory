import { useState, useEffect, useRef } from 'react';
import { fetchPipelineStatus } from '../api.js';

const MONO = "'JetBrains Mono', monospace";
const TEAL = '#1de9b6';

// Fallback if the backend is older and doesn't send `stages`.
const DEFAULT_STAGE_COUNT = 11;

/**
 * Always-visible status strip: backend health + KG size, plus a live
 * pipeline progress bar whenever any ingest job is queued/running —
 * including jobs started before this page was loaded.
 */
export default function StatusBar({ serverConnected, kgStats, onPipelineComplete }) {
  const [activeJob, setActiveJob] = useState(null);
  const [stages, setStages] = useState([]);
  const prevStatusRef = useRef({});

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    async function poll() {
      let active = null;
      try {
        const data = await fetchPipelineStatus();
        if (cancelled) return;
        const jobs = data.jobs || [];
        if (data.stages?.length) setStages(data.stages);
        active = jobs.find(j => j.status === 'running') || jobs.find(j => j.status === 'queued') || null;
        setActiveJob(active);

        // Refresh the KG when any job transitions to completed.
        const prev = prevStatusRef.current;
        if (jobs.some(j => j.status === 'completed' && prev[j.job_id] && prev[j.job_id] !== 'completed')) {
          onPipelineComplete?.();
        }
        prevStatusRef.current = Object.fromEntries(jobs.map(j => [j.job_id, j.status]));
      } catch {
        if (cancelled) return;
        setActiveJob(null);
      }
      timer = setTimeout(poll, active ? 2500 : 10000);
    }

    poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [onPipelineComplete]);

  const totalStages = stages.length || DEFAULT_STAGE_COUNT;
  const ckpt = activeJob?.checkpoint || {};
  const completedCount = ckpt.completed_stages?.length || 0;
  const pct = Math.min(100, Math.round((completedCount / totalStages) * 100));

  // `current_stage` in the manifest is the last *completed* stage — the stage
  // actually executing is the next one in the canonical order.
  let runningLabel = 'starting…';
  if (completedCount > 0 && stages.length) {
    const idx = stages.findIndex(s => s.id === ckpt.current_stage);
    const next = idx >= 0 ? stages[idx + 1] : null;
    runningLabel = next ? next.label : 'finishing…';
  }

  const healthColor = serverConnected ? '#22c55e' : '#ef4444';
  const healthText = serverConnected
    ? (kgStats ? `${kgStats.entities ?? 0} entities · ${kgStats.triples ?? 0} triples` : 'connected')
    : 'offline — demo data';

  return (
    <div style={{
      position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
      zIndex: 240, pointerEvents: 'none',
      display: 'flex', alignItems: 'center', gap: 14,
      padding: '6px 14px', borderRadius: 4,
      background: 'rgba(24,28,38,0.92)', border: '1px solid rgba(255,255,255,0.08)',
      fontFamily: MONO, fontSize: 10, color: '#8892a4', letterSpacing: '0.04em',
      whiteSpace: 'nowrap',
    }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%', background: healthColor,
          boxShadow: `0 0 6px ${healthColor}`,
        }} />
        {healthText}
      </span>

      {activeJob && (
        <span style={{ display: 'flex', alignItems: 'center', gap: 8, color: TEAL }}>
          <span style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {activeJob.filename || activeJob.job_id}
          </span>
          <span style={{
            width: 110, height: 5, borderRadius: 3,
            background: 'rgba(255,255,255,0.08)', overflow: 'hidden',
          }}>
            <span style={{
              display: 'block', height: '100%', width: `${pct}%`,
              background: TEAL, transition: 'width 0.6s ease',
            }} />
          </span>
          <span>
            {completedCount}/{totalStages} · {activeJob.status === 'queued' ? 'queued' : runningLabel}
          </span>
          {(ckpt.entity_count || ckpt.triple_count) ? (
            <span style={{ color: '#8892a4' }}>
              {ckpt.entity_count ?? 0}e / {ckpt.triple_count ?? 0}t
            </span>
          ) : null}
        </span>
      )}
    </div>
  );
}
