import { useState, useEffect, useRef } from 'react';
import { ingestDocument, fetchPipelineStatus } from '../api.js';

const TEAL   = '#1de9b6';
const INDIGO = '#7c83e8';

function StatusBadge({ status }) {
  const map = {
    queued:    { color: '#f59e0b', label: 'queued' },
    running:   { color: INDIGO,     label: 'running' },
    completed: { color: '#22c55e',  label: 'done' },
    failed:    { color: '#ef4444',  label: 'failed' },
  };
  const s = map[status] || { color: '#6a7585', label: status || 'idle' };
  return (
    <span style={{
      fontSize: 9, padding: '1px 6px', borderRadius: 2,
      color: s.color, background: `${s.color}1a`,
      fontFamily: "'JetBrains Mono', monospace",
    }}>{s.label}</span>
  );
}

export default function UploadPanel({ onIngestComplete }) {
  const [open, setOpen] = useState(false);
  const [job, setJob]   = useState(null); // current job state from /pipeline/status
  const [error, setError] = useState('');
  const fileRef = useRef(null);
  const pollRef = useRef(null);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPolling = (jobId) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const data = await fetchPipelineStatus(jobId);
        setJob(data);
        if (data.status === 'completed' || data.status === 'failed') {
          stopPolling();
          if (data.status === 'completed' && onIngestComplete) onIngestComplete();
        }
      } catch (e) {
        setError(e.message || 'status check failed');
        stopPolling();
      }
    }, 2000);
  };

  const handleFile = async (file) => {
    if (!file) return;
    setError('');
    try {
      const resp = await ingestDocument(file);
      setJob({ ...resp });
      startPolling(resp.job_id);
    } catch (e) {
      setError(e.message || 'upload failed');
    }
  };

  return (
    <div style={{ position: 'absolute', top: 16, right: 60, zIndex: 250 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          padding: '6px 12px', borderRadius: 4,
          background: open ? 'rgba(29,233,182,0.15)' : 'rgba(24,28,38,0.92)',
          border: `1px solid ${open ? 'rgba(29,233,182,0.4)' : 'rgba(255,255,255,0.1)'}`,
          color: open ? TEAL : '#8892a4',
          fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
          cursor: 'pointer', letterSpacing: '0.05em',
        }}>
        + INGEST
      </button>

      {open && (
        <div style={{
          marginTop: 8, width: 280, padding: 14,
          background: '#181c26', border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 4, boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        }}>
          <div style={{ fontSize: 10, color: '#3d4555', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 8 }}>
            Upload .txt / .pdf for pipeline
          </div>

          <input
            ref={fileRef}
            type="file"
            accept=".txt,.pdf"
            onChange={e => handleFile(e.target.files?.[0])}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={job && (job.status === 'queued' || job.status === 'running')}
            style={{
              width: '100%', padding: '8px', borderRadius: 4,
              background: 'rgba(124,131,232,0.12)',
              border: '1px dashed rgba(124,131,232,0.35)',
              color: INDIGO, fontSize: 11, cursor: 'pointer',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
            choose file…
          </button>

          {error && (
            <div style={{ marginTop: 8, fontSize: 10, color: '#ef4444', fontFamily: "'JetBrains Mono', monospace" }}>
              {error}
            </div>
          )}

          {job && (
            <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontSize: 10, color: '#8892a4', fontFamily: "'JetBrains Mono', monospace", overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>
                  {job.filename || job.job_id}
                </span>
                <StatusBadge status={job.status} />
              </div>
              {job.checkpoint && (job.checkpoint.entity_count || job.checkpoint.triple_count) ? (
                <div style={{ fontSize: 9, color: '#3d4555', fontFamily: "'JetBrains Mono', monospace" }}>
                  stage: {job.checkpoint.current_stage || '?'} ·
                  entities: {job.checkpoint.entity_count || 0} ·
                  triples: {job.checkpoint.triple_count || 0}
                </div>
              ) : null}
              {job.log_file && (
                <div style={{ fontSize: 9, color: '#3d4555', marginTop: 4, fontFamily: "'JetBrains Mono', monospace", wordBreak: 'break-all' }}>
                  log: {job.log_file.split('/').pop()}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
