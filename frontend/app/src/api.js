/**
 * API client for the MaKG backend.
 *
 * Dev: Vite proxies /api/* to localhost:8000 (see vite.config.js).
 * Prod: set VITE_API_BASE at build time to an absolute URL
 *   (e.g. VITE_API_BASE=http://localhost:8000 npm run build).
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

async function jsonOrThrow(resp) {
  if (!resp.ok) {
    let detail = '';
    try { detail = (await resp.json()).detail ?? ''; } catch { /* ignore */ }
    throw new Error(`HTTP ${resp.status}${detail ? `: ${detail}` : ''}`);
  }
  return resp.json();
}

export async function fetchHealth() {
  const resp = await fetch(`${API_BASE}/health`, {
    signal: AbortSignal.timeout(3000),
  });
  return jsonOrThrow(resp);
}

export async function fetchKGData() {
  const resp = await fetch(`${API_BASE}/kg/data`);
  return jsonOrThrow(resp);
}

export async function fetchKGStats() {
  const resp = await fetch(`${API_BASE}/kg/stats`);
  return jsonOrThrow(resp);
}

export async function fetchModels() {
  const resp = await fetch(`${API_BASE}/models`, {
    signal: AbortSignal.timeout(3000),
  });
  return jsonOrThrow(resp);
}

// Combine a caller-provided abort signal (e.g. a cancel button) with a timeout.
function withTimeout(signal, ms) {
  const t = AbortSignal.timeout(ms);
  if (!signal) return t;
  return typeof AbortSignal.any === 'function' ? AbortSignal.any([signal, t]) : signal;
}

export async function submitQuestion(question, model, { signal } = {}) {
  const resp = await fetch(`${API_BASE}/qa`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, model }),
    signal: withTimeout(signal, 600_000),
  });
  return jsonOrThrow(resp);
}

/**
 * Streaming QA via Server-Sent Events over POST.
 * Calls onProgress({stage, info}) for each progress event.
 * Resolves with the final result payload, or rejects on error.
 */
export async function submitQuestionStream(question, model, onProgress, { signal } = {}) {
  const resp = await fetch(`${API_BASE}/qa/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, model }),
    signal: withTimeout(signal, 600_000),
  });
  if (!resp.ok || !resp.body) {
    let detail = '';
    try { detail = (await resp.json()).detail ?? ''; } catch { /* ignore */ }
    throw new Error(`HTTP ${resp.status}${detail ? `: ${detail}` : ''}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop(); // keep incomplete chunk
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue; // skip keep-alive comments
      let msg;
      try { msg = JSON.parse(line.slice(5)); } catch { continue; }
      if (msg.event === 'progress' && onProgress) onProgress(msg);
      else if (msg.event === 'result') result = msg.payload;
      else if (msg.event === 'error') throw new Error(msg.message || 'QA stream failed');
    }
  }

  if (!result) throw new Error('Stream ended without a result');
  return result;
}

export async function fetchGovernanceAudit(params = {}) {
  const qs = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  ).toString();
  const resp = await fetch(`${API_BASE}/governance/audit${qs ? `?${qs}` : ''}`);
  return jsonOrThrow(resp);
}

export async function fetchGovernancePending() {
  const resp = await fetch(`${API_BASE}/governance/pending`);
  return jsonOrThrow(resp);
}

export async function submitGovernanceReview(index, action, rationale) {
  const resp = await fetch(`${API_BASE}/governance/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ index, action, rationale }),
  });
  return jsonOrThrow(resp);
}

export async function reloadKG() {
  const resp = await fetch(`${API_BASE}/kg/reload`, { method: 'POST' });
  return jsonOrThrow(resp);
}

export async function ingestDocument(file) {
  const form = new FormData();
  form.append('file', file);
  const resp = await fetch(`${API_BASE}/ingest`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(30_000),
  });
  return jsonOrThrow(resp);
}

export async function fetchPipelineStatus(jobId) {
  const url = jobId
    ? `${API_BASE}/pipeline/status?job_id=${encodeURIComponent(jobId)}`
    : `${API_BASE}/pipeline/status`;
  const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
  return jsonOrThrow(resp);
}

// ── Eval run viewer ──────────────────────────────────────────────────────────

export async function fetchEvalRuns() {
  const resp = await fetch(`${API_BASE}/eval/runs`, { signal: AbortSignal.timeout(5000) });
  return jsonOrThrow(resp);
}

export async function fetchEvalGraph(dir, strategy, doc = 'all') {
  const qs = new URLSearchParams({ dir, strategy, doc }).toString();
  const resp = await fetch(`${API_BASE}/eval/graph?${qs}`);
  return jsonOrThrow(resp);
}

export async function fetchEvalMetrics(dir, strategy) {
  const qs = new URLSearchParams({ dir, strategy }).toString();
  const resp = await fetch(`${API_BASE}/eval/metrics?${qs}`, { signal: AbortSignal.timeout(30_000) });
  return jsonOrThrow(resp);
}
