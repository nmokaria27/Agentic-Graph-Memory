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

export async function submitQuestion(question, model) {
  const resp = await fetch(`${API_BASE}/qa`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, model }),
    signal: AbortSignal.timeout(600_000),
  });
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
