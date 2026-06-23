# MaKG Frontend Improvements

This document summarizes the comprehensive frontend improvements implemented for the Multi-Agent Knowledge Graph (MaKG) project.

## Summary of Changes

### Phase 1: Fix Ingest-to-Reload Loop

Fixed broken upload-to-graph pipeline and added progress reporting:

- **Fixed pipeline invocation** (`scripts/api_server.py`):
  - Changed from positional argument to `--input` flag
  - Added `--checkpoint-dir` scoped per job to prevent checkpoint collisions
  - Added `_reload_kg()` function to hot-reload KG after pipeline completion
  - Added `POST /kg/reload` endpoint for manual reload

- **Real progress reporting** (`scripts/api_server.py`):
  - Replaced legacy `pipeline_checkpoint.json` reading with per-job checkpoint manifest reading
  - `_read_checkpoint_progress()` now reads from `checkpoints/<job_id>/<doc_id>/manifest.json`
  - Reports stage, entity count, and triple count accurately

- **Job persistence** (`scripts/api_server.py`):
  - Added `_save_jobs()` function to persist job table to `pipeline_logs/jobs.json`
  - Jobs survive server restarts

### Phase 2: Governance UI

Implemented full governance interface to surface the core contribution:

- **Backend endpoints** (`scripts/api_server.py`):
  - `GET /governance/audit` - paginated audit log with action, rationale, domain
  - `GET /governance/pending` - pending-review queue
  - `POST /governance/review` - approve/reject/revise pending triples
  - Extended `GET /kg/data` with topics and relation schema per domain

- **Frontend GovernancePanel** (`frontend/app/src/components/GovernancePanel.jsx`):
  - New component with tabs for Domains, Pending Review, and Audit Log
  - Domain panel shows entity counts, owner labels, scopes
  - Clicking domain highlights owned subgraph on canvas
  - Pending-review queue with Approve/Reject buttons
  - Audit log filterable by action and domain

- **API client** (`frontend/app/src/api.js`):
  - Added `fetchGovernanceAudit()`, `fetchGovernancePending()`, `submitGovernanceReview()`
  - Added `reloadKG()` for KG hot-reload

- **App integration** (`frontend/app/src/App.jsx`):
  - Added `GovernancePanel` import and integration
  - Added governance mode state (`govMode`)
  - Updated highlight logic to support domain/triple highlighting

- **Sidebar** (`frontend/app/src/components/Sidebar.jsx`):
  - Added "Govern" mode button to mode switcher
  - Added governance hint text
  - Added KG stats display in footer

### Phase 3: QA Experience

Enhanced QA with streaming progress and model switching:

- **Streaming QA** (`scripts/api_server.py`):
  - Added `POST /qa/stream` endpoint using Server-Sent Events (SSE)
  - Emits stage events: routing, decomposed, domain_answered, debate, synthesizing, critic

- **Progress callbacks** (`multi_agent_kg/core/qa_orchestrator.py`, `multi_agent_kg/core/advanced_qa.py`):
  - Added optional `progress_callback` parameter
  - Emits progress events at each stage of query processing

- **Frontend streaming** (`frontend/app/src/App.jsx`):
  - `handleSubmitQuestion()` uses `submitQuestionStream()` with `onProgress` callback
  - Falls back to `submitQuestion()` if streaming fails

- **QA Panel stage indicator** (`frontend/app/src/components/QAPanel.jsx`):
  - Shows current stage during loading (e.g., "domain answered", "synthesizing")

### Phase 4: Cleanup & Polish

- **Deleted legacy prototypes**:
  - Removed `frontend/MaKG.html`
  - Removed `frontend/MaKG Upload.html`
  - Removed `frontend/makg-app.jsx`
  - Removed `frontend/makg-data.js`
  - Removed `frontend/makg-graph.jsx`
  - Removed `frontend/makg-qa-panel.jsx`
  - Removed `frontend/makg-sidebar.jsx`
  - Removed `frontend/tweaks-panel.jsx`
  - Removed `frontend/package-lock.json`

- **Graph performance** (`frontend/app/src/App.jsx`):
  - Added `MAX_RENDER_NODES = 1500` cap
  - Degree-based node capping for large graphs
  - Shows warning when cap is applied

- **PDF ingestion** (`scripts/api_server.py`):
  - `/ingest` endpoint now accepts `.pdf` files
  - Uses `pypdf` for text extraction before pipeline

## Files Modified

### Backend
- `scripts/api_server.py` - Core API changes for all phases
- `multi_agent_kg/core/governed_kg.py` - Exposed `pending_review` property and `resolve_pending()` method
- `multi_agent_kg/core/qa_orchestrator.py` - Added progress callback
- `multi_agent_kg/core/advanced_qa.py` - Added progress callback

### Frontend
- `frontend/app/src/App.jsx` - Governance integration, streaming QA, node capping
- `frontend/app/src/api.js` - New API client functions
- `frontend/app/src/components/GovernancePanel.jsx` - New governance UI
- `frontend/app/src/components/Sidebar.jsx` - Governance mode, KG stats
- `frontend/app/src/components/QAPanel.jsx` - Stage indicator in loading state

## Build Verification

```bash
cd frontend/app && npm run build
```

Build completes successfully with:
- `dist/index.html` (0.76 kB)
- `dist/assets/index-H0LRvwqi.css` (1.24 kB)
- `dist/assets/index-CWNvG3y-.js` (314.91 kB)

## Key Features Now Available

1. **Upload documents** - Progress updates during pipeline, graph auto-refreshes when complete
2. **Governance mode** - View domains, approve/reject pending triples, audit log
3. **Enhanced QA** - Streaming progress indicators, model switching
4. **Large graph handling** - Node capping with 1500 node limit, performance protection
5. **PDF support** - Direct PDF upload without manual text extraction
