# HANDOFF: Orphan Node Investigation & Fix Plan

## Current State (as of session end)

### Pipeline Version
- **Architecture:** 10-stage deliberative multi-agent pipeline (`DeliberativeOrchestrator`)
- **Agents present:** DocumentProcessor, DomainClassifier, EntityExtractor, RelationExtractor, EvidenceLinker, ExtractionValidator, ExtractionVerificationAgent, KnowledgeOrganizer, EntityResolver, CriticAgent, CorrectorAgent
- **Note:** `OrphanLinker`, `TriplexExtractor`, `SchemaAligner`, `CriticAgent`, `CorrectorAgent` were referenced in earlier design docs but **do not currently exist on disk** in `multi_agent_kg/agents/`. The current agents directory has only the 8 files listed above.

### Latest KG Export Stats (`kg_export.json`)
| Metric | Value |
|--------|-------|
| Total entities | 183 |
| Total triples | 159 |
| Orphan entities | 166 |
| **Orphan rate** | **90.7%** |
| UNRESOLVED entities | 3 |

This is a **medical/biological document** (cardio-renal-immune domain). Top orphan types:
- `BIOLOGICAL_MARKER`: 58 orphans
- `CARDIOVASCULAR_MEASUREMENT`: 22 orphans
- `BIOLOGICAL_PROCESS`: 20 orphans
- `MEDICATION`: 17 orphans
- `METABOLIC_CONDITION`: 12 orphans

---

## Root Causes Identified (Confirmed by Code Inspection)

### RC1 — Entity ID / Triple Endpoint Format Mismatch (Most Critical)

**Confirmed by data:**
- Entity IDs in KG: `snake_case` format — e.g., `"chronic_metabolic_inflammation"`, `"complement_activation_markers"`, `"il_6"`
- Triple subjects/objects: **original text** — e.g., `"Chronic metabolic inflammation"`, `"IL-6"`, `"TNF-α"`, `"soluble C5b-9 (sC5b-9)"`

**76 unique triple subjects** and **51 unique triple objects** do NOT match any entity ID.

**Why:** `KnowledgeOrganizer._integrate_to_kg()` builds `name_to_id` mapping from entity `text` (lowercased) → `id` (snake_case). When `_resolve_entity_name("Chronic metabolic inflammation")` is called, it lowercases to `"chronic metabolic inflammation"` which **should** match the key `"chronic metabolic inflammation"` → but fails for anything with punctuation, abbreviations, or phrasing differences (e.g., `"IL-6"` vs `"il-6"`, `"soluble C5b-9 (sC5b-9)"` which has no match at all).

**Location:**
- `multi_agent_kg/agents/knowledge_organizer.py` — `_resolve_entity_name()` inner function, lines ~520–529
- Fallback phantom creation: lines ~577–597

### RC2 — Entity Cap: RelationExtractor Only Sees First 50 Entities

**File:** `multi_agent_kg/agents/relation_extractor.py`, line 351

```python
entities_str = ", ".join(
    f'"{e.get("text", str(e))}" ({e.get("type", "?")})'
    for e in entities[:50]   # ← hard cap at 50
)
```

With 183+ resolved entities, **all entities beyond index 50 are invisible to the LLM** during relation extraction. They will never appear in any triple — guaranteed orphans.

### RC3 — Entity Aliases Not Passed to RelationExtractor

**File:** `multi_agent_kg/agents/relation_extractor.py`, lines 349–352

The EntityResolver (`entity_resolver.py`) correctly produces `aliases` and `mentions` lists for every canonical entity. But the relation extractor prompt only shows the canonical name (`e.get("text")`). The LLM sees `"Chronic metabolic inflammation"` but may produce triples with `"chronic inflammation"` or `"metabolic inflammation"` — variants that then fail resolution.

### RC4 — Triples Reference Phrases That Were Never Extracted as Entities

Many triple subjects/objects like `"endothelial function surrogate via arginine/ADMA ratio"`, `"IESS quartile"`, `"Lower CFR"` are **complex descriptive phrases** the LLM generated that don't correspond to any extracted entity. The LLM invents inline descriptions rather than reusing entity names.

**Cause:** The prompt says "reference entities from the provided list" but this is advisory — LLMs routinely ignore it and use contextually natural phrasing.

### RC5 — Property-Like Entities Extracted as Standalone Nodes

Despite the EntityExtractor prompt saying "DO NOT EXTRACT generic dates/times", the gleaning pass encourages "entities mentioned only in passing" which pulls in dates, metric values, and abstract descriptors. These become orphans because they have no natural place in a triple.

---

## What Was Built in This Session

### 1. Investigation & Analysis
- Statistical breakdown of kg_export.json orphan patterns (27–31% in earlier run, now 90.7%)
- Full pipeline trace: how entities flow from EntityExtractor → EntityResolver → RelationExtractor → KnowledgeOrganizer
- Confirmed exact code locations for all root causes

### 2. OrphanLinker Agent (designed, NOT yet on disk)
The following was designed but the files may not exist on disk (verify before assuming):
- **`multi_agent_kg/agents/orphan_linker.py`** — post-verification agent that classifies orphan entities as link/reify/prune and generates new triples
- **Schemas** in `extraction_schemas.py`: `OrphanLinkerResponse`, `OrphanClassification`, `OrphanNewTriple`
- **Orchestrator integration**: Stage 10 (new), KnowledgeOrganizer moved to Stage 11

**⚠️ Check before proceeding:** Run `ls multi_agent_kg/agents/` — if `orphan_linker.py` is absent, this work was lost and needs to be re-implemented.

### 3. kg_explorer.html Physics Fix (done, on disk)
Fixed nodes clumping together by replacing the empty physics config with `forceAtlas2Based`:
- `gravitationalConstant: -50`, `centralGravity: 0.01`, `springLength: 200`, `springConstant: 0.08`
- Added stabilization progress bar (CSS + HTML + JS event listeners)

---

## The Fix Plan (Not Yet Implemented)

These fixes address root causes in priority order. None of these have been written to disk yet.

### Fix 1 — KnowledgeOrganizer: Replace Naive Resolution with rapidfuzz (Highest Impact)

**File:** `multi_agent_kg/agents/knowledge_organizer.py`, `_resolve_entity_name()` ~line 520

Replace the substring fallback with `rapidfuzz.process.extractOne` at 80-score threshold. `rapidfuzz` is already a dependency (used in `entity_resolver.py`) — just needs importing here.

```python
# ADD at top of knowledge_organizer.py:
try:
    from rapidfuzz import process as fuzz_process, fuzz as fuzz_scorer
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

# REPLACE _resolve_entity_name inner function:
def _resolve_entity_name(name: str) -> Optional[str]:
    key = name.lower().strip()
    # 1. Exact match
    if key in name_to_id:
        return name_to_id[key]
    # 2. rapidfuzz token sort (handles word-order variants)
    if _RAPIDFUZZ_AVAILABLE and len(key) > 2:
        match = fuzz_process.extractOne(
            key, name_to_id.keys(),
            scorer=fuzz_scorer.token_sort_ratio,
            score_cutoff=80,
        )
        if match:
            return name_to_id[match[0]]
    # 3. Substring fallback (kept for short names)
    for known_name, known_id in name_to_id.items():
        if len(key) > 3 and (key in known_name or known_name in key):
            return known_id
    return None
```

**Also:** When resolution still fails after all attempts, **skip the triple** instead of creating phantom UNRESOLVED entities:
```python
# REPLACE lines 577–597 (phantom creation):
if not resolved_subj or not resolved_obj:
    skipped_triples += 1
    continue   # don't create phantom entities
```

### Fix 2 — RelationExtractor: Remove Hard Entity Cap, Add Aliases

**File:** `multi_agent_kg/agents/relation_extractor.py`, lines 349–352

```python
# REPLACE:
entities_str = ", ".join(
    f'"{e.get("text", str(e))}" ({e.get("type", "?")})'
    for e in entities[:50]
)

# WITH:
def _relevant_entities(entities, segment_text, cap=80):
    """Return entities most likely mentioned in this segment."""
    text_lower = segment_text.lower()
    relevant, other = [], []
    for e in entities:
        name = e.get("text", "")
        all_names = [name] + e.get("aliases", []) + e.get("mentions", [])
        if any(n.lower() in text_lower for n in all_names if n):
            relevant.append(e)
        else:
            other.append(e)
    return (relevant + other)[:cap]

def _format_entity_with_aliases(e):
    name = e.get("text", str(e))
    etype = e.get("type", "?")
    aliases = [a for a in e.get("aliases", [])[:3] if a != name]
    alias_str = f' [aka: {", ".join(repr(a) for a in aliases)}]' if aliases else ""
    return f'"{name}"{alias_str} ({etype})'

scoped = _relevant_entities(entities, text, cap=80)
entities_str = ", ".join(_format_entity_with_aliases(e) for e in scoped)
```

**Also strengthen the prompt** — add to `JOINT_TRIPLE_EXTRACTION_PROMPT`:
```
IMPORTANT: When writing triple subjects and objects, use entity names EXACTLY
as they appear in the entity list above. Do not abbreviate, paraphrase, or
invent new names. If you want to refer to "Apple Computer Company", write
exactly "Apple Computer Company", not "Apple" or "Apple Inc."
```

### Fix 3 — OrphanLinker Agent (Re-implement if missing)

**File:** `multi_agent_kg/agents/orphan_linker.py` (create if absent)

Post-verification agent (runs between CriticCorrectorLoop and KnowledgeOrganizer). For each orphan entity, LLM classifies as:
- **link** — find missing relationships from source text, generate new triples
- **reify** — convert property values (dates, metrics) to typed attributes on existing entities
- **prune** — remove noise entities

**Pydantic schemas needed** in `multi_agent_kg/schemas/extraction_schemas.py`:
```python
class OrphanClassification(BaseModel):
    entity_id: str
    action: str  # "link" | "reify" | "prune"
    reason: str = ""
    target_entity: str = ""
    property_name: str = ""

class OrphanNewTriple(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = 0.7
    evidence: str = ""

class OrphanLinkerResponse(BaseModel):
    classifications: list[OrphanClassification] = []
    new_triples: list[OrphanNewTriple] = []
```

**Orchestrator integration:** Import and instantiate in `_init_agents()`. Insert Stage 10 (OrphanLinker) between Stage 9 (CriticCorrector) and the final Stage (KnowledgeOrganizer). Update `PipelineProgress(total_stages=...)` accordingly.

### Fix 4 — Property-Type Filter Before RelationExtractor

**File:** `multi_agent_kg/core/deliberative_orchestrator.py`, between Stage 6 and Stage 7 (~line 525)

```python
_PROPERTY_TYPES = frozenset({
    "date", "DATE", "Time", "TIME", "TIME_PERIOD",
    "FINANCIAL_METRIC", "Metric", "MONEY", "Money",
    "Cardinal", "Currency", "NOUN_PHRASE",
})

resolved_entities_for_relations = [
    e for e in resolved_entities
    if e.get("type", "") not in _PROPERTY_TYPES
]
# Pass filtered list to RelationExtractor only
# KnowledgeOrganizer still gets the full resolved_entities
```

---

## Recommended Implementation Order

1. **Fix 1** (KnowledgeOrganizer rapidfuzz + skip phantoms) — standalone, low risk, high impact
2. **Fix 2** (RelationExtractor entity cap + aliases) — standalone, medium risk, very high impact  
3. **Fix 4** (property-type filter) — 5-line addition to orchestrator, zero risk
4. **Fix 3** (OrphanLinker) — new file + orchestrator wiring, implement after verifying the above reduce orphan rate

## Verification Command

Run after each fix:
```bash
python3 -c "
import json
kg = json.load(open('kg_export.json'))['knowledge_graph']
connected = {t['subject'] for t in kg['triples']} | {t['object'] for t in kg['triples']}
orphans = [e for e in kg['entities'] if e['id'] not in connected]
print(f\"{len(orphans)}/{len(kg['entities'])} orphans ({100*len(orphans)/len(kg['entities']):.1f}%)\")
unresolved = [e for e in kg['entities'] if e.get('type') == 'UNRESOLVED']
print(f'UNRESOLVED phantom entities: {len(unresolved)}')
"
```

**Target:** Orphan rate below 10% (down from current 90.7%).

---

## Key File Locations

| File | Purpose |
|------|---------|
| `multi_agent_kg/agents/relation_extractor.py` | Fix entity cap (line 351) and prompt |
| `multi_agent_kg/agents/knowledge_organizer.py` | Fix `_resolve_entity_name` (~line 520) and phantom creation (~line 577) |
| `multi_agent_kg/agents/orphan_linker.py` | New file — create if missing |
| `multi_agent_kg/schemas/extraction_schemas.py` | Add `OrphanLinker*` schemas at bottom |
| `multi_agent_kg/core/deliberative_orchestrator.py` | Property filter (~line 525), OrphanLinker stage wiring |
| `kg_explorer.html` | Physics fix already applied — do not revert |
