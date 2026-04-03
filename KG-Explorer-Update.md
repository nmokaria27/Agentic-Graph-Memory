# KG Explorer Update Summary

## Issue Resolved
`kg_explorer.html` had **hardcoded static data** (183 medical entities) that didn't update when new extractions were run.

## Solution Implemented
Created `generate_kg_explorer.py` to dynamically generate the HTML file from your current KG export.

## Changes Made

### 1. Created Generator Script
**File:** `generate_kg_explorer.py`

**Features:**
- Loads KG data from `kg_export.json`
- Loads QA results from `qa_results.json`
- Generates interactive vis.js graph with all entities and relations
- Builds QA panel with evidence highlighting
- Color-codes entity types automatically
- Creates legend and statistics overlay

### 2. Regenerated kg_explorer.html

**Current stats:**
- **558 entities** (was 183 old medical entities)
- **553 triples** (was 153 old triples)
- **381 connected entities**
- **4 QA examples** integrated

**Entity types now include:**
- PRODUCT (131)
- ORGANIZATION (85)
- INDIVIDUAL (37)
- TECHNOLOGICAL_FEATURE (42)
- And 45+ other types from your Apple/tech extraction

## Usage

### To update kg_explorer.html after new extractions:

```bash
python3 generate_kg_explorer.py kg_export.json qa_results.json kg_explorer.html
```

### To view the updated graph:

1. Open `kg_explorer.html` in your browser
2. Explore the interactive graph (558 entities visible)
3. Click QA cards to highlight evidence triples
4. Use the "Ask" feature (requires `python qa_server.py` running)

## Files Modified
- ✅ `generate_kg_explorer.py` - New generator script
- ✅ `kg_explorer.html` - Updated with current KG data (558 entities, 553 triples)

## Next Steps
After running new extractions with `run_pipeline_on_text.py`:
1. Run `python3 generate_kg_explorer.py` to update the HTML
2. Optionally run `python3 run_domain_qa.py` to generate fresh QA results
3. Refresh browser to see updated graph
