# Neighborhood MaKG Rerun Decision
Updated: 2026-05-03T19:02:49

## 10-doc smoke result
| Run | Strict Triple F1 | Fuzzy/Mapped Triple F1 | Hallucination | Triples | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Prior GPT-5 MaKG 10-doc | 0.158 | 0.292 | 87.9% | n/a | n/a |
| Neighborhood GPT-5 MaKG 10-doc | 0.165 | 0.362 | 87.3% | 166 | 21 / 145 / 67 |

Decision: rerun 50-doc MaKG. The smoke improves fuzzy/mapped F1 substantially and slightly improves strict F1 and hallucination.

## Active 50-doc run
- Output: `evaluation/results/scierc_governed_gpt5_test_50_neighborhood.json`
- Log: `logs/scierc_gpt5_gov_50_neighborhood.log`
- Usage: `evaluation/results/gpt5_scierc_gov_50_neighborhood_usage.jsonl`
- Started from the completed 10-doc checkpoint, so only docs 11-50 are being processed.
