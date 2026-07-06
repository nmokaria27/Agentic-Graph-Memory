"""LongMemEval offline scorer — runs on run_eval.py's cached records only.

Two metrics, both offline (never re-runs the pipeline):
  1. substring — cheap signal: lowercase gold answer contained in the hypothesis.
     Meaningless for abstention (_abs) and preference questions; those report
     as None under substring and need --judge.
  2. --judge — the benchmark's real metric: LLM-as-judge with the OFFICIAL
     LongMemEval prompts (copied verbatim from
     xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py), answered by the
     local Nemotron instead of GPT-4o. Temperature 0. A generous token budget
     replaces the official max_tokens=10 because Nemotron reasons before
     answering; the verdict is parsed from the LAST line of the response so
     a "yes" inside the reasoning cannot leak into the label.

Usage:
  python evaluation/LongMemEval/score_longmemeval.py \
      --cache-dir evaluation/results/lme_kg_cache_smoke \
      --output evaluation/results/lme_scores_smoke.json [--judge]
"""
import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)


# ── official LongMemEval judge prompts (verbatim) ────────────────────────────

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return template.format(question, answer, response)


def parse_verdict(response):
    """yes/no from the judge. Last non-empty line only — Nemotron's visible
    reasoning may contain both words; the official 10-token GPT-4o setup never
    had that problem."""
    if not response:
        return None
    lines = [l.strip().lower() for l in response.strip().splitlines() if l.strip()]
    last = lines[-1]
    if "yes" in last and "no" not in last.replace("not", ""):
        return True
    if "no" in last:
        return False
    # fall back to whole-response scan, yes must be unambiguous
    full = response.lower()
    if "yes" in full and " no" not in full:
        return True
    if "no" in full:
        return False
    return None


def substring_label(record):
    """Cheap containment check. None where containment is not meaningful."""
    qid = str(record["question_id"])
    if "_abs" in qid or record["question_type"] == "single-session-preference":
        return None
    gold = str(record["answer"]).strip().lower()
    hyp = str(record.get("hypothesis", "")).lower()
    if not gold or not hyp:
        return False
    return gold in hyp


def judge_label(record, model, calls_log):
    import multi_agent_kg.llm.openai_client as oc

    qid = str(record["question_id"])
    prompt = get_anscheck_prompt(
        record["question_type"], record["question"], str(record["answer"]),
        str(record.get("hypothesis", "")), abstention="_abs" in qid)
    resp = oc.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=model, temperature=0.0, max_tokens=4096)
    label = parse_verdict(resp or "")
    calls_log.append({"question_id": qid, "judge_raw": (resp or "")[-200:],
                      "label": label})
    return label


def aggregate(rows, key):
    """accuracy per question_type + overall, skipping None labels."""
    by_type = {}
    for r in rows:
        label = r[key]
        if label is None:
            continue
        bucket = r["question_type"] + ("  [abstention]" if r["abstention"] else "")
        by_type.setdefault(bucket, []).append(1 if label else 0)
    out = {t: {"acc": round(sum(v) / len(v), 3), "n": len(v)}
           for t, v in sorted(by_type.items())}
    scored = [x for v in by_type.values() for x in v]
    if scored:
        out["OVERALL"] = {"acc": round(sum(scored) / len(scored), 3), "n": len(scored)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge", action="store_true",
                    help="LLM-as-judge with the official LongMemEval prompts")
    ap.add_argument("--judge-model",
                    default=os.environ.get("LLM_DEFAULT_MODEL", "nvidia/nemotron-3-nano"))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.json")))
    if not files:
        sys.exit(f"no cached records in {args.cache_dir}")

    rows, judge_calls = [], []
    for path in files:
        with open(path) as f:
            record = json.load(f)
        if "question_id" not in record:   # skip non-record json (e.g. ckpt dirs)
            continue
        qid = str(record["question_id"])
        row = {
            "question_id": qid,
            "question_type": record["question_type"],
            "abstention": "_abs" in qid,
            "hypothesis": record.get("hypothesis", ""),
            "answer": record["answer"],
            "error": (record.get("counts") or {}).get("error"),
            "substring": substring_label(record),
            "judge": None,
        }
        if args.judge:
            print(f"[SCORE] judging {qid} ({row['question_type']})", flush=True)
            row["judge"] = judge_label(record, args.judge_model, judge_calls)
        rows.append(row)

    report = {
        "cache_dir": args.cache_dir,
        "n_records": len(rows),
        "n_errors": sum(1 for r in rows if r["error"]),
        "substring": aggregate(rows, "substring"),
        "judge": aggregate(rows, "judge") if args.judge else None,
        "judge_model": args.judge_model if args.judge else None,
        "rows": rows,
        "judge_calls": judge_calls if args.judge else None,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SCORE] {len(rows)} records ({report['n_errors']} with errors)")
    for metric in ("substring", "judge"):
        agg = report[metric]
        if not agg:
            continue
        print(f"\n  {metric}:")
        for t, v in agg.items():
            print(f"    {t:38s} acc={v['acc']:.3f}  (n={v['n']})")
    print(f"\n[SCORE] -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
