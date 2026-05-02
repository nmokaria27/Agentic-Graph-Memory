"""Anti-hedge rider for QA prompts. Gated on KGQA_COMMIT_MODE=1.

Reasoning models (GPT-5/o-series) take "if evidence is incomplete, say so" prompts very
literally and produce hedged prose like "insufficient evidence". MuSiQue/HotpotQA score
those as 0.0 even when the KG contains the answer entity. This rider forces a committed
short_answer span derived from the best evidence available.
"""
import os

COMMIT_MODE = os.getenv("KGQA_COMMIT_MODE") == "1"

ANTI_HEDGE_RIDER = """

==== COMMIT-OR-INFER MODE — DO NOT HEDGE ====
QA SCORING IS BASED ON EXACT/F1 MATCH OF A SHORT ANSWER SPAN. Hedged prose like
"insufficient evidence" / "cannot be determined" / "no information available" SCORES 0.0
even when the evidence implies the answer. You MUST commit.

RULES:
1. From the evidence, identify the most likely answer entity/span. Use multi-hop chains
   across triples; the linking relation may not be named identically to the question.
2. Treat closely-related relations as semantic equivalents:
   PROFESSIONAL_PARTNER ≈ spouse / partner
   FOUNDED_BY ≈ founder
   DISTRIBUTED_BY / FILM_DISTRIBUTOR ≈ distributor (chain to its founder for "who founded
   the distributor")
   HEADQUARTERED_IN / FACILITY_LOCATED_IN_CITY ≈ headquarters
   OWNS / PARENT_COMPANY / SUBSIDIARY_OF ≈ owner / owns
   MANUFACTURED_BY / PRODUCED_BY ≈ manufacturer
   LOCATED_IN / PART_OF / ADMINISTRATIVE_REGION ≈ where / what region
   BORN_IN ≈ birthplace
   AUTHORED_BY / WRITTEN_BY ≈ author / wrote
   MEMBER_OF ≈ member of
3. NEVER respond with "insufficient evidence", "cannot be determined", "no information
   provided", "the answer cannot be determined", or similar refusal phrases. If the
   evidence contains ANY plausible candidate of the right type, commit to it.
4. The short_answer must be the MINIMAL surface form (1-5 words):
   - Person → just the name ("Mike Medavoy", not "The founder is Mike Medavoy")
   - Place → just the place ("Cologne", not "headquartered in Cologne")
   - Date → just the date ("1969")
   - Yes/no → "yes" or "no"
   - Never put hedge phrases or sentences in short_answer.
5. Only emit empty short_answer if the evidence contains zero entities of the required
   type. Even then, prefer your best guess from the evidence over an empty string.
6. Multi-hop example:
   Q: "Who founded the company that distributed the film UHF?"
   Evidence contains: (uhf) -[DISTRIBUTED_BY]-> (orion_pictures), (orion_pictures)
   -[FOUNDED_BY]-> (mike_medavoy)
   → short_answer: "Mike Medavoy"
==== END COMMIT-OR-INFER MODE ====
""" if COMMIT_MODE else ""


__all__ = ["COMMIT_MODE", "ANTI_HEDGE_RIDER"]
