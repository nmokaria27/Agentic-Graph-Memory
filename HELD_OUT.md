# Held-out data manifest (PLAYBOOK §7.4)

Slices listed here are scored **at most once per accepted change** and are
NEVER hand-read, debugged against, or iterated on. Anything not listed that has
appeared in an experiment is development data forever.

| benchmark | held-out slice | status |
|---|---|---|
| DocRED | slice B = docs 30–34 | reserved; never tuned on |
| DocRED | docs 120–139 (Phase-4 tail) | reserved (100–119 became dev via SPGOV/PAIR series) |
| LongMemEval | per question-type, filtered-list indices 20+ | reserved (0–19 = dev; 0–7 used by EXP-LME-BREADTH) |
| SciERC | test split docs 0–99 | used ONCE per protocol run (paper-compare, headroom probe used docs 0–9 → those 10 are now dev) |
| MuSiQue | questions 100+ | reserved (0–99 = paper-era dev) |

Rules of use:
1. A held-out score may be taken only for an ACCEPTED change, once, and the
   result goes in EXPERIMENT_LOG.md whether it is good or bad.
2. If a held-out slice is ever inspected (hand-read, error-analyzed), move it
   to dev in this table in the same commit.
3. Run scripts must not reference held-out indices except through a
   sanctioned `*_heldout.sh` script that logs its invocation here.
