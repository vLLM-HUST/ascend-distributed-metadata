# Issue #7 — proposed maintainer update, not posted

- [x] Reconstruct the previous-MOD to DP4 word version chain and distinguish
  the Qwen3 W8A8 service comparison, Qwen3.5 DP2, and CPU/Gloo call measurements.
- [x] Correct documentation: the observed +0.80% is against the previous MOD,
  not Native; the historical worker marker only establishes attempted branch
  entry before the collective and fallback decision.
- [ ] Calculate historical measurement-window entry and fallback rates: the
  original per-round payloads and worker logs have not been recovered or
  independently rehashed. Both rates are **UNKNOWN**, never inferred as zero.
- [ ] Establish repeatable gain against Native on the same software stack:
  the existing original-DP2 Native comparison is a different implementation;
  no qualified current DP4 Native comparison is available.
- [ ] Merge/review this evidence and capture implementation, run CI, and
  independently verify the owner-approved closure scope.

The new opt-in diagnostics add worker identity, window and coverage checks;
they cannot reconstruct missing historical logs or certify runtime effectiveness.
See [the evidence review](performance-evidence-review.md). Ask the maintainer
whether a scoped historical-review closure with the two UNKNOWNs is acceptable;
otherwise keep #7 open for the missing evidence or a separately approved study.
