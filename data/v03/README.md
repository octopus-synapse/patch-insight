# Refinement evaluation 0.3

240 synthetic texts from 60 authored scenarios, four variants per origin.
Train/validation/calibration/test: 96/48/32/64. Group split seed 314, frozen
before model scoring. Same AI writes, annotates and reviews: no human or
independent annotation. CC BY 4.0. Investigation alone is not a result.

These are controlled compositions with natural method phrasing, not 240
independent free-written narratives. Shared composition patterns limit external
validity. Compare the known v0.2 test only after selection, as a diagnostic.

Protocol: candidates are old BERT with threshold tuning, continued BERT at 1e-5,
and base BERT at 2e-5. Six epochs maximum, seed 42, batch 8, accumulation 2,
192 tokens. Select epoch by validation macro-F1 at .5, log loss tie breaker.
One global temperature fitted on calibration; component thresholds selected on
validation from .1 to .9 in .05 steps, ties closest to .5. Select candidate by
validation macro-F1, Brier tie breaker; freeze before test. Target +.10 macro-F1
on the same new test versus old checkpoint, without higher Brier or log loss.
Do not tune again after opening test. Historical tests and weights are retained.
