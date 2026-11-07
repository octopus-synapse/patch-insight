# Refinement 0.3 — local experimental checkpoint

Selection was frozen before test predictions. No human or independent annotation.

| Method / dataset | Macro-F1 | AP macro | Brier | Log loss | Exact match |
|---|---:|---:|---:|---:|---:|
| original | 0.729561 | 0.925493 | 0.15903552 | 0.501503 | 0.281250 |
| original_v02_diagnostic | 0.558371 | 0.776780 | 0.16481213 | 0.505283 | 0.250000 |
| refined | 1.000000 | 1.000000 | 0.00000002 | 0.000020 | 1.000000 |
| refined_v02_diagnostic | 0.975185 | 0.999024 | 0.01703561 | 0.086040 | 0.888889 |
| heuristic | 0.534557 | 0.731155 | 0.31562500 | 5.087274 | 0.156250 |
| frozen_logistic | 0.678631 | 0.931342 | 0.20968241 | 0.931406 | 0.328125 |
| threshold_only | 0.918774 | 0.925493 | 0.15874409 | 0.501640 | 0.546875 |
| base | 1.000000 | 1.000000 | 0.00000000 | 0.000005 | 1.000000 |

Selected: continued; temperature 0.25; thresholds [0.5, 0.5, 0.5, 0.5, 0.5].
Paired bootstrap: [0.24177595644703706, 0.30348257650988547] (16 origins; 1,000 resamples; seed 42).

The target of +0.10 macro-F1 with no worse Brier/log loss was met against the original checkpoint on the same new test.
The new 64-text test shares composition patterns with development and training and is artificially easy. It is not evidence of real-resume performance.
The 36-text v0.2 diagnostic was already known; its improvement is exploratory, not a new independent test.

Two preliminary executions were stopped before test predictions: one to correct the log-loss tie breaker, one to remove repeated generic negative texts across origins. Their logs/checkpoints remain under artifacts. The final protocol and text hashes were frozen before the completed training.

Additional manual diagnostic after selection: “Revisei as consultas com SQL e reduzi a espera em 20%.” still predicts result absent (score about 0.4993). No threshold was changed for this example. The model is not perfect outside the controlled benchmark.

Full per-label metrics, baselines, candidates, selection and saved predictions are in the adjacent JSON files.
