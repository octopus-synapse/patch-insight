# Avaliação posterior da v0.1

Referência sintética revisada pela mesma IA. Sem revisão humana ou evidência populacional.

| Método | Macro-F1 | Exact match | Hamming ↓ | AP macro | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| heuristic | 0.6791 | 0.3333 | 0.2067 | 0.8248 | 0.2067 | 3.3311 |
| frozen_logistic | 0.9888 | 0.9333 | 0.0133 | 1.0000 | 0.0089 | 0.0297 |
| bert_finetuned | 0.9911 | 0.9500 | 0.0100 | 1.0000 | 0.0383 | 0.1941 |

Métricas por rótulo, suportes, ROC-AUC, MCC, confiabilidade e risco/cobertura: `results.json`.
Scores heurísticos são decisões binárias, sem interpretação probabilística calibrada.
Log loss usa clipping em 1e-7. Esta análise não altera a seleção da v0.1.
