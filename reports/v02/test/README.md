# Avaliação do novo teste

Referência sintética revisada pela mesma IA. Sem revisão humana ou evidência populacional.

| Método | Macro-F1 | Exact match | Hamming ↓ | AP macro | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| heuristic | 0.6910 | 0.3056 | 0.1833 | 0.7441 | 0.1833 | 2.9550 |
| frozen_logistic | 0.6196 | 0.3056 | 0.1889 | 0.8401 | 0.1601 | 0.8540 |
| bert_finetuned | 0.5584 | 0.2500 | 0.2333 | 0.7768 | 0.1648 | 0.5053 |
| bert_finetuned_temperature | 0.5584 | 0.2500 | 0.2333 | 0.7768 | 0.1647 | 0.5051 |
| frozen_logistic_temperature | 0.6196 | 0.3056 | 0.1889 | 0.8401 | 0.1355 | 0.4292 |

Métricas por rótulo, suportes, ROC-AUC, MCC, confiabilidade e risco/cobertura: `results.json`.
Scores heurísticos são decisões binárias, sem interpretação probabilística calibrada.
Log loss usa clipping em 1e-7. Esta análise não altera a seleção da v0.1.

| Política experimental | Temperatura | Cutoff | Cobertura | Erro nas decisões aceitas |
|---|---:|---:|---:|---:|
| bert_finetuned | 0.9659 | desativado | 1.0000 | 0.23333333333333334 |
| frozen_logistic | 4.0000 | desativado | 1.0000 | 0.18888888888888888 |

Política ajustada somente em desenvolvimento; desempenho em teste não é garantia de risco.
Nenhum cutoff cumpriu a meta de erro/cobertura em desenvolvimento. O erro na tabela é sem abstenção, não o risco de uma política aprovada.
