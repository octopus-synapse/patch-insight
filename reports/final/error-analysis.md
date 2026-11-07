# Análise final de erros e limites

Análise descritiva posterior à avaliação; não orientou ajustes da v0.1.

| Método | Rótulo | FP | FN | Suporte |
|---|---|---:|---:|---:|
| heuristic | action | 0 | 31 | 31 |
| heuristic | context | 0 | 12 | 36 |
| heuristic | method | 0 | 19 | 33 |
| heuristic | result | 0 | 0 | 31 |
| heuristic | measurement | 0 | 0 | 31 |
| frozen_logistic | action | 0 | 0 | 31 |
| frozen_logistic | context | 3 | 0 | 36 |
| frozen_logistic | method | 0 | 0 | 33 |
| frozen_logistic | result | 1 | 0 | 31 |
| frozen_logistic | measurement | 0 | 0 | 31 |
| bert_finetuned | action | 0 | 0 | 31 |
| bert_finetuned | context | 1 | 0 | 36 |
| bert_finetuned | method | 1 | 0 | 33 |
| bert_finetuned | result | 1 | 0 | 31 |
| bert_finetuned | measurement | 0 | 0 | 31 |

## Erros do BERTimbau ajustado

- `pi-0350`: result: esperado 0, previsto 1. Texto sintético: No sistema de chat o histórico desaparecia ao reconectar. Avaliei a sincronização das conversas.
- `pi-0386`: context: esperado 0, previsto 1. Texto sintético: Inspecionei a identificação dos checkpoints. Abordagem: hashes de artefatos no Hugging Face. Cada checkpoint passou a ter uma identidade verificável.
- `pi-0390`: method: esperado 0, previsto 1. Texto sintético: Em um portal de pedidos as abas exibiam contagens divergentes. Examinei o compartilhamento dos dados da tela. As abas passaram a mostrar a mesma contagem de pedidos.

## Incerteza e alcance

O ganho observado de macro-F1 foi 0.002275. Bootstrap pareado por origem (15 grupos, 2.000 reamostragens, seed 42) produziu intervalo percentil de 95% [-0.010818, 0.014463]. Esse procedimento descreve a amostra sintética e não valida generalização para currículos reais.

Os textos compartilham moldes e marcadores explícitos. A ausência de revisão humana e a dependência entre variantes impedem extrapolar estes números para avaliação de pessoas. A baseline congelada permanece competitiva. Não houve novo treino, ajuste de limiares ou geração de dados a partir destes erros.
