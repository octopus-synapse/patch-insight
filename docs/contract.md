# Contrato CLI/JSON 0.2.0

`patchinsight predict --text '...' --json` usa BERT e exige um bundle local
ajustado. `--backend heuristic` seleciona explicitamente o baseline de regras.
Não existe fallback silencioso quando o modelo não carrega.

Entrada: string Unicode normalizada em NFC, espaços condensados, preservando
maiúsculas e acentos. Vazio, controle não espacial e mais de 20.000 caracteres
geram erro. O encoder limita a 192 tokens, incluindo tokens especiais.

Saída: um objeto JSON em stdout, validado por Pydantic. Erros de execução e
entrada retornam código 2 e JSON em stderr; erros de argumentos usam o formato
padrão do argparse em stderr. Logs das bibliotecas também podem ir para stderr.

| Campo | Significado |
|---|---|
| `components` | Exatamente os cinco rótulos; cada um com `state` e `score` |
| `state` | `present`, `absent` ou `unknown` |
| `score` | Sigmoid não calibrada por padrão; temperatura experimental opt-in; regra binária no baseline; `null` para desconhecido |
| `structures` | Zero ou mais de `XYZ`, `STAR`, `Action–Result`, `Action–Technology`, `Action–Method` |
| `task` | `present` se há objetivo/responsabilidade explícito, senão `unknown` |
| `technologies` | Nomes canônicos encontrados no léxico, sem extração de spans |
| `diagnostics` | Componentes ausentes/desconhecidos, `task_unknown`, `unclassified`, truncamento |
| `truncated` / `tokens` | Corte pelo tokenizer; tokens totais antes do corte; `null` no baseline |
| `versions` | `schema_version`, `model_version`, `rules_version` |
| `backend` / `score_kind` | Método efetivo e interpretação de seus valores |

Se houver truncamento, uma previsão negativa no prefixo fica `unknown` com
score `null`. Predições positivas referem-se ao trecho observado. A detecção
lexical de tecnologias e objetivo lê o texto normalizado completo; os
componentes semânticos do BERT leem o prefixo. Essa diferença pode limitar
a interpretação das combinações em entradas longas.

Regras, sempre exigindo componentes `present`:

| Estrutura | Componentes e auxiliares |
|---|---|
| XYZ | action + result + measurement + method |
| STAR | context + action + result + objetivo/responsabilidade explícito |
| Action–Result | action + result |
| Action–Technology | action + tecnologia do léxico |
| Action–Method | action + method |

Estas combinações verificam coocorrência, sem validar alinhamento causal,
ordem narrativa, qualidade ou adequação universal da estrutura. `context`
aproxima Situation; Task só é detectada por padrões explícitos como “meu
objetivo era” e “fui encarregado de”. A ausência desses padrões permanece
desconhecida. Não há score de qualidade ou ranking de pessoas. Na v0.2, a demo e o endpoint
HTTP são locais e restritos a loopback; não há API publicada.

O léxico usa limites de palavras, aliases como Postgres/K8s/sklearn e evita
Java dentro de JavaScript. Uma menção lexical não prova uso da tecnologia
pela pessoa; a estrutura Action–Technology é somente coocorrência.

## Extensões da v0.2

`versions.schema_version` e `rules_version` são 0.2.0; model_version continua a
identificar o checkpoint efetivo, inclusive 0.1.0 para os pesos preservados.
`score_kind` aceita binary_rule, sigmoid_uncalibrated e
temperature_scaled_experimental. A política opt-in exige método BERT e hash de
manifest do mesmo bundle; sua identificação aparece em diagnostics.

Quando uma política de confiança válida encaminhar componente para revisão,
state é unknown e score é null; nenhum match de estrutura pode usar unknown.
A política fornecida nesta rodada não atingiu a meta de desenvolvimento e tem
cutoff null, sem abstenção por confiança. Truncamento continua independente.

`batch` lê JSONL com text e id opcional; retorna envelope por linha com line,
id e prediction ou error. Código 1 indica erro por item; 2 erro global. O
carregamento ocorre uma vez e as previsões válidas são agrupadas. Limites,
timeout cooperativo e formato HTTP: [v02.md](v02.md).

A inferência em lote usa o mesmo contrato de componentes da inferência única.
O timeout entre lotes não interrompe inferência ativa ou stdin bloqueado.
