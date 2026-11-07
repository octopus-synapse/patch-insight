# Contrato CLI/JSON 0.1.0

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
| `score` | Sigmoid não calibrada; regra binária no baseline; `null` para desconhecido |
| `structures` | Zero ou mais de `XYZ`, `STAR`, `Action–Result`, `Action–Technology` |
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

Estas combinações verificam coocorrência, sem validar alinhamento causal,
ordem narrativa, qualidade ou adequação universal da estrutura. `context`
aproxima Situation; Task só é detectada por padrões explícitos como “meu
objetivo era” e “fui encarregado de”. A ausência desses padrões permanece
desconhecida. Não há score de qualidade, ranking, API REST ou demo na v0.1.

O léxico usa limites de palavras, aliases como Postgres/K8s/sklearn e evita
Java dentro de JavaScript. Uma menção lexical não prova uso da tecnologia
pela pessoa; a estrutura Action–Technology é somente coocorrência.
