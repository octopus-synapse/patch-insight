# Revisão dos dados sintéticos

As cláusulas em `data/source/*.tsv` foram escritas e inspecionadas pelo mesmo
assistente IA desta implementação. As variantes são renderizadas por código;
os rótulos acompanham as cláusulas incluídas. Não houve anotação independente,
votação entre modelos ou revisão humana. O identificador exato do modelo gerador
não está disponível de forma verificável; não é inferido a partir da interface.

| Lote | Registros | Origens | Verificação |
|---|---:|---:|---|
| 1 | 100 | 25 | Piloto de 50; ação de processo versus efeito, método nominal, medida de escopo |
| 2 | 100 | 25 | Negação, intenção futura, efeitos sem números, separação de origem |
| 3 | 100 | 25 | Amplitude de software/dados/IA, cobertura dos rótulos |
| 4 | 100 | 25 | 40 exemplos de desenvolvimento e 60 de teste de origens separadas |

O teste foi escrito em arquivo separado antes de rodar qualquer baseline. Não
foi composto a partir de erros do modelo. Cada cenário possui uma variante
completa, duas remoções de componentes e uma ação negada ou intenção futura.
Os textos de saída foram inspecionados em conjunto com os testes das regras de
composição; o histórico não representa uma avaliação cega texto a texto.

A auditoria é reproduzível por `python -m patchinsight.data`. Há 26 combinações
de rótulos, 100 grupos, zero duplicatas exatas e nenhum grupo entre splits.
O algoritmo de similaridade em desenvolvimento (SequenceMatcher ≥ 0,85)
sinalizou `pi-0216` e `pi-0312`: “Não cheguei a analisar os passos da instalação”
e “Não cheguei a analisar os passos da rotina diária”. Foram mantidos: os
objetos são diferentes, mas o molde é o mesmo. Esse caso evidencia a limitação
de diversidade da geração composicional; não constitui prova de independência.

Revisão semântica: as ações de análise/revisão/investigação não afirmam entrega;
os efeitos impessoais afirmam mudança identificável; a cláusula “Escopo” mede
volume sem presumir resultado. As variantes negativas preservam método,
contexto ou medida quando esses componentes são explicitamente incluídos.
“OAuth 2.0” aparece como tecnologia, não como medida; quantidades adjacentes
continuam válidas. A distinção está coberta por fixtures externas ao dataset.

Esta versão não contém narrativas livres suficientes para estimar desempenho
em currículos reais. Prefixos como “Abordagem” e “Escopo” podem tornar a tarefa
artificialmente fácil. Há vocabulário compartilhado entre splits, mesmo sem
origens compartilhadas. Esses atalhos devem ser relatados junto às métricas.
