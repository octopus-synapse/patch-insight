> Protocolo histórico da v0.1. A avaliação exploratória posterior, sem retreino,
> tem protocolo próprio em [data/v02/README.md](../data/v02/README.md) e resultados
> em [v02.md](v02.md). A métrica primária e a seleção originais não foram alteradas.

# Protocolo experimental 0.1

Pergunta: fine-tuning completo de BERTimbau melhora a classificação de cinco
componentes em relação a regras e embeddings fixos, contra referências
sintéticas revisadas por IA?

## Dados e isolamento

Congelamento em `data/freeze.json`; 280/60/60 registros e 70/15/15 cenários.
Cada cenário é uma origem com quatro variantes correlacionadas. Separação por
origem impede derivados literais em splits diferentes, mas não elimina moldes
compartilhados. Ver `reports/data-audit.json` e `docs/data-review.md`.

Só treino e validação orientam configuração e complementação. O módulo
`experiment` não abre o teste; a verificação dos hashes lê bytes de integridade,
sem usar textos ou rótulos do teste no treinamento. `evaluate --final` é o
único fluxo de métricas de teste. Após o primeiro benchmark, os erros não
podem orientar a v0.1; uma nova pesquisa exige outro teste e outra versão.

## Métodos fixados

| Método | Representação | Aprendizado | Decisão |
|---|---|---|---|
| Heurístico | Regex e léxico curado | Nenhum | Regras binárias |
| Congelado + regressão | Média dos estados do encoder com máscara | StandardScaler e uma LogisticRegression por rótulo | 0,5 |
| Fine-tuned | BERTimbau Base + cabeça multilabel | Todos os pesos, BCE com logits | 0,5 |

O scaler e a regressão são ajustados apenas em treino. C=1 e max_iter=2000;
não há busca de hiperparâmetros. Encoder congelado tem `requires_grad=False`
e modo eval. Pooling usa média mascarada, incluindo tokens especiais, fixado
para todas as execuções da baseline. Regressão serializada em JSON com pesos,
interceptos, média e escala; não é necessário carregar pickle.

Fine-tuning: seed 42 antes de inicializar a cabeça; até cinco épocas, AdamW
lr 2e-5 e weight decay 0,01; batch 8; acumulação 2; clipping 1; max_length 192.
O melhor macro-F1 de validação determina o checkpoint; empate mantém a época
mais antiga. Não há tuning de limiar nesta versão. Menor batch por falta de
memória deve ser registrado em uma nova configuração antes da avaliação final.

Reprodutibilidade não garante identidade bit a bit entre GPUs e versões de
drivers. Seeds, determinismo solicitado, ambiente, revisão e hashes devem
acompanhar os resultados; avisos de operações não determinísticas permanecem
visíveis. Preservar artefatos fora da sessão Kaggle/Colab ao concluir o treino.

## Avaliação

Métrica primária: macro-F1. Secundárias: micro-F1 e exact match multilabel.
Por rótulo: precision, recall, F1, suporte positivo, TP, FP, FN e TN.
Divisão por zero resulta em zero. Os três métodos compartilham a implementação
`metrics.py` e o mesmo teste. Guardar previsões por ID para auditar erros.

O JSON final registra hashes dos bundles e dos dados, ambiente e limiares.
Publicar os três métodos mesmo quando fine-tuning perde. Não atribuir melhora
a pessoas reais ou a recrutamento; o alvo é consistência com anotações
sintéticas dependentes do gerador e do guia.

## Evidência disponível

`reports/development/heuristic-validation.json` contém a execução real do
baseline na validação: macro-F1 0,852965, micro-F1 0,862319, exact match
0,533333, n=60. Essa execução não exigiu bibliotecas de ML. Nenhuma alteração
de regra foi feita com base em erros do teste; o teste não foi pontuado.

A baseline congelada e o fine-tuning foram executados na GPU local NVIDIA RTX
A1000 6 GB, com Python 3.11.14. Seleção: época 5, macro-F1 de validação 0,996923.
O teste final está em `reports/final/benchmark.json`; modelos ajustado e congelado
obtiveram macro-F1 0,991101 e 0,988825. O bootstrap pareado por origem inclui zero
no intervalo de ganho; ver `reports/final/uncertainty.json`.

O retreino de reprodução usa a mesma configuração, sem decisões adicionais
baseadas no teste; é uma verificação do pipeline, não um novo candidato. Logs
de treinamento, ambiente e aviso de atenção não determinística estão em
`reports/training`. A identidade bit a bit dos pesos entre retreinos não é
prometida. Testes com BERT minúsculo validam somente o fluxo técnico e não
entram no benchmark. O teste de checkpoint real passou após o ajuste completo.
