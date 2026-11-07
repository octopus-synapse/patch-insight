# Erros e interpretação da avaliação 0.2

Análise posterior ao teste; não orientou correções, regras, limiares ou retreino nesta versão.

| Método | Componente | FP | FN | Suporte positivo |
|---|---|---:|---:|---:|
| heuristic | action | 0 | 16 | 20 |
| heuristic | context | 1 | 3 | 14 |
| heuristic | method | 0 | 4 | 14 |
| heuristic | result | 0 | 7 | 11 |
| heuristic | measurement | 0 | 2 | 12 |
| frozen_logistic | action | 1 | 0 | 20 |
| frozen_logistic | context | 2 | 3 | 14 |
| frozen_logistic | method | 2 | 11 | 14 |
| frozen_logistic | result | 0 | 5 | 11 |
| frozen_logistic | measurement | 0 | 10 | 12 |
| bert_finetuned | action | 1 | 2 | 20 |
| bert_finetuned | context | 3 | 5 | 14 |
| bert_finetuned | method | 0 | 11 | 14 |
| bert_finetuned | result | 2 | 7 | 11 |
| bert_finetuned | measurement | 2 | 9 | 12 |

## Exemplos de erros do BERT

Amostra determinística: primeiros 12 textos com erro na ordem dos IDs. Referência revisada pela mesma IA, sem revisão humana.

- `v02-test-002` — method: FN. Auditei as permissões por meio de uma lista de verificação.
  Interpretação de referência registrada antes dos scores: Procedimento explícito; auditoria não afirma correção.
- `v02-test-004` — context: FN; method: FN. Durante a mudança de sede, examinei a conectividade usando captura de tráfego.
  Interpretação de referência registrada antes dos scores: Circunstância e meio, sem efeito declarado.
- `v02-test-005` — result: FN. Entreguei o roteiro de recuperação.
  Interpretação de referência registrada antes dos scores: Entrega concreta sem números.
- `v02-test-006` — method: FN; result: FN. Disponibilizei a biblioteca usando Docker; o pacote ficou acessível.
  Interpretação de referência registrada antes dos scores: Ação, método e disponibilidade explícitos.
- `v02-test-007` — context: FN; method: FN; result: FN; measurement: FN. No processo de conciliação, ajustei as consultas com PostgreSQL e o tempo caiu 22%.
  Interpretação de referência registrada antes dos scores: Cenário, atividade, ferramenta, efeito e medida.
- `v02-test-008` — measurement: FN. Na equipe de entregas, reorganizei os turnos. O atraso caiu de 30 para 10 minutos.
  Interpretação de referência registrada antes dos scores: Mudança e medida explícitas, sem procedimento separado.
- `v02-test-010` — measurement: FP. Em 2024. Ocorrência OPS-719.
  Interpretação de referência registrada antes dos scores: Data e identificador não medem realização.
- `v02-test-012` — context: FN; measurement: FN. Tenho seis anos de experiência no setor de pagamentos.
  Interpretação de referência registrada antes dos scores: Duração e setor; experiência genérica não é contribuição concreta.
- `v02-test-013` — context: FP; measurement: FP. Perfil colaborativo com foco em excelência e alta performance.
  Interpretação de referência registrada antes dos scores: Qualidades alegadas não são componentes de uma realização.
- `v02-test-015` — method: FN; measurement: FN. Planejo reduzir os custos em 17% por meio de compressão.
  Interpretação de referência registrada antes dos scores: Futuro preserva método e medida, sem efeito realizado.
- `v02-test-016` — action: FN; result: FN. Não migrei os arquivos; entreguei o plano de contingência.
  Interpretação de referência registrada antes dos scores: Negação restrita à primeira oração; segunda afirma entrega.
- `v02-test-017` — method: FN. Comparação de checksums como procedimento.
  Interpretação de referência registrada antes dos scores: Abordagem nominal explícita, sem contribuição afirmada.

## Sensibilidade a marcadores

Dois pares de origens, cada um com versão marcada e livre, preservam os rótulos. Isso é uma sondagem pequena, não uma estimativa populacional.

| Método | Origem | Componentes que mudaram de decisão | Mudança absoluta média do score |
|---|---|---:|---:|
| heuristic | v02-test-probe-1 | 1 | 0.2000 |
| heuristic | v02-test-probe-2 | 1 | 0.2000 |
| frozen_logistic | v02-test-probe-1 | 2 | 0.3921 |
| frozen_logistic | v02-test-probe-2 | 2 | 0.4020 |
| bert_finetuned | v02-test-probe-1 | 3 | 0.3497 |
| bert_finetuned | v02-test-probe-2 | 3 | 0.3063 |

## Leitura dos resultados

O BERT e a regressão acertam action com frequência maior que a heurística, mas perdem cobertura em method e measurement. A vantagem global das heurísticas nesta amostra não significa que elas sejam superiores em qualquer domínio.

Os pares sem marcadores mudam duas decisões por origem na regressão e três no BERT, embora os rótulos de referência permaneçam iguais. O comportamento é compatível com dependência de sinais de superfície; dois pares não isolam uma explicação causal.

A temperatura da regressão reduziu Brier/log loss, mantendo F1. A temperatura do BERT alterou pouco os scores. Nenhum cutoff de confiança cumpriu a meta de desenvolvimento; as políticas salvam abstenção desativada. Não selecionar um cutoff pelos erros deste teste.

Recortes podem não ter positivos em todos os rótulos: macro-F1 com zero_division=0 pode ser zero apesar de todas as decisões negativas estarem corretas. Usar suportes, Hamming e FP/FN na leitura desses recortes.

As fontes de desenvolvimento e teste têm categorias e algumas construções paralelas; a autoria/revisão única e a pequena amostra limitam independência. Não são textos reais, validação humana, avaliação de spans ou benchmark de estruturas. O teste original continua preservado.
