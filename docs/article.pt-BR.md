# PatchInsight: componentes textuais, dados sintéticos e um experimento auditável

O PatchInsight compara regras, embeddings fixos e fine-tuning completo de um
encoder em descrições profissionais em português. No teste sintético congelado,
o BERTimbau ajustado obteve macro-F1 de 0,9911 e a regressão sobre embeddings,
0,9888. A diferença foi pequena e seu intervalo por reamostragem inclui zero;
a experiência não sustenta uma alegação de superioridade robusta do fine-tuning.
Há código, dados, checkpoint e relatórios para auditar a execução.

## O problema

Uma descrição pode dizer o que foi feito, em que contexto, por qual meio, com
qual efeito e em que quantidade. Esses componentes podem aparecer juntos ou
separados. “Revisei as consultas” descreve uma ação; “a espera diminuiu” descreve
um efeito sem números. “OAuth 2.0” contém dígitos, mas não mede uma realização.

O projeto usa cinco rótulos independentes: action, context, method, result e
measurement. Reconhecer sua presença ajuda a descrever a construção textual.
Não permite concluir que a pessoa é mais competente ou que precisa seguir
uma fórmula narrativa. Por isso a interface não retorna notas nem ranking.

## Arquitetura e hipótese

O classificador é BERTimbau Base, encoder pré-treinado para português,
com uma cabeça de cinco saídas sigmoid. O treinamento atualiza todos os pesos
com perda binária por rótulo. A revisão do encoder é fixada e os parâmetros
constam de `configs/experiment.json`.

Dois controles tornam a hipótese comparável. O primeiro usa regex e um léxico
de tecnologias; o segundo extrai embeddings do encoder congelado e ajusta uma
regressão logística por rótulo. A hipótese é que o fine-tuning melhore macro-F1
contra as mesmas referências sintéticas. Um resultado negativo será igualmente
informativo e deve ser publicado.

As estruturas XYZ, STAR, Action–Result e Action–Technology são regras sobre
componentes e sinais auxiliares. STAR exige objetivo ou responsabilidade
explícitos; sem evidência, Task fica desconhecida. Coocorrência não demonstra
causalidade ou qualidade, e essas regras são avaliadas por fixtures documentadas.

## O que os dados permitem concluir

O conjunto tem 400 textos sintéticos, criados, anotados e revisados pela mesma
IA. Não houve revisão humana nem independência entre geração e revisão. Há
100 cenários originais e quatro variantes controladas por cenário: completa,
remoções de componentes e negação ou intenção futura. Essa construção permite
controlar rótulos, mas restringe a diversidade e pode criar atalhos.

Treino, validação e teste têm 280, 60 e 60 textos, respectivamente. Todas as
variantes de um cenário ficam no mesmo split. O teste foi criado em arquivo
separado antes dos experimentos, e seus textos e rótulos possuem hashes de
congelamento. O PDF privado do currículo não foi usado nem será publicado.

Prefixos como “Abordagem” e “Escopo” tornam parte da tarefa explícita. O léxico
e o estilo também se repetem entre splits. Portanto, mesmo um F1 elevado não
comprovaria generalização para currículos reais. A separação de grupos protege
contra derivações da mesma origem, sem garantir independência semântica.

## Resultado disponível: baseline de desenvolvimento

A execução do baseline heurístico em 60 exemplos de **validação**, com limiar
binário fixo, produziu:

| Métrica | Valor |
|---|---:|
| Macro-F1 | 0,8530 |
| Micro-F1 | 0,8623 |
| Exact match multilabel | 0,5333 |

| Rótulo | Precision | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| action | 1,0000 | 0,5313 | 0,6939 | 32 |
| context | 0,9677 | 0,8571 | 0,9091 | 35 |
| method | 1,0000 | 0,9063 | 0,9508 | 32 |
| result | 1,0000 | 0,6923 | 0,8182 | 26 |
| measurement | 1,0000 | 0,8065 | 0,8929 | 31 |

A evidência está em `reports/development/heuristic-validation.json`. São números
de desenvolvimento, sem comparação neural e sem avaliação do teste. O recall
de ação indica a fragilidade da cobertura lexical; as precisões elevadas
devem ser lidas junto à regularidade artificial dos dados.

## Comparação final e incerteza

Após encerrar a seleção pela validação, os três métodos foram avaliados nos
mesmos 60 textos de teste. O checkpoint escolhido foi o da época 5, com macro-F1
de validação 0,996923. Os limiares permaneceram em 0,5; não houve ajuste pelo teste.

| Método | Macro-F1 | Micro-F1 | Exact match |
|---|---:|---:|---:|
| Heurísticas | 0,6791 | 0,7634 | 0,3333 |
| Encoder congelado + regressão | 0,9888 | 0,9878 | 0,9333 |
| BERTimbau ajustado | 0,9911 | 0,9908 | 0,9500 |

A heurística não reconheceu nenhuma das ações positivas do teste, cujo vocabulário
não estava coberto por seus verbos. Esse resultado negativo foi preservado;
não foram ampliadas as regras para melhorar o teste. A baseline congelada ficou
próxima do fine-tuning, mostrando que o ajuste completo acrescentou pouco nesta
amostra. Os três erros binários do modelo ajustado foram falsos positivos de
contexto, método e resultado, um de cada.

A diferença de macro-F1 entre fine-tuning e regressão foi 0,002275. Um bootstrap
pareado pelas 15 origens, com 2.000 reamostragens e seed 42, produziu intervalo
percentil de 95% de −0,010818 a 0,014463. O intervalo descreve essa amostra
sintética, que ainda compartilha moldes; não é uma validação populacional de
currículos reais. Nem o ponto estimado nem os F1 elevados justificam extrapolar
para avaliação de profissionais.

As métricas por rótulo, contagens TP/FP/FN/TN e previsões por ID estão em
`reports/final/benchmark.json` e `predictions.jsonl`. A análise de erros está em
`reports/final/error-analysis.md`. Esses erros foram usados somente para relatar
limites, sem novo treino, ajuste de limiar ou geração de dados da v0.1.

## Reprodução

O treino foi executado em uma GPU local NVIDIA RTX A1000 de 6 GB, com Python
3.11.14 e dependências fixadas por uv.lock. O notebook chama os mesmos módulos
versionados para reprodução em Kaggle/Colab gratuitos; não se alega que a
execução publicada ocorreu nessas plataformas. Nenhuma computação paga foi
contratada. A impossibilidade inicial de rede e acesso ao hardware foi resolvida
antes dos experimentos reais.

Quinze testes passaram, incluindo contratos, regras, dados, atualização de pesos
em BERT minúsculo e carregamento do checkpoint real. O BERT minúsculo é apenas
um teste técnico e não entra no benchmark. O PyTorch emitiu aviso de atenção
backward não determinística; seeds e ambiente estão registrados, mas não se
promete identidade bit a bit entre retreinos. A evidência de reprodução em
ambiente separado e de container está em `docs/status.md` e `reports/`.

O projeto é independente e motivado por uma candidatura profissional. A
contribuição pretendida é demonstrar o ciclo experimental com evidência do
tamanho das afirmações, incluindo limitações e resultados negativos.

## Referências e licença

- Souza, Nogueira e Lotufo, [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
- [Guia e protocolo locais](annotation-guide.md), [experimentos](experiments.md), [Dataset Card](../data/README.md).
- Código: MIT. Este artigo e os dados: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Continuação experimental: avaliação 0.2

Os resultados anteriores pertencem à v0.1. Na continuação, os mesmos checkpoints
foram avaliados sem retreino em 36 textos novos, de 34 origens, com outro conjunto
de 36 textos usado somente em desenvolvimento. O próprio assistente escreveu,
anotou e revisou os dados, conforme solicitado pelo responsável; não houve
revisão humana, independência entre geração e revisão ou concordância medida.

No novo teste, macro-F1 foi 0,6910 para heurísticas, 0,6196 para regressão e
0,5584 para o BERT ajustado. A regressão teve AP macro de 0,8401, contra 0,7768
do BERT. Os testes têm composições distintas e ambos são sintéticos; a queda
não equivale a medir generalização em currículos reais. Ela mostra que o
resultado original próximo do teto era restrito ao seu desenho experimental.

A temperatura global foi ajustada somente no novo desenvolvimento e congelada
antes do teste. Na regressão, Brier caiu de 0,1601 para 0,1355 e log loss de
0,8540 para 0,4292, mantendo decisões no limiar 0,5. Isso melhora a avaliação
probabilística neste conjunto, sem demonstrar calibração externa. Nenhum cutoff
cumpriu erro observado de até 5% com pelo menos 50% de cobertura de decisões em
desenvolvimento; a abstenção permanece desativada nas políticas fornecidas.

A v0.2 também acrescenta inferência em lote, Action–Method e demonstração local
com checkpoint real e comparação por métricas. Protocolo, recortes, scores,
confiabilidade, incerteza e limitações: [avaliação 0.2](v02.md). Pesos, splits e
relatórios finais da v0.1 foram preservados; não houve ajuste orientado pelos erros do novo teste.
