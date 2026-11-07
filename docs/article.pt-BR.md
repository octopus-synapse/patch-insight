# PatchInsight: componentes textuais, dados sintéticos e um experimento auditável

**Artigo de desenvolvimento — comparação final pendente.** O PatchInsight
investiga classificação de componentes de descrições profissionais em português.
Até aqui, há dados congelados, pipelines implementados e uma execução real do
baseline heurístico na validação. Ainda não há evidência de fine-tuning ou
superioridade de um modelo neural.

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

O classificador proposto é BERTimbau Base, encoder pré-treinado para português,
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

## Reprodução e trabalho pendente

Os módulos de treino local e notebook são os mesmos. A seleção usa macro-F1
de validação; os limiares permanecem em 0,5. O comando final compara os três
métodos e salva contagens de erros por rótulo e previsões por ID. Não usar os
erros de teste para ajustar essa versão.

Nesta implementação, o terminal não conseguiu resolver DNS, não tinha as
dependências de ML nem Python 3.11, a GPU não respondeu e o daemon Docker
estava inacessível. Assim, o lock completo, o treino, o checkpoint, o benchmark
final e a reprodução limpa continuam pendentes. Dez testes do núcleo passaram
com Python 3.13.5; isso não substitui a validação da stack completa.

O projeto é independente e motivado por uma candidatura profissional. A
contribuição pretendida é demonstrar o ciclo experimental com evidência do
tamanho das afirmações, incluindo limitações e resultados negativos.

## Referências e licença

- Souza, Nogueira e Lotufo, [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased).
- [Guia e protocolo locais](annotation-guide.md), [experimentos](experiments.md), [Dataset Card](../data/README.md).
- Código: MIT. Este artigo e os dados: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
