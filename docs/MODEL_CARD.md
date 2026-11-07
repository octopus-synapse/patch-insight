---
language:
- pt
license: mit
base_model: neuralmind/bert-base-portuguese-cased
pipeline_tag: text-classification
tags:
- multilabel-classification
- synthetic-data
---

# PatchInsight 0.1 — Model Card em preparação

**Status: template de Model Card usado antes de preparar uma publicação.**
A card com os resultados reais está no [modelo publicado](https://huggingface.co/enzoferracini/patchinsight-bertimbau-ptbr).
O treinamento copia este template para cada novo bundle; `prepare_hub.py`
preenche os resultados do respectivo run antes da publicação.

O modelo proposto é BERTimbau Base com ajuste completo dos pesos e cabeça de
cinco saídas sigmoid na ordem `action`, `context`, `method`, `result`,
`measurement`. Modelo base: `neuralmind/bert-base-portuguese-cased`, revisão
`94d69c95f98f7d5b2a8700c420230ae10def0baa`. A Model Card do base declara MIT;
o aviso original está em `licenses/BERTIMBAU-MIT.txt`.

Treino previsto: seed 42, até cinco épocas, AdamW, learning rate 2e-5, máximo
de 192 tokens, batch 8 e acumulação 2. Seleção por macro-F1 de validação;
empate mantém a época mais antiga. Limiar 0,5 para todos os rótulos, sem ajuste
usando teste. `patchinsight.json` registra a configuração efetiva, a época
selecionada, hashes dos dados e ambiente. Sigmoid não representa probabilidade
calibrada nem qualidade do texto.

Dados: PatchInsight Synthetic pt-BR 0.1, 280/60/60 variantes, 70/15/15 cenários.
São exemplos sintéticos anotados e revisados pela mesma IA, sem revisão humana.
O teste é fixado antes dos experimentos e fica fora de seleção e ajuste.

Avaliação pendente: heurísticas, encoder congelado com regressão logística por
rótulo, BERTimbau ajustado, no mesmo teste. Relatar macro/micro-F1, exact match,
precision/recall/F1, suporte e TP/FP/FN/TN de cada rótulo. Nenhuma métrica de
fine-tuning foi produzida. Publicar resultado negativo se não superar baselines.

Inferência: `patchinsight predict --model ./model --text '...' --json`.
O bundle é carregado localmente com hashes verificados. O retorno sinaliza
truncamento; componentes negativos no prefixo truncado viram `unknown` para
evitar afirmar ausência no texto completo. O score desses componentes é `null`.
O modelo não extrai spans nem comprova causalidade entre ação e efeito.

Limitações: dados pequenos, composicionais, com atalhos lexicais e estilísticos;
domínio concentrado em software/dados/IA; avaliação sem anotação humana;
generalização externa desconhecida. Estruturas são regras de coocorrência e
fixtures, não um benchmark humano de XYZ ou STAR.

Uso pretendido: experimento aberto de classificação de componentes textuais.
Não usar para ranking de pessoas, contratação automática ou atribuir qualidade
universal a textos profissionais. O projeto é autoral, independente, motivado
por candidatura; não é patrocinado, supervisionado ou endossado por empresa.

Referência do encoder: Souza, Nogueira e Lotufo (2020), *BERTimbau: Pretrained
BERT Models for Brazilian Portuguese*, BRACIS.
