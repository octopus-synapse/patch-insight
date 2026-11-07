---
language:
- pt
license: cc-by-4.0
task_categories:
- text-classification
tags:
- synthetic
- multilabel
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: splits/train.jsonl
  - split: validation
    path: splits/validation.jsonl
  - split: test
    path: splits/test.jsonl
---

# PatchInsight Synthetic pt-BR 0.1

400 exemplos sintéticos de descrições profissionais em software, dados e IA,
anotados e revisados pela mesma IA. **Não há revisão humana nem independência
entre geração e revisão.** O PDF de currículo não foi usado na geração.

| Split | Exemplos | Grupos | action | context | method | result | measurement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Treino | 280 | 70 | 150 | 176 | 153 | 140 | 140 |
| Validação | 60 | 15 | 32 | 35 | 32 | 26 | 31 |
| Teste | 60 | 15 | 31 | 36 | 33 | 31 | 31 |

Valores dos rótulos são suportes positivos. Cada grupo representa um cenário
com quatro variantes composicionais correlacionadas; o tamanho efetivo é menor
que 400 observações independentes. Guia: `../docs/annotation-guide.md`.

`dataset.jsonl` reúne todos os registros; `splits/*.jsonl` fixa os subconjuntos.
Campos: `id`, `text`, `labels`, `group_id`, `origin_id`, `batch`, `pilot`,
`split`, `source_ref`, `transformation`, `prompt_ref`, `review_prompt_ref`,
`generator`, `reviewer`, `review_history`, `guide_version`, `dataset_version`,
`synthetic`, `human_reviewed`, `language`, `domain` e `license`.

`source/prompts.json` registra a especificação efetivamente usada para autoria
dos cenários. O nome exato do modelo gerador não foi exposto de forma verificável.
`scripts/build_dataset.py` documenta a transformação composicional determinística;
o texto originalmente escrito não depende de uma API para ser reproduzido.
Nenhum exemplo foi copiado de currículos, do PDF local ou de datasets externos.

O teste possui origens próprias em `source/test.tsv`. `freeze.json` contém os
hashes dos textos, anotações, fontes e splits anteriores aos experimentos.
O treino verifica esses hashes, mas só utiliza treino e validação. Não usar
erros do teste para melhorar dados, limiares ou seleção da mesma versão.

Limites: prefixos explícitos e léxico repetido facilitam atalhos; português
formal e afirmações curtas predominam; não há evidência de generalização para
currículos reais, outras profissões, dialetos ou narrativas longas. Os rótulos
descrevem componentes textuais e não qualidade ou competência profissional.
As frases são inventadas e não verificam impacto de pessoas ou organizações.

Uso: pesquisa de classificação multilabel. Não usar para seleção automática,
ranking de candidatos ou inferência de habilidade. Medidas de desempenho são
contra referências sintéticas revisadas por IA.

Licença: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), atribuição
“PatchInsight contributors — PatchInsight Synthetic pt-BR 0.1; dados sintéticos
gerados, anotados e revisados por IA”. Direitos concedidos na extensão aplicável
ao material. A elegibilidade autoral de conteúdo de IA pode variar por jurisdição.
