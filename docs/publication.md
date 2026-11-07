# Publicação e progressão

Código: https://github.com/octopus-synapse/patch-insight . Commits em inglês,
Conventional Commits, sem trailer Co-authored-by. Nenhum push é considerado
concluído sem conferir o SHA e os arquivos remotos. O PDF fica fora da publicação.

Artefatos publicados e revisões conferidas em `reports/publication.json`:
dataset `enzoferracini/patchinsight-synthetic-ptbr` e modelo
`enzoferracini/patchinsight-bertimbau-ptbr`. O histórico preserva a Model Card
inicial sem pesos e o commit posterior com o checkpoint avaliado.

No Hugging Face, há dois repositórios de artefatos. Isso permite manter pesos e
dados com seus respectivos históricos e licenças, enquanto o código permanece
em um único GitHub. Criar os espaços agora é útil para documentar a progressão;
uma página sem pesos deve dizer claramente “treinamento pendente”.

1. Acessar https://huggingface.co/new com login. Proprietário: seu usuário;
   nome sugerido `patchinsight-bertimbau-ptbr`; licença MIT; Public.
2. Acessar https://huggingface.co/new-dataset. Nome sugerido
   `patchinsight-synthetic-ptbr`; licença CC BY 4.0; Public.
3. Não criar Space nem contratar computação. A política vigente treina localmente;
   Kaggle e Colab gratuitos são reservas.

O Hub permite iniciar com README/Model Card e adicionar arquivos em commits
posteriores. Guia: https://huggingface.co/docs/hub/repositories-getting-started .
Não é necessário publicar pesos inexistentes para reservar o nome.

## Antes de publicar o dataset

Conferir `data/freeze.json`, auditoria, guia, declaração de revisão pela mesma IA
e licença. Copiar a pasta `data` como artefato do dataset. A Dataset Card usa
links relativos ao repositório de código; ao publicar no Hub, substituir
esses links por URLs do GitHub e incluir uma cópia do guia e da revisão.
Os JSONL contêm os splits, IDs e proveniência; não enviar o PDF.

## Antes de publicar o modelo

Executar treino real, teste `real_model`, benchmark dos três métodos, inferência
da imagem sem rede e reprodução limpa em Python 3.11 com lock. Atualizar a
Model Card com dataset e revisões reais, métricas finais, ambiente e links.
Remover o status de preparação somente quando houver evidência.

O diretório ajustado contém `model.safetensors`, configuração, tokenizer,
`patchinsight.json`, `manifest.json`, licença MIT e aviso NeuralMind. Não enviar
somente os pesos do encoder base como se fossem um modelo treinado da tarefa.
Ao modificar a Model Card dentro do bundle, regenerar seu manifest.

Com Hugging Face autenticado localmente, preparar e enviar uma nova publicação:

```bash
hf auth whoami
python3 scripts/prepare_hub.py --namespace enzoferracini --stage trained \
  --output artifacts/hub-next
hf upload enzoferracini/patchinsight-synthetic-ptbr artifacts/hub-next/dataset . \
  --type dataset --commit-message 'feat(data): publish frozen synthetic dataset'
hf upload enzoferracini/patchinsight-bertimbau-ptbr artifacts/hub-next/model . \
  --type model --commit-message 'feat(model): publish evaluated multilabel checkpoint'
```

Não colar tokens no chat nem versioná-los. Conferir os SHAs retornados pelo Hub,
baixar a revisão fixa em um ambiente limpo e testar o carregamento real.
Atualizar README, artigo e links somente com os destinos efetivos.

## Estado local da v0.2

A versão 0.2.0 do software e seus novos relatórios estão preparados localmente.
Não foram enviados ao GitHub/Hub nem transformados em release pública nesta
rodada. O checkpoint avaliado continua o da v0.1. Uma publicação posterior deve
preservar essa distinção, os hashes históricos e a autoria/revisão por IA da
nova suite. A demo é local, sem hospedagem externa ou Space.
