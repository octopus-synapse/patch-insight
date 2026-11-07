# Publicação e progressão

Código: https://github.com/octopus-synapse/patch-insight . Commits em inglês,
Conventional Commits, sem trailer Co-authored-by. Nenhum push é considerado
concluído sem conferir o SHA e os arquivos remotos. O PDF fica fora da publicação.

No Hugging Face, há dois repositórios de artefatos. Isso permite manter pesos e
dados com seus respectivos históricos e licenças, enquanto o código permanece
em um único GitHub. Criar os espaços agora é útil para documentar a progressão;
uma página sem pesos deve dizer claramente “treinamento pendente”.

1. Acessar https://huggingface.co/new com login. Proprietário: seu usuário;
   nome sugerido `patchinsight-bertimbau-ptbr`; licença MIT; Public.
2. Acessar https://huggingface.co/new-dataset. Nome sugerido
   `patchinsight-synthetic-ptbr`; licença CC BY 4.0; Public.
3. Não criar Space nem contratar computação. O plano treina em Kaggle gratuito,
   com Colab gratuito como reserva.

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

Com Hugging Face autenticado localmente, o SDK pode enviar a pasta completa:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(repo_id="SEU_USUARIO/patchinsight-synthetic-ptbr",
                  repo_type="dataset", folder_path="data",
                  commit_message="feat(data): publish frozen synthetic dataset")
api.upload_folder(repo_id="SEU_USUARIO/patchinsight-bertimbau-ptbr",
                  repo_type="model", folder_path="artifacts/run-42/model",
                  commit_message="feat(model): publish evaluated multilabel checkpoint")
```

Não colar tokens no chat nem versioná-los. Conferir os SHAs retornados pelo Hub,
baixar a revisão fixa em um ambiente limpo e testar o carregamento real.
Atualizar README, artigo e links somente com os destinos efetivos.
