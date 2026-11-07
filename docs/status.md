# Estado da v0.1 — 07/11/2026

Implementação, treino, avaliação, publicação dos artefatos e verificações locais
concluídos. O experimento usa referências sintéticas revisadas pela mesma IA;
não há validação humana ou evidência de generalização para currículos reais.

| Entregável | Evidência |
|---|---|
| Código e histórico | GitHub `octopus-synapse/patch-insight`; Conventional Commits em inglês, sem Co-authored-by |
| Dataset | 400 variantes de 100 origens, quatro lotes; 280/60/60; hashes em `data/freeze.json` |
| Guia e auditoria | `annotation-guide.md`, `data-review.md`, `reports/data-audit.json` |
| Heurísticas | Validação e teste final registrados |
| Encoder congelado + regressão | Executado, macro-F1 de teste 0,988825 |
| Fine-tuning completo | Cinco épocas, seleção da época 5 por validação; macro-F1 de teste 0,991101 |
| Checkpoint real | Hugging Face `enzoferracini/patchinsight-bertimbau-ptbr` |
| Dataset publicado | Hugging Face `enzoferracini/patchinsight-synthetic-ptbr` |
| Benchmark | `reports/final`: três métodos, suportes, contagens, previsões, erros e incerteza |
| CLI JSON | Pydantic, schema em `configs/output.schema.json`, componentes e estruturas versionados |
| Testes | 15 passaram em Python 3.11.14, incluindo modelo real; `reports/verification.json` |
| Lock | `uv.lock` resolvido; instalação com `uv sync --frozen` |
| Docker | Build e inferência sem rede passaram; `reports/docker.json` |
| Reprodução limpa | Clone SSH novo e nova virtualenv; treino, avaliação e carga real repetidos; `reports/clean-room.json` |
| Artigo e cards | `article.pt-BR.md`, Dataset Card e Model Card publicada com resultados reais |

## Desvios e limites

O ambiente inicialmente restringia rede, GPU, Docker e escrita em `.git`.
Depois da liberação, os impedimentos foram resolvidos. A execução ocorreu
na GPU local NVIDIA RTX A1000 6 GB, sem contratação de computação. O notebook
de reprodução para Kaggle/Colab foi preparado, mas não se alega execução nessas
plataformas. Os checkpoints estão também no Hub, fora da sessão de treino.

O teste tem apenas 15 origens e variantes dependentes. O ganho de macro-F1 do
fine-tuning frente à regressão foi 0,002275; intervalo bootstrap por origem
de 95%: [−0,010818; 0,014463]. A baseline congelada permanece competitiva.
Prefixos e moldes artificiais podem facilitar atalhos. Não há benchmark humano
das estruturas nem revisão humana dos dados.

O PyTorch emitiu aviso de atenção backward não determinística. Nesta máquina,
o retreino com a mesma configuração reproduziu inclusive o hash dos pesos;
isso não garante identidade em outro hardware. A reprodução reutilizou caches
de pacotes e downloads, mas criou clone e ambiente Python separados.

O Docker foi validado localmente; não foi enviado a um registry. O código
permite reconstruir a imagem com a revisão publicada. A documentação local
`MODEL_CARD.md` é um template para novos runs; a card efetiva está no Hub.
Demo visual, API REST e integração com Patch Careers permanecem fora do escopo.
