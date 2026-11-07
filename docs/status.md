# Refinamento local 0.3

Checkpoint treinado e selecionado; meta de melhoria atingida no novo teste.
Modelo local padrão atualizado, versões anteriores preservadas. Os novos pesos
ainda não foram publicados. [Resultados e limitações](v03.md).

# Publicação da 0.2 — 07/11/2026

Release GitHub v0.2.0 publicada; workflow de release concluído com sucesso.
Imagem GHCR pública com acesso anônimo verificado. Model e dataset no Hugging
Face têm tags v0.2.0; os pesos continuam 0.1.0. Revisões e digest constam em
`reports/releases/`. O registro local abaixo é histórico da preparação.

A automação publica por tag e pode retomar uma publicação do Hub já verificada.
Futuras tags inéditas exigem cadastrar o secret HF_TOKEN restrito aos dois repos.
A autenticação OAuth local não foi copiada para secrets.

---

# Estado da v0.2 — 07/11/2026

Implementada e verificada localmente, sem retreino ou republicação de checkpoint.
Detalhes, resultados e comandos: [v02.md](v02.md). Revisão dos novos dados feita
pelo próprio assistente IA, conforme solicitado, sem revisão humana independente.

- 72 textos novos congelados: 36 desenvolvimento e 36 teste, 34 origens por split.
- Macro-F1 no novo teste: heurística 0,6910; regressão 0,6196; BERT 0,5584.
- Avaliação por F1, AP, ROC-AUC, MCC, Hamming, exact match, Brier e log loss;
  confiabilidade, recortes, risco/cobertura e bootstrap por origem.
- Temperatura ajustada só em desenvolvimento; melhora probabilística da regressão.
  Nenhum cutoff atingiu o critério de abstenção; confiança não vira promessa de risco.
- Inferência JSONL em lote, Action–Method e demo local com checkpoint real;
  comparação interativa por métrica e acesso ao JSON versionado.
- Pesos e dados/relatórios finais da v0.1 preservados; software/schema/regras 0.2.0.
- Sem publicação externa da v0.2, integração ao Patch Careers ou computação paga.

A evidência de verificação desta rodada está em `reports/v02/verification.json`.
O registro histórico abaixo se refere exclusivamente à v0.1.

---

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
