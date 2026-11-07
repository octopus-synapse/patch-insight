# Estado da implementação — 07/11/2026

Atualização: rede e acesso à GPU/Docker liberados; Python 3.11 e lock resolvidos.
Dataset e Model Card inicial publicados; 14 testes aprovados, um de checkpoint
real aguardando o fine-tuning em execução. A tabela abaixo registra a etapa
anterior e será substituída pelas evidências finais após a execução.

**A v0.1 ainda não está pronta para release.** Há implementação e evidência
local do núcleo; faltam treinamento real, benchmark completo, lock resolvido,
publicação dos pesos e validação de inferência em container.

| Entregável | Estado | Evidência ou próximo passo |
|---|---|---|
| Guia e contrato | Implementados | `annotation-guide.md`, `contract.md`, Pydantic |
| Dataset | 400 textos congelados | 100 origens, 4 lotes, 280/60/60, `data/freeze.json` |
| Auditoria | Executada | `reports/data-audit.json`, revisão pela mesma IA |
| Heurísticas | Validação executada | Macro-F1 0,852965; teste intocado |
| Encoder congelado + regressão | Código implementado, não executado | Falta stack de ML e download do encoder |
| Fine-tuning | Código e notebook implementados, não executados | Falta ambiente Kaggle/Colab com modelo base |
| Checkpoint no Hugging Face | Pendente | Precisa de treino, avaliação e destino |
| Benchmark final | Pendente | Os três métodos devem estar prontos antes de `--final` |
| CLI JSON | Implementada, testes de contrato não executados | Falta instalar Pydantic |
| Regras estruturais | Fixtures passaram | Não há benchmark humano de estruturas |
| Testes do núcleo | 10 passaram | Python 3.13.5; `reports/verification.json` |
| Testes ML e modelo real | Implementados, não executados | Dependências/modelo ausentes |
| `uv.lock` | Pendente | Sem uv e sem DNS para resolver dependências |
| Docker | Dockerfile e smoke test implementados | Daemon inacessível; precisa do lock e checkpoint |
| Reprodução limpa em Python 3.11 | Pendente | Só Python 3.13 disponível nesta sessão |
| GitHub | Destino informado | `octopus-synapse/patch-insight`; inicialmente vazio e privado |
| Artigo | Texto de desenvolvimento escrito | Resultado real de validação; comparação final pendente |

## Restrições observadas

O terminal não resolve `huggingface.co` ou `github.com`; instalar uv por pip
falhou por DNS. `torch`, `transformers`, `sklearn`, `pydantic` e `pytest` não
estão instalados. `nvidia-smi` não se comunica com o driver. `docker info`
retorna permissão negada no socket. `.git` está protegido e não contém um
repositório inicializado. Não houve escalada de permissões.

`gh api` falhou na conexão. SSH padrão falhou nas permissões de uma configuração
de sistema; teste com configuração isolada confirmou também falha de DNS.
A integração GitHub conseguiu ler o repositório e confirmou permissão de push.
O histórico de publicação deve ser conferido pelo SHA remoto; acesso à API
não significa que um push já ocorreu.

Busca por plugins de Kaggle/Hugging Face não retornou integrações disponíveis
nesta sessão. O navegador de consulta consegue ler documentação pública, mas
não substitui um ambiente de treino ou autenticação de publicação.

## Próxima execução com rede e recursos gratuitos

1. Resolver e versionar `uv.lock` em Python 3.11; executar pytest completo.
2. No Kaggle, usar o notebook versionado, baseline congelada e fine-tuning.
   Colab gratuito é reserva; nenhum serviço pago foi contratado.
3. Preservar modelo e metadados fora da sessão. Conferir seleção na validação.
4. Executar benchmark final uma vez, publicar os três métodos e seus erros.
5. Rodar teste com checkpoint real e imagem Docker sem rede.
6. Atualizar cards, artigo e links; publicar e reproduzir em ambiente limpo.

Cronograma original: dados/baselines até 09/11; treino 10/11; avaliação 11/11;
CLI 12/11; Docker/publicação 13/11; documentação 14/11; reprodução 15/11;
release 16/11. Código adiantado não muda os gates pendentes de execução real.
