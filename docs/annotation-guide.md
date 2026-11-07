# Guia de anotação — 0.1.0

Cada exemplo recebe cinco booleanos independentes. A referência é o que o texto
afirma, sem inferir experiência a partir de cargo, empresa ou tecnologia.

| Rótulo | Positivo | Negativo / fronteira |
|---|---|---|
| `action` | Contribuição identificável realizada: “Investiguei os timeouts.” | “Atuei com tecnologia”, intenções futuras e ações negadas. |
| `context` | Situação, domínio, equipe ou problema explícito: “Na equipe de pagamentos havia falhas.” | Nome de ferramenta isolado; não inferir o domínio. |
| `method` | Meio, técnica, processo ou ferramenta usado/indicado como abordagem: “Abordagem: revisão por pares.” | Lista de competências sem relação com uma abordagem. |
| `result` | Efeito ou entrega identificável: “O acesso foi restabelecido”, “Entreguei o painel.” | Investigação sem conclusão, “bons resultados”, propósito futuro. Não exige números. |
| `measurement` | Quantidade com referente: “Escopo: 12 serviços”, “redução de 20%”, “duas horas”. | “OAuth 2.0”, “Python 3.11”, “2026”, “INC-42”. Número pode medir escopo, sem resultado. |

“Entreguei três relatórios” tem `action`, `result` e `measurement`.
“Investiguei a lentidão” tem `action`, sem entrega ou efeito afirmado.
“O tempo de espera caiu” tem `result`, sem ação pessoal explícita.
“Não implementei cache; corrigi o timeout” contém uma ação realizada na segunda
oração; não aplicar negação à frase inteira. “Pretendo reduzir 20% usando SQL”
tem método e medida explícitos, sem ação realizada nem resultado realizado.
“Trabalho há dois anos com dados” contém medida e contexto, mas uma atividade
vaga não estabelece contribuição específica. A anotação é descritiva; falsidade
ou causalidade de uma alegação não podem ser verificadas por este dataset.

As cláusulas de contexto, abordagem, efeito e escopo podem ocorrer isoladamente
nos exemplos composicionais. “Escopo: 8 jobs” é uma medida, sem presumir entrega
ou situação de negócio. Uma abordagem nominal conta como método mesmo sem verbo
de realização; essa convenção limita a interpretação dos dados.

## Piloto e lotes

O piloto são os primeiros 50 registros do lote 1. O guia foi fixado antes dos
experimentos; o piloto usa as mesmas definições dos demais registros. A geração
é composicional, com 100 cenários originais e quatro variantes controladas por
cenário. A variante completa e as remoções/negações são deliberadamente
correlacionadas e permanecem no mesmo split. Não são 400 origens independentes.

Revisão feita pela mesma IA: inspecionar as cláusulas de cada cenário quanto a
efeito concreto, contribuição, quantidades, negação e intenção; revisar os
textos renderizados e auditar duplicatas, cobertura e grupos. Histórico por
registro descreve o procedimento realizado, sem alegar um segundo anotador.
Não houve revisão humana nem medida de concordância.

O teste tem arquivo de cenários separado, produzido antes dos experimentos.
Congelar hashes de textos e rótulos; usar só treino/validação em decisões.
O teste final poderá ser lido para relatar erros, nunca para orientar novos dados
ou ajustes da v0.1. Mudanças posteriores exigem outra versão e outro teste.
