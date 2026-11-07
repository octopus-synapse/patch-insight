# Avaliação exploratória 0.2

72 textos sintéticos, escritos, anotados e revisados pelo mesmo assistente IA.
Não há revisão humana, anotador independente ou concordância interavaliadores.
São 36 textos em desenvolvimento e 36 em teste, com 34 origens em cada split:
32 descrições livres e dois pares controlados de marcadores por split. Não são
currículos reais nem uma amostra representativa. Não houve coleta de dados pessoais.

A revisão é semântica, com justificativa por registro, antes de executar modelos.
`review_note` documenta a interpretação; não representa uma segunda anotação cega.
O gerador exato não é inferido. A taxonomia é a do guia 0.1.0. Números por
extenso com referente também contam como measurement. Licença: CC BY 4.0.

As fontes livres foram escritas separadamente para cada split; os pares ficam
na mesma origem. Não há expansão por combinação das fontes da v0.1. Temas e
construções ainda se repetem entre os splits, e a mesma IA conhece ambos.
A auditoria exclui duplicatas exatas contra a v0.1; não comprova independência
semântica. Negação, futuro, versões, perguntas, contrafactuais, escrita informal,
resultado qualitativo e medidas de escopo têm tags explícitas.

## Protocolo definido antes dos scores

1. Congelar fontes, textos, rótulos, script e este protocolo com SHA-256.
2. Usar checkpoints da v0.1 sem retreino, regras heurísticas sem ampliar léxico.
3. Em desenvolvimento, ajustar uma temperatura global por modelo probabilístico
   na grade geométrica de 81 valores entre 0,25 e 4, incluindo 1. Minimizar a
   log loss binária média; empate favorece temperatura próxima de 1.
4. Manter limiar de classe 0,5. Separadamente escolher o primeiro cutoff de
   confiança entre 0,5/0,6/0,7/0,8/0,85/0,9/0,95/0,975/0,99 que observe erro
   de até 5% e cobertura de pelo menos 50% das decisões de componentes em
   desenvolvimento. Se nenhum atender, não ativar abstenção: registrar falha.
   Isso não garante esse risco em teste ou em textos reais.
5. Salvar e congelar políticas antes de pontuar o novo teste. Relatar todas as
   métricas predefinidas: F1 macro/micro e por rótulo, precision/recall, exact
   match, Hamming, MCC macro e por rótulo, AP, ROC-AUC, Brier e log loss.
   ROC-AUC/AP indefinidas ficam null, sem média parcial otimista.
6. Relatar confiabilidade em cinco bins por componente, risco versus cobertura,
   recortes por tags e diagnóstico pareado com/sem marcadores. Não usar estes
   resultados para escolher novos hiperparâmetros, limiares ou textos desta versão.
7. Bootstrap pareado de ganho macro-F1 (BERT menos regressão), 1.000 reamostragens
   de origens completas, seed 42. IC percentil 95%, exploratório nesta amostra.
8. Preservar os arquivos de data/freeze.json e reports/final da v0.1. Métricas
   adicionais da v0.1 são análise posterior, sem trocar seu critério de seleção.

A avaliação de componentes não valida as estruturas STAR/XYZ/Action–Method.
Regras continuam avaliadas por fixtures de contrato e coocorrência, sem gold set
humano de estruturas. Spans, novos rótulos e outros idiomas ficam para outro ciclo.
