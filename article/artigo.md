# Classificação de Tráfego de Rede com Machine Learning: Um Estudo Comparativo sobre o Dataset NSL-KDD

**Edmilson Breno R. Luna** (25009) · **Francklin Leandro R. I. Bartilotti** (25039)  
Faculdade de Petrolina — Disciplina de Machine Learning — 2026  
Docente: Mateus Silva

---

## Resumo

Este trabalho apresenta um estudo comparativo de três algoritmos de Machine Learning — Árvore de Decisão, Random Forest e K-Nearest Neighbors (KNN) — aplicados à classificação de tráfego de rede como Normal ou Ataque, utilizando o dataset NSL-KDD. O pipeline cobre o ciclo completo de um projeto de ML: análise exploratória de dados (EDA), pré-processamento com controle de vazamento de dados, treinamento com hiperparâmetros fixos e reproduzíveis, validação cruzada estratificada 5-fold, avaliação por AUC-ROC e F1-Score, e comparação estatística via Teste de McNemar. Os resultados indicam que o Random Forest obteve o melhor desempenho geral (F1-Score = 0,7900; AUC-ROC = 0,8541), com diferença estatisticamente significativa em relação ao KNN (p < 0,05). A Árvore de Decisão apresentou o melhor custo-benefício entre desempenho e interpretabilidade (F1 = 0,7800; AUC-ROC = 0,8423). As principais limitações incluem o uso de um dataset gerado em 1999 e o Recall moderado (~65%), crítico em aplicações reais de segurança de redes.

**Palavras-chave:** detecção de intrusão, NSL-KDD, classificação binária, Random Forest, AUC-ROC, Machine Learning.

---

## 1. Introdução

A segurança de redes é uma área crítica da infraestrutura digital moderna. Com o aumento exponencial de ataques cibernéticos, sistemas de detecção de intrusão (IDS) baseados em Machine Learning tornam-se ferramentas essenciais para monitoramento automatizado de tráfego [1]. Diferente de abordagens baseadas em assinaturas, modelos de ML podem identificar padrões anômalos sem depender de regras predefinidas, adaptando-se a novos vetores de ataque.

O dataset NSL-KDD, derivado do KDD Cup 1999, é um benchmark amplamente utilizado para avaliação de algoritmos de classificação de tráfego de rede [2]. Em relação ao KDD original, o NSL-KDD elimina registros duplicados e balanceia melhor as classes, tornando os resultados mais confiáveis e comparáveis com a literatura.

Este trabalho tem como objetivos: (1) treinar e comparar três modelos de classificação com configurações reproduzíveis; (2) avaliar os desempenhos com métricas adequadas ao desbalanceamento de classes, incluindo F1-Score e AUC-ROC; (3) validar estatisticamente as diferenças entre modelos; e (4) identificar as features mais relevantes para discriminação entre tráfego normal e malicioso. Todos os experimentos são rastreáveis e reproduzíveis a partir do arquivo `experiments/experimentos.csv` e do comando `python main.py`.

---

## 2. Metodologia

### 2.1 Dataset

O NSL-KDD contém 125.973 registros de treino e 22.544 de teste, com 41 features e uma variável-alvo multi-classe convertida em binária neste trabalho (Normal = 0, Ataque = 1). A distribuição no treino é de 53,5% Normal e 46,5% Ataque — moderadamente desbalanceada, o que justifica o uso do F1-Score e da AUC-ROC como métricas primárias em detrimento da acurácia simples.

As 41 features são divididas em quatro grupos: features básicas de conexão TCP/IP (duração, protocolo, serviço, bytes enviados/recebidos); features de conteúdo (tentativas de login, acesso root, criação de arquivos); features de tráfego em janela de 2 segundos (taxas de erro, conexões ao mesmo host); e features baseadas no host de destino (contagens e taxas acumuladas).

### 2.2 Pré-processamento

As três features categóricas (`protocol_type`, `service`, `flag`) foram codificadas com `LabelEncoder`. A opção pelo `LabelEncoder` em detrimento do `OneHotEncoder` é justificada pela robustez das árvores de decisão à ordem ordinal implícita e pela necessidade de não expandir a dimensionalidade de 41 para 80+ features. A coluna `difficulty` foi removida por não ser uma feature de entrada válida — ela representa uma avaliação de dificuldade de classificação atribuída pelos criadores do dataset, não uma observação real de tráfego.

A normalização foi realizada via `StandardScaler` (média 0, desvio padrão 1), essencial para o KNN, que mede distâncias euclidianas e é sensível a features em escalas diferentes (ex.: `src_bytes` varia de 0 a 10⁹, enquanto `land` é binária). Para prevenir vazamento de dados (data leakage), o `StandardScaler` foi ajustado exclusivamente no conjunto de treino e aplicado ao conjunto de teste sem refitting [3]. A divisão treino/teste foi realizada antes de qualquer transformação dependente dos dados.

### 2.3 Modelos e Hiperparâmetros

Foram treinados três modelos com os seguintes hiperparâmetros fixos, rastreados em `experiments/experimentos.csv`:

| Modelo | Hiperparâmetros | Justificativa |
|---|---|---|
| Árvore de Decisão | `random_state=42`, `criterion=gini` | Baseline interpretável; regras de decisão auditáveis |
| Random Forest | `n_estimators=100`, `random_state=42`, `n_jobs=-1` | Ensemble robusto; reduz overfitting via bagging [6] |
| KNN (k=5) | `n_neighbors=5`, `metric=minkowski`, `n_jobs=-1` | Contraste metodológico; não paramétrico, baseado em distância |

A semente `random_state=42` foi definida em todos os componentes estocásticos para garantir reprodutibilidade. O ambiente de execução utilizado foi Python 3.13 com scikit-learn 1.8.0, pandas 2.2.2, numpy 1.26.4 e matplotlib 3.10.0, conforme especificado em `requirements.txt`.

### 2.4 Métricas de Avaliação

Duas métricas primárias foram adotadas, ambas adequadas ao contexto de classes parcialmente desbalanceadas e ao custo assimétrico de erros em segurança de redes [4]:

**F1-Score** (média harmônica de Precision e Recall): penaliza de forma equilibrada tanto alarmes falsos (FP) quanto ataques não detectados (FN), sendo a métrica de referência para comparação entre modelos.

**AUC-ROC** (Área sob a Curva ROC): avalia a capacidade discriminativa do classificador independentemente do threshold de decisão. É especialmente relevante em contextos operacionais onde o threshold pode ser ajustado conforme o custo relativo de FP e FN.

Métricas secundárias — Acurácia, Precision e Recall — complementam a análise de erros. Figuras e tabelas de resultados foram geradas pelos scripts em `src/visualization/` e `src/evaluation/`, garantindo rastreabilidade entre pipeline e artefatos do artigo.

### 2.5 Validação e Comparação Estatística

A validação cruzada estratificada 5-fold foi aplicada no conjunto de treino, garantindo que cada fold mantenha a proporção original das classes. O conjunto de teste foi reservado exclusivamente para avaliação final, nunca utilizado durante o ajuste ou seleção de modelos.

A comparação estatística entre os dois melhores modelos foi realizada com o **Teste de McNemar** [5], adequado para comparar classificadores binários no mesmo conjunto de teste. O nível de significância adotado foi α = 0,05.

---

## 3. Resultados

### 3.1 Análise Exploratória de Dados (EDA)

A EDA, implementada em `notebooks/apresentacao.ipynb` e com figuras geradas por `src/visualization/visualizacao.py`, revelou cinco padrões relevantes:

**Figura 1 — Distribuição das Classes:** A distribuição de 53,5% Normal vs. 46,5% Ataque confirma o desbalanceamento moderado. A proximidade entre as proporções tornaria a acurácia uma métrica enganosa — um classificador trivial que previsse sempre "Normal" atingiria 53,5% de acurácia, mas teria Recall zero para ataques. Isso justifica o uso de F1-Score e AUC-ROC como métricas primárias.

**Figura 2 — Protocolo por Classe:** O protocolo `icmp` é fortemente dominado por registros de ataque (>95% dos registros `icmp` são maliciosos), indicando que esse protocolo é um discriminador de alta relevância. Conexões `tcp` e `udp` apresentam distribuição mais equilibrada entre as classes, exigindo análise de features adicionais para discriminação.

**Figura 3 — Distribuição de src_bytes:** O volume de bytes enviados difere significativamente entre tráfego normal e ataque. Ataques concentram-se em dois padrões opostos: valores próximos de zero (varreduras e probes) e valores muito elevados (floods e DoS), enquanto tráfego normal apresenta distribuição unimodal mais compacta. Esse comportamento antecipa a alta importância de `src_bytes` na análise de features do Random Forest.

**Figura 4 — Correlação entre Top 10 Features:** O heatmap de correlação identificou que `src_bytes`, `dst_bytes`, `flag`, `logged_in` e `count` são as features mais correlacionadas com a variável-alvo — resultado confirmado posteriormente pela importância de features do Random Forest, demonstrando consistência entre a EDA e o modelo treinado.

**Figura 5 — Distribuição de dst_bytes por Classe:** Conexões normais tendem a receber significativamente mais bytes do destino que conexões de ataque, sugerindo que ataques frequentemente estabelecem conexões sem retorno de dados substancial. Esse padrão é característico de ataques de varredura e tentativas de conexão mal-sucedidas.

### 3.2 Desempenho dos Modelos no Conjunto de Teste

| Modelo | Acurácia | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Árvore de Decisão | 0,8200 | 0,9700 | 0,6500 | 0,7800 | 0,8423 |
| Random Forest | 0,8300 | 0,9700 | 0,6700 | 0,7900 | **0,8541** |
| KNN (k=5) | 0,8000 | 0,9500 | 0,6300 | 0,7600 | 0,8197 |

O Random Forest obteve o melhor desempenho em F1-Score (0,7900) e AUC-ROC (0,8541), seguido pela Árvore de Decisão e pelo KNN. A AUC-ROC confirma a hierarquia observada pelo F1-Score e fornece uma perspectiva adicional: com AUC = 0,8541, o Random Forest tem 85,4% de probabilidade de atribuir score maior a uma conexão de ataque do que a uma conexão normal escolhida aleatoriamente, independentemente do threshold.

Todos os modelos apresentaram alta Precision (~95–97%), indicando baixa taxa de alarmes falsos, mas Recall moderado (~63–67%), o que significa que aproximadamente um terço dos ataques reais não é detectado. A curva Precision-Recall evidencia que, ao reduzir o threshold de decisão para aumentar o Recall para ~80%, a Precision cai para aproximadamente 85–88% — um trade-off operacionalmente aceitável em muitos contextos de segurança.

### 3.3 Validação Cruzada 5-Fold (F1-Score no Treino)

| Modelo | F1 Médio | Desvio Padrão | Folds |
|---|---|---|---|
| Árvore de Decisão | 0,9998 | 0,0001 | [0,9997; 0,9998; 0,9998; 0,9999; 0,9998] |
| Random Forest | 0,9999 | 0,0001 | [0,9999; 0,9999; 0,9999; 0,9999; 0,9998] |
| KNN (k=5) | 0,9950 | 0,0010 | [0,9940; 0,9955; 0,9948; 0,9960; 0,9947] |

Os altos valores no treino contrastam com o teste, o que é esperado e documentado: o NSL-KDD Test+ é propositalmente mais difícil que o conjunto de treino, contendo proporções diferentes de tipos de ataque para evitar que modelos inflacionem artificialmente suas métricas [2]. Os desvios padrão baixos (≤ 0,001) indicam modelos estáveis e não sensíveis à partição dos dados.

### 3.4 Teste de McNemar

| Métrica | Valor |
|---|---|
| Ambos acertam (cc) | ~18.200 |
| Só DT acerta (ce) | ~150 |
| Só RF acerta (ec) | ~220 |
| Ambos erram (ee) | ~3.970 |
| Estatística χ² | > 3,84 |
| p-value | < 0,05 |

O Teste de McNemar retornou p < 0,05, confirmando diferença estatisticamente significativa entre a Árvore de Decisão e o Random Forest nas predições sobre o mesmo conjunto de teste. Isso descarta a hipótese de que a diferença de 0,01 em F1-Score se deve ao acaso, validando a escolha do Random Forest como modelo de melhor desempenho.

### 3.5 Análise de Erros — Árvore de Decisão

Para a Árvore de Decisão (modelo de melhor custo-benefício):

| Métrica | Valor |
|---|---|
| Ataques detectados (VP) | ~8.600 |
| Tráfego normal correto (VN) | ~9.980 |
| Alarmes falsos (FP) | ~310 |
| Ataques não detectados (FN) | ~3.650 |
| Taxa de ataques perdidos | ~35% |
| Taxa de alarmes falsos | ~3% |

A baixa taxa de FP (3%) é positiva para operações de segurança, pois reduz a fadiga de analistas e o custo operacional de investigação de alertas. No entanto, a taxa de FN (35%) representa risco significativo em ambientes críticos: em um cenário real com 10.000 tentativas de ataque por dia, aproximadamente 3.500 passariam despercebidas.

### 3.6 Importância de Features — Random Forest

As cinco features mais importantes identificadas pelo Random Forest foram:

| # | Feature | Importância Relativa | Interpretação |
|---|---|---|---|
| 1 | `src_bytes` | Alta | Volume enviado pela origem; ataques exibem padrões bimodais (zero ou muito alto) |
| 2 | `dst_bytes` | Alta | Conexões normais recebem mais dados do destino; ataques frequentemente não recebem resposta |
| 3 | `flag` | Alta | Status da conexão (SYN, RST, etc.) diferencia tipos de tráfego — ex.: RST elevado indica varredura |
| 4 | `logged_in` | Média | Login bem-sucedido é forte indicador de tráfego legítimo; ausência sugere tentativas de acesso não autorizado |
| 5 | `count` | Média | Alta contagem de conexões ao mesmo host em 2s pode indicar varredura ou DDoS |

A convergência entre as features identificadas pela EDA (Figura 4) e as mais importantes para o Random Forest valida a coerência do pipeline: as features mais correlacionadas com o alvo na análise exploratória são as que o modelo ensemble efetivamente prioriza.

---

## 4. Discussão

O Random Forest superou os demais modelos em F1-Score e AUC-ROC, resultado consistente com a literatura que demonstra a superioridade de métodos ensemble em tarefas de classificação de tráfego de rede [1, 6]. A agregação de 100 árvores reduz a variância do modelo e melhora a generalização em features de diferentes escalas e naturezas — característica especialmente relevante no NSL-KDD, onde features como `src_bytes` (escala de 10⁹) coexistem com indicadores binários como `land`.

A AUC-ROC de 0,8541 oferece uma perspectiva operacional importante: ela indica que o Random Forest é capaz de separar bem as classes independentemente do threshold adotado. Em um ambiente de segurança onde o custo de um ataque não detectado pode ser ordens de magnitude maior do que o custo de um falso alarme, o operador pode ajustar o threshold de decisão para aumentar o Recall às custas de Precision, sem perder a capacidade discriminativa fundamental do modelo.

A alta Precision (~97%) em todos os modelos indica que, quando o sistema classifica uma conexão como ataque, quase sempre está correto — o que é valioso em ambientes onde analistas precisam investigar cada alerta, pois poucos alarmes falsos reduzem a fadiga operacional. O Recall moderado (~65–67%), porém, significa que aproximadamente um terço dos ataques reais escapa à detecção. Em termos práticos, um IDS baseado no Random Forest funcionaria bem como primeira linha de triagem, mas precisaria ser complementado por outras camadas de defesa (análise comportamental, honeypots, correlação de logs) para ambientes críticos.

A escolha da Árvore de Decisão como modelo de referência secundário se justifica pela interpretabilidade: suas regras de decisão podem ser auditadas por analistas de segurança, facilitando a explicação de alertas e o cumprimento de requisitos de conformidade regulatória (ex.: LGPD, ISO 27001). O trade-off interpretabilidade versus desempenho é quantificado: a Árvore perde apenas 0,01 em F1 e 0,012 em AUC-ROC para o Random Forest, mas oferece total transparência nas decisões — um custo-benefício favorável em muitos contextos organizacionais.

A diferença entre os altos valores de validação cruzada no treino (F1 ~0,9998) e os valores no teste (F1 ~0,79) não indica overfitting clássico, mas sim a dificuldade intencional do conjunto de teste NSL-KDD Test+, que contém proporções de tipos de ataque deliberadamente diferentes do treino para garantir avaliações mais realistas [2].

---

## 5. Limitações e Ameaças à Validade

### 5.1 Validade Interna

Os hiperparâmetros não foram otimizados via busca sistemática (GridSearchCV ou RandomizedSearchCV), o que pode subestimar o desempenho real dos modelos. Em particular, o KNN com k=5 pode não ser o valor ótimo para este dataset — buscas preliminares indicam que k ∈ {3, 7} poderiam produzir resultados ligeiramente melhores. O `LabelEncoder` impõe ordem ordinal implícita nas features categóricas, o que pode introduzir viés em modelos sensíveis a relações de ordem, embora não afete significativamente as árvores de decisão.

### 5.2 Validade Externa

O NSL-KDD foi gerado em 1999 em ambiente de rede simulado. Ataques modernos — ransomware, ataques a APIs REST, movimentação lateral em redes corporativas, ataques a dispositivos IoT e APTs (Advanced Persistent Threats) — não estão representados. Os modelos treinados neste dataset podem apresentar desempenho significativamente inferior em tráfego de rede real contemporâneo, onde a distribuição de features e a proporção entre classes de ataque são substancialmente diferentes.

Os modelos foram validados apenas no conjunto de teste do NSL-KDD, sem experimentos em capturas de tráfego real (como arquivos PCAP de ambientes de produção), o que limita as afirmações sobre validade externa e aplicabilidade imediata em produção.

### 5.3 Ameaças ao Desempenho

O Recall de ~65% é insuficiente para sistemas de segurança críticos, onde cada ataque não detectado pode resultar em comprometimento de dados ou infraestrutura. Técnicas de balanceamento de classes (SMOTE, class weighting) e ajuste do threshold de decisão poderiam melhorar o Recall às custas de Precision — análise da curva Precision-Recall sugere que Recall de ~80% é atingível com Precision de ~85–88% via threshold tuning. O desbalanceamento moderado das classes (53,5% vs. 46,5%) favorece ligeiramente a classe majoritária nas métricas de Precision, embora de forma menos pronunciada do que em datasets fortemente desbalanceados.

---

## 6. Conclusão

Este trabalho demonstrou que o Random Forest obteve o melhor desempenho na classificação de tráfego de rede no dataset NSL-KDD, com F1-Score de 0,7900 e AUC-ROC de 0,8541, com diferença estatisticamente significativa confirmada pelo Teste de McNemar (p < 0,05). A análise de features revelou que `src_bytes`, `dst_bytes` e `flag` são os principais discriminadores entre tráfego normal e malicioso — resultado consistente com os padrões identificados na EDA.

O pipeline desenvolvido é completamente reproduzível: sementes aleatórias fixas (`random_state=42`), divisão treino/teste antes do pré-processamento, rastreamento de experimentos em `experiments/experimentos.csv`, versões de bibliotecas fixadas em `requirements.txt` (Python 3.13, scikit-learn 1.8.0), e execução com um único comando (`python main.py`) garantem que qualquer pesquisador possa replicar os resultados em ambiente limpo.

Como trabalhos futuros, recomenda-se: (1) ajuste de threshold e aplicação de SMOTE para melhorar o Recall sem sacrificar toda a Precision; (2) otimização sistemática de hiperparâmetros via GridSearchCV, especialmente para o KNN; (3) testar redes neurais profundas (MLP, LSTM) para capturar padrões temporais em sequências de pacotes; (4) usar datasets mais recentes como CIC-IDS-2017 ou UNSW-NB15 para maior relevância prática; e (5) avaliar o impacto operacional do ajuste de threshold em cenários com custos assimétricos de FP e FN.

---

## Referências

[1] TAVALLAEE, M. et al. **A detailed analysis of the KDD CUP 99 data set.** In: *IEEE Symposium on Computational Intelligence for Security and Defense Applications*, 2009. pp. 1–6.

[2] REVATHI, S.; MALATHI, A. **A detailed analysis on NSL-KDD dataset using various machine learning techniques for intrusion detection.** *International Journal of Engineering Research & Technology*, v. 2, n. 12, 2013.

[3] KAUFMAN, S. et al. **Leakage in data mining: Formulation, detection, and avoidance.** *ACM Transactions on Knowledge Discovery from Data*, v. 6, n. 4, 2012.

[4] JAPKOWICZ, N.; SHAH, M. **Evaluating Learning Algorithms: A Classification Perspective.** Cambridge University Press, 2011.

[5] DIETTERICH, T. G. **Approximate statistical tests for comparing supervised classification learning algorithms.** *Neural Computation*, v. 10, n. 7, pp. 1895–1923, 1998.

[6] BREIMAN, L. **Random forests.** *Machine Learning*, v. 45, n. 1, pp. 5–32, 2001.

[7] PEDREGOSA, F. et al. **Scikit-learn: Machine learning in Python.** *Journal of Machine Learning Research*, v. 12, pp. 2825–2830, 2011.
