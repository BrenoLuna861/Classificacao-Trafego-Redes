# Decisões Técnicas e Justificativas Metodológicas

**Projeto:** Classificação de Tráfego de Rede — NSL-KDD  
**Disciplina:** Machine Learning — AV2 — Faculdade de Petrolina — 2026  
**Discentes:** Edmilson Breno R. Luna (25009) · Francklin Leandro R. I. Bartilotti (25039)

---

## 1. Escolha do Dataset

**Dataset:** NSL-KDDs (Network Security Lab — KDD)  
**Fonte:** https://github.com/defcom17/NSL_KDD

**Justificativa:** O NSL-KDD é a versão corrigida do KDD Cup 1999, eliminando registros duplicados que distorciam os resultados e inflavam artificialmente as métricas. É amplamente utilizado como benchmark em pesquisas de detecção de intrusão, permitindo comparação direta com a literatura existente. O dataset foi mantido como referência histórica por sua estabilidade e documentação extensiva.

**Trade-off:** O dataset é de 1999 e não representa ataques modernos (ransomware, IoT, APT). Para produção real, datasets mais recentes como CIC-IDS-2017 seriam preferíveis.

---

## 2. Definição da Métrica Primária

**Métrica escolhida:** F1-Score  

**Justificativa:** O dataset apresenta desbalanceamento moderado entre classes (Normal: 53,5% × Ataque: 46,5%). Em problemas de segurança de redes, ambos os tipos de erro têm custo significativo:
- **Falsos Positivos (FP):** alarmes falsos que sobrecarregam analistas de segurança e geram interrupções operacionais desnecessárias.
- **Falsos Negativos (FN):** ataques reais não detectados, que podem resultar em comprometimento de sistemas.

O F1-Score equilibra Precision e Recall, sendo mais adequado que a acurácia simples neste cenário. Métricas complementares (Precision, Recall, Acurácia, AUC-ROC) foram reportadas para completude.

---

## 3. Pré-processamento

### 3.1 Codificação de Variáveis Categóricas

**Decisão:** `LabelEncoder` nas colunas `protocol_type`, `service` e `flag`

**Justificativa:** Os modelos sklearn (DecisionTree, RandomForest, KNN) não aceitam strings como entrada. O LabelEncoder converte categorias em inteiros de forma eficiente e sem expansão de dimensionalidade.

**Trade-off analisado:**

| Abordagem | Vantagem | Desvantagem |
|---|---|---|
| `LabelEncoder` | Mantém 41 features; eficiente | Impõe ordem ordinal implícita |
| `OneHotEncoder` | Sem ordem implícita | Expande para 80+ features; prejudica KNN |

**Decisão final:** `LabelEncoder`, pois Árvore de Decisão e Random Forest são invariantes à ordem ordinal imposta, e a expansão de dimensionalidade prejudicaria o KNN (curse of dimensionality).

### 3.2 Normalização

**Decisão:** `StandardScaler` (média 0, desvio padrão 1)

**Justificativa:** Essencial para o KNN, que mede distância euclidiana e é sensível a features em escalas muito diferentes. Exemplo: `src_bytes` varia de 0 a 10⁹, enquanto `land` é binária (0 ou 1). Sem normalização, `src_bytes` dominaria completamente o cálculo de distância.

**Prevenção de Data Leakage:** O scaler foi ajustado (`fit`) exclusivamente no conjunto de treino e aplicado (`transform`) ao conjunto de teste, garantindo que nenhuma informação do teste influencie o pré-processamento.

### 3.3 Remoção da Coluna `difficulty`

**Decisão:** Remover a coluna `difficulty` antes do treinamento.

**Justificativa:** A coluna `difficulty` representa uma avaliação de dificuldade de classificação atribuída pelos criadores do dataset — ela não é uma observação real de tráfego de rede. Utilizá-la como feature seria uma forma de vazamento de dados (data leakage), pois essa informação não estaria disponível em um IDS real.

### 3.4 Criação do Alvo Binário

**Decisão:** Converter os 23 tipos de ataque originais em uma única classe "Ataque (1)".

**Justificativa:** O objetivo do projeto é detecção de intrusão (Normal vs. Ataque), não classificação multi-classe dos tipos de ataque (DoS, Probe, R2L, U2R). A conversão binária simplifica o problema, aumenta os exemplos por classe e é consistente com o objetivo de negócio: alertar sobre qualquer tráfego anômalo.

---

## 4. Escolha dos Modelos

| Modelo | Justificativa |
|---|---|
| **Árvore de Decisão** | Interpretável (regras de decisão visíveis), baseline clássico, permite auditoria por analistas de segurança |
| **Random Forest** | Ensemble que reduz overfitting via bagging; robusto a outliers e features irrelevantes; amplamente validado na literatura |
| **KNN (k=5)** | Contraste metodológico; não paramétrico; baseado em distância; sem fase de treinamento explícita |

**Trade-off interpretabilidade vs. desempenho:**
- Árvore de Decisão: totalmente interpretável, desempenho ligeiramente inferior.
- Random Forest: "caixa cinza" (feature importances disponíveis, mas regras não são diretas), melhor desempenho.
- KNN: não paramétrico, sem modelo explícito, desempenho inferior e alta latência de inferência.

Optamos por incluir os três modelos para cobrir diferentes paradigmas de aprendizado e justificar a escolha final com embasamento comparativo.

---

## 5. Hiperparâmetros e Reprodutibilidade

**Decisão:** Hiperparâmetros padrão do scikit-learn, com `random_state=42` em todos os componentes estocásticos.

**Justificativa:** O objetivo é estabelecer uma linha de base sólida (baseline) antes de otimização. Hiperparâmetros padrão do sklearn são bem validados para a maioria dos problemas. A fixação de `random_state=42` garante reprodutibilidade total dos resultados.

**Trade-off:** Não foi realizado GridSearchCV ou RandomizedSearchCV, o que pode subestimar o desempenho real dos modelos. Isso está documentado como limitação no artigo.

### 5.1 Critério de Impureza: Gini vs. Entropia

**Decisão:** `criterion='gini'` na Árvore de Decisão e no Random Forest.

**Justificativa:** O índice de Gini é computacionalmente mais eficiente que a Entropia por evitar o cálculo de logaritmos, produzindo resultados comparáveis na maioria dos problemas práticos. O Gini é o critério padrão do scikit-learn e adequado para classificação binária como a deste projeto. A Entropia conecta a decisão ao conceito de ganho de informação, mas o custo computacional adicional não se justifica neste contexto (Aula 17 — Árvores de Decisão, slide 7).

---

## 6. Estratégia de Validação

**Decisão:** Validação cruzada estratificada 5-fold no conjunto de treino + OOB Score no Random Forest.

**Justificativa da Validação Cruzada:** 
- **Estratificada:** garante que cada fold mantenha a proporção original das classes (53,5% Normal / 46,5% Ataque).
- **5-fold:** oferece boa estimativa de generalização com custo computacional aceitável para um dataset de 125.973 registros.
- **Conjunto de teste reservado:** usado exclusivamente para avaliação final, nunca para seleção de hiperparâmetros ou comparação de modelos durante o desenvolvimento.

**Justificativa do OOB Score (Out-of-Bag):**  
Em cada bootstrap do Random Forest, aproximadamente 36,8% das amostras não são incluídas na amostra de treino daquela árvore. O OOB Score usa essas amostras para estimar o erro de generalização sem necessidade de um conjunto de validação separado, funcionando como uma validação cruzada automática e não enviesada. O OOB Score é computado gratuitamente durante o treinamento e complementa a validação cruzada 5-fold como segunda estimativa de generalização (Aula 19 — Ensembles, slides 7 e 30).

---

## 7. Teste Estatístico

**Decisão:** Teste de McNemar entre Árvore de Decisão e Random Forest.

**Justificativa:** Os dois modelos apresentaram F1-Scores próximos (0.78 vs. 0.79). Uma diferença pequena pode ser ruído estatístico. O Teste de McNemar é o teste adequado para comparar classificadores binários aplicados ao mesmo conjunto de teste, pois leva em conta a correlação entre as predições dos modelos. O nível de significância adotado foi α = 0,05 (p < 0.05 → diferença significativa).

---

## 8. Feature Importance e Suas Limitações

**Decisão:** Utilizar Mean Decrease in Impurity (MDI) do Random Forest para ranquear a importância das features.

**Justificativa:** O MDI mede a redução média de impureza (Gini) que cada feature produz em todas as árvores da floresta. Features com alta redução de impureza são candidatas a discriminadores relevantes entre Normal e Ataque.

**Limitação conhecida — viés de alta cardinalidade:**  
A feature `service` possui 66 valores únicos (alta cardinalidade), o que a torna candidata natural a splits em qualquer nó, mesmo que sua relevância real seja menor do que outras features. Features contínuas como `src_bytes` e `dst_bytes` também tendem a ser favorecidas por terem mais pontos de corte candidatos. Isso pode inflar artificialmente a importância de `service` no ranking MDI.

Uma alternativa mais confiável é a **Permutation Feature Importance**, que embaralha cada feature individualmente e mede a queda na métrica de avaliação (F1-Score ou AUC-ROC), sendo menos sensível à cardinalidade. Essa alternativa não foi implementada neste projeto por limitação de escopo, mas é documentada como melhoria futura (Aula 19 — Ensembles, slides 9 e 33–34).

---

## 9. Análise do Viés-Variância nos Resultados

**Observação:** A Árvore de Decisão apresentou CV F1 de 0,9998 no treino e F1 de 0,78 no teste — uma queda expressiva que indica **alta variância** (overfitting). O Random Forest, ao combinar 100 árvores via bagging com aleatoriedade nas features (√p por nó), reduz essa variância e obtém F1 de 0,79 no teste com menor lacuna treino-teste.

Esse resultado está alinhado com o conceito de que o bagging combate variância sem reduzir viés, enquanto o boosting combateria viés em casos onde o modelo simples sistematicamente erra — não o caso aqui (Aula 19 — Ensembles, slides 3–4 e 19).

---

## 10. Estrutura do Projeto

**Decisão:** Separação clara entre `data/raw/` (dados brutos, somente leitura) e `data/processed/` (dados transformados pelos scripts).

**Justificativa:** Garante rastreabilidade e reprodutibilidade. Os dados brutos nunca são sobrescritos por scripts de transformação, permitindo reexecutar o pipeline do zero a qualquer momento.

**Decisão:** Código modularizado em `src/` com responsabilidades bem definidas.

| Módulo | Responsabilidade |
|---|---|
| `preprocessamento.py` | Carregamento, limpeza e preparação dos dados |
| `treinamento.py` | Definição, treinamento e avaliação dos modelos |
| `validacao.py` | Validação cruzada, teste estatístico e análise de erros |
| `visualizacao.py` | Geração de todos os gráficos do projeto |
| `main.py` | Orquestração do pipeline completo |

Essa estrutura facilita testes isolados de cada componente e manutenção do código.
