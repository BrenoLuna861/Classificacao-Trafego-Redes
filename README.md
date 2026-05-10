# Classificação de Tráfego de Rede — NSL-KDD

**Disciplina:** Machine Learning — AV2  
**Docente:** Mateus Silva  
**Instituição:** Faculdade de Petrolina — 2026

| Discente | Matrícula |
|---|---|
| Edmilson Breno R. Luna | 25009 |
| Francklin Leandro R. I. Bartilotti | 25039 |

---

## Problema e Objetivo

O projeto classifica conexões de rede como **Normal (0)** ou **Ataque (1)** usando o dataset NSL-KDD. O objetivo é comparar três algoritmos de Machine Learning — Árvore de Decisão, Random Forest e KNN — e identificar o melhor modelo para detecção de intrusão em redes, com validação estatística dos resultados.

**Variável-alvo:** `alvo` — binária (0 = Normal, 1 = Ataque)  
**Métrica primária:** F1-Score (justificado pelo desbalanceamento das classes e pelo duplo custo de erros em segurança)

---

## Origem dos Dados

- **Dataset:** NSL-KDD (versão melhorada do KDD Cup 1999)
- **Fonte:** https://github.com/defcom17/NSL_KDD
- **Treino:** 125.973 amostras × 41 features
- **Teste:** 22.544 amostras × 41 features
- **Features:** 38 numéricas + 3 categóricas (`protocol_type`, `service`, `flag`)
- **Classes:** Normal (53,5%) e Ataque (46,5%) no treino

---

## Instalação e Execução

### 1. Entrar na pasta do projeto
```bash
cd CTDR
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Baixar os datasets
```bash
curl -o data/raw/KDDTrain.txt "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
curl -o data/raw/KDDTest.txt  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
```

### 4. Executar o pipeline completo
```bash
python main.py
```

Os gráficos são salvos automaticamente em `outputs/` e os resultados em `experiments/experimentos.csv`.

---

## Resumo dos Resultados

| Modelo | Acurácia | Precision | Recall | F1-Score | CV F1 (média ± dp) |
|---|---|---|---|---|---|
| Árvore de Decisão | 0.8200 | 0.9700 | 0.6500 | 0.7800 | 0.9998 ± 0.0001 |
| Random Forest | 0.8300 | 0.9700 | 0.6700 | **0.7900** | 0.9999 ± 0.0001 |
| KNN (k=5) | 0.8000 | 0.9500 | 0.6300 | 0.7600 | 0.9950 ± 0.0010 |

- O **Teste de McNemar** confirmou diferença estatisticamente significativa entre Árvore de Decisão e Random Forest (p < 0.05)
- O **Random Forest** obteve o melhor F1-Score geral
- A **Árvore de Decisão** apresentou o melhor custo-benefício entre interpretabilidade e desempenho
- Features mais importantes: `src_bytes`, `dst_bytes`, `flag`, `logged_in`, `count`

---

## Limitações

- Dataset de 1999 — pode não representar ataques modernos (ransomware, IoT, APT)
- Recall de ~65% implica que ~35% dos ataques reais não são detectados
- Hiperparâmetros não otimizados via busca sistemática (GridSearchCV)
- Modelos não testados em tráfego de rede real

---

## Estrutura do Repositório

```
CTDR/
├── data/
│   ├── raw/                    ← datasets brutos (não modificar)
│   └── processed/              ← dados após pré-processamento
├── notebooks/
│   └── apresentacao.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── preprocessamento.py     ← carregamento e preparação dos dados
│   ├── features/                   ← engenharia de features (reservado)
│   ├── models/
│   │   └── treinamento.py          ← definição e treinamento dos modelos
│   ├── evaluation/
│   │   └── validacao.py            ← validação cruzada e teste estatístico
│   └── visualization/
│       └── visualizacao.py         ← geração de todos os gráficos
├── experiments/
│   └── experimentos.csv        ← rastreamento de experimentos
├── article/
│   ├── artigo.md               ← artigo técnico-científico
│   ├── referencias.bib         ← referências bibliográficas
│   ├── figures/                ← figuras geradas pelo pipeline
│   └── tables/                 ← tabelas do artigo
├── docs/
│   ├── decisoes-tecnicas.md    ← justificativas metodológicas
│   └── dicionario-de-dados.md  ← descrição de todas as features
├── outputs/                    ← gráficos gerados pelo pipeline
├── main.py                     ← ponto de entrada do pipeline
├── requirements.txt            ← dependências com versões fixas
└── .gitignore
```

---

## Versões

- **Python:** 3.13
- **pandas:** 2.2.2
- **numpy:** 1.26.4
- **scikit-learn:** 1.8.0
- **statsmodels:** 0.14.2
- **matplotlib:** 3.10.0
- **seaborn:** 0.13.2
