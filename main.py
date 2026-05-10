# ============================================================
# main.py — Ponto de entrada do projeto
# Classificação de Tráfego de Rede — NSL-KDD
# Discentes: Edmilson Breno R. Luna (25009)
#            Francklin Leandro R. I. Bartilotti (25039)
# Docente: Mateus Silva — Faculdade de Petrolina — 2026
# ============================================================

import os

# Garante que a pasta outputs existe antes de salvar gráficos
os.makedirs('outputs', exist_ok=True)

from src.data.preprocessamento   import carregar_dados, criar_alvo_binario, preparar_dados
from src.models.treinamento      import definir_modelos, treinar_modelos, gerar_predicoes, exibir_tabela_resultados
from src.evaluation.validacao    import validacao_cruzada, teste_mcnemar, analise_erros
from src.visualization.visualizacao import (gerar_todos_graficos_eda,
                                             grafico_matrizes_confusao,
                                             grafico_importancia_features)

# ── 1. CARREGAR DADOS ──────────────────────────────────────
print("\n" + "=" * 55)
print("COMPONENTE 1 — Carregamento e Descrição do Dataset")
print("=" * 55)

treino_raw, teste_raw = carregar_dados(
    caminho_treino='data/raw/KDDTrain.txt',
    caminho_teste='data/raw/KDDTest.txt'
)

treino, teste = criar_alvo_binario(treino_raw, teste_raw)

# ── 2. ANÁLISE EXPLORATÓRIA ────────────────────────────────
print("\n" + "=" * 55)
print("COMPONENTE 2 — Análise Exploratória (EDA)")
print("=" * 55)

gerar_todos_graficos_eda(treino)

# ── 3. PRÉ-PROCESSAMENTO ──────────────────────────────────
print("\n" + "=" * 55)
print("COMPONENTE 2 — Pré-processamento e Features")
print("=" * 55)

X_treino, X_teste, y_treino, y_teste = preparar_dados(treino, teste)

# ── 4. TREINAMENTO ─────────────────────────────────────────
print("\n" + "=" * 55)
print("COMPONENTE 3 — Treinamento dos Modelos")
print("=" * 55)

modelos           = definir_modelos()
modelos_treinados = treinar_modelos(modelos, X_treino, y_treino)
predicoes         = gerar_predicoes(modelos_treinados, X_teste)

exibir_tabela_resultados(predicoes, y_teste)
grafico_matrizes_confusao(predicoes, y_teste)

# ── 5. VALIDAÇÃO ───────────────────────────────────────────
print("\n" + "=" * 55)
print("COMPONENTE 4 — Validação e Teste Estatístico")
print("=" * 55)

validacao_cruzada(modelos_treinados, X_treino, y_treino)
teste_mcnemar(predicoes, y_teste)
analise_erros(predicoes, y_teste)

# ── 6. IMPORTÂNCIA DE FEATURES ────────────────────────────
print("\n" + "=" * 55)
print("COMPONENTE 5 — Importância de Features e Insights")
print("=" * 55)

grafico_importancia_features(modelos_treinados, treino)

print("""
INSIGHTS TÉCNICOS:
1. Alta Precision (97%) — quando classifica como ataque, quase sempre acerta.
2. Recall (~65%) indica que ~35% dos ataques reais não são detectados.
3. NSL-KDD Test+ é propositalmente mais difícil — explica queda vs treino.
4. Features 'src_bytes', 'dst_bytes' e 'flag' são as mais discriminativas.

LIMITAÇÕES:
- Dataset de 1999 pode não representar ataques modernos.
- Recall baixo é crítico em segurança real.

MELHORIAS SUGERIDAS:
- Aplicar SMOTE para melhorar o Recall.
- Testar redes neurais (MLP) para padrões mais complexos.
- Usar dados mais recentes (CIC-IDS-2017).
""")

print("✅ Pipeline completo executado com sucesso!")
