# ============================================================
# treinamento.py
# Definição, treinamento e avaliação dos 3 modelos ML
# ============================================================

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)


def definir_modelos() -> dict:
    """
    Retorna dicionário com os 3 modelos instanciados.

    Justificativas:
    - Árvore de Decisão : interpretável, baseline clássico
    - Random Forest     : ensemble robusto, reduz overfitting
    - KNN               : baseado em distância, contraste metodológico
    """
    return {
        "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=100,
                                                    random_state=42,
                                                    n_jobs=-1),
        "KNN (k=5)":         KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def treinar_modelos(modelos: dict, X_treino, y_treino) -> tuple[dict, dict]:
    """
    Treina todos os modelos e retorna:
    - modelos_treinados : dict com objetos sklearn ajustados
    - predicoes         : dict com arrays de predição no teste
    """
    modelos_treinados = {}
    predicoes         = {}

    for nome, modelo in modelos.items():
        print(f"⏳ Treinando {nome}...")
        modelo.fit(X_treino, y_treino)
        modelos_treinados[nome] = modelo
        print(f"✅ {nome} concluído!")

    return modelos_treinados


def gerar_predicoes(modelos_treinados: dict, X_teste) -> dict:
    """Gera predições de todos os modelos no conjunto de teste."""
    return {
        nome: modelo.predict(X_teste)
        for nome, modelo in modelos_treinados.items()
    }


def exibir_tabela_resultados(predicoes: dict, y_teste):
    """Imprime tabela comparativa de métricas."""
    print("\n" + "=" * 68)
    print(f"{'Modelo':<22} {'Acurácia':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("=" * 68)
    for nome, y_pred in predicoes.items():
        print(
            f"{nome:<22} "
            f"{accuracy_score(y_teste, y_pred):>9.4f} "
            f"{precision_score(y_teste, y_pred):>10.4f} "
            f"{recall_score(y_teste, y_pred):>8.4f} "
            f"{f1_score(y_teste, y_pred):>8.4f}"
        )
    print("=" * 68)
