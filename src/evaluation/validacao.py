# ============================================================
# validacao.py
# Validação cruzada 5-fold e Teste de McNemar
# ============================================================

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar


def validacao_cruzada(modelos_treinados: dict, X_treino, y_treino):
    """
    Executa validação cruzada estratificada 5-fold (F1-Score).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 55)
    print("VALIDAÇÃO CRUZADA 5-FOLD — F1-Score no treino")
    print("=" * 55)

    for nome, modelo in modelos_treinados.items():
        scores = cross_val_score(
            modelo, X_treino, y_treino,
            cv=cv, scoring='f1', n_jobs=-1
        )
        print(f"\n{nome}:")
        print(f"  Folds:  {[round(s, 4) for s in scores]}")
        print(f"  Média:  {scores.mean():.4f}")
        print(f"  Desvio: {scores.std():.4f}")


def teste_mcnemar(predicoes: dict, y_teste):
    """
    Compara Árvore de Decisão vs Random Forest via Teste de McNemar.
    """
    pred_dt = predicoes["Árvore de Decisão"]
    pred_rf = predicoes["Random Forest"]
    y       = np.array(y_teste)

    cc = np.sum((pred_dt == y) & (pred_rf == y))
    ce = np.sum((pred_dt == y) & (pred_rf != y))
    ec = np.sum((pred_dt != y) & (pred_rf == y))
    ee = np.sum((pred_dt != y) & (pred_rf != y))

    resultado = mcnemar([[cc, ce], [ec, ee]], exact=False, correction=True)

    print("\n" + "=" * 55)
    print("TESTE DE McNEMAR — Árvore de Decisão vs Random Forest")
    print("=" * 55)
    print(f"\nAmbos acertam:       {cc}")
    print(f"Só DT acerta:        {ce}")
    print(f"Só RF acerta:        {ec}")
    print(f"Ambos erram:         {ee}")
    print(f"\nEstatística:  {resultado.statistic:.4f}")
    print(f"p-value:      {resultado.pvalue:.6f}")

    if resultado.pvalue < 0.05:
        print("\n✅ Diferença SIGNIFICATIVA (p < 0.05)")
        print("   Os modelos têm desempenho estatisticamente diferente.")
    else:
        print("\n⚠️  Diferença NÃO significativa (p >= 0.05)")
        print("   Preferir o modelo mais simples (Árvore de Decisão).")

    return resultado


def analise_erros(predicoes: dict, y_teste):
    """
    Detalha VP, VN, FP e FN do melhor modelo (Árvore de Decisão).
    """
    y_pred = predicoes["Árvore de Decisão"]
    y      = np.array(y_teste)

    vp = np.sum((y_pred == 1) & (y == 1))
    vn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))

    print("\n" + "=" * 55)
    print("ANÁLISE DE ERROS — Árvore de Decisão (melhor F1)")
    print("=" * 55)
    print(f"\n✅ Ataques detectados (VP):        {vp:>6}")
    print(f"✅ Normais corretos  (VN):          {vn:>6}")
    print(f"⚠️  Alarmes falsos   (FP):            {fp:>6}")
    print(f"🚨 Ataques NÃO detectados (FN):   {fn:>6}")
    print(f"\n🔴 Taxa de ataques perdidos: {fn / (fn + vp) * 100:.1f}%")
    print(f"🟡 Taxa de alarmes falsos:   {fp / (fp + vn) * 100:.1f}%")
