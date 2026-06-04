# ============================================================
# visualizacao.py
# Todos os gráficos do projeto (5 gráficos EDA + matrizes + ROC)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

COLUNAS_CATEGORICAS = ['protocol_type', 'service', 'flag']

# Tempo (segundos) que cada gráfico fica visível antes de fechar sozinho
_PAUSA = 3


def _exibir(caminho: str):
    """Salva o gráfico, exibe por _PAUSA segundos e fecha."""
    plt.savefig(caminho, dpi=150)
    plt.pause(_PAUSA)
    plt.close()


def grafico_distribuicao_classes(treino: pd.DataFrame):
    """Gráfico 1 — Distribuição das classes Normal vs Ataque."""
    plt.figure(figsize=(6, 4))
    contagem = treino['alvo'].value_counts().sort_index()
    cores = ['steelblue', 'tomato']
    bars = plt.bar(['Normal (0)', 'Ataque (1)'], contagem.values,
                   color=cores, edgecolor='white')
    for bar, v in zip(bars, contagem.values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 100,
                 str(v), ha='center', fontweight='bold', fontsize=11)
    plt.title('Gráfico 1 — Distribuição das Classes')
    plt.ylabel('Quantidade de amostras')
    plt.tight_layout()
    _exibir('outputs/g1_distribuicao_classes.png')


def grafico_protocolo_por_classe(treino: pd.DataFrame):
    """Gráfico 2 — Tipo de protocolo por classe."""
    temp = pd.DataFrame({
        'protocol': treino['protocol_type'],
        'alvo': treino['alvo']
    })
    ct = pd.crosstab(temp['protocol'], temp['alvo'])
    ct.columns = ['Normal', 'Ataque']

    ct.plot(kind='bar', figsize=(7, 4),
            color=['steelblue', 'tomato'], edgecolor='white')
    plt.title('Gráfico 2 — Tipo de Protocolo por Classe')
    plt.xlabel('Protocolo')
    plt.ylabel('Quantidade')
    plt.xticks(rotation=0)
    plt.tight_layout()
    _exibir('outputs/g2_protocolo.png')
    print("💡 Protocolo 'icmp' é dominado por ataques — feature relevante.")


def grafico_src_bytes(treino: pd.DataFrame):
    """Gráfico 3 — Histograma de src_bytes por classe."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (classe, cor, label) in zip(axes,
            [(0, 'steelblue', 'Normal'), (1, 'tomato', 'Ataque')]):
        dados = treino[treino['alvo'] == classe]['src_bytes']
        dados_clip = dados.clip(upper=dados.quantile(0.95))
        ax.hist(dados_clip, bins=40, color=cor, edgecolor='white', alpha=0.85)
        ax.set_title(f'Gráfico 3 — src_bytes ({label})')
        ax.set_xlabel('Bytes enviados (percentil 95)')
        ax.set_ylabel('Frequência')
    plt.tight_layout()
    _exibir('outputs/g3_src_bytes.png')
    print("💡 Padrão de bytes enviados difere entre tráfego normal e ataque.")


def grafico_correlacao(treino: pd.DataFrame):
    """Gráfico 4 — Heatmap de correlação das top 10 features."""
    treino_num = treino.drop(['label', 'difficulty', 'alvo'], axis=1).copy()
    for col in COLUNAS_CATEGORICAS:
        treino_num[col] = LabelEncoder().fit_transform(treino_num[col])

    top10 = (treino_num.corrwith(treino['alvo'])
             .abs().sort_values(ascending=False).head(10).index)

    plt.figure(figsize=(10, 7))
    sns.heatmap(treino_num[top10].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=0.5)
    plt.title('Gráfico 4 — Correlação entre Top 10 Features')
    plt.tight_layout()
    _exibir('outputs/g4_correlacao.png')

    print("Top 10 features mais correlacionadas com o alvo:")
    print(treino_num.corrwith(treino['alvo']).abs()
          .sort_values(ascending=False).head(10))


def grafico_dst_bytes(treino: pd.DataFrame):
    """Gráfico 5 — Boxplot de dst_bytes por classe."""
    temp_box = treino[['dst_bytes', 'alvo']].copy()
    temp_box['Classe'] = temp_box['alvo'].map({0: 'Normal', 1: 'Ataque'})
    temp_box['dst_bytes'] = temp_box['dst_bytes'].clip(
        upper=temp_box['dst_bytes'].quantile(0.95))

    plt.figure(figsize=(7, 5))
    sns.boxplot(x='Classe', y='dst_bytes', data=temp_box,
                palette={'Normal': 'steelblue', 'Ataque': 'tomato'})
    plt.title('Gráfico 5 — Distribuição de dst_bytes por Classe')
    plt.ylabel('Bytes recebidos (percentil 95)')
    plt.tight_layout()
    _exibir('outputs/g5_dst_bytes.png')
    print("💡 Conexões normais tendem a receber mais bytes que ataques.")


def grafico_matrizes_confusao(predicoes: dict, y_teste):
    """Matrizes de confusão dos 3 modelos lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (nome, y_pred) in zip(axes, predicoes.items()):
        cm = confusion_matrix(y_teste, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                    xticklabels=['Normal', 'Ataque'],
                    yticklabels=['Normal', 'Ataque'])
        ax.set_title(nome)
        ax.set_ylabel('Real')
        ax.set_xlabel('Previsto')
    plt.tight_layout()
    _exibir('outputs/matrizes_confusao.png')


def grafico_importancia_features(modelos_treinados: dict, treino: pd.DataFrame):
    """Top 15 features mais importantes — Random Forest."""
    nomes_feat   = treino.drop(['label', 'difficulty', 'alvo'], axis=1).columns
    importancias = modelos_treinados["Random Forest"].feature_importances_

    df_imp = (pd.DataFrame({'Feature': nomes_feat, 'Importância': importancias})
              .sort_values('Importância', ascending=False).head(15))

    plt.figure(figsize=(9, 6))
    sns.barplot(x='Importância', y='Feature', data=df_imp, palette='Blues_r')
    plt.title('Top 15 Features mais Importantes — Random Forest')
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    _exibir('outputs/importancia_features.png')

    print("Top 5 features:")
    print(df_imp.head(5).to_string(index=False))


def gerar_todos_graficos_eda(treino: pd.DataFrame):
    """Atalho para rodar todos os gráficos de análise exploratória."""
    plt.ion()   # modo interativo — não bloqueia o pipeline
    grafico_distribuicao_classes(treino)
    grafico_protocolo_por_classe(treino)
    grafico_src_bytes(treino)
    grafico_correlacao(treino)
    grafico_dst_bytes(treino)
    plt.ioff()  # volta ao modo normal após a EDA
