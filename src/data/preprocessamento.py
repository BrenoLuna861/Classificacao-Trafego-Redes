# ============================================================
# preprocessamento.py
# Carregamento, limpeza e preparação do dataset NSL-KDD
# ============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Nomes das 43 colunas (41 features + label + difficulty)
COLUNAS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

COLUNAS_CATEGORICAS = ['protocol_type', 'service', 'flag']


def carregar_dados(caminho_treino: str, caminho_teste: str):
    """
    Lê os arquivos TXT do NSL-KDD e retorna DataFrames brutos.
    """
    treino = pd.read_csv(caminho_treino, names=COLUNAS)
    teste  = pd.read_csv(caminho_teste,  names=COLUNAS)

    print(f"✅ Treino: {treino.shape[0]} amostras x {treino.shape[1]} colunas")
    print(f"✅ Teste:  {teste.shape[0]} amostras x {teste.shape[1]} colunas")

    # Verificação de valores faltantes
    faltantes = treino.isnull().sum().sum()
    if faltantes == 0:
        print("✅ Nenhum valor faltante encontrado!")
    else:
        print(f"⚠️  {faltantes} valores faltantes no treino!")

    return treino, teste


def criar_alvo_binario(treino: pd.DataFrame, teste: pd.DataFrame):
    """
    Cria coluna 'alvo': 0 = Normal, 1 = Ataque.
    """
    treino = treino.copy()
    teste  = teste.copy()

    treino['alvo'] = treino['label'].apply(lambda x: 0 if x == 'normal' else 1)
    teste['alvo']  = teste['label'].apply(lambda x: 0 if x == 'normal' else 1)

    print("\n--- Distribuição binária (treino) ---")
    print(f"Normal (0): {(treino['alvo']==0).sum()} ({(treino['alvo']==0).mean()*100:.1f}%)")
    print(f"Ataque (1): {(treino['alvo']==1).sum()} ({(treino['alvo']==1).mean()*100:.1f}%)")
    print("⚠️  Dataset desbalanceado — usaremos F1-Score como métrica principal.")

    return treino, teste


def preparar_dados(treino: pd.DataFrame, teste: pd.DataFrame):
    """
    Separa features/alvo, aplica LabelEncoder e StandardScaler.
    Retorna X_treino, X_teste, y_treino, y_teste.

    Data Leakage evitado:
    - LabelEncoder: fit no treino, transform no teste
    - StandardScaler: fit no treino, transform no teste
    """
    X_treino = treino.drop(['label', 'difficulty', 'alvo'], axis=1).copy()
    X_teste  = teste.drop(['label', 'difficulty', 'alvo'], axis=1).copy()
    y_treino = treino['alvo'].copy()
    y_teste  = teste['alvo'].copy()

    # Codificação das variáveis categóricas
    for col in COLUNAS_CATEGORICAS:
        le = LabelEncoder()
        X_treino[col] = le.fit_transform(X_treino[col])
        X_teste[col]  = le.transform(X_teste[col])

    # Normalização
    scaler   = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste  = scaler.transform(X_teste)

    print("\n✅ Pré-processamento concluído!")
    print(f"X_treino: {X_treino.shape} | X_teste: {X_teste.shape}")
    print("✅ Data Leakage evitado: scaler ajustado APENAS no treino.")

    return X_treino, X_teste, y_treino, y_teste
