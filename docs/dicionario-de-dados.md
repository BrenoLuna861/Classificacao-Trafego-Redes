# Dicionário de Dados — NSL-KDD

**Projeto:** Classificação de Tráfego de Rede — NSL-KDD  
**Disciplina:** Machine Learning — AV2 — Faculdade de Petrolina — 2026

---

## Visão Geral do Dataset

| Atributo | Valor |
|---|---|
| Nome | NSL-KDD |
| Fonte | https://github.com/defcom17/NSL_KDD |
| Formato | CSV sem cabeçalho (.txt) |
| Total de colunas | 43 (41 features + label + difficulty) |
| Registros de treino | 125.973 |
| Registros de teste | 22.544 |
| Valores faltantes | Nenhum |
| Features numéricas | 38 |
| Features categóricas | 3 (`protocol_type`, `service`, `flag`) |
| Distribuição treino | Normal: 53,5% — Ataque: 46,5% |

---

## Variável-Alvo e Colunas Auxiliares

| Coluna | Tipo | Uso no Projeto | Descrição |
|---|---|---|---|
| `label` | string | Removida antes do treino | Tipo original da conexão: `normal` + 22 categorias de ataque |
| `alvo` | int (0 ou 1) | **Variável-alvo criada pelo projeto** | 0 = Normal, 1 = Ataque (binarização de `label`) |
| `difficulty` | int | **Removida antes do treino** | Nível de dificuldade de classificação atribuído pelos autores do dataset — não é uma feature de tráfego real |

### Categorias originais de `label`

| Categoria | Tipo | Descrição |
|---|---|---|
| `normal` | Normal | Tráfego legítimo |
| `neptune`, `smurf`, `pod`, `teardrop`, `land`, `back`, `apache2`, `udpstorm`, `processtable`, `mailbomb` | DoS | Negação de serviço |
| `ipsweep`, `portsweep`, `nmap`, `satan`, `mscan`, `saint` | Probe | Varredura/reconhecimento |
| `ftp_write`, `guess_passwd`, `imap`, `multihop`, `phf`, `spy`, `warezclient`, `warezmaster` | R2L | Acesso remoto não autorizado |
| `buffer_overflow`, `loadmodule`, `perl`, `rootkit`, `xterm`, `ps`, `sqlattack` | U2R | Escalada de privilégio |

---

## Features Básicas de Conexão TCP/IP (1–9)

| # | Feature | Tipo | Pré-processamento | Descrição |
|---|---|---|---|---|
| 1 | `duration` | numérico contínuo | StandardScaler | Duração da conexão em segundos |
| 2 | `protocol_type` | categórico | LabelEncoder + StandardScaler | Protocolo de rede: `tcp`, `udp`, `icmp` |
| 3 | `service` | categórico | LabelEncoder + StandardScaler | Serviço de destino: `http`, `ftp`, `smtp`, etc. (70 valores distintos) |
| 4 | `flag` | categórico | LabelEncoder + StandardScaler | Status da conexão TCP: `SF` (normal), `S0`, `REJ`, `RSTO`, etc. |
| 5 | `src_bytes` | numérico contínuo | StandardScaler | Bytes enviados da origem ao destino |
| 6 | `dst_bytes` | numérico contínuo | StandardScaler | Bytes enviados do destino à origem |
| 7 | `land` | binário (0/1) | StandardScaler | 1 se origem e destino são o mesmo host/porta (ataque land) |
| 8 | `wrong_fragment` | numérico discreto | StandardScaler | Número de fragmentos com erros |
| 9 | `urgent` | numérico discreto | StandardScaler | Número de pacotes com flag URG |

---

## Features de Conteúdo da Conexão (10–22)

| # | Feature | Tipo | Pré-processamento | Descrição |
|---|---|---|---|---|
| 10 | `hot` | numérico discreto | StandardScaler | Número de indicadores "hot" (acessos a diretórios sensíveis, etc.) |
| 11 | `num_failed_logins` | numérico discreto | StandardScaler | Número de tentativas de login malsucedidas |
| 12 | `logged_in` | binário (0/1) | StandardScaler | 1 se o login foi bem-sucedido |
| 13 | `num_compromised` | numérico discreto | StandardScaler | Número de condições comprometidas detectadas |
| 14 | `root_shell` | binário (0/1) | StandardScaler | 1 se um shell root foi obtido |
| 15 | `su_attempted` | binário (0/1) | StandardScaler | 1 se o comando `su` foi tentado |
| 16 | `num_root` | numérico discreto | StandardScaler | Número de acessos como root |
| 17 | `num_file_creations` | numérico discreto | StandardScaler | Número de operações de criação de arquivo |
| 18 | `num_shells` | numérico discreto | StandardScaler | Número de shells iniciados |
| 19 | `num_access_files` | numérico discreto | StandardScaler | Número de acessos a arquivos de controle |
| 20 | `num_outbound_cmds` | numérico discreto | StandardScaler | Número de comandos de saída em sessão FTP |
| 21 | `is_host_login` | binário (0/1) | StandardScaler | 1 se o login é do tipo host |
| 22 | `is_guest_login` | binário (0/1) | StandardScaler | 1 se o login é do tipo guest |

---

## Features de Tráfego — Janela Temporal de 2 Segundos (23–31)

Estas features descrevem o comportamento de tráfego nos últimos 2 segundos, capturando padrões de varredura e flood.

| # | Feature | Tipo | Pré-processamento | Descrição |
|---|---|---|---|---|
| 23 | `count` | numérico discreto | StandardScaler | Número de conexões ao mesmo host destino nas últimas 2s |
| 24 | `srv_count` | numérico discreto | StandardScaler | Número de conexões ao mesmo serviço nas últimas 2s |
| 25 | `serror_rate` | numérico [0,1] | StandardScaler | % de conexões com erros SYN |
| 26 | `srv_serror_rate` | numérico [0,1] | StandardScaler | % de conexões ao mesmo serviço com erros SYN |
| 27 | `rerror_rate` | numérico [0,1] | StandardScaler | % de conexões com erros REJ (rejeição) |
| 28 | `srv_rerror_rate` | numérico [0,1] | StandardScaler | % de conexões ao mesmo serviço com erros REJ |
| 29 | `same_srv_rate` | numérico [0,1] | StandardScaler | % de conexões ao mesmo serviço |
| 30 | `diff_srv_rate` | numérico [0,1] | StandardScaler | % de conexões a serviços diferentes |
| 31 | `srv_diff_host_rate` | numérico [0,1] | StandardScaler | % de conexões ao mesmo serviço vindas de hosts diferentes |

---

## Features Baseadas no Host de Destino (32–41)

Estas features descrevem o comportamento histórico de tráfego em direção ao host de destino, capturando padrões de longo prazo.

| # | Feature | Tipo | Pré-processamento | Descrição |
|---|---|---|---|---|
| 32 | `dst_host_count` | numérico discreto | StandardScaler | Número de conexões ao mesmo host destino (janela longa) |
| 33 | `dst_host_srv_count` | numérico discreto | StandardScaler | Número de conexões ao mesmo serviço no host destino |
| 34 | `dst_host_same_srv_rate` | numérico [0,1] | StandardScaler | % de conexões ao mesmo serviço no host destino |
| 35 | `dst_host_diff_srv_rate` | numérico [0,1] | StandardScaler | % de conexões a serviços diferentes no host destino |
| 36 | `dst_host_same_src_port_rate` | numérico [0,1] | StandardScaler | % de conexões pela mesma porta de origem |
| 37 | `dst_host_srv_diff_host_rate` | numérico [0,1] | StandardScaler | % de conexões a hosts diferentes para o mesmo serviço |
| 38 | `dst_host_serror_rate` | numérico [0,1] | StandardScaler | % de conexões com erros SYN (perspectiva do host destino) |
| 39 | `dst_host_srv_serror_rate` | numérico [0,1] | StandardScaler | % de conexões ao mesmo serviço com erros SYN |
| 40 | `dst_host_rerror_rate` | numérico [0,1] | StandardScaler | % de conexões com erros REJ (perspectiva do host destino) |
| 41 | `dst_host_srv_rerror_rate` | numérico [0,1] | StandardScaler | % de conexões ao mesmo serviço com erros REJ |

---

## Top 5 Features Mais Importantes (Random Forest)

| Rank | Feature | Importância Relativa | Interpretação para o Problema |
|---|---|---|---|
| 1 | `src_bytes` | Alta | Volume de dados enviados pela origem; ataques DoS enviam volumes anômalos |
| 2 | `dst_bytes` | Alta | Conexões normais geralmente recebem mais dados do servidor |
| 3 | `flag` | Alta | Status TCP diferencia conexões completas (SF) de tentativas abortadas |
| 4 | `logged_in` | Média | Login bem-sucedido é forte indicador de sessão legítima |
| 5 | `count` | Média | Alta frequência de conexões ao mesmo host pode indicar DDoS ou varredura |

---

## Observações sobre Qualidade dos Dados

- **Valores faltantes:** nenhum detectado no treino ou teste.
- **Outliers:** features como `src_bytes` e `dst_bytes` possuem valores extremos; os gráficos EDA aplicam clipping no percentil 95 para visualização, mas os dados originais são mantidos para treino.
- **Features constantes:** `num_outbound_cmds` apresenta variância próxima de zero (quase sempre 0) — informação registrada, mas não removida, pois modelos baseados em árvore a ignoram naturalmente.
