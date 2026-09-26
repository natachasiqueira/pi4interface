from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from data_processing import obter_colunas_modelagem


@dataclass(frozen=True)
class ConfiguracaoClustering:
    vizinhos_imputacao: int = 5
    intervalo_k: tuple[int, int] = (2, 5)
    random_state: int = 42
    n_init: int = 25
    max_iter: int = 500
    tolerancia: float = 0.00005


class ErroModelagem(Exception):
    pass


def obter_configuracao_padrao() -> ConfiguracaoClustering:
    return ConfiguracaoClustering()


def descrever_metodologia_modelagem() -> dict[str, Any]:
    configuracao = obter_configuracao_padrao()
    return {
        "tratamento_nulos": (
            f"KNNImputer com n_neighbors={configuracao.vizinhos_imputacao} "
            "nas variáveis numéricas de treino"
        ),
        "padronizacao": "StandardScaler aplicado nas variáveis de treino",
        "algoritmo": "KMeans com escolha dinâmica de k no intervalo de 2 a 5",
        "metricas": [
            "Método do Cotovelo",
            "Silhouette Score",
            "Davies-Bouldin",
            "Calinski-Harabasz",
        ],
        "variaveis_chave": [
            "Notas das disciplinas",
            "Média Global",
            "Faltas Médias",
            "Transporte",
            "Bolsa Família",
        ],
        "configuracao_modelo": {
            "random_state": configuracao.random_state,
            "n_init": configuracao.n_init,
            "max_iter": configuracao.max_iter,
            "tolerancia": configuracao.tolerancia,
        },
    }


def obter_intervalo_k_disponivel(total_registros: int) -> tuple[int, int]:
    configuracao = obter_configuracao_padrao()
    if total_registros < configuracao.vizinhos_imputacao:
        raise ErroModelagem(
            "A modelagem requer pelo menos 5 alunos no recorte para respeitar "
            "a imputação KNN com n_neighbors=5."
        )

    k_min, k_max = configuracao.intervalo_k
    k_max_ajustado = min(k_max, total_registros - 1)
    if k_max_ajustado < k_min:
        raise ErroModelagem(
            "Não há alunos suficientes no recorte para avaliar o clustering."
        )
    return k_min, k_max_ajustado


def _obter_colunas_treino_validas(df: pd.DataFrame) -> list[str]:
    colunas_modelagem = obter_colunas_modelagem(df)
    colunas_validas = [
        coluna
        for coluna in colunas_modelagem
        if coluna in df.columns and df[coluna].notna().any()
    ]
    if len(colunas_validas) < 2:
        raise ErroModelagem(
            "O conjunto filtrado não possui variáveis suficientes para o clustering."
        )
    return colunas_validas


def _preparar_matriz_modelagem(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    configuracao = obter_configuracao_padrao()
    colunas_treino = _obter_colunas_treino_validas(df)
    dados_treino = df[colunas_treino].apply(pd.to_numeric, errors="coerce")

    imputer = KNNImputer(n_neighbors=configuracao.vizinhos_imputacao)
    dados_imputados = imputer.fit_transform(dados_treino)

    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(dados_imputados)
    df_imputado = pd.DataFrame(dados_imputados, columns=colunas_treino, index=df.index)
    return df_imputado, colunas_treino, dados_imputados, dados_padronizados


def _calcular_curva_cotovelo(dados_padronizados: np.ndarray) -> pd.DataFrame:
    limite_superior = min(10, dados_padronizados.shape[0])
    wcss: list[dict[str, float]] = []
    for valor_k in range(1, limite_superior + 1):
        modelo = KMeans(
            n_clusters=valor_k,
            init="k-means++",
            n_init=10,
            random_state=42,
        )
        modelo.fit(dados_padronizados)
        wcss.append(
            {
                "k": valor_k,
                "inercia": float(modelo.inertia_),
            }
        )
    return pd.DataFrame(wcss)


def _calcular_metricas_validacao(dados_padronizados: np.ndarray) -> pd.DataFrame:
    configuracao = obter_configuracao_padrao()
    _, k_max = obter_intervalo_k_disponivel(dados_padronizados.shape[0])

    metricas: list[dict[str, float]] = []
    for valor_k in range(configuracao.intervalo_k[0], k_max + 1):
        modelo = KMeans(
            n_clusters=valor_k,
            init="k-means++",
            n_init=configuracao.n_init,
            max_iter=configuracao.max_iter,
            tol=configuracao.tolerancia,
            random_state=configuracao.random_state,
        )
        labels = modelo.fit_predict(dados_padronizados)
        metricas.append(
            {
                "k": valor_k,
                "silhouette_score": round(
                    float(silhouette_score(dados_padronizados, labels)),
                    3,
                ),
                "davies_bouldin": round(
                    float(davies_bouldin_score(dados_padronizados, labels)),
                    3,
                ),
                "calinski_harabasz": round(
                    float(calinski_harabasz_score(dados_padronizados, labels)),
                    1,
                ),
            }
        )
    return pd.DataFrame(metricas)


def sugerir_k_recomendado(tabela_metricas: pd.DataFrame) -> int:
    tabela_ranqueada = tabela_metricas.copy()
    tabela_ranqueada["rank_silhouette"] = tabela_ranqueada["silhouette_score"].rank(
        ascending=False,
        method="min",
    )
    tabela_ranqueada["rank_davies"] = tabela_ranqueada["davies_bouldin"].rank(
        ascending=True,
        method="min",
    )
    tabela_ranqueada["rank_calinski"] = tabela_ranqueada["calinski_harabasz"].rank(
        ascending=False,
        method="min",
    )
    tabela_ranqueada["score_recomendacao"] = (
        tabela_ranqueada["rank_silhouette"]
        + tabela_ranqueada["rank_davies"]
        + tabela_ranqueada["rank_calinski"]
    )
    melhor_k = tabela_ranqueada.sort_values(
        by=[
            "score_recomendacao",
            "rank_silhouette",
            "rank_davies",
            "rank_calinski",
        ],
        ascending=[True, True, True, True],
    )["k"].iloc[0]
    return int(melhor_k)


def _rotular_perfis(base_clusterizada: pd.DataFrame, nome_coluna_grupo: str) -> dict[int, str]:
    resumo_clusters = (
        base_clusterizada.groupby(nome_coluna_grupo)
        .agg(
            media_global=("media_global", "mean"),
            faltas_medias=("faltas_medias", "mean"),
        )
        .reset_index()
    )

    desvio_media = resumo_clusters["media_global"].std(ddof=0) or 1.0
    desvio_faltas = resumo_clusters["faltas_medias"].std(ddof=0) or 1.0
    resumo_clusters["score_perfil"] = (
        (resumo_clusters["media_global"] - resumo_clusters["media_global"].mean())
        / desvio_media
    ) - (
        (resumo_clusters["faltas_medias"] - resumo_clusters["faltas_medias"].mean())
        / desvio_faltas
    )

    resumo_clusters = resumo_clusters.sort_values(
        by=["score_perfil", "media_global", "faltas_medias"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    total_clusters = len(resumo_clusters)
    mapeamento: dict[int, str] = {}
    for posicao, grupo in enumerate(resumo_clusters[nome_coluna_grupo].tolist()):
        if posicao == 0:
            mapeamento[int(grupo)] = "Avançado"
        elif posicao == total_clusters - 1:
            mapeamento[int(grupo)] = "Necessita de Reforço"
        elif total_clusters == 3:
            mapeamento[int(grupo)] = "Regular"
        else:
            mapeamento[int(grupo)] = f"Regular {posicao}"
    return mapeamento


def _montar_resumo_perfis(
    base_clusterizada: pd.DataFrame,
    nome_coluna_grupo: str,
    colunas_treino: list[str],
) -> pd.DataFrame:
    colunas_resumo = list(dict.fromkeys(colunas_treino + ["media_global", "faltas_medias"]))
    agregacoes = {coluna: "mean" for coluna in colunas_resumo if coluna in base_clusterizada.columns}
    agregacoes["matricula"] = "count"

    resumo = base_clusterizada.groupby(
        [nome_coluna_grupo, "perfil_cluster", "rotulo_grupo"],
        dropna=False,
    ).agg(agregacoes)
    resumo = resumo.rename(columns={"matricula": "qtd_alunos"}).reset_index()
    return resumo.sort_values(by=[nome_coluna_grupo]).reset_index(drop=True)


def gerar_lista_acao(base_clusterizada: pd.DataFrame) -> dict[str, pd.DataFrame]:
    colunas_acao = [
        "matricula",
        "arquivo_origem",
        "ano_letivo",
        "etapa",
        "perfil_cluster",
        "media_global",
        "faltas_medias",
        "transporte_descricao",
        "bolsa_familia_descricao",
    ]
    colunas_existentes = [coluna for coluna in colunas_acao if coluna in base_clusterizada.columns]

    alunos_reforco = (
        base_clusterizada[base_clusterizada["perfil_cluster"] == "Necessita de Reforço"]
        .sort_values(by=["faltas_medias", "media_global"], ascending=[False, True])
        [colunas_existentes]
        .reset_index(drop=True)
    )
    alunos_avancados = (
        base_clusterizada[base_clusterizada["perfil_cluster"] == "Avançado"]
        .sort_values(by=["media_global", "faltas_medias"], ascending=[False, True])
        [colunas_existentes]
        .reset_index(drop=True)
    )
    return {
        "avancados": alunos_avancados,
        "reforco": alunos_reforco,
    }


@st.cache_data(show_spinner="Executando a modelagem de clustering...")
def executar_pipeline_clustering(
    df: pd.DataFrame,
    k_escolhido: int,
) -> dict[str, Any]:
    total_registros = len(df)
    k_min, k_max = obter_intervalo_k_disponivel(total_registros)
    if not k_min <= k_escolhido <= k_max:
        raise ErroModelagem(
            f"O valor de k deve estar entre {k_min} e {k_max} para o recorte atual."
        )

    configuracao = obter_configuracao_padrao()
    df_imputado, colunas_treino, _, dados_padronizados = _preparar_matriz_modelagem(df)
    curva_cotovelo = _calcular_curva_cotovelo(dados_padronizados)
    tabela_metricas = _calcular_metricas_validacao(dados_padronizados)
    k_recomendado = sugerir_k_recomendado(tabela_metricas)

    modelo_final = KMeans(
        n_clusters=k_escolhido,
        init="k-means++",
        n_init=configuracao.n_init,
        max_iter=configuracao.max_iter,
        tol=configuracao.tolerancia,
        random_state=configuracao.random_state,
    )
    labels = modelo_final.fit_predict(dados_padronizados)

    nome_coluna_grupo = "grupo_cluster"
    base_clusterizada = df.copy().reset_index(drop=True)
    base_clusterizada[nome_coluna_grupo] = labels
    mapeamento_perfis = _rotular_perfis(base_clusterizada, nome_coluna_grupo)
    base_clusterizada["perfil_cluster"] = base_clusterizada[nome_coluna_grupo].map(
        mapeamento_perfis
    )
    base_clusterizada["rotulo_grupo"] = (
        "Grupo " + base_clusterizada[nome_coluna_grupo].astype(str)
    )

    resumo_perfis = _montar_resumo_perfis(
        base_clusterizada,
        nome_coluna_grupo,
        colunas_treino,
    )
    metricas_k_escolhido = (
        tabela_metricas[tabela_metricas["k"] == k_escolhido].iloc[0].to_dict()
    )

    return {
        "base_clusterizada": base_clusterizada,
        "dados_imputados": df_imputado.reset_index(drop=True),
        "colunas_treino": colunas_treino,
        "curva_cotovelo": curva_cotovelo,
        "tabela_metricas": tabela_metricas,
        "metricas_k_escolhido": metricas_k_escolhido,
        "resumo_perfis": resumo_perfis,
        "lista_acao": gerar_lista_acao(base_clusterizada),
        "k_recomendado": k_recomendado,
        "k_minimo": k_min,
        "k_maximo": k_max,
    }
