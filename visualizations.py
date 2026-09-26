from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

PALETA_PERFIS = {
    "Avançado": "#2A9D8F",
    "Regular": "#577590",
    "Regular 1": "#4D908E",
    "Regular 2": "#7A9E9F",
    "Necessita de Reforço": "#E76F51",
}

PALETA_CATEGORIAS = ["#4C78A8", "#72B7B2", "#F4A261", "#E76F51"]


def listar_recursos_dashboard() -> list[str]:
    return [
        "Filtros globais por Ano Letivo, Base de dados e Etapa na sidebar.",
        "Indicadores executivos com foco em desempenho, assiduidade e segmentação.",
        "Gráficos de impacto socioeconômico e logístico para Transporte e Bolsa Família.",
        "Mapa de calor de correlação entre notas, faltas e fatores sociais.",
        "Dispersão interativa Média Global x Faltas Médias com clusters e hover de Matrícula.",
        "Tabela de ação para alunos avançados e alunos com necessidade de reforço.",
    ]


def _titulo_categoria(coluna: str) -> str:
    return {
        "transporte_descricao": "Transporte",
        "bolsa_familia_descricao": "Bolsa Família",
    }.get(coluna, coluna.replace("_", " ").title())


def _gerar_base_resumo_categoria(df: pd.DataFrame, coluna_categoria: str) -> pd.DataFrame:
    resumo = (
        df.groupby(coluna_categoria, dropna=False)
        .agg(
            qtd_alunos=("matricula", "count"),
            media_global=("media_global", "mean"),
            faltas_medias=("faltas_medias", "mean"),
        )
        .reset_index()
        .rename(columns={coluna_categoria: "categoria"})
    )
    resumo["media_global"] = resumo["media_global"].round(2)
    resumo["faltas_medias"] = resumo["faltas_medias"].round(2)
    return resumo


def criar_grafico_impacto_categoria(df: pd.DataFrame, coluna_categoria: str) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    resumo = _gerar_base_resumo_categoria(df, coluna_categoria)
    titulo = _titulo_categoria(coluna_categoria)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{titulo} x Média Global",
            f"{titulo} x Faltas Médias",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=resumo["categoria"],
            y=resumo["media_global"],
            marker_color=PALETA_CATEGORIAS[: len(resumo)],
            text=resumo["media_global"],
            textposition="outside",
            name="Média Global",
            hovertemplate="%{x}<br>Média Global: %{y:.2f}<br>Alunos: %{customdata}<extra></extra>",
            customdata=resumo["qtd_alunos"],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=resumo["categoria"],
            y=resumo["faltas_medias"],
            marker_color=PALETA_CATEGORIAS[: len(resumo)],
            text=resumo["faltas_medias"],
            textposition="outside",
            name="Faltas Médias",
            hovertemplate="%{x}<br>Faltas Médias: %{y:.2f}<br>Alunos: %{customdata}<extra></extra>",
            customdata=resumo["qtd_alunos"],
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
        title=f"Impacto de {titulo} no Desempenho e na Assiduidade",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Média Global", row=1, col=1)
    fig.update_yaxes(title_text="Faltas Médias", row=1, col=2)
    return fig


def criar_tabela_resumo_categoria(df: pd.DataFrame, coluna_categoria: str) -> pd.DataFrame:
    resumo = _gerar_base_resumo_categoria(df, coluna_categoria)
    return resumo.rename(
        columns={
            "categoria": _titulo_categoria(coluna_categoria),
            "qtd_alunos": "Quantidade de Alunos",
            "media_global": "Média Global",
            "faltas_medias": "Faltas Médias",
        }
    )


def criar_mapa_calor_correlacao(df: pd.DataFrame, colunas_modelagem: list[str]) -> Any:
    import plotly.graph_objects as go

    colunas_correlacao = [
        coluna
        for coluna in colunas_modelagem
        if coluna in df.columns and df[coluna].notna().any()
    ]
    matriz = df[colunas_correlacao].corr(numeric_only=True).round(2)

    figura = go.Figure(
        data=go.Heatmap(
            z=matriz.values,
            x=matriz.columns.tolist(),
            y=matriz.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=matriz.values,
            texttemplate="%{text:.2f}",
            hovertemplate=(
                "Variável X: %{x}<br>"
                "Variável Y: %{y}<br>"
                "Correlação: %{z:.2f}<extra></extra>"
            ),
            colorbar=dict(title="Correlação"),
        )
    )
    figura.update_layout(
        title="Mapa de Calor de Correlação",
        template="plotly_white",
        height=720,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    figura.update_xaxes(tickangle=45)
    return figura


def criar_grafico_dispersao_clusters(base_clusterizada: pd.DataFrame) -> Any:
    import plotly.express as px

    categorias_ordenadas = [
        categoria
        for categoria in ["Avançado", "Regular", "Regular 1", "Regular 2", "Necessita de Reforço"]
        if categoria in base_clusterizada["perfil_cluster"].dropna().unique().tolist()
    ]
    fig = px.scatter(
        base_clusterizada,
        x="media_global",
        y="faltas_medias",
        color="perfil_cluster",
        color_discrete_map=PALETA_PERFIS,
        category_orders={"perfil_cluster": categorias_ordenadas},
        hover_data={
            "matricula": True,
            "arquivo_origem": True,
            "ano_letivo": True,
            "etapa": True,
            "grupo_cluster": True,
            "media_global": ":.2f",
            "faltas_medias": ":.2f",
        },
        title="Dispersão dos Perfis de Aprendizagem",
        labels={
            "media_global": "Média Global",
            "faltas_medias": "Faltas Médias",
            "perfil_cluster": "Perfil",
        },
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.7, color="white")))
    fig.update_layout(
        template="plotly_white",
        legend_title="Perfil",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def criar_grafico_cotovelo(curva_cotovelo: pd.DataFrame) -> Any:
    import plotly.express as px

    fig = px.line(
        curva_cotovelo,
        x="k",
        y="inercia",
        markers=True,
        title="Método do Cotovelo",
        labels={"k": "Número de Grupos (k)", "inercia": "WCSS (Inércia)"},
    )
    fig.update_traces(line_color="#4C78A8", marker_color="#4C78A8")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def criar_grafico_metricas_validacao(tabela_metricas: pd.DataFrame) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Silhouette (maior é melhor)",
            "Davies-Bouldin (menor é melhor)",
            "Calinski-Harabasz (maior é melhor)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=tabela_metricas["k"],
            y=tabela_metricas["silhouette_score"],
            mode="lines+markers",
            line=dict(color="#2A9D8F"),
            name="Silhouette",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tabela_metricas["k"],
            y=tabela_metricas["davies_bouldin"],
            mode="lines+markers",
            line=dict(color="#E76F51"),
            name="Davies-Bouldin",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=tabela_metricas["k"],
            y=tabela_metricas["calinski_harabasz"],
            mode="lines+markers",
            line=dict(color="#577590"),
            name="Calinski-Harabasz",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title="Comparativo das Métricas de Validação",
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(title_text="k", dtick=1)
    return fig


def gerar_mensagem_impacto(df: pd.DataFrame, coluna_categoria: str) -> str:
    resumo = _gerar_base_resumo_categoria(df, coluna_categoria)
    titulo = _titulo_categoria(coluna_categoria)

    if resumo.empty or len(resumo) < 2:
        return f"Não há variação suficiente em {titulo.lower()} para comparar os grupos."

    maior_media = resumo.loc[resumo["media_global"].idxmax()]
    maior_falta = resumo.loc[resumo["faltas_medias"].idxmax()]
    return (
        f"Em {titulo}, o grupo '{maior_media['categoria']}' apresenta a maior média global "
        f"({maior_media['media_global']:.2f}), enquanto '{maior_falta['categoria']}' concentra "
        f"as maiores faltas médias ({maior_falta['faltas_medias']:.2f})."
    )


def converter_dataframe_csv(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


def criar_tabela_acao_visual(df: pd.DataFrame) -> pd.DataFrame:
    colunas_renomeadas = {
        "matricula": "Matrícula",
        "arquivo_origem": "Arquivo de Origem",
        "ano_letivo": "Ano Letivo",
        "etapa": "Etapa",
        "perfil_cluster": "Perfil",
        "media_global": "Média Global",
        "faltas_medias": "Faltas Médias",
        "transporte_descricao": "Transporte",
        "bolsa_familia_descricao": "Bolsa Família",
    }
    tabela = df.rename(columns=colunas_renomeadas).copy()
    if "Média Global" in tabela.columns:
        tabela["Média Global"] = tabela["Média Global"].round(2)
    if "Faltas Médias" in tabela.columns:
        tabela["Faltas Médias"] = tabela["Faltas Médias"].round(2)
    return tabela


def criar_indicadores_acao(lista_acao: dict[str, pd.DataFrame]) -> dict[str, Any]:
    avancados = lista_acao["avancados"]
    reforco = lista_acao["reforco"]
    return {
        "alunos_avancados": int(len(avancados)),
        "alunos_reforco": int(len(reforco)),
        "maior_media_avancados": round(float(avancados["media_global"].max()), 2)
        if not avancados.empty
        else 0.0,
        "maior_falta_reforco": round(float(reforco["faltas_medias"].max()), 2)
        if not reforco.empty
        else 0.0,
    }
