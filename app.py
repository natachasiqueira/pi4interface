from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data_processing import (
    ErroDadosEducacionais,
    carregar_bases_consolidadas,
    descrever_contrato_saida,
    descrever_estrategia_concatenacao,
    filtrar_base,
    gerar_relatorio_qualidade,
    gerar_resumo_base,
    listar_bases_disponiveis,
    obter_amostra_exibicao,
    obter_colunas_modelagem,
    obter_opcoes_filtros,
)
from ml_models import (
    ErroModelagem,
    descrever_metodologia_modelagem,
    executar_pipeline_clustering,
    obter_intervalo_k_disponivel,
)
from visualizations import (
    converter_dataframe_csv,
    criar_grafico_cotovelo,
    criar_grafico_dispersao_clusters,
    criar_grafico_impacto_categoria,
    criar_grafico_metricas_validacao,
    criar_indicadores_acao,
    criar_mapa_calor_correlacao,
    criar_tabela_acao_visual,
    criar_tabela_resumo_categoria,
    descrever_paineis_planejados,
    gerar_mensagem_impacto,
)


st.set_page_config(
    page_title="Dashboard Educacional",
    page_icon="📚",
    layout="wide",
)


def carregar_contexto_dados() -> tuple[pd.DataFrame, dict[str, list[int] | list[str]]]:
    try:
        base_consolidada = carregar_bases_consolidadas()
    except ErroDadosEducacionais as erro:
        st.error(str(erro))
        st.stop()
    except Exception:
        st.error(
            "Ocorreu um problema ao preparar a base educacional. "
            "Revise os arquivos Excel e tente novamente."
        )
        st.stop()

    opcoes = obter_opcoes_filtros(base_consolidada)
    return base_consolidada, opcoes


def renderizar_sidebar(
    opcoes: dict[str, list[int] | list[str]],
) -> tuple[list[str], list[int], list[int]]:
    st.sidebar.title("Filtros Globais")
    st.sidebar.success("Etapa 5 concluída")
    st.sidebar.caption("Experiência final integrada no Streamlit")

    bases = listar_bases_disponiveis()
    st.sidebar.markdown("**Bases localizadas**")
    for base in bases:
        st.sidebar.write(f"- {base}")

    arquivos_origem = st.sidebar.multiselect(
        "Base de dados",
        options=opcoes["arquivos_origem"],
        default=opcoes["arquivos_origem"],
    )
    anos_letivos = st.sidebar.multiselect(
        "Ano Letivo",
        options=opcoes["anos_letivos"],
        default=opcoes["anos_letivos"],
    )
    etapas = st.sidebar.multiselect(
        "Etapa",
        options=opcoes["etapas"],
        default=opcoes["etapas"],
        format_func=lambda valor: f"{valor}º ano",
    )

    st.sidebar.markdown("**Próxima etapa**")
    st.sidebar.write("6. Empacotamento e deploy")
    return arquivos_origem, anos_letivos, etapas


def renderizar_controles_modelagem(base_filtrada: pd.DataFrame) -> int | None:
    st.sidebar.markdown("**Parâmetros do Modelo**")
    try:
        k_min, k_max = obter_intervalo_k_disponivel(len(base_filtrada))
    except ErroModelagem as erro:
        st.sidebar.warning(str(erro))
        return None

    valor_padrao = 3 if k_min <= 3 <= k_max else k_min
    return st.sidebar.slider(
        "Número de grupos (k)",
        min_value=k_min,
        max_value=k_max,
        value=valor_padrao,
        step=1,
        help="Use este controle para testar diferentes segmentações entre 2 e 5 grupos.",
    )


def gerar_indicadores_executivos(
    base_filtrada: pd.DataFrame,
    resultado_modelagem: dict[str, Any] | None,
) -> dict[str, Any]:
    resumo = gerar_resumo_base(base_filtrada)
    indicadores: dict[str, Any] = {
        "alunos": resumo["total_alunos"],
        "bases": resumo["total_bases"],
        "anos": len(resumo["anos_letivos"]),
        "media_global": round(float(base_filtrada["media_global"].mean()), 2),
        "faltas_medias": round(float(base_filtrada["faltas_medias"].mean()), 2),
        "variaveis_modelagem": resumo["total_colunas_modelagem"],
    }

    if resultado_modelagem is None:
        indicadores["alunos_avancados"] = 0
        indicadores["alunos_reforco"] = 0
        indicadores["percentual_reforco"] = 0.0
        return indicadores

    lista_acao = resultado_modelagem["lista_acao"]
    qtd_reforco = len(lista_acao["reforco"])
    qtd_avancados = len(lista_acao["avancados"])
    indicadores["alunos_avancados"] = qtd_avancados
    indicadores["alunos_reforco"] = qtd_reforco
    indicadores["percentual_reforco"] = round((qtd_reforco / max(len(base_filtrada), 1)) * 100, 1)
    return indicadores


def gerar_mensagens_executivas(
    base_filtrada: pd.DataFrame,
    resultado_modelagem: dict[str, Any] | None,
    k_escolhido: int | None,
) -> list[str]:
    mensagens = [
        (
            f"O recorte atual reúne {len(base_filtrada)} alunos, com média global de "
            f"{base_filtrada['media_global'].mean():.2f} e faltas médias de "
            f"{base_filtrada['faltas_medias'].mean():.2f}."
        )
    ]

    if resultado_modelagem is None or k_escolhido is None:
        mensagens.append(
            "A modelagem ainda não foi executada neste recorte. Ajuste os filtros ou aumente a base para ativar os perfis."
        )
        return mensagens

    lista_acao = resultado_modelagem["lista_acao"]
    metricas = resultado_modelagem["metricas_k_escolhido"]
    mensagens.append(
        f"Com k={k_escolhido}, o modelo identificou {len(lista_acao['reforco'])} alunos com necessidade de reforço e "
        f"{len(lista_acao['avancados'])} alunos avançados."
    )
    mensagens.append(
        f"As métricas do recorte atual ficaram em Silhouette {metricas['silhouette_score']}, "
        f"Davies-Bouldin {metricas['davies_bouldin']} e Calinski-Harabasz {metricas['calinski_harabasz']}."
    )
    return mensagens


def renderizar_cabecalho(
    base_filtrada: pd.DataFrame,
    resultado_modelagem: dict[str, Any] | None,
    k_escolhido: int | None,
) -> None:
    st.title("Dashboard Educacional")
    st.caption(
        "Etapa 5: experiência integrada no Streamlit, com navegação orientada à decisão pedagógica."
    )

    indicadores = gerar_indicadores_executivos(base_filtrada, resultado_modelagem)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Alunos no recorte", indicadores["alunos"])
    col2.metric("Bases carregadas", indicadores["bases"])
    col3.metric("Anos letivos", indicadores["anos"])
    col4.metric("Média Global", indicadores["media_global"])
    col5.metric("Faltas Médias", indicadores["faltas_medias"])
    col6.metric("Alunos para reforço", indicadores["alunos_reforco"])

    for mensagem in gerar_mensagens_executivas(base_filtrada, resultado_modelagem, k_escolhido):
        st.info(mensagem)


def renderizar_estado_vazio() -> None:
    st.warning(
        "Os filtros selecionados não retornaram alunos. Ajuste Base de dados, Ano Letivo ou Etapa para continuar a análise."
    )


def renderizar_estado_modelagem_indisponivel() -> None:
    st.info(
        "O ETL está pronto, mas o recorte atual ainda não possui alunos suficientes para executar a modelagem com KNNImputer(n_neighbors=5)."
    )


def renderizar_painel_executivo(
    base_filtrada: pd.DataFrame,
    resultado_modelagem: dict[str, Any] | None,
    k_escolhido: int | None,
) -> None:
    st.subheader("Painel Executivo")
    st.caption(
        "Visão geral do recorte selecionado, com foco no que precisa de atenção imediata."
    )

    indicadores = gerar_indicadores_executivos(base_filtrada, resultado_modelagem)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variáveis de modelagem", indicadores["variaveis_modelagem"])
    col2.metric("Alunos avançados", indicadores["alunos_avancados"])
    col3.metric("% em reforço", indicadores["percentual_reforco"])
    col4.metric("k aplicado", k_escolhido if k_escolhido is not None else "N/A")

    renderizar_impactos_socioeconomicos(base_filtrada)

    if resultado_modelagem is None:
        renderizar_estado_modelagem_indisponivel()
        return

    st.markdown("**Dispersão dos perfis de aprendizagem**")
    st.plotly_chart(
        criar_grafico_dispersao_clusters(resultado_modelagem["base_clusterizada"]),
        use_container_width=True,
    )


def renderizar_impactos_socioeconomicos(base_filtrada: pd.DataFrame) -> None:
    st.subheader("Impacto Socioeconômico e Logístico")
    st.caption(
        "Estas visualizações ajudam a responder se transporte e Bolsa Família estão associados a mudanças de desempenho ou de assiduidade."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            criar_grafico_impacto_categoria(base_filtrada, "transporte_descricao"),
            use_container_width=True,
        )
        st.info(gerar_mensagem_impacto(base_filtrada, "transporte_descricao"))
        st.dataframe(
            criar_tabela_resumo_categoria(base_filtrada, "transporte_descricao"),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.plotly_chart(
            criar_grafico_impacto_categoria(base_filtrada, "bolsa_familia_descricao"),
            use_container_width=True,
        )
        st.info(gerar_mensagem_impacto(base_filtrada, "bolsa_familia_descricao"))
        st.dataframe(
            criar_tabela_resumo_categoria(base_filtrada, "bolsa_familia_descricao"),
            use_container_width=True,
            hide_index=True,
        )


def renderizar_diagnostico(
    base_filtrada: pd.DataFrame,
    resultado_modelagem: dict[str, Any] | None,
) -> None:
    st.subheader("Diagnóstico")
    st.caption(
        "Leituras para entender a qualidade do agrupamento e a relação entre as variáveis do recorte."
    )

    st.markdown("**Mapa de calor de correlação**")
    st.pyplot(
        criar_mapa_calor_correlacao(base_filtrada, obter_colunas_modelagem(base_filtrada)),
        clear_figure=True,
    )

    if resultado_modelagem is None:
        renderizar_estado_modelagem_indisponivel()
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Comparativo visual das métricas**")
        st.plotly_chart(
            criar_grafico_metricas_validacao(resultado_modelagem["tabela_metricas"]),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Método do Cotovelo**")
        st.plotly_chart(
            criar_grafico_cotovelo(resultado_modelagem["curva_cotovelo"]),
            use_container_width=True,
        )

    with st.expander("Ver tabelas numéricas da validação"):
        st.dataframe(
            resultado_modelagem["tabela_metricas"],
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            resultado_modelagem["curva_cotovelo"],
            use_container_width=True,
            hide_index=True,
        )


def renderizar_perfis(
    resultado_modelagem: dict[str, Any] | None,
) -> None:
    st.subheader("Perfis de Aprendizagem")
    st.caption(
        "Leitura dos grupos encontrados pelo modelo e de como eles se distribuem no recorte."
    )

    if resultado_modelagem is None:
        renderizar_estado_modelagem_indisponivel()
        return

    metricas = resultado_modelagem["metricas_k_escolhido"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("k recomendado", int(resultado_modelagem["k_recomendado"]))
    col2.metric("Silhouette", metricas["silhouette_score"])
    col3.metric("Davies-Bouldin", metricas["davies_bouldin"])
    col4.metric("Calinski-Harabasz", metricas["calinski_harabasz"])
    col5.metric("Grupos gerados", resultado_modelagem["base_clusterizada"]["grupo_cluster"].nunique())

    st.plotly_chart(
        criar_grafico_dispersao_clusters(resultado_modelagem["base_clusterizada"]),
        use_container_width=True,
    )

    st.markdown("**Resumo consolidado dos perfis**")
    st.dataframe(
        resultado_modelagem["resumo_perfis"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Amostra da base clusterizada**")
    base_clusterizada = resultado_modelagem["base_clusterizada"]
    colunas_amostra = [
        "matricula",
        "arquivo_origem",
        "ano_letivo",
        "etapa",
        "media_global",
        "faltas_medias",
        "grupo_cluster",
        "perfil_cluster",
        "transporte_descricao",
        "bolsa_familia_descricao",
    ]
    colunas_existentes = [coluna for coluna in colunas_amostra if coluna in base_clusterizada.columns]
    st.dataframe(
        base_clusterizada[colunas_existentes],
        use_container_width=True,
        hide_index=True,
    )


def renderizar_acao(resultado_modelagem: dict[str, Any] | None) -> None:
    st.subheader("Lista de Ação")
    st.caption(
        "Apoio direto à intervenção pedagógica, com listas exportáveis para acompanhamento."
    )

    if resultado_modelagem is None:
        renderizar_estado_modelagem_indisponivel()
        return

    lista_acao = resultado_modelagem["lista_acao"]
    indicadores = criar_indicadores_acao(lista_acao)

    ind1, ind2, ind3, ind4 = st.columns(4)
    ind1.metric("Alunos avançados", indicadores["alunos_avancados"])
    ind2.metric("Alunos para reforço", indicadores["alunos_reforco"])
    ind3.metric("Maior média dos avançados", indicadores["maior_media_avancados"])
    ind4.metric("Maior falta no reforço", indicadores["maior_falta_reforco"])

    tabela_reforco = criar_tabela_acao_visual(lista_acao["reforco"])
    tabela_avancados = criar_tabela_acao_visual(lista_acao["avancados"])
    tabela_clusterizada = resultado_modelagem["base_clusterizada"].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Alunos com necessidade de reforço**")
        st.dataframe(tabela_reforco, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar reforço (.csv)",
            data=converter_dataframe_csv(tabela_reforco),
            file_name="alunos_reforco.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("**Alunos avançados**")
        st.dataframe(tabela_avancados, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar avançados (.csv)",
            data=converter_dataframe_csv(tabela_avancados),
            file_name="alunos_avancados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("**Exportação do recorte clusterizado**")
    st.download_button(
        "Exportar base clusterizada (.csv)",
        data=converter_dataframe_csv(tabela_clusterizada),
        file_name="base_clusterizada.csv",
        mime="text/csv",
        use_container_width=True,
    )


def renderizar_dados_metodo(
    base_filtrada: pd.DataFrame,
    resultado_modelagem: dict[str, Any] | None,
) -> None:
    st.subheader("Dados e Método")
    st.caption(
        "Área de apoio para leitura técnica, auditoria do recorte e entendimento do pipeline."
    )

    tab1, tab2, tab3 = st.tabs(["Base", "Qualidade", "Metodologia"])

    with tab1:
        resumo = gerar_resumo_base(base_filtrada)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Arquivos de origem presentes**")
            st.write(resumo["arquivos_origem"])
            st.markdown("**Anos letivos presentes**")
            st.write(resumo["anos_letivos"])
            st.markdown("**Etapas presentes**")
            st.write([f"{etapa}º ano" for etapa in resumo["etapas"]])
        with col2:
            st.markdown("**Colunas selecionadas para a modelagem**")
            st.write(resumo["features_modelagem"])
        st.markdown("**Amostra pronta para análise**")
        st.dataframe(
            obter_amostra_exibicao(base_filtrada),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.dataframe(
            gerar_relatorio_qualidade(base_filtrada),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Contrato analítico da saída do ETL**")
        st.json(descrever_contrato_saida())

    with tab3:
        st.markdown("**Estratégia de dados para múltiplos anos e múltiplas bases**")
        for item in descrever_estrategia_concatenacao():
            st.write(f"- {item}")

        st.markdown("**Metodologia de modelagem aprovada**")
        st.json(descrever_metodologia_modelagem())

        if resultado_modelagem is not None:
            st.markdown("**Status do conjunto analítico**")
            st.write(
                {
                    "Colunas de modelagem detectadas": obter_colunas_modelagem(base_filtrada),
                    "Registros após filtros": len(base_filtrada),
                    "Arquivos ativos no recorte": gerar_resumo_base(base_filtrada)["arquivos_origem"],
                    "k recomendado": int(resultado_modelagem["k_recomendado"]),
                }
            )

        with st.expander("Painéis previstos para o dashboard"):
            for painel in descrever_paineis_planejados():
                st.write(f"- {painel}")


def renderizar_tabs_principais(
    base_filtrada: pd.DataFrame,
    resultado_modelagem: dict[str, Any] | None,
    k_escolhido: int | None,
) -> None:
    aba1, aba2, aba3, aba4, aba5 = st.tabs(
        [
            "Painel Executivo",
            "Diagnóstico",
            "Perfis",
            "Ação",
            "Dados e Método",
        ]
    )

    with aba1:
        renderizar_painel_executivo(base_filtrada, resultado_modelagem, k_escolhido)
    with aba2:
        renderizar_diagnostico(base_filtrada, resultado_modelagem)
    with aba3:
        renderizar_perfis(resultado_modelagem)
    with aba4:
        renderizar_acao(resultado_modelagem)
    with aba5:
        renderizar_dados_metodo(base_filtrada, resultado_modelagem)


def main() -> None:
    base_consolidada, opcoes = carregar_contexto_dados()
    arquivos_origem, anos_letivos, etapas = renderizar_sidebar(opcoes)
    base_filtrada = filtrar_base(
        base_consolidada,
        arquivos_origem=arquivos_origem,
        anos_letivos=anos_letivos,
        etapas=etapas,
    )

    if base_filtrada.empty:
        st.title("Dashboard Educacional")
        renderizar_estado_vazio()
        return

    k_escolhido = renderizar_controles_modelagem(base_filtrada)
    resultado_modelagem: dict[str, Any] | None = None

    if k_escolhido is not None:
        try:
            resultado_modelagem = executar_pipeline_clustering(base_filtrada, k_escolhido)
        except ErroModelagem as erro:
            st.warning(str(erro))
        except Exception:
            st.error(
                "Não foi possível concluir a modelagem deste recorte. Revise os filtros aplicados e tente novamente."
            )

    renderizar_cabecalho(base_filtrada, resultado_modelagem, k_escolhido)
    renderizar_tabs_principais(base_filtrada, resultado_modelagem, k_escolhido)


if __name__ == "__main__":
    main()
