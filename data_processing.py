from __future__ import annotations

import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as xml

import pandas as pd
import streamlit as st


PADRAO_ANO = re.compile(r"(20\d{2})")
PADRAO_ETAPA = re.compile(r"^Etapa_(20\d{2})_\d+$")
PADRAO_TRANSPORTE = re.compile(r"^Transporte_(20\d{2})_\d+$")
PADRAO_BF = re.compile(r"^BF_(20\d{2})_\d+$")
PADRAO_MEDIA_DISCIPLINA = re.compile(r"^M_(.+)_(20\d{2})$")
PADRAO_NOTA_BIMESTRE = re.compile(r"^(.+?)_(20\d{2})_(\d+)$")


@dataclass(frozen=True)
class ContratoDadosEducacionais:
    coluna_identificador: str = "Matrícula"
    coluna_media_global: str = "Média Global"
    coluna_faltas_medias: str = "Faltas Médias"
    coluna_ano_letivo: str = "ano_letivo"
    coluna_etapa: str = "etapa"
    coluna_transporte: str = "transporte"
    coluna_bolsa_familia: str = "bolsa_familia"


class ErroDadosEducacionais(Exception):
    pass


def obter_raiz_projeto() -> Path:
    return Path(__file__).resolve().parent


def obter_pasta_dados() -> Path:
    pasta_dados = obter_raiz_projeto() / "data"
    return pasta_dados if pasta_dados.exists() else obter_raiz_projeto()


def obter_pastas_busca_dados() -> list[Path]:
    raiz = obter_raiz_projeto()
    pasta_dados = raiz / "data"
    pastas = [raiz]
    if pasta_dados.exists():
        pastas.insert(0, pasta_dados)
    return pastas


def listar_arquivos_excel() -> list[Path]:
    arquivos_por_caminho: dict[Path, Path] = {}
    for pasta in obter_pastas_busca_dados():
        for arquivo in pasta.glob("*.xlsx"):
            arquivos_por_caminho[arquivo.resolve()] = arquivo
    return sorted(arquivos_por_caminho.values(), key=lambda arquivo: arquivo.name.lower())


def extrair_ano_do_nome_arquivo(nome_arquivo: str) -> int | None:
    correspondencia = PADRAO_ANO.search(nome_arquivo)
    return int(correspondencia.group(1)) if correspondencia else None


def _normalizar_identificador_coluna(texto: str) -> str:
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    texto_ascii = texto_normalizado.encode("ascii", "ignore").decode("ascii")
    texto_ascii = re.sub(r"[^a-zA-Z0-9]+", "_", texto_ascii).strip("_").lower()
    return texto_ascii


def _ler_texto_planilha(
    celula: xml.Element,
    shared_strings: list[str],
    namespace: dict[str, str],
) -> str:
    tipo = celula.attrib.get("t")
    valor = celula.find("a:v", namespace)
    if valor is None or valor.text is None:
        return ""
    if tipo == "s":
        return shared_strings[int(valor.text)]
    return valor.text


def _obter_caminho_primeira_planilha(planilha_zip: zipfile.ZipFile) -> str:
    namespace = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    workbook = xml.fromstring(planilha_zip.read("xl/workbook.xml"))
    primeira_sheet = workbook.find("a:sheets/a:sheet", namespace)
    if primeira_sheet is None:
        raise ValueError("A planilha não possui abas válidas para leitura.")

    rel_id = primeira_sheet.attrib.get(
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    )
    rels = xml.fromstring(planilha_zip.read("xl/_rels/workbook.xml.rels"))
    for relacao in rels.findall("rel:Relationship", namespace):
        if relacao.attrib.get("Id") == rel_id:
            destino = relacao.attrib.get("Target", "")
            return destino if destino.startswith("xl/") else f"xl/{destino}"

    raise ValueError("Não foi possível localizar a primeira aba da planilha.")


@st.cache_data(show_spinner=False)
def extrair_cabecalhos_excel(caminho_arquivo: str) -> list[str]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho.name}")

    with zipfile.ZipFile(caminho) as planilha_zip:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in planilha_zip.namelist():
            raiz_strings = xml.fromstring(planilha_zip.read("xl/sharedStrings.xml"))
            for item in raiz_strings.findall("a:si", namespace):
                texto = "".join(no.text or "" for no in item.iterfind(".//a:t", namespace))
                shared_strings.append(texto)

        caminho_planilha = _obter_caminho_primeira_planilha(planilha_zip)
        worksheet = xml.fromstring(planilha_zip.read(caminho_planilha))
        primeira_linha = worksheet.find("a:sheetData/a:row", namespace)
        if primeira_linha is None:
            return []

        return [
            _ler_texto_planilha(celula, shared_strings, namespace).strip()
            for celula in primeira_linha.findall("a:c", namespace)
        ]


def validar_contrato_basico(colunas: list[str]) -> dict[str, Any]:
    contrato = ContratoDadosEducacionais()
    colunas_encontradas = set(colunas)

    obrigatorias = [
        contrato.coluna_identificador,
        contrato.coluna_media_global,
        contrato.coluna_faltas_medias,
    ]
    obrigatorias_ausentes = [
        coluna for coluna in obrigatorias if coluna not in colunas_encontradas
    ]

    colunas_etapa = [coluna for coluna in colunas if PADRAO_ETAPA.match(coluna)]
    colunas_transporte = [
        coluna for coluna in colunas if PADRAO_TRANSPORTE.match(coluna)
    ]
    colunas_bf = [coluna for coluna in colunas if PADRAO_BF.match(coluna)]
    anos_disponiveis = sorted(
        {
            int(ano)
            for coluna in colunas
            for ano in PADRAO_ANO.findall(coluna)
        }
    )

    return {
        "colunas_obrigatorias_ausentes": obrigatorias_ausentes,
        "colunas_etapa": colunas_etapa,
        "colunas_transporte": colunas_transporte,
        "colunas_bf": colunas_bf,
        "anos_identificados": anos_disponiveis,
        "contrato_valido": not obrigatorias_ausentes and bool(colunas_etapa),
    }


def resumir_estrutura_planilha(caminho_arquivo: Path) -> dict[str, Any]:
    colunas = extrair_cabecalhos_excel(str(caminho_arquivo))
    validacao = validar_contrato_basico(colunas)

    return {
        "arquivo": caminho_arquivo.name,
        "ano_sugerido": extrair_ano_do_nome_arquivo(caminho_arquivo.name),
        "quantidade_colunas": len(colunas),
        "colunas_principais": {
            "identificador": ContratoDadosEducacionais().coluna_identificador,
            "media_global": ContratoDadosEducacionais().coluna_media_global,
            "faltas_medias": ContratoDadosEducacionais().coluna_faltas_medias,
        },
        "validacao": validacao,
        "amostra_colunas": colunas[:20],
    }


@st.cache_data(show_spinner=False)
def listar_bases_disponiveis() -> list[str]:
    return [arquivo.name for arquivo in listar_arquivos_excel()]


def obter_fontes_disponiveis(df: pd.DataFrame) -> list[str]:
    if "arquivo_origem" not in df.columns:
        return []
    return sorted(df["arquivo_origem"].dropna().astype(str).unique().tolist())


def _selecionar_coluna_contexto(
    colunas: list[str],
    padrao: re.Pattern[str],
    ano: int,
    descricao: str,
) -> str:
    candidatas = [
        coluna for coluna in colunas if padrao.match(coluna) and f"_{ano}_" in coluna
    ]
    if not candidatas:
        raise ErroDadosEducacionais(
            f"A coluna de {descricao} do ano {ano} não foi localizada na base."
        )
    return sorted(candidatas)[0]


def _obter_anos_disponiveis_colunas(colunas: list[str], nome_arquivo: str) -> list[int]:
    anos_colunas = sorted(
        {
            int(ano)
            for coluna in colunas
            for ano in PADRAO_ANO.findall(coluna)
        }
    )
    ano_arquivo = extrair_ano_do_nome_arquivo(nome_arquivo)
    if ano_arquivo and ano_arquivo in anos_colunas:
        return [ano_arquivo]
    return anos_colunas or ([ano_arquivo] if ano_arquivo else [])


def _extrair_notas_agregadas(df: pd.DataFrame, ano: int) -> dict[str, pd.Series]:
    notas: dict[str, pd.Series] = {}
    for coluna in df.columns:
        correspondencia = PADRAO_MEDIA_DISCIPLINA.match(coluna)
        if not correspondencia or int(correspondencia.group(2)) != ano:
            continue

        disciplina = correspondencia.group(1)
        nome_padronizado = f"nota_{_normalizar_identificador_coluna(disciplina)}"
        notas[nome_padronizado] = pd.to_numeric(df[coluna], errors="coerce")
    return notas


def _extrair_notas_bimestrais(df: pd.DataFrame, ano: int) -> dict[str, pd.Series]:
    colunas_agrupadas: dict[str, list[str]] = {}
    colunas_ignorar = {
        "etapa",
        "transporte",
        "bf",
        str(ano),
    }
    for coluna in df.columns:
        correspondencia = PADRAO_NOTA_BIMESTRE.match(coluna)
        if not correspondencia or int(correspondencia.group(2)) != ano:
            continue

        base_disciplina = correspondencia.group(1)
        identificador_base = _normalizar_identificador_coluna(base_disciplina)
        if identificador_base.startswith("f") or identificador_base in colunas_ignorar:
            continue

        nome_padronizado = f"nota_{identificador_base}"
        colunas_agrupadas.setdefault(nome_padronizado, []).append(coluna)

    notas_bimestrais: dict[str, pd.Series] = {}
    for nome_padronizado, colunas_disciplina in colunas_agrupadas.items():
        dados_disciplina = df[colunas_disciplina].apply(pd.to_numeric, errors="coerce")
        notas_bimestrais[nome_padronizado] = dados_disciplina.mean(axis=1)
    return notas_bimestrais


def _extrair_notas_padronizadas(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    notas_agregadas = _extrair_notas_agregadas(df, ano)
    notas_bimestrais = _extrair_notas_bimestrais(df, ano)

    notas_finais = notas_bimestrais | notas_agregadas
    if not notas_finais:
        raise ErroDadosEducacionais(
            f"Nenhuma coluna de notas foi identificada para o ano {ano}."
        )

    return pd.DataFrame(notas_finais)


def _normalizar_booleano_binario(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce").round().astype("Int64")


def _normalizar_inteiros(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce").round().astype("Int64")


def _montar_base_padronizada(
    df_bruto: pd.DataFrame,
    ano: int,
    arquivo_origem: str,
) -> pd.DataFrame:
    contrato = ContratoDadosEducacionais()
    colunas = df_bruto.columns.tolist()

    coluna_etapa = _selecionar_coluna_contexto(colunas, PADRAO_ETAPA, ano, "etapa")
    coluna_transporte = _selecionar_coluna_contexto(
        colunas,
        PADRAO_TRANSPORTE,
        ano,
        "transporte",
    )
    coluna_bf = _selecionar_coluna_contexto(colunas, PADRAO_BF, ano, "Bolsa Família")

    notas_padronizadas = _extrair_notas_padronizadas(df_bruto, ano)

    base = pd.DataFrame(
        {
            "arquivo_origem": arquivo_origem,
            contrato.coluna_ano_letivo: ano,
            "matricula": df_bruto[contrato.coluna_identificador].astype(str).str.strip(),
            contrato.coluna_etapa: _normalizar_inteiros(df_bruto[coluna_etapa]),
            contrato.coluna_transporte: _normalizar_booleano_binario(
                df_bruto[coluna_transporte]
            ),
            contrato.coluna_bolsa_familia: _normalizar_booleano_binario(df_bruto[coluna_bf]),
            "media_global": pd.to_numeric(
                df_bruto[contrato.coluna_media_global],
                errors="coerce",
            ),
            "faltas_medias": pd.to_numeric(
                df_bruto[contrato.coluna_faltas_medias],
                errors="coerce",
            ),
        }
    )

    base = pd.concat([base, notas_padronizadas], axis=1)
    base = base.dropna(how="all")
    base = base[base["matricula"].ne("")].copy()
    base = base.dropna(subset=[contrato.coluna_etapa, "media_global", "faltas_medias"])
    base["transporte_descricao"] = base[contrato.coluna_transporte].map(
        {
            0: "Sem transporte",
            1: "Com transporte",
        }
    ).fillna("Não informado")
    base["bolsa_familia_descricao"] = base[contrato.coluna_bolsa_familia].map(
        {
            0: "Não recebe",
            1: "Recebe",
        }
    ).fillna("Não informado")
    return base.reset_index(drop=True)


@st.cache_data(show_spinner="Carregando planilha Excel...")
def carregar_planilha_excel(caminho_arquivo: str) -> pd.DataFrame:
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise ErroDadosEducacionais(
            f"O arquivo '{caminho.name}' não foi encontrado na pasta esperada."
        )

    try:
        dados = pd.read_excel(caminho, engine="openpyxl").dropna(how="all")
    except Exception as erro:  # pragma: no cover - proteção para erros externos
        raise ErroDadosEducacionais(
            f"Não foi possível ler o arquivo '{caminho.name}'. "
            "Verifique se ele é um Excel válido."
        ) from erro

    if dados.empty:
        raise ErroDadosEducacionais(
            f"O arquivo '{caminho.name}' não contém linhas válidas para análise."
        )

    dados.columns = [str(coluna).strip() for coluna in dados.columns]
    return dados


def normalizar_base_educacional(
    df_bruto: pd.DataFrame,
    nome_arquivo: str,
) -> pd.DataFrame:
    contrato = ContratoDadosEducacionais()
    colunas = df_bruto.columns.tolist()
    validacao = validar_contrato_basico(colunas)
    if not validacao["contrato_valido"]:
        raise ErroDadosEducacionais(
            "A base não possui as colunas mínimas exigidas para o dashboard."
        )

    anos_base = _obter_anos_disponiveis_colunas(colunas, nome_arquivo)
    if not anos_base:
        raise ErroDadosEducacionais(
            f"Não foi possível identificar o ano letivo da base '{nome_arquivo}'."
        )

    bases_normalizadas: list[pd.DataFrame] = []
    for ano in anos_base:
        try:
            base_ano = _montar_base_padronizada(df_bruto, ano, nome_arquivo)
        except ErroDadosEducacionais:
            continue
        if not base_ano.empty:
            bases_normalizadas.append(base_ano)

    if not bases_normalizadas:
        raise ErroDadosEducacionais(
            f"Nenhum recorte anual válido foi gerado a partir do arquivo '{nome_arquivo}'."
        )

    base_consolidada = pd.concat(bases_normalizadas, ignore_index=True)
    colunas_modelagem = obter_colunas_modelagem(base_consolidada)
    base_consolidada["quantidade_features_modelagem"] = (
        base_consolidada[colunas_modelagem].notna().sum(axis=1)
    )
    base_consolidada = base_consolidada.sort_values(
        by=[contrato.coluna_ano_letivo, contrato.coluna_etapa, "matricula"]
    ).reset_index(drop=True)
    return base_consolidada


@st.cache_data(show_spinner="Consolidando bases educacionais...")
def carregar_bases_consolidadas() -> pd.DataFrame:
    arquivos = listar_arquivos_excel()
    if not arquivos:
        raise ErroDadosEducacionais(
            "Nenhuma base Excel foi encontrada na raiz do projeto ou na pasta data/."
        )

    bases_consolidadas: list[pd.DataFrame] = []
    erros: list[str] = []
    for arquivo in arquivos:
        try:
            df_bruto = carregar_planilha_excel(str(arquivo))
            bases_consolidadas.append(normalizar_base_educacional(df_bruto, arquivo.name))
        except ErroDadosEducacionais as erro:
            erros.append(str(erro))

    if not bases_consolidadas:
        raise ErroDadosEducacionais(
            "Nenhuma base pôde ser consolidada com sucesso. "
            + " | ".join(erros[:3])
        )

    return pd.concat(bases_consolidadas, ignore_index=True)


def filtrar_base(
    df: pd.DataFrame,
    anos_letivos: list[int] | None = None,
    etapas: list[int] | None = None,
    arquivos_origem: list[str] | None = None,
) -> pd.DataFrame:
    base_filtrada = df.copy()
    if arquivos_origem:
        base_filtrada = base_filtrada[
            base_filtrada["arquivo_origem"].isin(arquivos_origem)
        ]
    if anos_letivos:
        base_filtrada = base_filtrada[
            base_filtrada["ano_letivo"].isin(anos_letivos)
        ]
    if etapas:
        base_filtrada = base_filtrada[base_filtrada["etapa"].isin(etapas)]
    return base_filtrada.reset_index(drop=True)


def obter_opcoes_filtros(df: pd.DataFrame) -> dict[str, list[int]]:
    anos = sorted(df["ano_letivo"].dropna().astype(int).unique().tolist())
    etapas = sorted(df["etapa"].dropna().astype(int).unique().tolist())
    return {
        "arquivos_origem": obter_fontes_disponiveis(df),
        "anos_letivos": anos,
        "etapas": etapas,
    }


def obter_colunas_modelagem(df: pd.DataFrame) -> list[str]:
    colunas_base = [
        coluna
        for coluna in ["media_global", "faltas_medias", "transporte", "bolsa_familia"]
        if coluna in df.columns
    ]
    colunas_notas = sorted(
        [
            coluna
            for coluna in df.columns
            if coluna.startswith("nota_")
        ]
    )
    return colunas_notas + colunas_base


def gerar_resumo_base(df: pd.DataFrame) -> dict[str, Any]:
    colunas_modelagem = obter_colunas_modelagem(df)
    return {
        "total_alunos": int(len(df)),
        "total_bases": len(obter_fontes_disponiveis(df)),
        "arquivos_origem": obter_fontes_disponiveis(df),
        "anos_letivos": sorted(df["ano_letivo"].dropna().astype(int).unique().tolist()),
        "etapas": sorted(df["etapa"].dropna().astype(int).unique().tolist()),
        "total_colunas_modelagem": len(colunas_modelagem),
        "features_modelagem": colunas_modelagem,
    }


def gerar_relatorio_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    colunas_prioritarias = [
        "matricula",
        "ano_letivo",
        "etapa",
        "transporte",
        "bolsa_familia",
        "media_global",
        "faltas_medias",
        *obter_colunas_modelagem(df),
    ]
    colunas_validas = list(
        dict.fromkeys(
            coluna for coluna in colunas_prioritarias if coluna in df.columns
        )
    )
    relatorio = pd.DataFrame(
        {
            "coluna": colunas_validas,
            "nulos": [int(df[coluna].isna().sum()) for coluna in colunas_validas],
            "percentual_nulos": [
                round(float(df[coluna].isna().mean() * 100), 2)
                for coluna in colunas_validas
            ],
            "tipo_dado": [str(df[coluna].dtype) for coluna in colunas_validas],
        }
    )
    return relatorio.sort_values(by=["percentual_nulos", "coluna"], ascending=[False, True])


def obter_amostra_exibicao(df: pd.DataFrame) -> pd.DataFrame:
    colunas_exibicao = [
        "matricula",
        "ano_letivo",
        "etapa",
        "transporte_descricao",
        "bolsa_familia_descricao",
        "media_global",
        "faltas_medias",
        "quantidade_features_modelagem",
        *obter_colunas_modelagem(df)[:8],
    ]
    colunas_existentes = [coluna for coluna in colunas_exibicao if coluna in df.columns]
    return df[colunas_existentes].copy()


def descrever_contrato_saida() -> dict[str, Any]:
    return {
        "chaves_analiticas": [
            "arquivo_origem",
            "matricula",
            "ano_letivo",
            "etapa",
        ],
        "variaveis_contexto": [
            "transporte",
            "bolsa_familia",
            "transporte_descricao",
            "bolsa_familia_descricao",
        ],
        "variaveis_desempenho": [
            "media_global",
            "faltas_medias",
            "nota_*",
        ],
        "regra_de_padronizacao": (
            "As médias anuais por disciplina são priorizadas. Quando elas não "
            "existirem, a camada de dados calcula a média das colunas bimestrais "
            "equivalentes do mesmo ano."
        ),
    }


def descrever_estrategia_concatenacao() -> list[str]:
    return [
        "Ler automaticamente todos os arquivos Excel localizados na raiz do projeto e na pasta data/.",
        "Permitir que novas planilhas versionadas no GitHub ampliem a análise sem ajuste manual de código.",
        "Extrair o ano letivo pelo nome do arquivo e pelas colunas para permitir múltiplas bases futuras.",
        "Padronizar colunas essenciais e notas em um formato analítico único antes da concatenação.",
        "Aplicar filtros globais por Ano Letivo e Etapa já na camada de dados.",
    ]
