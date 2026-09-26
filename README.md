# Dashboard Educacional

Aplicação em Streamlit para análise de desempenho, faltas e perfis de aprendizagem de alunos do Ensino Fundamental 2.

## Estrutura do projeto

- `app.py`: interface principal do dashboard.
- `data_processing.py`: leitura, padronização e consolidação das bases.
- `ml_models.py`: pipeline de clustering com KNNImputer, StandardScaler e KMeans.
- `visualizations.py`: gráficos e tabelas analíticas.
- `requirements.txt`: dependências para execução local e deploy.
- `.gitignore`: arquivos e pastas que não devem ser versionados.

## Como adicionar novas bases

1. Coloque a nova planilha `.xlsx` na raiz do projeto ou na pasta `data/`.
2. Mantenha o mesmo padrão estrutural essencial da base atual.
3. Faça commit e push para o GitHub.
4. O dashboard passará a considerar automaticamente a nova base nos filtros.

Exemplos:

- `Dados Fundamental 2 Limpos 2025.xlsx`
- `Dados Fundamental 2 Limpos 2026.xlsx`
- `Dados Fundamental 2 Limpos 2027.xlsx`

## Como rodar localmente

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Inicie a aplicação:

```bash
python -m streamlit run app.py
```

4. Abra no navegador o endereço mostrado no terminal, normalmente `http://localhost:8501`.

## Como publicar no GitHub

1. Crie um repositório no GitHub.
2. Envie os arquivos do projeto, incluindo as planilhas `.xlsx` que deseja analisar.
3. Garanta que o arquivo principal continue sendo `app.py`.

## Como fazer deploy no Streamlit Community Cloud

1. Acesse `https://share.streamlit.io/`.
2. Entre com a sua conta conectada ao GitHub.
3. Escolha o repositório do projeto.
4. Defina:
   - Branch: a branch principal do projeto
   - Main file path: `app.py`
5. Clique em **Deploy**.

## Observações importantes

- O app usa caminhos relativos, então funciona localmente e na nuvem.
- Novas bases no GitHub ampliam automaticamente os filtros de Base, Ano Letivo e Etapa.
- O modelo de clustering exige pelo menos 5 alunos no recorte para respeitar o `KNNImputer(n_neighbors=5)`.
