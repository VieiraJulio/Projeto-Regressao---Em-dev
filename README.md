### Previsão de Preços de Imóveis | Machine Learning

### Objetivo 

Atráves de dados do censo do estado da Califórnia, desenvolveremos um modelo de regressão de Machine Learning para prever preços de imóveis. 

### Organização do projeto

```
├── home.py            <- Arquivo gerador do aplicativo no Streamlit.
├── .env               <- Arquivo de variáveis de ambiente. (não versionar)
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git.
├── requirements.txt   <- O arquivo de requisitos para reproduzir o ambiente de análise.
├── LICENSE            <- Licença de código aberto (MIT).
├── README.md          <- README principal para apresentação e desenvolvedores que usam este projeto.
|
├── dados              <- Arquivos de dados para o projeto.
     │
|    ├── california_counties.geojson  <- Dados espaciais com detalhamento de localidade(s).
|    ├── gdf_counties.parquet         <- Conjunto de dados espaciais com ETL.
|    └── housing.csv                  <- Conjunto de dados importado (csv)do Kaggle.
|    └── housing.csv                  <- Conjunto de dados acima reduzido.
|
|
├── modelos            <- Modelos treinados e serializados, previsões de modelos ou resumos de modelos.
|     |
|     └── ridge_polyfeat_target_quantile.joblib  <- Modelo de Regressão
|
├── notebooks          <- Cadernos Jupyter.
│
|      ├── 01-jv-eda.ipynb  
|      ├── 02-jv-mapas.ipynb    
|      └── 03-jv-geo.ipynb 
|      └── 04-jv-modelos.ipynb 
|
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├──auxiliares.py <- Funçoes auxiliares.
|      ├── __init__.py  <- Torna um módulo Python.
|      ├── config.py    <- Configurações básicas do projeto.
|      └── graficos.py  <- Scripts para criar visualizações exploratórias e orientadas a resultados.
|      └── models.py.   <- Funções que auxiliam a contrução do modelo de previsão.

```


### DataSet

Origem: [Conjunto de dados](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)

Este conjunto de dados foi derivado do censo dos EUA de 1990, usando uma linha por grupo
de blocos censitários. Um grupo de blocos é a menor unidade geográfica para a qual o
Escritório do Censo dos EUA publica dados amostrais (um grupo de blocos geralmente tem
uma população de 600 a 3.000 pessoas).

### Detalhes do DataSet

Um domicílio (*household*) é um grupo de pessoas que reside em uma casa. Como o número
médio de cômodos e quartos neste conjunto de dados é fornecido por domicílio, essas
colunas podem apresentar valores surpreendentemente altos para grupos de blocos com
poucos domicílios e muitas casas vazias, como em resorts de férias.

A **variável alvo (Target)** é o valor mediano das casas para os distritos da Califórnia, expressa em
dólares, na coluna median_house_value.

colunas:

- `median_income`: renda mediana no grupo de blocos (em dezenas de milhares de dólares)
- `housing_median_age`: idade mediana das casas no grupo de blocos
- `total_rooms`: número cômodos no grupo de blocos
- `total_bedrooms`: número de quartos no grupo de blocos
- `population`: população do grupo de blocos
- `households`: domicílios no grupo de blocos
- `latitude`: latitude do grupo de blocos
- `longitude`: longitude do grupo de blocos
- `ocean_proximity`: proximidade do oceano
  - `NEAR BAY`: perto da baía
  - `<1H OCEAN`: a menos de uma hora do oceano
  - `INLAND`: no interior
  - `NEAR OCEAN`: perto do oceano
  - `ISLAND`: ilha
- `median_house_value`: valor mediano das casas no grupo de blocos (em dólares) 

### Configuração do ambiente

1. Faça o clone do repositório que será criado a partir deste modelo.

    ```bash
    git clone ENDERECO_DO_SEU_REPOSITORIO
    ```

2. Crie um ambiente virtual para o seu projeto utilizando o gerenciador de ambientes de sua preferência.

    a. Caso esteja utilizando o `conda`, exporte as dependências do ambiente para um novo arquivo, como exemplo: `ambiente.yml`:

      ```bash
      conda env export > ambiente.yml
      ```

    b. Caso esteja utilizando outro gerenciador de ambientes, exporte as dependências
    para o arquivo `requirements.txt` ou outro formato de sua preferência. Adicione o
    arquivo ao controle de versão, removendo o arquivo `ambiente.yml`.


Para mais informações sobre como usar Git e GitHub, [clique aqui](https://cienciaprogramada.com.br/2021/09/guia-definitivo-git-github/). Sobre ambientes virtuais, [clique aqui](https://cienciaprogramada.com.br/2020/08/ambiente-virtual-projeto-python/).
