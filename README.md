### Previsão de Preços de Imóveis | Machine Learning

### Aplicativo no Streamlit

    https://priceprediction-vieirajulio.streamlit.app/

### Objetivo 

Atráves de dados do censo do estado da Califórnia, desenvolvemos um modelo de regressão de Machine Learning para prever preços de imóveis. 

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
|    └── housing_clean.parquet        <- Conjunto de dados acima reduzido.
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


### Mapa de Distribuição

![image](https://github.com/user-attachments/assets/9426a726-eb07-4686-abf1-0c93a7e24913)

### Metodologia

A avaliação dos dados concentrou-se nas métricas detalhada das features que impactam o nosso target. 
Após a análise exploratória, selecionamos 13 features. É possível acompanhar o processo no arquivo:

```
    01-jv-eda.pynb
```
Durante o desenvolvimento do modelo, aplicamos o GridSearch para otimizar os hiperparâmetros, o que nos permitiu identificar a combinação ideal de parâmetros para maximizar o desempenho. 
Esse processo de avaliação sistemática aprimorou a robustez e a precisão do modelo, contribuindo significativamente para a sua generalização.

A pipeline de melhor desempenho foi composta pelo modelo de regressão Ridge, combinada com uma transformação no target utilizando o Quantile Transformer. 

![image](https://github.com/user-attachments/assets/539ebff6-6533-4e82-acbe-a17fa2a7f1d6)

É possível testar e modificar os hiperparâmetros existentes no arquivo:

```
    04-jv-modelos.ipynb 
```

### Resultados

Este modelo prevê preços de imóveis com base nos dados do censo da Califórnia dos anos 90. 
O projeto envolveu uma análise exploratória de dados (EDA), testes de distribuição baseada nas posições dos imóveis nos mapas e um extenso processo de pré-processamento implementado em pipelines que resultaram na construção final do modelo.

Além disso, desenvolvemos um aplicativo interativo em Streamlit com uma interface personalizada que facilita a utilização e iteração com o modelo. No aplicativo, o usuário pode visualizar a previsão do preço do imóvel, bem como explorar otimizações realizadas com um conjunto reduzido de features — uma abordagem pensada para simplificar a experiência do usuário sem comprometer a precisão das estimativas.
Por fim, incluímos um mapa interativo que destaca o condado do imóvel, oferecendo uma perspectiva geográfica detalhada e auxiliando na análise regional dos preços.

### Ferramentas usadas: 


Linguagem de programação:<p> <br> ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
```
Bibliotecas:
    geopandas==1.0.1
    joblib==1.4.2
    matplotlib==3.10.0
    numpy==2.2.1
    pandas ==2.2.3
    pydeck==0.8.0
    scikit-learn==1.6.1
    scipy==1.14.1
    seaborn==0.13.2
    Shapely==2.0.7
    streamlit==1.42.2
```

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
