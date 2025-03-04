import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def construir_pipeline_modelo_regressao(
    regressor, preprocessor=None, target_transformer=None
):
    """
    Constrói uma pipeline para um modelo de regressão, incluindo opcionalmente um pré-processador
    e uma transformação no alvo (target).

    Parâmetros:
    -----------
    regressor : estimator (modelo de regressão)
        Modelo de regressão que será treinado e avaliado.

    preprocessor : sklearn.pipeline.Pipeline ou transformer, opcional (padrão=None)
        Pipeline ou transformador para pré-processamento das variáveis independentes (X), como
        normalização, codificação de variáveis categóricas, etc.

    target_transformer : transformer, opcional (padrão=None)
        Transformador para aplicar ao target (y), como `QuantileTransformer` ou `StandardScaler`.
        Útil quando a variável alvo não segue uma distribuição normal e requer transformação.

    Retorna:
    --------
    model : sklearn.pipeline.Pipeline ou sklearn.compose.TransformedTargetRegressor
        Modelo final encapsulado em uma pipeline, com os passos necessários de pré-processamento 
        e transformação do target, caso aplicáveis.

    Exemplo de uso:
    ---------------
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.compose import ColumnTransformer
    >>> import numpy as np

    >>> preprocessor = ColumnTransformer([("scaler", StandardScaler(), ["feature1", "feature2"])])
    >>> target_transformer = StandardScaler()
    >>> regressor = Ridge(alpha=1.0)

    >>> pipeline_modelo = construir_pipeline_modelo_regressao(regressor, preprocessor, target_transformer)
    >>> pipeline_modelo.fit(X_train, y_train)
    >>> y_pred = pipeline_modelo.predict(X_test)
    """
    # Se houver um pré-processador, adiciona à pipeline antes do regressor
    if preprocessor is not None:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    # Se houver um transformador de target, encapsula a pipeline no TransformedTargetRegressor
    if target_transformer is not None:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline
        
      return model

def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=None,
):
    """
    Treina e valida um modelo de regressão utilizando validação cruzada (K-Fold).

    Parâmetros:
    -----------
    X : array-like ou pd.DataFrame
        Conjunto de variáveis independentes (features).

    y : array-like ou pd.Series
        Variável alvo (target) a ser prevista pelo modelo.

    regressor : estimator (modelo de regressão)
        Modelo de regressão que será treinado e avaliado.

    preprocessor : sklearn.pipeline.Pipeline ou transformer, opcional (padrão=None)
        Pipeline ou transformador para pré-processamento das features (X), como normalização ou
        codificação de variáveis categóricas.

    target_transformer : transformer, opcional (padrão=None)
        Transformador para aplicar ao target (y), útil quando a variável alvo precisa de transformação,
        como `QuantileTransformer` ou `StandardScaler`.

    n_splits : int, opcional (padrão=5)
        Número de divisões (folds) para a validação cruzada K-Fold.

    random_state : int, opcional (padrão=None)
        Semente para garantir reprodutibilidade dos resultados ao embaralhar os dados.

    Retorna:
    --------
    scores : dict
        Dicionário contendo as métricas da validação cruzada. As chaves do dicionário são:
        - 'fit_time': Tempo gasto para treinar o modelo.
        - 'score_time': Tempo gasto para calcular as métricas.
        - 'test_r2': R² médio do modelo nas validações.
        - 'test_neg_mean_absolute_error': Erro Médio Absoluto Negativo (MAE).
        - 'test_neg_root_mean_squared_error': Raiz do Erro Quadrático Médio Negativo (RMSE).

    Exemplo de uso:
    ---------------
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.compose import ColumnTransformer
    >>> import pandas as pd
    >>> import numpy as np

    >>> X = pd.DataFrame(np.random.rand(100, 3), columns=["feature1", "feature2", "feature3"])
    >>> y = np.random.rand(100)

    >>> preprocessor = ColumnTransformer([("scaler", StandardScaler(), ["feature1", "feature2"])])
    >>> regressor = Ridge(alpha=1.0)

    >>> resultados = treinar_e_validar_modelo_regressao(X, y, regressor, preprocessor, n_splits=5, random_state=42)
    >>> print(resultados)
    """
    # Constrói o pipeline do modelo
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    # Define a estratégia de validação cruzada K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Executa a validação cruzada com métricas específicas
    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
    )

    return scores


def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=None,
    return_train_score=False,
):
    """
    Executa uma busca de hiperparâmetros (Grid Search) com validação cruzada para um modelo de regressão.

    Parâmetros:
    -----------
    regressor : estimator
        Modelo de regressão que será ajustado e otimizado.

    param_grid : dict
        Dicionário contendo os hiperparâmetros e seus respectivos valores a serem testados.

    preprocessor : sklearn.pipeline.Pipeline ou transformer, opcional (padrão=None)
        Pipeline ou transformador para pré-processamento das features (X), como normalização 
        ou codificação de variáveis categóricas.

    target_transformer : transformer, opcional (padrão=None)
        Transformador para aplicar ao target (y), útil quando a variável alvo precisa de transformação,
        como `QuantileTransformer` ou `StandardScaler`.

    n_splits : int, opcional (padrão=5)
        Número de divisões (folds) para a validação cruzada K-Fold.

    random_state : int, opcional (padrão=None)
        Semente para garantir reprodutibilidade dos resultados ao embaralhar os dados.

    return_train_score : bool, opcional (padrão=False)
        Indica se os scores de treino devem ser incluídos nos resultados.

    Retorna:
    --------
    grid_search : sklearn.model_selection.GridSearchCV
        Objeto `GridSearchCV` ajustado, que pode ser utilizado para acessar o melhor modelo encontrado,
        os melhores hiperparâmetros e as métricas de validação cruzada.

    Exemplo de uso:
    ---------------
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.compose import ColumnTransformer
    >>> import numpy as np

    >>> param_grid = {"reg__alpha": [0.1, 1.0, 10.0]}

    >>> preprocessor = ColumnTransformer([("scaler", StandardScaler(), ["feature1", "feature2"])])
    >>> regressor = Ridge()

    >>> grid_search = grid_search_cv_regressor(
    ...     regressor, param_grid, preprocessor, n_splits=5, random_state=42
    ... )
    >>> grid_search.fit(X_train, y_train)
    >>> print(grid_search.best_params_)
    """
    # Constrói a pipeline do modelo com pré-processador e transformação do target, se houver
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    # Define a estratégia de validação cruzada K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Configuração da busca em grade (Grid Search) com validação cruzada
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=kf,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",  # Otimiza o modelo com base no RMSE
        n_jobs=-1,  # Paraleliza os cálculos para maior eficiência
        return_train_score=return_train_score,
        verbose=1,  # Exibe progresso durante a busca
    )

    return grid_search


def organiza_resultados(resultados):
    """
    Organiza os resultados da validação cruzada em um DataFrame formatado.

    Parâmetros:
    -----------
    resultados : dict
        Dicionário contendo os resultados da validação cruzada, onde cada chave representa um modelo 
        e seu valor é um outro dicionário com métricas como `fit_time`, `score_time`, `test_r2`, etc.

    Retorna:
    --------
    df_resultados_expandido : pd.DataFrame
        DataFrame contendo os resultados organizados, incluindo:
        - 'model': Nome do modelo avaliado.
        - 'fit_time': Tempo gasto no ajuste do modelo.
        - 'score_time': Tempo gasto para calcular as métricas.
        - 'time_seconds': Soma de 'fit_time' e 'score_time'.
        - Outras métricas como R², MAE e RMSE (dependendo do conteúdo de `resultados`).

    Exemplo de uso:
    ---------------
    >>> resultados = {
    ...     "Ridge": {
    ...         "fit_time": [0.05, 0.06, 0.05],
    ...         "score_time": [0.002, 0.002, 0.003],
    ...         "test_r2": [0.85, 0.86, 0.84],
    ...     },
    ...     "Lasso": {
    ...         "fit_time": [0.03, 0.04, 0.03],
    ...         "score_time": [0.001, 0.002, 0.001],
    ...         "test_r2": [0.80, 0.81, 0.79],
    ...     },
    ... }

    >>> df_resultados = organiza_resultados(resultados)
    >>> print(df_resultados)
    """
    # Adiciona uma nova métrica 'time_seconds' (soma de fit_time e score_time) para cada modelo
    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    # Converte o dicionário de resultados para um DataFrame e reorganiza a estrutura
    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    # Expande as listas dentro do DataFrame para formato tabular
    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    # Converte colunas para valores numéricos, se possível
    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass  # Mantém os valores originais caso não seja possível converter

    return df_resultados_expandido
