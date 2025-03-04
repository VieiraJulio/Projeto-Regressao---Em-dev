import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from .models import RANDOM_STATE

sns.set_theme(palette="bright")

PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2


def plot_coeficientes(df_coefs, titulo="Coeficientes"):
    """
    Gera um gráfico de barras horizontais para visualizar os coeficientes de um modelo.

    Parâmetros:
    -----------
    df_coefs : pd.DataFrame
        DataFrame contendo os coeficientes do modelo. Deve ter uma única coluna com os valores
        dos coeficientes e os índices representando os nomes das features.

    titulo : str, opcional (padrão="Coeficientes")
        Título do gráfico.

    Retorna:
    --------
    None
        A função exibe um gráfico e não retorna nenhum valor.

    Exemplo:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> coefs = np.array([0.5, -1.2, 0.8])
    >>> colunas = ["feature_1", "feature_2", "feature_3"]
    >>> df_coefs = pd.DataFrame(data=coefs, index=colunas, columns=["coeficiente"])
    >>> plot_coeficientes(df_coefs)
    """
    df_coefs.plot.barh()
    plt.title(titulo)
    plt.axvline(x=0, color=".5")  # Linha de referência no eixo X
    plt.xlabel("Coeficientes")
    plt.gca().get_legend().remove()  # Remove a legenda do gráfico
    plt.show()



def plot_residuos(y_true, y_pred):
    """
    Gera gráficos para análise dos resíduos do modelo de regressão, permitindo avaliar sua distribuição e padrão.

    Parâmetros:
    -----------
    y_true : array-like
        Valores reais do target - dados de teste (variável dependente).
    
    y_pred : array-like
        Valores preditos pelo modelo.

    Retorna:
    --------
    None
        A função exibe três gráficos para análise dos resíduos e não retorna nenhum valor.

    Descrição dos gráficos:
    -----------------------
    1. **Histograma dos resíduos**: Exibe a distribuição dos erros (resíduos), ajudando a identificar padrões ou 
       desvios da normalidade.
    2. **Resíduos vs. Valores previstos**: Avalia a homocedasticidade (dispersão dos erros) e possíveis padrões 
       indesejados.
    3. **Valores reais vs. previstos**: Permite verificar o quão próximos os valores preditos estão dos reais.

    Exemplo de uso:
    ---------------
    >>> import numpy as np
    >>> y_true = np.array([100, 200, 300, 400, 500])
    >>> y_pred = np.array([110, 190, 290, 410, 490])
    >>> plot_residuos(y_true, y_pred)
    """
    # Calcula os resíduos (erro entre valores reais e previstos)
    residuos = y_true - y_pred

    # Cria uma figura com 3 subgráficos lado a lado
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    # Gráfico 1: Histograma dos resíduos
    sns.histplot(residuos, kde=True, ax=axs[0])
    axs[0].set_title("Distribuição dos Resíduos")

    # Gráfico 2: Resíduos vs. Valores previstos
    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )
    axs[1].set_title("Resíduos vs. Valores Previstos")

    # Gráfico 3: Valores reais vs. Valores previstos
    PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )
    axs[2].set_title("Valores Reais vs. Previstos")

    # Ajusta o layout para evitar sobreposição de elementos
    plt.tight_layout()

    # Exibe os gráficos
    plt.show()












































def plot_residuos_estimador(estimator, X, y, eng_formatter=False, fracao_amostra=0.25):

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    residuos = error_display_01.y_true - error_display_01.y_pred

    sns.histplot(residuos, kde=True, ax=axs[0])

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()

    plt.show()


def plot_comparar_metricas_modelos(df_resultados):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    comparar_metricas = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    nomes_metricas = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        sns.boxplot(
            x="model",
            y=metrica,
            data=df_resultados,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(nome)
        ax.set_ylabel(nome)
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    plt.show()
