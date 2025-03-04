
import pandas as pd

def dataframe_coeficientes(coefs, colunas):
    """
    Cria um DataFrame com os coeficientes de um modelo, associando-os às suas respectivas features
    e ordenando-os em ordem crescente.

    Parâmetros:
    -----------
    coefs : array-like
        Lista ou array contendo os coeficientes do modelo.
    
    colunas : list
        Lista ou Série com os nomes das features correspondentes aos coeficientes.

    Retorna:
    --------
    pd.DataFrame
        DataFrame contendo os coeficientes organizados em uma única coluna chamada "coeficiente",
        ordenados de forma ascendente.
    
    Exemplo:
    --------
    >>> import numpy as np
    >>> coefs = np.array([0.5, -1.2, 0.8])
    >>> colunas = ["feature_1", "feature_2", "feature_3"]
    >>> dataframe_coeficientes(coefs, colunas)
    
                   coeficiente
    feature_2       -1.2
    feature_1        0.5
    feature_3        0.8
    """
    return pd.DataFrame(data=coefs, index=colunas, columns=["coeficiente"]).sort_values(
        by="coeficiente"
    )

