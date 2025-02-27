import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from joblib import load
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL


def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)

def carregar_modelo():
    return load(MODELO_FINAL)

st.title("Previsão de preços de imóveis")


