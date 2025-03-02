import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from joblib import load
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)
    
@st.cache_data
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)
    
@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()
    

st.title("Previsão de preços de imóveis")


condados = list(gdf_geo["name"].sort_values())

selecionar_condado = st.selectbox("Condado", condados)

longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values
latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values

housing_median_age = st.number_input("Idade do imóvel", value = 10, min_value = 1, max_value = 50)

total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values
total_bedrooms = gdf_geo.query("name == @selecionar_condado")["total_bedrooms"].values
population = gdf_geo.query("name == @selecionar_condado")["population"].values
households = gdf_geo.query("name == @selecionar_condado")["households"].values

median_income = st.slider("Renda média (multiplos de US $ 10k)", 0.5, 15.0, 0.5)

ocean_proximity = st.selectbox("Proximidade do oceano", df["ocean_proximity"].unique())

median_income_cat = st.number_input("Categoria de Renda", value = 4)

rooms_per_househould = st.number_input("Quartos por domicílio", value = 7)
bedrooms_per_room = st.number_input("Quartos por cômodo", value = 0.2)
population_per_household = st.number_input("População por domicílio", value = 2)

entrada_modelo = {
    "longtiude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
    "median_income_cat": median_income_cat,
    "rooms_per_household": rooms_per_household.
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_househould" : population_per_househould
    
}

df_entrada_modelo = pd.DataFrame(entrada_modelo, index = [0])

botao_previsao = st.button("Prever preço")

if botao_previsao:
    preco = modelo.predict(df_entrada_modelo)
    st.write(f"Preço previsto : $ {preco[0][0]:.2f}")

    

