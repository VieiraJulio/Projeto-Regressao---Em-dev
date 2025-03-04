import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st
    
from joblib import load
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)
    
@st.cache_data
def carregar_dados_geo():
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    
    gdf_geo = gdf_geo.explode(ignore_index=True)

    
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if isinstance(
            geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)
        ):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo
    
@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()
    

st.title( " 🏠 Previsão de preços de imóveis", help = "Para sugestões entrar em contato pelo GitHub: VieiraJulio")

st.caption(":gray[Selecione o condado de interesse, a idade do imóvel e a faixa de renda média anual.]")


condados = sorted(gdf_geo["name"].unique())

coluna1, coluna2 = st.columns(spec = 2)

with coluna1:

    with st.form(key = "formulario"):
    
        selecionar_condado = st.selectbox("Condado", condados)
        
        longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values
        latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values
        
        housing_median_age = st.number_input("Idade do imóvel", value = 10, min_value = 1, max_value = 50)
        
        total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values
        total_bedrooms = gdf_geo.query("name == @selecionar_condado")["total_bedrooms"].values
        population = gdf_geo.query("name == @selecionar_condado")["population"].values
        households = gdf_geo.query("name == @selecionar_condado")["households"].values
        
        median_income = st.slider("Renda média (multiplos de US $ 10k)", 5.0, 100.0, 45.0, 5.0)
        
        ocean_proximity = gdf_geo.query("name == @selecionar_condado")["ocean_proximity"].values
        
        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        
        median_income_cat = np.digitize(median_income / 10 , bins = bins_income)
        
        rooms_per_household = gdf_geo.query("name == @selecionar_condado")["rooms_per_household"].values
        bedrooms_per_room = gdf_geo.query("name == @selecionar_condado")["bedrooms_per_room"].values
        population_per_household = gdf_geo.query("name == @selecionar_condado")["population_per_household"].values
        
        entrada_modelo = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income / 10, 
            "ocean_proximity": ocean_proximity,
            "median_income_cat": median_income_cat,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "population_per_household" : population_per_household
            
        }
        
        df_entrada_modelo = pd.DataFrame(entrada_modelo)
        
        botao_previsao = st.form_submit_button("Prever preço")
    
    if botao_previsao:
        preco = modelo.predict(df_entrada_modelo)
        st.metric(label = "Possível Preço previsto: US($)", value =  f"{preco[0][0]:,.2f}")

with coluna2:
    view_state = pdk.ViewState(
        latitude = float(latitude[0]), 
        longitude = float(longitude[0]),
        zoom = 5,
        min_zoom = 5,
        max_zoom = 15
        
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data = gdf_geo[["name", "geometry"]],
        get_polygon = "geometry",
        get_fill_color = [128, 128, 128, 100],
        get_line_color = [255, 255, 255],
        get_line_width = 500,
        pickable = True,
        auto_highlight = True
    )

    condado_selecionado = gdf_geo.query("name == @selecionar_condado")
    
    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data = condado_selecionado[["name", "geometry"]],
        get_polygon = "geometry",
        get_fill_color = [255, 0, 0, 100],
        get_line_color = [255, 0, 0], # COR AO REDOR DO CONDADO SELECIONADO
        get_line_width = 500,
        pickable = True,
        auto_highlight = True
    )

    tooltip = {
        "html": "<b>Condado:</b> {name}" ,
        "style": {"backgroundColor": "gray", "color": "white", "fontsize": "10px"}
    }
    
    mapa = pdk.Deck(
        initial_view_state = view_state,
        map_style = "dark",
        layers = [polygon_layer, highlight_layer],
        tooltip = tooltip
    )

    st.pydeck_chart(mapa)
    

    

