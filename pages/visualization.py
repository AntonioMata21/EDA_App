import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import data_visualization as dv

def show_visualization_page(df):
    """
    Muestra la página de visualización interactiva del dataset.
    
    Args:
        df: DataFrame de Pandas
    """
    st.markdown("<div class='section-header'>📊 Visualización Interactiva</div>", unsafe_allow_html=True)
    
    # Usar DataFrame filtrado si existe
    if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
        data = st.session_state.filtered_data
    else:
        data = df.copy()
    
    # Crear pestañas para diferentes tipos de visualizaciones
    tabs = st.tabs(["Univariante", "Bivariante", "Multivariante", "Series Temporales", "Personalizado"])
    
    # Pestaña de visualizaciones univariantes
    with tabs[0]:
        st.markdown("<div class='subsection-header'>📈 Análisis Univariante</div>", unsafe_allow_html=True)
        
        # Seleccionar columna para visualizar
        col_to_viz = st.selectbox(
            "Selecciona una columna para visualizar",
            data.columns,
            key="univariate_col"
        )
        
        if col_to_viz:
            # Obtener tipo de columna
            col_type = data[col_to_viz].dtype
            
            # Seleccionar tipo de gráfico según el tipo de datos
            if pd.api.types.is_numeric_dtype(col_type):
                viz_type = st.radio(
                    "Tipo de visualización",
                    ["Histograma", "Boxplot", "Violín", "KDE"],
                    horizontal=True,
                    key="univariate_viz_type_num"
                )
                
                # Configuración adicional según el tipo de gráfico
                if viz_type == "Histograma":
                    # Número de bins
                    n_bins = st.slider("Número de bins", 5, 100, 20, key="hist_bins")
                    
                    # Mostrar KDE
                    show_kde = st.checkbox("Mostrar curva KDE", value=True, key="hist_kde")
                    
                    # Generar gráfico
                    fig = dv.create_histogram(data, col_to_viz, n_bins=n_bins, kde=show_kde)
                    st.plotly_chart(fig, use_container_width=True, key="univariate_numerical_hist")
                
                elif viz_type == "Boxplot":
                    # Generar gráfico
                    fig = dv.create_boxplot(data, col_to_viz)
                    st.plotly_chart(fig, use_container_width=True, key="univariate_numerical_box")
                
                elif viz_type == "Violín":
                    # Generar gráfico
                    fig = dv.create_violin(data, col_to_viz)
                    st.plotly_chart(fig, use_container_width=True, key="univariate_numerical_violin")
                
                elif viz_type == "KDE":
                    # Generar gráfico
                    fig = dv.create_kde(data, col_to_viz)
                    st.plotly_chart(fig, use_container_width=True, key="univariate_numerical_kde")
            
            elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
                viz_type = st.radio(
                    "Tipo de visualización",
                    ["Barras", "Pie", "Treemap"],
                    horizontal=True,
                    key="univariate_viz_type_cat"
                )
                
                # Limitar valores para visualización
                max_categories = st.slider(
                    "Máximo de categorías a mostrar", 
                    5, 50, 10, 
                    key="max_categories"
                )
                
                # Obtener conteo de valores
                value_counts = data[col_to_viz].value_counts().reset_index()
                value_counts.columns = [col_to_viz, 'count']
                
                # Limitar a las top categorías
                if len(value_counts) > max_categories:
                    top_categories = value_counts.nlargest(max_categories, 'count')
                    other_count = value_counts.iloc[max_categories:]['count'].sum()
                    other_row = pd.DataFrame({col_to_viz: ['Otros'], 'count': [other_count]})
                    value_counts = pd.concat([top_categories, other_row], ignore_index=True)
                
                # Generar gráfico según el tipo seleccionado
                if viz_type == "Barras":
                    fig = dv.create_bar(value_counts, col_to_viz, 'count', title=f'Distribución de {col_to_viz}')
                    st.plotly_chart(fig, use_container_width=True, key="univariate_categorical_bar")
                
                elif viz_type == "Pie":
                    fig = dv.create_pie(value_counts, col_to_viz, 'count', title=f'Distribución de {col_to_viz}')
                    st.plotly_chart(fig, use_container_width=True, key="univariate_categorical_pie")
                
                elif viz_type == "Treemap":
                    fig = px.treemap(value_counts, path=[col_to_viz], values='count', 
                                    title=f'Treemap de {col_to_viz}')
                    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                    st.plotly_chart(fig, use_container_width=True, key="univariate_categorical_treemap")
            
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                viz_type = st.radio(
                    "Tipo de visualización",
                    ["Histograma", "Línea de tiempo", "Calendario"],
                    horizontal=True,
                    key="univariate_viz_type_time"
                )
                
                # Generar gráfico según el tipo seleccionado
                if viz_type == "Histograma":
                    # Número de bins
                    n_bins = st.slider("Número de bins", 5, 100, 20, key="time_hist_bins")
                    
                    # Generar gráfico
                    fig = dv.create_histogram(data, col_to_viz, n_bins=n_bins, kde=False)
                    st.plotly_chart(fig, use_container_width=True, key="univariate_temporal_hist")
                
                elif viz_type == "Línea de tiempo":
                    # Agrupar por fecha
                    data_copy = data.copy()
                    data_copy['count'] = 1
                    time_df = data_copy.groupby(pd.Grouper(key=col_to_viz, freq='D')).count().reset_index()
                    
                    # Generar gráfico
                    fig = dv.create_line(time_df, col_to_viz, 'count', 
                                        title=f'Distribución temporal de {col_to_viz}')
                    st.plotly_chart(fig, use_container_width=True, key="univariate_temporal_line")
                
                elif viz_type == "Calendario":
                    st.info("La visualización de calendario requiere la biblioteca calplot que no está incluida por defecto. Se muestra un gráfico de línea alternativo.")
                    
                    # Agrupar por fecha
                    data_copy = data.copy()
                    data_copy['count'] = 1
                    time_df = data_copy.groupby(pd.Grouper(key=col_to_viz, freq='D')).count().reset_index()
                    
                    # Generar gráfico
                    fig = dv.create_line(time_df, col_to_viz, 'count', 
                                        title=f'Distribución temporal de {col_to_viz}')
                    st.plotly_chart(fig, use_container_width=True, key="univariate_temporal_calendar")
            
            elif pd.api.types.is_bool_dtype(col_type):
                # Obtener conteo de valores
                value_counts = data[col_to_viz].value_counts().reset_index()
                value_counts.columns = [col_to_viz, 'count']
                
                # Generar gráfico
                fig = dv.create_pie(value_counts, col_to_viz, 'count', 
                                   title=f'Distribución de {col_to_viz}')
                st.plotly_chart(fig, use_container_width=True, key="univariate_boolean_pie")
    
    # Pestaña de visualizaciones bivariantes
    with tabs[1]:
        st.markdown("<div class='subsection-header'>📊 Análisis Bivariante</div>", unsafe_allow_html=True)
        
        # Seleccionar columnas para visualizar
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox(
                "Selecciona columna para eje X",
                data.columns,
                key="bivariate_x_col"
            )
        
        with col2:
            y_col = st.selectbox(
                "Selecciona columna para eje Y",
                [col for col in data.columns if col != x_col],
                key="bivariate_y_col"
            )
        
        if x_col and y_col:
            # Obtener tipos de columnas
            x_type = data[x_col].dtype
            y_type = data[y_col].dtype
            
            # Determinar tipo de gráfico según los tipos de datos
            if pd.api.types.is_numeric_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
                viz_type = st.radio(
                    "Tipo de visualización",
                    ["Scatter", "Hexbin", "Regresión"],
                    horizontal=True,
                    key="bivariate_viz_type_num_num"
                )
                
                # Configuración adicional según el tipo de gráfico
                if viz_type == "Scatter":
                    # Seleccionar columna para color (opcional)
                    color_col = st.selectbox(
                        "Columna para color (opcional)",
                        [None] + [col for col in data.columns if col not in [x_col, y_col]],
                        key="scatter_color_col"
                    )
                    
                    # Generar gráfico
                    fig = dv.create_scatter(data, x_col, y_col, color_col=color_col)
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_scatter_num_num")
                 
                elif viz_type == "Hexbin":
                     # Generar gráfico
                     fig = px.density_heatmap(data, x=x_col, y=y_col, marginal_x="histogram", marginal_y="histogram")
                     st.plotly_chart(fig, use_container_width=True, key="bivariate_hexbin_num_num")
                 
                elif viz_type == "Regresión":
                     # Seleccionar columna para color (opcional)
                     color_col = st.selectbox(
                         "Columna para color (opcional)",
                         [None] + [col for col in data.columns if col not in [x_col, y_col]],
                         key="regression_color_col"
                     )
                     
                     # Generar gráfico
                     fig = px.scatter(data, x=x_col, y=y_col, color=color_col, trendline="ols",
                                     trendline_color_override="red")
                     st.plotly_chart(fig, use_container_width=True, key="bivariate_regression_num_num")
                     
                     # Mostrar estadísticas de regresión
                     if st.checkbox("Mostrar estadísticas de regresión", key="show_regression_stats_bivar"):
                         import statsmodels.api as sm
                         X = sm.add_constant(data[x_col])
                         model = sm.OLS(data[y_col], X).fit()
                         st.code(model.summary().as_text())
            
            elif pd.api.types.is_numeric_dtype(y_type) and (pd.api.types.is_object_dtype(x_type) or pd.api.types.is_categorical_dtype(x_type) or pd.api.types.is_string_dtype(x_type) or pd.api.types.is_bool_dtype(x_type)):
                viz_type = st.radio(
                    "Tipo de visualización",
                    ["Boxplot", "Violín", "Barras"],
                    horizontal=True,
                    key="bivariate_viz_type_cat_num"
                )
                
                # Limitar categorías para visualización
                max_categories = st.slider(
                    "Máximo de categorías a mostrar", 
                    5, 30, 10, 
                    key="bivariate_max_categories"
                )
                
                # Filtrar las categorías más frecuentes
                top_categories = data[x_col].value_counts().nlargest(max_categories).index.tolist()
                filtered_df = data[data[x_col].isin(top_categories)]
                
                # Generar gráfico según el tipo seleccionado
                if viz_type == "Boxplot":
                    fig = dv.create_boxplot_by_category(filtered_df, x_col, y_col)
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_boxplot_cat_num")
                
                elif viz_type == "Violín":
                    fig = dv.create_violin_by_category(filtered_df, x_col, y_col)
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_violin_cat_num")
                
                elif viz_type == "Barras":
                    # Calcular estadística para barras
                    stat = st.selectbox(
                        "Estadística a mostrar",
                        ["Media", "Mediana", "Suma", "Conteo", "Min", "Max"],
                        key="bar_stat"
                    )
                    
                    # Calcular estadística por grupo
                    if stat == "Media":
                        agg_df = filtered_df.groupby(x_col)[y_col].mean().reset_index()
                    elif stat == "Mediana":
                        agg_df = filtered_df.groupby(x_col)[y_col].median().reset_index()
                    elif stat == "Suma":
                        agg_df = filtered_df.groupby(x_col)[y_col].sum().reset_index()
                    elif stat == "Conteo":
                        agg_df = filtered_df.groupby(x_col)[y_col].count().reset_index()
                    elif stat == "Min":
                        agg_df = filtered_df.groupby(x_col)[y_col].min().reset_index()
                    elif stat == "Max":
                        agg_df = filtered_df.groupby(x_col)[y_col].max().reset_index()
                    
                    # Generar gráfico
                    fig = dv.create_bar(agg_df, x_col, y_col, 
                                       title=f'{stat} de {y_col} por {x_col}')
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_bar_cat_num")
            
            elif (pd.api.types.is_object_dtype(x_type) or pd.api.types.is_categorical_dtype(x_type) or pd.api.types.is_string_dtype(x_type) or pd.api.types.is_bool_dtype(x_type)) and \
                 (pd.api.types.is_object_dtype(y_type) or pd.api.types.is_categorical_dtype(y_type) or pd.api.types.is_string_dtype(y_type) or pd.api.types.is_bool_dtype(y_type)):
                viz_type = st.radio(
                    "Tipo de visualización",
                    ["Heatmap", "Mosaico", "Barras Apiladas"],
                    horizontal=True,
                    key="bivariate_viz_type_cat_cat"
                )
                
                # Limitar categorías para visualización
                max_categories_x = st.slider(
                    f"Máximo de categorías a mostrar para {x_col}", 
                    5, 20, 10, 
                    key="bivariate_max_categories_x"
                )
                
                max_categories_y = st.slider(
                    f"Máximo de categorías a mostrar para {y_col}", 
                    5, 20, 10, 
                    key="bivariate_max_categories_y"
                )
                
                # Filtrar las categorías más frecuentes
                top_categories_x = data[x_col].value_counts().nlargest(max_categories_x).index.tolist()
                top_categories_y = data[y_col].value_counts().nlargest(max_categories_y).index.tolist()
                filtered_df = data[data[x_col].isin(top_categories_x) & 
                                      data[y_col].isin(top_categories_y)]
                
                # Generar gráfico según el tipo seleccionado
                if viz_type == "Heatmap":
                    # Crear tabla de contingencia
                    contingency = pd.crosstab(filtered_df[y_col], filtered_df[x_col])
                    
                    # Generar gráfico
                    fig = dv.create_heatmap(contingency, 
                                           title=f'Heatmap de {x_col} vs {y_col}')
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_heatmap_cat_cat")
                
                elif viz_type == "Mosaico":
                    st.info("La visualización de mosaico requiere la biblioteca squarify que no está incluida por defecto. Se muestra un heatmap alternativo.")
                    
                    # Crear tabla de contingencia
                    contingency = pd.crosstab(filtered_df[y_col], filtered_df[x_col])
                    
                    # Generar gráfico
                    fig = dv.create_heatmap(contingency, 
                                           title=f'Heatmap de {x_col} vs {y_col}')
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_mosaic_cat_cat")
                
                elif viz_type == "Barras Apiladas":
                    # Crear tabla de contingencia
                    contingency = pd.crosstab(filtered_df[x_col], filtered_df[y_col], normalize="index")
                    
                    # Generar gráfico
                    fig = go.Figure()
                    
                    for col in contingency.columns:
                        fig.add_trace(go.Bar(
                            x=contingency.index,
                            y=contingency[col],
                            name=str(col)
                        ))
                    
                    fig.update_layout(
                        title=f'Distribución de {y_col} por {x_col}',
                        xaxis_title=x_col,
                        yaxis_title='Proporción',
                        barmode='stack',
                        legend_title=y_col
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_stackedbar_cat_cat")
            
            elif pd.api.types.is_datetime64_any_dtype(x_type) and pd.api.types.is_numeric_dtype(y_type):
                # Generar gráfico de serie temporal
                fig = dv.create_timeseries(data, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True, key="univariate_temporal_box")
                
                # Opciones adicionales
                if st.checkbox("Mostrar tendencia", key="show_trend"):
                    # Agregar línea de tendencia
                    fig = px.scatter(data, x=x_col, y=y_col, trendline="ols")
                    st.plotly_chart(fig, use_container_width=True, key="bivariate_timeseries_trend")
            
            else:
                st.info("La combinación de tipos de datos seleccionada no tiene una visualización predefinida.")
    
    # Pestaña de visualizaciones multivariantes
    with tabs[2]:
        st.markdown("<div class='subsection-header'>🔍 Análisis Multivariante</div>", unsafe_allow_html=True)
        
        # Seleccionar tipo de visualización
        multi_viz_type = st.selectbox(
            "Tipo de visualización",
            ["Matriz de Correlación", "Pairplot", "Scatter 3D", "Parallel Coordinates"],
            key="multivariate_viz_type"
        )
        
        if multi_viz_type == "Matriz de Correlación":
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if len(numeric_cols) > 1:
                # Seleccionar columnas para la matriz
                selected_cols = st.multiselect(
                    "Selecciona columnas para la matriz de correlación",
                    numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))],
                    key="corr_matrix_cols"
                )
                
                if selected_cols and len(selected_cols) > 1:
                    # Seleccionar método de correlación
                    corr_method = st.radio(
                        "Método de correlación",
                        ["pearson", "spearman", "kendall"],
                        horizontal=True,
                        key="corr_method"
                    )
                    
                    # Generar matriz de correlación
                    corr_matrix = data[selected_cols].corr(method=corr_method)
                    
                    # Generar gráfico
                    fig = dv.create_heatmap(corr_matrix, 
                                           title=f'Matriz de Correlación ({corr_method})')
                    st.plotly_chart(fig, use_container_width=True, key="multivariate_corr_matrix")
                    
                    # Mostrar tabla de correlación
                    if st.checkbox("Mostrar tabla de correlación", key="show_corr_table"):
                        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))
                else:
                    st.info("Selecciona al menos dos columnas numéricas para generar la matriz de correlación.")
            else:
                st.info("No hay suficientes columnas numéricas para generar una matriz de correlación.")
        
        elif multi_viz_type == "Pairplot":
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if len(numeric_cols) > 1:
                # Seleccionar columnas para el pairplot
                selected_cols = st.multiselect(
                    "Selecciona columnas para el pairplot",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key="pairplot_cols"
                )
                
                # Seleccionar columna para color (opcional)
                color_col = st.selectbox(
                    "Columna para color (opcional)",
                    [None] + [col for col in data.columns if col not in selected_cols and 
                             (pd.api.types.is_categorical_dtype(data[col].dtype) or 
                              pd.api.types.is_object_dtype(data[col].dtype) or 
                              pd.api.types.is_string_dtype(data[col].dtype) or 
                              pd.api.types.is_bool_dtype(data[col].dtype))],
                    key="pairplot_color_col"
                )
                
                if selected_cols and len(selected_cols) > 1:
                    # Limitar número de filas para rendimiento
                    max_rows = st.slider(
                        "Máximo de filas a incluir", 
                        1000, min(10000, len(data)), 
                        min(5000, len(data)),
                        key="pairplot_max_rows"
                    )
                    
                    # Subconjunto de datos
                    if len(data) > max_rows:
                        sample_df = data.sample(max_rows, random_state=42)
                    else:
                        sample_df = data
                    
                    # Generar pairplot
                    if st.button("Generar Pairplot", key="generate_pairplot"):
                        with st.spinner("Generando pairplot..."):
                            fig = dv.create_pairplot(sample_df, selected_cols, color_col)
                            st.pyplot(fig)
                else:
                    st.info("Selecciona al menos dos columnas numéricas para generar el pairplot.")
            else:
                st.info("No hay suficientes columnas numéricas para generar un pairplot.")
        
        elif multi_viz_type == "Scatter 3D":
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if len(numeric_cols) >= 3:
                # Seleccionar columnas para los ejes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox(
                        "Eje X",
                        numeric_cols,
                        key="scatter3d_x_col"
                    )
                
                with col2:
                    y_col = st.selectbox(
                        "Eje Y",
                        [col for col in numeric_cols if col != x_col],
                        key="scatter3d_y_col"
                    )
                
                with col3:
                    z_col = st.selectbox(
                        "Eje Z",
                        [col for col in numeric_cols if col not in [x_col, y_col]],
                        key="scatter3d_z_col"
                    )
                
                # Seleccionar columna para color (opcional)
                color_col = st.selectbox(
                    "Columna para color (opcional)",
                    [None] + [col for col in data.columns if col not in [x_col, y_col, z_col]],
                    key="scatter3d_color_col"
                )
                
                # Limitar número de filas para rendimiento
                max_rows = st.slider(
                    "Máximo de puntos a mostrar", 
                    1000, min(10000, len(data)), 
                    min(5000, len(data)),
                    key="scatter3d_max_rows"
                )
                
                # Subconjunto de datos
                if len(data) > max_rows:
                    sample_df = data.sample(max_rows, random_state=42)
                else:
                    sample_df = data
                
                # Generar gráfico
                fig = px.scatter_3d(sample_df, x=x_col, y=y_col, z=z_col, color=color_col, size=size_col,
                                   title=title)
                st.plotly_chart(fig, use_container_width=True, key="custom_viz_3d_scatter")
            else:
                st.info("Se necesitan al menos tres columnas numéricas para generar un Scatter 3D.")
        
        elif multi_viz_type == "Parallel Coordinates":
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if len(numeric_cols) >= 3:
                # Seleccionar columnas para el gráfico
                selected_cols = st.multiselect(
                    "Selecciona columnas para el gráfico",
                    numeric_cols,
                    default=numeric_cols[:min(6, len(numeric_cols))],
                    key="parallel_coords_cols"
                )
                
                # Seleccionar columna para color (opcional)
                color_col = st.selectbox(
                    "Columna para color (opcional)",
                    [None] + [col for col in data.columns],
                    key="parallel_coords_color_col"
                )
                
                if selected_cols and len(selected_cols) >= 3:
                    # Limitar número de filas para rendimiento
                    max_rows = st.slider(
                        "Máximo de filas a incluir", 
                        1000, min(5000, len(data)), 
                        min(2000, len(data)),
                        key="parallel_coords_max_rows"
                    )
                    
                    # Subconjunto de datos
                    if len(data) > max_rows:
                        sample_df = data.sample(max_rows, random_state=42)
                    else:
                        sample_df = data
                    
                    # Generar gráfico
                    fig = px.parallel_coordinates(sample_df, dimensions=selected_cols, color=color_col,
                                                title='Parallel Coordinates Plot')
                    st.plotly_chart(fig, use_container_width=True, key="multivariate_parallel_coords")
                else:
                    st.info("Selecciona al menos tres columnas numéricas para generar el gráfico.")
            else:
                st.info("No hay suficientes columnas numéricas para generar un gráfico de coordenadas paralelas.")
    
    # Pestaña de series temporales
    with tabs[3]:
        st.markdown("<div class='subsection-header'>📅 Análisis de Series Temporales</div>", unsafe_allow_html=True)
        
        # Filtrar columnas temporales
        temporal_cols = [col for col in data.columns 
                       if pd.api.types.is_datetime64_any_dtype(data[col].dtype)]
        
        if temporal_cols:
            # Seleccionar columna temporal
            time_col = st.selectbox(
                "Selecciona columna temporal",
                temporal_cols,
                key="timeseries_time_col"
            )
            
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if numeric_cols:
                # Seleccionar columna numérica
                value_col = st.selectbox(
                    "Selecciona columna de valores",
                    numeric_cols,
                    key="timeseries_value_col"
                )
                
                # Seleccionar tipo de agregación temporal
                agg_freq = st.selectbox(
                    "Frecuencia de agregación",
                    ["Diaria", "Semanal", "Mensual", "Trimestral", "Anual"],
                    key="timeseries_agg_freq"
                )
                
                # Mapear frecuencia a formato pandas
                freq_map = {
                    "Diaria": "D",
                    "Semanal": "W",
                    "Mensual": "M",
                    "Trimestral": "Q",
                    "Anual": "Y"
                }
                
                # Seleccionar función de agregación
                agg_func = st.selectbox(
                    "Función de agregación",
                    ["Media", "Suma", "Min", "Max", "Conteo"],
                    key="timeseries_agg_func"
                )
                
                # Mapear función a método pandas
                func_map = {
                    "Media": "mean",
                    "Suma": "sum",
                    "Min": "min",
                    "Max": "max",
                    "Conteo": "count"
                }
                
                # Generar serie temporal
                try:
                    # Asegurar que la columna temporal es datetime
                    if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
                        data_copy = data.copy()
                        data_copy[time_col] = pd.to_datetime(data_copy[time_col], errors='coerce')
                    else:
                        data_copy = data.copy()
                    
                    # Agrupar por frecuencia temporal
                    time_df = data_copy.groupby(pd.Grouper(key=time_col, freq=freq_map[agg_freq]))[value_col]\
                                     .agg(func_map[agg_func]).reset_index()
                    
                    # Eliminar filas con fechas nulas
                    time_df = time_df.dropna(subset=[time_col])
                    
                    # Opciones de visualización
                    viz_options = st.multiselect(
                        "Opciones de visualización",
                        ["Línea", "Tendencia", "Media Móvil", "Descomposición"],
                        default=["Línea"],
                        key="timeseries_viz_options"
                    )
                    
                    # Generar gráficos según opciones seleccionadas
                    if "Línea" in viz_options:
                        fig = dv.create_timeseries(time_df, time_col, value_col, 
                                                 title=f'{agg_func} {value_col} ({agg_freq})')
                        st.plotly_chart(fig, use_container_width=True, key="timeseries_line_plot")
                    
                    if "Tendencia" in viz_options:
                        fig = px.scatter(time_df, x=time_col, y=value_col, trendline="ols",
                                        title=f'Tendencia de {value_col} ({agg_freq})')
                        st.plotly_chart(fig, use_container_width=True, key="timeseries_trend_plot")
                    
                    if "Media Móvil" in viz_options:
                        # Seleccionar ventana para media móvil
                        window = st.slider(
                            "Ventana para media móvil", 
                            2, min(30, len(time_df)), 
                            7,
                            key="timeseries_ma_window"
                        )
                        
                        # Calcular media móvil
                        time_df['MA'] = time_df[value_col].rolling(window=window).mean()
                        
                        # Generar gráfico
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=time_df[time_col],
                            y=time_df[value_col],
                            name=value_col,
                            line=dict(color='blue')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=time_df[time_col],
                            y=time_df['MA'],
                            name=f'Media Móvil ({window})',
                            line=dict(color='red')
                        ))
                        
                        fig.update_layout(
                            title=f'Media Móvil de {value_col} ({agg_freq})',
                            xaxis_title=time_col,
                            yaxis_title=value_col
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="timeseries_moving_average_plot")
                    
                    if "Descomposición" in viz_options:
                        try:
                            from statsmodels.tsa.seasonal import seasonal_decompose
                            
                            # Verificar que hay suficientes datos
                            if len(time_df) >= 4:
                                # Configurar índice temporal
                                time_df_idx = time_df.set_index(time_col)
                                
                                # Seleccionar modelo
                                model = st.radio(
                                    "Modelo de descomposición",
                                    ["additive", "multiplicative"],
                                    horizontal=True,
                                    key="decomposition_model"
                                )
                                
                                # Realizar descomposición
                                decomposition = seasonal_decompose(time_df_idx[value_col], model=model, extrapolate_trend='freq')
                                
                                # Crear figura para componentes
                                fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
                                
                                # Graficar componentes
                                decomposition.observed.plot(ax=axes[0], title='Observado')
                                decomposition.trend.plot(ax=axes[1], title='Tendencia')
                                decomposition.seasonal.plot(ax=axes[2], title='Estacionalidad')
                                decomposition.resid.plot(ax=axes[3], title='Residuos')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.warning("Se necesitan al menos 4 puntos de datos para la descomposición.")
                        except Exception as e:
                            st.error(f"Error en la descomposición: {str(e)}")
                except Exception as e:
                    st.error(f"Error al generar serie temporal: {str(e)}")
            else:
                st.info("No se encontraron columnas numéricas para analizar junto con la serie temporal.")
        else:
            st.info("No se encontraron columnas temporales en el dataset. Para análisis de series temporales, necesitas al menos una columna de tipo fecha/hora.")
    
    # Pestaña de visualización personalizada
    with tabs[4]:
        st.markdown("<div class='subsection-header'>🎨 Visualización Personalizada</div>", unsafe_allow_html=True)
        
        # Seleccionar tipo de gráfico
        custom_viz_type = st.selectbox(
            "Tipo de gráfico",
            ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Pie", "Heatmap", "3D Scatter"],
            key="custom_viz_type"
        )
        
        # Configuración según tipo de gráfico
        if custom_viz_type in ["Scatter", "Line", "Bar"]:
            # Seleccionar columnas para ejes
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "Columna para eje X",
                    data.columns,
                    key="custom_x_col"
                )
            
            with col2:
                y_col = st.selectbox(
                    "Columna para eje Y",
                    [col for col in data.columns if col != x_col],
                    key="custom_y_col"
                )
            
            # Opciones adicionales
            color_col = st.selectbox(
                "Columna para color (opcional)",
                [None] + [col for col in data.columns if col not in [x_col, y_col]],
                key="custom_color_col"
            )
            
            size_col = st.selectbox(
                "Columna para tamaño (opcional)",
                [None] + [col for col in data.columns 
                         if pd.api.types.is_numeric_dtype(data[col].dtype) and col not in [x_col, y_col]],
                key="custom_size_col"
            )
            
            # Título y etiquetas
            title = st.text_input("Título del gráfico", f"{custom_viz_type} de {x_col} vs {y_col}", key="custom_title")
            x_label = st.text_input("Etiqueta eje X", x_col, key="custom_x_label")
            y_label = st.text_input("Etiqueta eje Y", y_col, key="custom_y_label")
            
            # Generar gráfico según tipo
            if custom_viz_type == "Scatter":
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col, size=size_col,
                               title=title, labels={x_col: x_label, y_col: y_label})
            
            elif custom_viz_type == "Line":
                fig = px.line(data, x=x_col, y=y_col, color=color_col,
                            title=title, labels={x_col: x_label, y_col: y_label})
            
            elif custom_viz_type == "Bar":
                # Limitar categorías para barras
                max_categories = st.slider(
                    "Máximo de categorías a mostrar", 
                    5, 50, 20, 
                    key="custom_bar_max_categories"
                )
                
                # Agrupar datos si es necesario
                if data[x_col].nunique() > max_categories:
                    top_categories = data[x_col].value_counts().nlargest(max_categories).index.tolist()
                    filtered_df = data[data[x_col].isin(top_categories)]
                else:
                    filtered_df = data
                
                fig = px.bar(filtered_df, x=x_col, y=y_col, color=color_col,
                           title=title, labels={x_col: x_label, y_col: y_label})
            
            st.plotly_chart(fig, use_container_width=True, key="custom_viz_scatter_line_bar")
        
        elif custom_viz_type in ["Histogram", "Box", "Violin"]:
            # Seleccionar columna
            col_to_viz = st.selectbox(
                "Columna para visualizar",
                data.columns,
                key="custom_col_to_viz"
            )
            
            # Opciones adicionales
            color_col = st.selectbox(
                "Columna para color/agrupación (opcional)",
                [None] + [col for col in data.columns if col != col_to_viz],
                key="custom_group_col"
            )
            
            # Título y etiquetas
            title = st.text_input("Título del gráfico", f"{custom_viz_type} de {col_to_viz}", key="custom_single_title")
            
            # Generar gráfico según tipo
            if custom_viz_type == "Histogram":
                # Opciones específicas para histograma
                n_bins = st.slider("Número de bins", 5, 100, 20, key="custom_hist_bins")
                
                fig = px.histogram(data, x=col_to_viz, color=color_col, nbins=n_bins,
                                 title=title)
            
            elif custom_viz_type == "Box":
                fig = px.box(data, x=color_col, y=col_to_viz, color=color_col,
                           title=title)
            
            elif custom_viz_type == "Violin":
                fig = px.violin(data, x=color_col, y=col_to_viz, color=color_col,
                              title=title, box=True)
            
            st.plotly_chart(fig, use_container_width=True, key="custom_viz_histogram_box_violin")
        
        elif custom_viz_type == "Pie":
            # Seleccionar columna para categorías
            cat_col = st.selectbox(
                "Columna para categorías",
                [col for col in data.columns 
                 if pd.api.types.is_categorical_dtype(data[col].dtype) or 
                    pd.api.types.is_object_dtype(data[col].dtype) or 
                    pd.api.types.is_string_dtype(data[col].dtype) or 
                    pd.api.types.is_bool_dtype(data[col].dtype)],
                key="custom_pie_cat_col"
            )
            
            # Seleccionar columna para valores (opcional)
            val_col = st.selectbox(
                "Columna para valores (opcional, por defecto es conteo)",
                [None] + [col for col in data.columns 
                         if pd.api.types.is_numeric_dtype(data[col].dtype)],
                key="custom_pie_val_col"
            )
            
            # Limitar categorías
            max_categories = st.slider(
                "Máximo de categorías a mostrar", 
                3, 20, 8, 
                key="custom_pie_max_categories"
            )
            
            # Título
            title = st.text_input("Título del gráfico", f"Distribución de {cat_col}", key="custom_pie_title")
            
            # Preparar datos para pie chart
            if val_col:
                # Agrupar por categoría y sumar valores
                pie_data = data.groupby(cat_col)[val_col].sum().reset_index()
            else:
                # Contar ocurrencias de cada categoría
                pie_data = data[cat_col].value_counts().reset_index()
                pie_data.columns = [cat_col, 'count']
                val_col = 'count'
            
            # Limitar a las top categorías
            if len(pie_data) > max_categories:
                top_categories = pie_data.nlargest(max_categories - 1, val_col)
                other_count = pie_data.iloc[max_categories - 1:][val_col].sum()
                other_row = pd.DataFrame({cat_col: ['Otros'], val_col: [other_count]})
                pie_data = pd.concat([top_categories, other_row], ignore_index=True)
            
            # Generar gráfico
            fig = px.pie(pie_data, names=cat_col, values=val_col, title=title)
            st.plotly_chart(fig, use_container_width=True, key="custom_viz_pie")
        
        elif custom_viz_type == "Heatmap":
            # Seleccionar columnas para ejes
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "Columna para eje X",
                    [col for col in data.columns 
                     if pd.api.types.is_categorical_dtype(data[col].dtype) or 
                        pd.api.types.is_object_dtype(data[col].dtype) or 
                        pd.api.types.is_string_dtype(data[col].dtype) or 
                        pd.api.types.is_bool_dtype(data[col].dtype)],
                    key="custom_heatmap_x_col"
                )
            
            with col2:
                y_col = st.selectbox(
                    "Columna para eje Y",
                    [col for col in data.columns 
                     if (pd.api.types.is_categorical_dtype(data[col].dtype) or 
                         pd.api.types.is_object_dtype(data[col].dtype) or 
                         pd.api.types.is_string_dtype(data[col].dtype) or 
                         pd.api.types.is_bool_dtype(data[col].dtype)) and col != x_col],
                    key="custom_heatmap_y_col"
                )
            
            # Seleccionar columna para valores (opcional)
            val_col = st.selectbox(
                "Columna para valores (opcional, por defecto es conteo)",
                [None] + [col for col in data.columns 
                         if pd.api.types.is_numeric_dtype(data[col].dtype)],
                key="custom_heatmap_val_col"
            )
            
            # Título
            title = st.text_input("Título del gráfico", f"Heatmap de {x_col} vs {y_col}", key="custom_heatmap_title")
            
            # Limitar categorías
            max_categories_x = st.slider(
                f"Máximo de categorías para {x_col}", 
                5, 30, 15, 
                key="custom_heatmap_max_x"
            )
            
            max_categories_y = st.slider(
                f"Máximo de categorías para {y_col}", 
                5, 30, 15, 
                key="custom_heatmap_max_y"
            )
            
            # Filtrar las categorías más frecuentes
            top_categories_x = data[x_col].value_counts().nlargest(max_categories_x).index.tolist()
            top_categories_y = data[y_col].value_counts().nlargest(max_categories_y).index.tolist()
            filtered_df = data[data[x_col].isin(top_categories_x) & 
                                  data[y_col].isin(top_categories_y)]
            
            # Crear tabla de contingencia
            if val_col:
                # Agrupar y agregar valores
                pivot_table = filtered_df.pivot_table(index=y_col, columns=x_col, values=val_col, aggfunc='mean')
            else:
                # Contar ocurrencias
                pivot_table = pd.crosstab(filtered_df[y_col], filtered_df[x_col])
            
            # Generar gráfico
            fig = px.imshow(pivot_table, title=title, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True, key="custom_viz_heatmap")
        
        elif custom_viz_type == "3D Scatter":
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if len(numeric_cols) >= 3:
                # Seleccionar columnas para los ejes
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox(
                        "Eje X",
                        numeric_cols,
                        key="custom_3d_x_col"
                    )
                
                with col2:
                    y_col = st.selectbox(
                        "Eje Y",
                        [col for col in numeric_cols if col != x_col],
                        key="custom_3d_y_col"
                    )
                
                with col3:
                    z_col = st.selectbox(
                        "Eje Z",
                        [col for col in numeric_cols if col not in [x_col, y_col]],
                        key="custom_3d_z_col"
                    )
                
                # Opciones adicionales
                color_col = st.selectbox(
                    "Columna para color (opcional)",
                    [None] + [col for col in data.columns if col not in [x_col, y_col, z_col]],
                    key="custom_3d_color_col"
                )
                
                size_col = st.selectbox(
                    "Columna para tamaño (opcional)",
                    [None] + [col for col in numeric_cols if col not in [x_col, y_col, z_col]],
                    key="custom_3d_size_col"
                )
                
                # Título
                title = st.text_input("Título del gráfico", f"Scatter 3D de {x_col}, {y_col}, {z_col}", key="custom_3d_title")
                
                # Limitar número de puntos
                max_points = st.slider(
                    "Máximo de puntos a mostrar", 
                    1000, min(10000, len(data)), 
                    min(5000, len(data)),
                    key="custom_3d_max_points"
                )
                
                # Subconjunto de datos
                if len(data) > max_points:
                    sample_df = data.sample(max_points, random_state=42)
                else:
                    sample_df = data
                
                # Generar gráfico
                fig = px.scatter_3d(sample_df, x=x_col, y=y_col, z=z_col, color=color_col,
                                   title=f'Scatter 3D: {x_col} vs {y_col} vs {z_col}')
                st.plotly_chart(fig, use_container_width=True, key="multivariate_scatter_3d")
            else:
                st.info("Se necesitan al menos tres columnas numéricas para generar un Scatter 3D.")
    
    # Sección de exportación de visualizaciones
    st.markdown("<div class='section-header'>💾 Exportar Visualización</div>", unsafe_allow_html=True)
    
    # Opciones de exportación
    export_format = st.radio(
        "Formato de exportación",
        ["PNG", "PDF", "HTML Interactivo"],
        horizontal=True,
        key="export_format"
    )
    
    # Botón para exportar
    if st.button("Exportar Visualización Actual", key="export_viz_button"):
        st.info("Para exportar la visualización actual, utiliza el botón de descarga que aparece al pasar el cursor sobre el gráfico.")
        
        if export_format == "HTML Interactivo":
            st.info("Para guardar como HTML interactivo, haz clic en el botón 'Export to HTML' que aparece en la barra de herramientas del gráfico.")
        
        elif export_format == "PNG":
            st.info("Para guardar como PNG, haz clic en el botón 'Download plot as a png' que aparece en la barra de herramientas del gráfico.")
        
        elif export_format == "PDF":
            st.info("Para exportar a PDF, primero descarga como PNG y luego utiliza una herramienta de conversión o la función de exportación en el módulo de exportación.")