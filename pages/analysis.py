import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.data_analysis import get_descriptive_stats, analyze_column_distribution, get_correlation_matrix

def show_analysis_page(df):
    """
    Muestra la página de análisis detallado del dataset.
    
    Args:
        df: DataFrame de Polars
    """
    st.markdown("<div class='section-header'>🔍 Análisis Detallado</div>", unsafe_allow_html=True)
    
    # Crear pestañas para diferentes tipos de análisis
    tabs = st.tabs(["Estadísticas Descriptivas", "Análisis de Columnas", "Correlaciones", "Valores Atípicos"])
    
    # Pestaña de estadísticas descriptivas
    with tabs[0]:
        st.markdown("<div class='subsection-header'>📊 Estadísticas Descriptivas</div>", unsafe_allow_html=True)
        
        # Opciones para seleccionar columnas y estadísticas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Selección de columnas
            all_columns = df.columns
            selected_columns = st.multiselect(
                "Selecciona columnas para analizar",
                all_columns,
                default=all_columns[:5] if len(all_columns) > 5 else all_columns
            )
        
        with col2:
            # Opciones adicionales
            include_percentiles = st.checkbox("Incluir percentiles", value=True)
        
        if selected_columns:
            # Calcular estadísticas descriptivas
            stats_df = get_descriptive_stats(df, columns=selected_columns, include_percentiles=include_percentiles)
            
            # Mostrar estadísticas
            st.dataframe(stats_df, use_container_width=True)
            
            # Opción para exportar
            if st.button("Exportar Estadísticas", key="export_stats"):
                from modules.data_export import export_stats
                buffer, filename, mime_type = export_stats(stats_df, format="excel")
                st.download_button(
                    label="Descargar Excel",
                    data=buffer,
                    file_name=filename,
                    mime=mime_type
                )
        else:
            st.info("Selecciona al menos una columna para ver las estadísticas.")
    
    # Pestaña de análisis de columnas
    with tabs[1]:
        st.markdown("<div class='subsection-header'>🔎 Análisis de Columnas</div>", unsafe_allow_html=True)
        
        # Selección de columna
        selected_column = st.selectbox(
            "Selecciona una columna para analizar en detalle",
            df.columns
        )
        
        if selected_column:
            # Analizar distribución de la columna
            analysis = analyze_column_distribution(df, selected_column)
            
            # Mostrar resultados según el tipo de columna
            col_type = df.schema[selected_column]
            
            # Dividir en columnas para mostrar información
            col1, col2 = st.columns([1, 1])
            
            # Análisis para columnas numéricas
            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                with col1:
                    # Estadísticas básicas
                    stats = pd.DataFrame({
                        "Estadística": ["Conteo", "Media", "Mediana", "Desv. Estándar", "Mínimo", "Máximo",
                                      "Asimetría", "Curtosis", "Valores Nulos", "% Nulos"],
                        "Valor": [
                            analysis["count"],
                            round(analysis["mean"], 4),
                            round(analysis["median"], 4),
                            round(analysis["std"], 4),
                            round(analysis["min"], 4),
                            round(analysis["max"], 4),
                            round(analysis["skewness"], 4),
                            round(analysis["kurtosis"], 4),
                            analysis["null_count"],
                            round(analysis["null_count"] / df.height * 100, 2)
                        ]
                    })
                    st.dataframe(stats, use_container_width=True)
                    
                    # Percentiles
                    percentiles = pd.DataFrame({
                        "Percentil": ["25%", "50%", "75%", "90%", "95%", "99%"],
                        "Valor": [
                            round(analysis["percentiles"]["25%"], 4),
                            round(analysis["percentiles"]["50%"], 4),
                            round(analysis["percentiles"]["75%"], 4),
                            round(analysis["percentiles"]["90%"], 4),
                            round(analysis["percentiles"]["95%"], 4),
                            round(analysis["percentiles"]["99%"], 4)
                        ]
                    })
                    st.dataframe(percentiles, use_container_width=True)
                    
                    # Información sobre outliers
                    st.metric("Valores Atípicos (Outliers)", 
                             f"{analysis['outliers_count']} ({analysis['outliers_percent']:.2f}%)")
                
                with col2:
                    # Histograma
                    fig = px.histogram(
                        df, x=selected_column,
                        title=f"Distribución de {selected_column}",
                        nbins=30,
                        marginal="box"
                    )
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True, key="numeric_histogram_chart")
                    
                    # Boxplot
                    fig = px.box(
                        df, y=selected_column,
                        title=f"Boxplot de {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="numeric_boxplot_chart")
            
            # Análisis para columnas categóricas
            elif pd.api.types.is_categorical_dtype(dtype):
                with col1:
                    # Estadísticas básicas
                    stats = pd.DataFrame({
                        "Estadística": ["Conteo", "Valores Únicos", "Valores Nulos", "% Nulos"],
                        "Valor": [
                            analysis["count"],
                            analysis["unique_values"],
                            analysis["null_count"],
                            round(analysis["null_count"] / df.height * 100, 2)
                        ]
                    })
                    st.dataframe(stats, use_container_width=True)
                    
                    # Valores más frecuentes
                    if "top_values" in analysis:
                        top_values = pd.DataFrame({
                            "Valor": list(analysis["top_values"].keys()),
                            "Frecuencia": list(analysis["top_values"].values()),
                            "Porcentaje": [round(p, 2) for p in list(analysis["top_values_percent"].values())]
                        })
                        st.dataframe(top_values, use_container_width=True)
                
                with col2:
                    # Gráfico de barras para valores más frecuentes
                    if "top_values" in analysis:
                        top_values = pd.DataFrame({
                            "Valor": list(analysis["top_values"].keys()),
                            "Frecuencia": list(analysis["top_values"].values()),
                            "Porcentaje": [round(p, 2) for p in list(analysis["top_values_percent"].values())]
                        })
                        
                        fig = px.bar(
                            top_values,
                            x="Valor",
                            y="Frecuencia",
                            title=f"Valores más frecuentes de {selected_column}",
                            color="Porcentaje",
                            color_continuous_scale="Blues"
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True, key="categorical_bar_chart")
                        
                        # Gráfico de torta
                        fig = px.pie(
                            top_values,
                            values="Frecuencia",
                            names="Valor",
                            title=f"Distribución de {selected_column}",
                            hole=0.4
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True, key="categorical_pie_chart")
            
            # Análisis para columnas booleanas
            elif pd.api.types.is_bool_dtype(dtype):
                with col1:
                    # Estadísticas básicas
                    stats = pd.DataFrame({
                        "Estadística": ["Conteo", "True", "False", "% True", "% False", "Valores Nulos", "% Nulos"],
                        "Valor": [
                            analysis["count"],
                            analysis["true_count"],
                            analysis["false_count"],
                            round(analysis["true_percent"], 2),
                            round(analysis["false_percent"], 2),
                            analysis["null_count"],
                            round(analysis["null_count"] / df.height * 100, 2)
                        ]
                    })
                    st.dataframe(stats, use_container_width=True)
                
                with col2:
                    # Gráfico de torta
                    fig = px.pie(
                        values=[analysis["true_count"], analysis["false_count"], analysis["null_count"]],
                        names=["True", "False", "Nulos"],
                        title=f"Distribución de {selected_column}",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True, key="boolean_pie_chart")
            
            # Análisis para columnas de fecha/hora
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                with col1:
                    # Estadísticas básicas
                    stats_data = {
                        "Estadística": ["Conteo", "Fecha Mínima", "Fecha Máxima", "Valores Nulos", "% Nulos"],
                        "Valor": [
                            analysis["count"],
                            analysis["min"],
                            analysis["max"],
                            analysis["null_count"],
                            round(analysis["null_count"] / df.height * 100, 2)
                        ]
                    }
                    
                    # Añadir rango en días si está disponible
                    if "range_days" in analysis:
                        stats_data["Estadística"].append("Rango (días)")
                        stats_data["Valor"].append(analysis["range_days"])
                    
                    stats = pd.DataFrame(stats_data)
                    st.dataframe(stats, use_container_width=True)
                
                with col2:
                    # Histograma de fechas
                    pandas_df = df.select([selected_column])
                    fig = px.histogram(
                        pandas_df,
                        x=selected_column,
                        title=f"Distribución de {selected_column}",
                        nbins=30
                    )
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig, use_container_width=True, key="temporal_histogram_chart")
    
    # Pestaña de correlaciones
    with tabs[2]:
        st.markdown("<div class='subsection-header'>🔗 Matriz de Correlación</div>", unsafe_allow_html=True)
        
        # Opciones para la matriz de correlación
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Selección de columnas numéricas
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            
            if len(numeric_cols) < 2:
                st.warning("Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")
            else:
                selected_numeric_cols = st.multiselect(
                    "Selecciona columnas numéricas para la matriz de correlación",
                    numeric_cols,
                    default=numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
                )
        
        with col2:
            # Método de correlación
            corr_method = st.radio(
                "Método de correlación",
                ["pearson", "spearman"]
            )
        
        if len(numeric_cols) >= 2 and selected_numeric_cols:
            # Filtrar DataFrame para incluir solo las columnas seleccionadas
            df_selected = df.select(selected_numeric_cols)
            
            # Calcular matriz de correlación
            corr_matrix = get_correlation_matrix(df_selected, method=corr_method)
            
            if corr_matrix is not None:
                # Convertir a pandas para visualización
                corr_pd = corr_matrix
                
                # Crear mapa de calor con Plotly
                fig = go.Figure(data=go.Heatmap(
                    z=corr_pd.values,
                    x=corr_pd.columns,
                    y=corr_pd.index,
                    colorscale="RdBu_r",
                    zmin=-1, zmax=1,
                    text=np.round(corr_pd.values, 2),
                    texttemplate="%{text}",
                    textfont={"size":10},
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title=f"Matriz de Correlación ({corr_method})",
                    height=600,
                    width=800,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, key="correlation_heatmap_chart")
                
                # Mostrar tabla de correlación
                with st.expander("Ver tabla de correlación"):
                    st.dataframe(corr_pd.round(3), use_container_width=True)
                
                # Correlaciones más fuertes
                with st.expander("Correlaciones más fuertes"):
                    # Obtener correlaciones en formato largo
                    corr_long = []
                    for i, col1 in enumerate(corr_pd.columns):
                        for j, col2 in enumerate(corr_pd.columns):
                            if i < j:  # Solo la mitad superior de la matriz
                                corr_long.append({
                                    "Variable 1": col1,
                                    "Variable 2": col2,
                                    "Correlación": corr_pd.iloc[i, j]
                                })
                    
                    # Convertir a DataFrame y ordenar
                    corr_long_df = pd.DataFrame(corr_long)
                    corr_long_df["Correlación Abs"] = corr_long_df["Correlación"].abs()
                    corr_long_df = corr_long_df.sort_values("Correlación Abs", ascending=False)
                    
                    # Mostrar top 10 correlaciones
                    st.dataframe(corr_long_df.head(10).drop("Correlación Abs", axis=1).round(3), use_container_width=True)
            else:
                st.warning("No se pudo calcular la matriz de correlación.")
    
    # Pestaña de valores atípicos
    with tabs[3]:
        st.markdown("<div class='subsection-header'>⚠️ Detección de Valores Atípicos</div>", unsafe_allow_html=True)
        
        # Selección de columna numérica
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            selected_col = st.selectbox(
                "Selecciona una columna numérica para detectar valores atípicos",
                numeric_cols
            )
            
            # Método de detección
            detection_method = st.radio(
                "Método de detección",
                ["IQR (Rango Intercuartil)", "Z-Score"]
            )
            
            # Parámetros según el método
            if detection_method == "IQR (Rango Intercuartil)":
                iqr_factor = st.slider("Factor IQR", 1.0, 3.0, 1.5, 0.1,
                                      help="Valores más altos son menos restrictivos. Estándar: 1.5")
            else:  # Z-Score
                z_threshold = st.slider("Umbral Z-Score", 2.0, 5.0, 3.0, 0.1,
                                      help="Valores más altos son menos restrictivos. Estándar: 3.0")
            
            # Detectar outliers
            if detection_method == "IQR (Rango Intercuartil)":
                # Calcular Q1, Q3 e IQR
                q1 = float(df.get_column(selected_col).quantile(0.25))
                q3 = float(df.get_column(selected_col).quantile(0.75))
                iqr = q3 - q1
                
                # Definir límites
                lower_bound = q1 - iqr_factor * iqr
                upper_bound = q3 + iqr_factor * iqr
                
                # Identificar outliers
                outliers_mask = (df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)
                outliers_df = df.filter(outliers_mask)
                
                # Mostrar información
                st.write(f"**Límites para detección de outliers:**")
                st.write(f"- Q1 (25%): {q1:.4f}")
                st.write(f"- Q3 (75%): {q3:.4f}")
                st.write(f"- IQR: {iqr:.4f}")
                st.write(f"- Límite inferior: {lower_bound:.4f}")
                st.write(f"- Límite superior: {upper_bound:.4f}")
            
            else:  # Z-Score
                # Calcular media y desviación estándar
                mean_val = float(df.get_column(selected_col).mean())
                std_val = float(df.get_column(selected_col).std())
                
                # Calcular Z-scores
                z_scores = (df.get_column(selected_col) - mean_val) / std_val
                
                # Identificar outliers
                outliers_mask = (z_scores.abs() > z_threshold)
                outliers_df = df.filter(outliers_mask)
                
                # Mostrar información
                st.write(f"**Parámetros para detección de outliers:**")
                st.write(f"- Media: {mean_val:.4f}")
                st.write(f"- Desviación estándar: {std_val:.4f}")
                st.write(f"- Umbral Z-Score: ±{z_threshold}")
                st.write(f"- Límite inferior: {mean_val - z_threshold * std_val:.4f}")
                st.write(f"- Límite superior: {mean_val + z_threshold * std_val:.4f}")
            
            # Mostrar resultados
            st.write(f"**Se encontraron {outliers_df.height} valores atípicos ({outliers_df.height / df.height * 100:.2f}% del total)**")
            
            # Visualización
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot
                fig = px.box(
                    df, y=selected_col,
                    title=f"Boxplot de {selected_col} con outliers"
                )
                st.plotly_chart(fig, use_container_width=True, key="outlier_boxplot_chart")
            
            with col2:
                # Histograma
                fig = px.histogram(
                    df, x=selected_col,
                    title=f"Distribución de {selected_col}",
                    nbins=30,
                    marginal="box"
                )
                fig.update_layout(bargap=0.1)
                
                # Añadir líneas para los límites
                if detection_method == "IQR (Rango Intercuartil)":
                    fig.add_vline(x=lower_bound, line_dash="dash", line_color="red")
                    fig.add_vline(x=upper_bound, line_dash="dash", line_color="red")
                else:  # Z-Score
                    fig.add_vline(x=mean_val - z_threshold * std_val, line_dash="dash", line_color="red")
                    fig.add_vline(x=mean_val + z_threshold * std_val, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True, key="outlier_histogram_chart")
            
            # Mostrar outliers
            if outliers_df.height > 0:
                with st.expander("Ver valores atípicos"):
                    st.dataframe(outliers_df, use_container_width=True)
                    
                    # Opción para exportar outliers
                    if st.button("Exportar Valores Atípicos"):
                        from modules.data_export import export_data
                        buffer, filename, mime_type = export_data(outliers_df, format="csv", filename=f"outliers_{selected_col}")
                        st.download_button(
                            label="Descargar CSV",
                            data=buffer,
                            file_name=filename,
                            mime=mime_type
                        )
        else:
            st.info("No se encontraron columnas numéricas en el dataset.")