import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import sys
import os
from datetime import datetime

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import data_export as de
from modules import data_analysis as da
from modules import data_visualization as dv

def show_export_page(df):
    """
    Muestra la página de exportación de datos, estadísticas y visualizaciones.
    
    Args:
        df: DataFrame de Pandas
    """
    st.markdown("<div class='section-header'>💾 Exportación de Resultados</div>", unsafe_allow_html=True)
    
    # Usar DataFrame filtrado si existe
    if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
        data = st.session_state.filtered_data
    elif "filtered_data_saved" in st.session_state and st.session_state.filtered_data_saved is not None:
        data = st.session_state.filtered_data_saved
    else:
        data = df.copy()
    
    # Crear pestañas para diferentes tipos de exportación
    tabs = st.tabs(["Exportar Dataset", "Exportar Estadísticas", "Exportar Visualizaciones", "Generar Reporte"])
    
    # Pestaña de exportación de dataset
    with tabs[0]:
        st.markdown("<div class='subsection-header'>📊 Exportar Dataset</div>", unsafe_allow_html=True)
        
        # Información sobre el dataset a exportar
        st.write(f"**Dataset actual:** {data.shape[0]} filas × {data.shape[1]} columnas")
        
        # Opciones de exportación
        export_format = st.radio(
            "Formato de exportación",
            ["CSV", "Excel", "Parquet", "JSON"],
            horizontal=True,
            key="dataset_export_format"
        )
        
        # Opciones específicas según formato
        if export_format == "CSV":
            # Opciones para CSV
            col1, col2, col3 = st.columns(3)
            
            with col1:
                delimiter = st.selectbox(
                    "Delimitador",
                    [",", ";", "\t", "|", " "],
                    index=0,
                    key="csv_delimiter"
                )
            
            with col2:
                include_header = st.checkbox("Incluir encabezado", value=True, key="csv_header")
            
            with col3:
                encoding = st.selectbox(
                    "Codificación",
                    ["utf-8", "latin-1", "iso-8859-1", "cp1252"],
                    index=0,
                    key="csv_encoding"
                )
            
            # Botón para exportar
            if st.button("Exportar a CSV", key="export_csv_button"):
                try:
                    # Generar CSV
                    csv_data = de.export_dataframe_to_csv(data, delimiter=delimiter, 
                                                        include_header=include_header, 
                                                        encoding=encoding)
                    
                    # Crear enlace de descarga
                    b64 = base64.b64encode(csv_data.encode(encoding)).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Descargar CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("CSV generado correctamente. Haz clic en el enlace para descargar.")
                except Exception as e:
                    st.error(f"Error al exportar a CSV: {str(e)}")
        
        elif export_format == "Excel":
            # Opciones para Excel
            col1, col2 = st.columns(2)
            
            with col1:
                sheet_name = st.text_input("Nombre de la hoja", "Dataset", key="excel_sheet_name")
            
            with col2:
                include_index = st.checkbox("Incluir índice", value=False, key="excel_index")
            
            # Botón para exportar
            if st.button("Exportar a Excel", key="export_excel_button"):
                try:
                    # Generar Excel
                    excel_data = de.export_dataframe_to_excel(data, sheet_name=sheet_name, 
                                                            include_index=include_index)
                    
                    # Crear enlace de descarga
                    b64 = base64.b64encode(excel_data).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Descargar Excel</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("Excel generado correctamente. Haz clic en el enlace para descargar.")
                except Exception as e:
                    st.error(f"Error al exportar a Excel: {str(e)}")
        
        elif export_format == "Parquet":
            # Botón para exportar
            if st.button("Exportar a Parquet", key="export_parquet_button"):
                try:
                    # Generar Parquet
                    parquet_data = de.export_dataframe_to_parquet(data)
                    
                    # Crear enlace de descarga
                    b64 = base64.b64encode(parquet_data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet">Descargar Parquet</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("Parquet generado correctamente. Haz clic en el enlace para descargar.")
                except Exception as e:
                    st.error(f"Error al exportar a Parquet: {str(e)}")
        
        elif export_format == "JSON":
            # Opciones para JSON
            col1, col2 = st.columns(2)
            
            with col1:
                orient = st.selectbox(
                    "Orientación",
                    ["records", "columns", "index", "split", "table"],
                    index=0,
                    key="json_orient"
                )
            
            with col2:
                indent = st.slider("Indentación", 0, 4, 2, key="json_indent")
            
            # Botón para exportar
            if st.button("Exportar a JSON", key="export_json_button"):
                try:
                    # Generar JSON
                    json_data = de.export_dataframe_to_json(data, orient=orient, indent=indent)
                    
                    # Crear enlace de descarga
                    b64 = base64.b64encode(json_data.encode('utf-8')).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="dataset_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Descargar JSON</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("JSON generado correctamente. Haz clic en el enlace para descargar.")
                except Exception as e:
                    st.error(f"Error al exportar a JSON: {str(e)}")
    
    # Pestaña de exportación de estadísticas
    with tabs[1]:
        st.markdown("<div class='subsection-header'>📈 Exportar Estadísticas</div>", unsafe_allow_html=True)
        
        # Seleccionar tipo de estadísticas
        stats_type = st.selectbox(
            "Tipo de estadísticas",
            ["Resumen General", "Estadísticas Descriptivas", "Matriz de Correlación"],
            key="stats_type"
        )
        
        # Opciones según tipo de estadísticas
        if stats_type == "Resumen General":
            # Generar resumen general
            summary = da.get_data_summary(data)
            
            # Mostrar vista previa
            st.write("**Vista previa:**")
            st.dataframe(summary, use_container_width=True)
            
            # Opciones de exportación
            export_format = st.radio(
                "Formato de exportación",
                ["CSV", "Excel"],
                horizontal=True,
                key="summary_export_format"
            )
            
            # Botón para exportar
            if st.button("Exportar Resumen", key="export_summary_button"):
                try:
                    if export_format == "CSV":
                        # Exportar a CSV
                        csv_data = de.export_statistics_to_csv(summary)
                        
                        # Crear enlace de descarga
                        b64 = base64.b64encode(csv_data.encode('utf-8')).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="resumen_general_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Descargar CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    elif export_format == "Excel":
                        # Exportar a Excel
                        excel_data = de.export_statistics_to_excel(summary, sheet_name="Resumen General")
                        
                        # Crear enlace de descarga
                        b64 = base64.b64encode(excel_data).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="resumen_general_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Descargar Excel</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    st.success(f"Resumen general exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                except Exception as e:
                    st.error(f"Error al exportar resumen: {str(e)}")
        
        elif stats_type == "Estadísticas Descriptivas":
            # Seleccionar columnas para estadísticas
            selected_cols = st.multiselect(
                "Selecciona columnas para estadísticas descriptivas",
                data.columns,
                default=data.columns[:min(5, len(data.columns))],
                key="desc_stats_cols"
            )
            
            if selected_cols:
                # Seleccionar percentiles
                percentiles = st.multiselect(
                    "Percentiles a incluir",
                    [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
                    default=[0.25, 0.5, 0.75],
                    key="desc_stats_percentiles"
                )
                
                # Generar estadísticas descriptivas
                desc_stats = da.get_descriptive_stats(data, selected_cols, percentiles)
                
                # Mostrar vista previa
                st.write("**Vista previa:**")
                st.dataframe(desc_stats, use_container_width=True)
                
                # Opciones de exportación
                export_format = st.radio(
                    "Formato de exportación",
                    ["CSV", "Excel"],
                    horizontal=True,
                    key="desc_stats_export_format"
                )
                
                # Botón para exportar
                if st.button("Exportar Estadísticas Descriptivas", key="export_desc_stats_button"):
                    try:
                        if export_format == "CSV":
                            # Exportar a CSV
                            csv_data = de.export_statistics_to_csv(desc_stats)
                            
                            # Crear enlace de descarga
                            b64 = base64.b64encode(csv_data.encode('utf-8')).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="estadisticas_descriptivas_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Descargar CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        
                        elif export_format == "Excel":
                            # Exportar a Excel
                            excel_data = de.export_statistics_to_excel(desc_stats, sheet_name="Estadísticas Descriptivas")
                            
                            # Crear enlace de descarga
                            b64 = base64.b64encode(excel_data).decode()
                            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="estadisticas_descriptivas_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Descargar Excel</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        
                        st.success(f"Estadísticas descriptivas exportadas correctamente a {export_format}. Haz clic en el enlace para descargar.")
                    except Exception as e:
                        st.error(f"Error al exportar estadísticas descriptivas: {str(e)}")
            else:
                st.warning("Selecciona al menos una columna para generar estadísticas descriptivas.")
        
        elif stats_type == "Matriz de Correlación":
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if len(numeric_cols) > 1:
                # Seleccionar columnas para la matriz
                selected_cols = st.multiselect(
                    "Selecciona columnas para la matriz de correlación",
                    numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))],
                    key="corr_matrix_export_cols"
                )
                
                if selected_cols and len(selected_cols) > 1:
                    # Seleccionar método de correlación
                    corr_method = st.radio(
                        "Método de correlación",
                        ["pearson", "spearman", "kendall"],
                        horizontal=True,
                        key="corr_export_method"
                    )
                    
                    # Generar matriz de correlación
                    corr_matrix = data[selected_cols].corr(method=corr_method)
                    
                    # Mostrar vista previa
                    st.write("**Vista previa:**")
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1), 
                                use_container_width=True)
                    
                    # Opciones de exportación
                    export_format = st.radio(
                        "Formato de exportación",
                        ["CSV", "Excel", "Imagen"],
                        horizontal=True,
                        key="corr_matrix_export_format"
                    )
                    
                    # Botón para exportar
                    if st.button("Exportar Matriz de Correlación", key="export_corr_matrix_button"):
                        try:
                            if export_format == "CSV":
                                # Exportar a CSV
                                csv_data = de.export_statistics_to_csv(corr_matrix)
                                
                                # Crear enlace de descarga
                                b64 = base64.b64encode(csv_data.encode('utf-8')).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="matriz_correlacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">Descargar CSV</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            elif export_format == "Excel":
                                # Exportar a Excel
                                excel_data = de.export_statistics_to_excel(corr_matrix, sheet_name="Matriz de Correlación")
                                
                                # Crear enlace de descarga
                                b64 = base64.b64encode(excel_data).decode()
                                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="matriz_correlacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx">Descargar Excel</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            elif export_format == "Imagen":
                                # Generar heatmap
                                fig = dv.create_heatmap(corr_matrix, 
                                                      title=f'Matriz de Correlación ({corr_method})')
                                
                                # Exportar a PNG
                                img_bytes = de.export_plotly_fig_to_image(fig, format="png")
                                
                                # Crear enlace de descarga
                                b64 = base64.b64encode(img_bytes).decode()
                                href = f'<a href="data:image/png;base64,{b64}" download="matriz_correlacion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png">Descargar Imagen</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            st.success(f"Matriz de correlación exportada correctamente a {export_format}. Haz clic en el enlace para descargar.")
                        except Exception as e:
                            st.error(f"Error al exportar matriz de correlación: {str(e)}")
                else:
                    st.warning("Selecciona al menos dos columnas numéricas para generar la matriz de correlación.")
            else:
                st.info("No hay suficientes columnas numéricas para generar una matriz de correlación.")
    
    # Pestaña de exportación de visualizaciones
    with tabs[2]:
        st.markdown("<div class='subsection-header'>🎨 Exportar Visualizaciones</div>", unsafe_allow_html=True)
        
        # Seleccionar tipo de visualización
        viz_type = st.selectbox(
            "Tipo de visualización",
            ["Histograma", "Boxplot", "Scatter", "Bar", "Pie", "Heatmap", "Violin", "Line"],
            key="export_viz_type"
        )
        
        # Configuración según tipo de visualización
        if viz_type == "Histograma":
            # Seleccionar columna
            col_to_viz = st.selectbox(
                "Columna para histograma",
                [col for col in data.columns 
                 if pd.api.types.is_numeric_dtype(data[col].dtype)],
                key="export_hist_col"
            )
            
            if col_to_viz:
                # Opciones para histograma
                n_bins = st.slider("Número de bins", 5, 100, 20, key="export_hist_bins")
                show_kde = st.checkbox("Mostrar curva KDE", value=True, key="export_hist_kde")
                
                # Generar histograma
                fig = dv.create_histogram(data, col_to_viz, n_bins=n_bins, kde=show_kde)
                
                # Mostrar visualización
                st.plotly_chart(fig, use_container_width=True)
                
                # Opciones de exportación
                export_format = st.radio(
                    "Formato de exportación",
                    ["PNG", "PDF", "SVG", "HTML"],
                    horizontal=True,
                    key="hist_export_format"
                )
                
                # Botón para exportar
                if st.button("Exportar Histograma", key="export_hist_button"):
                    try:
                        # Exportar visualización
                        if export_format == "HTML":
                            html_bytes = de.export_plotly_fig_to_html(fig)
                            b64 = base64.b64encode(html_bytes.encode()).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="histograma_{col_to_viz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                        else:
                            img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="histograma_{col_to_viz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"Histograma exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                    except Exception as e:
                        st.error(f"Error al exportar histograma: {str(e)}")
        
        elif viz_type == "Boxplot":
            # Seleccionar columna
            col_to_viz = st.selectbox(
                "Columna para boxplot",
                [col for col in data.columns 
                 if pd.api.types.is_numeric_dtype(data[col].dtype)],
                key="export_box_col"
            )
            
            if col_to_viz:
                # Seleccionar columna para agrupar (opcional)
                group_col = st.selectbox(
                    "Columna para agrupar (opcional)",
                    [None] + [col for col in data.columns 
                             if pd.api.types.is_categorical_dtype(data[col].dtype) or 
                                pd.api.types.is_object_dtype(data[col].dtype) or 
                                pd.api.types.is_string_dtype(data[col].dtype) or 
                                pd.api.types.is_bool_dtype(data[col].dtype)],
                    key="export_box_group_col"
                )
                
                # Generar boxplot
                if group_col:
                    fig = dv.create_boxplot_by_category(data, group_col, col_to_viz)
                else:
                    fig = dv.create_boxplot(data, col_to_viz)
                
                # Mostrar visualización
                st.plotly_chart(fig, use_container_width=True)
                
                # Opciones de exportación
                export_format = st.radio(
                    "Formato de exportación",
                    ["PNG", "PDF", "SVG", "HTML"],
                    horizontal=True,
                    key="box_export_format"
                )
                
                # Botón para exportar
                if st.button("Exportar Boxplot", key="export_box_button"):
                    try:
                        # Exportar visualización
                        if export_format == "HTML":
                            html_bytes = de.export_plotly_fig_to_html(fig)
                            b64 = base64.b64encode(html_bytes.encode()).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="boxplot_{col_to_viz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                        else:
                            img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="boxplot_{col_to_viz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"Boxplot exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                    except Exception as e:
                        st.error(f"Error al exportar boxplot: {str(e)}")
        
        elif viz_type == "Scatter":
            # Seleccionar columnas para ejes
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox(
                    "Columna para eje X",
                    [col for col in data.columns 
                     if pd.api.types.is_numeric_dtype(data[col].dtype)],
                    key="export_scatter_x_col"
                )
            
            with col2:
                y_col = st.selectbox(
                    "Columna para eje Y",
                    [col for col in data.columns 
                     if pd.api.types.is_numeric_dtype(data[col].dtype) and col != x_col],
                    key="export_scatter_y_col"
                )
            
            if x_col and y_col:
                # Seleccionar columna para color (opcional)
                color_col = st.selectbox(
                    "Columna para color (opcional)",
                    [None] + [col for col in data.columns if col not in [x_col, y_col]],
                    key="export_scatter_color_col"
                )
                
                # Generar scatter
                fig = dv.create_scatter(data, x_col, y_col, color_col=color_col)
                
                # Mostrar visualización
                st.plotly_chart(fig, use_container_width=True)
                
                # Opciones de exportación
                export_format = st.radio(
                    "Formato de exportación",
                    ["PNG", "PDF", "SVG", "HTML"],
                    horizontal=True,
                    key="scatter_export_format"
                )
                
                # Botón para exportar
                if st.button("Exportar Scatter", key="export_scatter_button"):
                    try:
                        # Exportar visualización
                        if export_format == "HTML":
                            html_bytes = de.export_plotly_fig_to_html(fig)
                            b64 = base64.b64encode(html_bytes.encode()).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="scatter_{x_col}_vs_{y_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                        else:
                            img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="scatter_{x_col}_vs_{y_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"Scatter exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                    except Exception as e:
                        st.error(f"Error al exportar scatter: {str(e)}")
        
        elif viz_type == "Bar":
            # Seleccionar columna para categorías
            cat_col = st.selectbox(
                "Columna para categorías",
                [col for col in data.columns 
                 if pd.api.types.is_categorical_dtype(data[col].dtype) or 
                    pd.api.types.is_object_dtype(data[col].dtype) or 
                    pd.api.types.is_string_dtype(data[col].dtype) or 
                    pd.api.types.is_bool_dtype(data[col].dtype)],
                key="export_bar_cat_col"
            )
            
            if cat_col:
                # Seleccionar columna para valores (opcional)
                val_col = st.selectbox(
                    "Columna para valores (opcional, por defecto es conteo)",
                    [None] + [col for col in data.columns 
                             if pd.api.types.is_numeric_dtype(data[col].dtype)],
                    key="export_bar_val_col"
                )
                
                # Limitar categorías
                max_categories = st.slider(
                    "Máximo de categorías a mostrar", 
                    5, 30, 10, 
                    key="export_bar_max_categories"
                )
                
                # Preparar datos para gráfico de barras
                
                if val_col:
                    # Agrupar por categoría y calcular estadística
                    agg_func = st.selectbox(
                        "Función de agregación",
                        ["Media", "Suma", "Conteo", "Min", "Max"],
                        key="export_bar_agg_func"
                    )
                    
                    # Aplicar agregación
                    if agg_func == "Media":
                        bar_data = data.groupby(cat_col)[val_col].mean().reset_index()
                    elif agg_func == "Suma":
                        bar_data = data.groupby(cat_col)[val_col].sum().reset_index()
                    elif agg_func == "Conteo":
                        bar_data = data.groupby(cat_col)[val_col].count().reset_index()
                    elif agg_func == "Min":
                        bar_data = data.groupby(cat_col)[val_col].min().reset_index()
                    elif agg_func == "Max":
                        bar_data = data.groupby(cat_col)[val_col].max().reset_index()
                else:
                    # Contar ocurrencias de cada categoría
                    bar_data = data[cat_col].value_counts().reset_index()
                    bar_data.columns = [cat_col, 'count']
                    val_col = 'count'
                
                # Limitar a las top categorías
                if len(bar_data) > max_categories:
                    bar_data = bar_data.nlargest(max_categories, val_col if val_col else 'count')
                
                # Generar gráfico de barras
                fig = dv.create_bar(bar_data, cat_col, val_col if val_col else 'count', 
                                   title=f'Distribución de {cat_col}')
                
                # Mostrar visualización
                st.plotly_chart(fig, use_container_width=True)
                
                # Opciones de exportación
                export_format = st.radio(
                    "Formato de exportación",
                    ["PNG", "PDF", "SVG", "HTML"],
                    horizontal=True,
                    key="bar_export_format"
                )
                
                # Botón para exportar
                if st.button("Exportar Gráfico de Barras", key="export_bar_button"):
                    try:
                        # Exportar visualización
                        if export_format == "HTML":
                            html_bytes = de.export_plotly_fig_to_html(fig)
                            b64 = base64.b64encode(html_bytes.encode()).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="barras_{cat_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                        else:
                            img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="barras_{cat_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"Gráfico de barras exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                    except Exception as e:
                        st.error(f"Error al exportar gráfico de barras: {str(e)}")
        
        elif viz_type == "Pie":
            # Seleccionar columna para categorías
            cat_col = st.selectbox(
                "Columna para categorías",
                [col for col in data.columns 
                 if pd.api.types.is_categorical_dtype(data[col].dtype) or 
                    pd.api.types.is_object_dtype(data[col].dtype) or 
                    pd.api.types.is_string_dtype(data[col].dtype) or 
                    pd.api.types.is_bool_dtype(data[col].dtype)],
                key="export_pie_cat_col"
            )
            
            if cat_col:
                # Seleccionar columna para valores (opcional)
                val_col = st.selectbox(
                    "Columna para valores (opcional, por defecto es conteo)",
                    [None] + [col for col in data.columns 
                             if pd.api.types.is_numeric_dtype(data[col].dtype)],
                    key="export_pie_val_col"
                )
                
                # Limitar categorías
                max_categories = st.slider(
                    "Máximo de categorías a mostrar", 
                    3, 15, 8, 
                    key="export_pie_max_categories"
                )
                
                # Preparar datos para gráfico de pie
                
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
                    top_categories = pie_data.nlargest(max_categories - 1, val_col if val_col else 'count')
                    other_count = pie_data.iloc[max_categories - 1:][val_col if val_col else 'count'].sum()
                    other_row = pd.DataFrame({cat_col: ['Otros'], val_col if val_col else 'count': [other_count]})
                    pie_data = pd.concat([top_categories, other_row], ignore_index=True)
                
                # Generar gráfico de pie
                fig = dv.create_pie(pie_data, cat_col, val_col if val_col else 'count', 
                                   title=f'Distribución de {cat_col}')
                
                # Mostrar visualización
                st.plotly_chart(fig, use_container_width=True)
                
                # Opciones de exportación
                export_format = st.radio(
                    "Formato de exportación",
                    ["PNG", "PDF", "SVG", "HTML"],
                    horizontal=True,
                    key="pie_export_format"
                )
                
                # Botón para exportar
                if st.button("Exportar Gráfico de Pie", key="export_pie_button"):
                    try:
                        # Exportar visualización
                        if export_format == "HTML":
                            html_bytes = de.export_plotly_fig_to_html(fig)
                            b64 = base64.b64encode(html_bytes.encode()).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="pie_{cat_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                        else:
                            img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="pie_{cat_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"Gráfico de pie exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                    except Exception as e:
                        st.error(f"Error al exportar gráfico de pie: {str(e)}")
        
        elif viz_type == "Heatmap":
            # Filtrar columnas numéricas
            numeric_cols = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col].dtype)]
            
            if len(numeric_cols) > 1:
                # Seleccionar columnas para la matriz
                selected_cols = st.multiselect(
                    "Selecciona columnas para el heatmap",
                    numeric_cols,
                    default=numeric_cols[:min(8, len(numeric_cols))],
                    key="export_heatmap_cols"
                )
                
                if selected_cols and len(selected_cols) > 1:
                    # Seleccionar tipo de heatmap
                    heatmap_type = st.radio(
                        "Tipo de heatmap",
                        ["Correlación", "Valores"],
                        horizontal=True,
                        key="export_heatmap_type"
                    )
                    
                    # Generar datos para heatmap
                    if heatmap_type == "Correlación":
                        # Seleccionar método de correlación
                        corr_method = st.selectbox(
                            "Método de correlación",
                            ["pearson", "spearman", "kendall"],
                            key="export_heatmap_corr_method"
                        )
                        
                        # Generar matriz de correlación
                        heatmap_data = data[selected_cols].corr(method=corr_method)
                        title = f'Matriz de Correlación ({corr_method})'
                    else:
                        # Usar valores directos
                        heatmap_data = data[selected_cols]
                        title = 'Heatmap de Valores'
                    
                    # Generar heatmap
                    fig = dv.create_heatmap(heatmap_data, title=title)
                    
                    # Mostrar visualización
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Opciones de exportación
                    export_format = st.radio(
                        "Formato de exportación",
                        ["PNG", "PDF", "SVG", "HTML"],
                        horizontal=True,
                        key="heatmap_export_format"
                    )
                    
                    # Botón para exportar
                    if st.button("Exportar Heatmap", key="export_heatmap_button"):
                        try:
                            # Exportar visualización
                            if export_format == "HTML":
                                html_bytes = de.export_plotly_fig_to_html(fig)
                                b64 = base64.b64encode(html_bytes.encode()).decode()
                                href = f'<a href="data:text/html;base64,{b64}" download="heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                            else:
                                img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                                b64 = base64.b64encode(img_bytes).decode()
                                href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                            
                            st.markdown(href, unsafe_allow_html=True)
                            st.success(f"Heatmap exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                        except Exception as e:
                            st.error(f"Error al exportar heatmap: {str(e)}")
                else:
                    st.warning("Selecciona al menos dos columnas numéricas para generar el heatmap.")
            else:
                st.info("No hay suficientes columnas numéricas para generar un heatmap.")
        
        elif viz_type == "Violin":
            # Seleccionar columna
            col_to_viz = st.selectbox(
                "Columna para violin plot",
                [col for col in data.columns 
                 if pd.api.types.is_numeric_dtype(data[col].dtype)],
                key="export_violin_col"
            )
            
            if col_to_viz:
                # Seleccionar columna para agrupar (opcional)
                group_col = st.selectbox(
                    "Columna para agrupar (opcional)",
                    [None] + [col for col in data.columns 
                             if pd.api.types.is_categorical_dtype(data[col].dtype) or 
                                pd.api.types.is_object_dtype(data[col].dtype) or 
                                pd.api.types.is_string_dtype(data[col].dtype) or 
                                pd.api.types.is_bool_dtype(data[col].dtype)],
                    key="export_violin_group_col"
                )
                
                # Generar violin plot
                if group_col:
                    fig = dv.create_violin_by_category(data, group_col, col_to_viz)
                else:
                    fig = dv.create_violin(data, col_to_viz)
                
                # Mostrar visualización
                st.plotly_chart(fig, use_container_width=True)
                
                # Opciones de exportación
                export_format = st.radio(
                    "Formato de exportación",
                    ["PNG", "PDF", "SVG", "HTML"],
                    horizontal=True,
                    key="violin_export_format"
                )
                
                # Botón para exportar
                if st.button("Exportar Violin Plot", key="export_violin_button"):
                    try:
                        # Exportar visualización
                        if export_format == "HTML":
                            html_bytes = de.export_plotly_fig_to_html(fig)
                            b64 = base64.b64encode(html_bytes.encode()).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="violin_{col_to_viz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                        else:
                            img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                            b64 = base64.b64encode(img_bytes).decode()
                            href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="violin_{col_to_viz}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                        
                        st.markdown(href, unsafe_allow_html=True)
                        st.success(f"Violin plot exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                    except Exception as e:
                        st.error(f"Error al exportar violin plot: {str(e)}")
        
        elif viz_type == "Line":
            # Filtrar columnas temporales
            temporal_cols = [col for col in data.columns 
                           if pd.api.types.is_datetime64_any_dtype(data[col].dtype)]
            
            if temporal_cols:
                # Seleccionar columna temporal
                time_col = st.selectbox(
                    "Columna temporal",
                    temporal_cols,
                    key="export_line_time_col"
                )
                
                # Filtrar columnas numéricas
                numeric_cols = [col for col in data.columns 
                              if pd.api.types.is_numeric_dtype(data[col].dtype)]
                
                if numeric_cols:
                    # Seleccionar columna numérica
                    value_col = st.selectbox(
                        "Columna de valores",
                        numeric_cols,
                        key="export_line_value_col"
                    )
                    
                    # Generar gráfico de línea
                    fig = dv.create_timeseries(data, time_col, value_col)
                    
                    # Mostrar visualización
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Opciones de exportación
                    export_format = st.radio(
                        "Formato de exportación",
                        ["PNG", "PDF", "SVG", "HTML"],
                        horizontal=True,
                        key="line_export_format"
                    )
                    
                    # Botón para exportar
                    if st.button("Exportar Gráfico de Línea", key="export_line_button"):
                        try:
                            # Exportar visualización
                            if export_format == "HTML":
                                html_bytes = de.export_plotly_fig_to_html(fig)
                                b64 = base64.b64encode(html_bytes.encode()).decode()
                                href = f'<a href="data:text/html;base64,{b64}" download="linea_{time_col}_{value_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                            else:
                                img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                                b64 = base64.b64encode(img_bytes).decode()
                                href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="linea_{time_col}_{value_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                            
                            st.markdown(href, unsafe_allow_html=True)
                            st.success(f"Gráfico de línea exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                        except Exception as e:
                            st.error(f"Error al exportar gráfico de línea: {str(e)}")
                else:
                    st.info("No se encontraron columnas numéricas para el gráfico de línea.")
            else:
                # Seleccionar columnas para ejes
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox(
                        "Columna para eje X",
                        data.columns,
                        key="export_line_x_col"
                    )
                
                with col2:
                    y_col = st.selectbox(
                        "Columna para eje Y",
                        [col for col in data.columns 
                         if pd.api.types.is_numeric_dtype(data[col].dtype) and col != x_col],
                        key="export_line_y_col"
                    )
                
                if x_col and y_col:
                    # Generar gráfico de línea
                    fig = dv.create_line(data, x_col, y_col)
                    
                    # Mostrar visualización
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Opciones de exportación
                    export_format = st.radio(
                        "Formato de exportación",
                        ["PNG", "PDF", "SVG", "HTML"],
                        horizontal=True,
                        key="line_export_format_2"
                    )
                    
                    # Botón para exportar
                    if st.button("Exportar Gráfico de Línea", key="export_line_button_2"):
                        try:
                            # Exportar visualización
                            if export_format == "HTML":
                                html_bytes = de.export_plotly_fig_to_html(fig)
                                b64 = base64.b64encode(html_bytes.encode()).decode()
                                href = f'<a href="data:text/html;base64,{b64}" download="linea_{x_col}_{y_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Descargar HTML</a>'
                            else:
                                img_bytes = de.export_plotly_fig_to_image(fig, format=export_format.lower())
                                b64 = base64.b64encode(img_bytes).decode()
                                href = f'<a href="data:image/{export_format.lower()};base64,{b64}" download="linea_{x_col}_{y_col}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{export_format.lower()}">Descargar {export_format}</a>'
                            
                            st.markdown(href, unsafe_allow_html=True)
                            st.success(f"Gráfico de línea exportado correctamente a {export_format}. Haz clic en el enlace para descargar.")
                        except Exception as e:
                            st.error(f"Error al exportar gráfico de línea: {str(e)}")
    
    # Pestaña de generación de reporte
    with tabs[3]:
        st.markdown("<div class='subsection-header'>📑 Generar Reporte</div>", unsafe_allow_html=True)
        
        # Opciones para el reporte
        st.write("**Configura el reporte:**")
        
        # Secciones a incluir
        st.write("**Secciones a incluir:**")
        include_info = st.checkbox("Información General", value=True, key="report_include_info")
        include_summary = st.checkbox("Resumen de Tipos de Datos", value=True, key="report_include_summary")
        include_stats = st.checkbox("Estadísticas Descriptivas", value=True, key="report_include_stats")
        include_viz = st.checkbox("Visualizaciones", value=True, key="report_include_viz")
        
        # Opciones para visualizaciones
        if include_viz:
            st.write("**Visualizaciones a incluir:**")
            viz_options = st.multiselect(
                "Selecciona visualizaciones",
                ["Distribución de Tipos de Datos", "Matriz de Correlación", "Histogramas de Variables Numéricas", 
                 "Gráficos de Barras para Variables Categóricas", "Boxplots de Variables Numéricas"],
                default=["Distribución de Tipos de Datos", "Matriz de Correlación"],
                key="report_viz_options"
            )
            
            # Limitar número de visualizaciones
            max_viz_per_type = st.slider(
                "Máximo de visualizaciones por tipo", 
                1, 10, 3, 
                key="report_max_viz"
            )
        
        # Opciones de formato
        st.write("**Opciones de formato:**")
        report_title = st.text_input("Título del reporte", "Reporte de Análisis Exploratorio de Datos", key="report_title")
        include_toc = st.checkbox("Incluir tabla de contenidos", value=True, key="report_include_toc")
        
        # Botón para generar reporte
        if st.button("Generar Reporte PDF", key="generate_report_button"):
            try:
                with st.spinner("Generando reporte... Esto puede tardar unos segundos."):
                    # Configurar opciones del reporte
                    report_options = {
                        "title": report_title,
                        "include_toc": include_toc,
                        "sections": {
                            "info": include_info,
                            "summary": include_summary,
                            "stats": include_stats,
                            "viz": include_viz
                        },
                        "viz_options": viz_options if include_viz else [],
                        "max_viz_per_type": max_viz_per_type if include_viz else 0
                    }
                    
                    # Generar reporte
                    pdf_bytes = de.generate_pdf_report(data, report_options)
                    
                    # Crear enlace de descarga
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="reporte_eda_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">Descargar Reporte PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.success("Reporte generado correctamente. Haz clic en el enlace para descargar.")
            except Exception as e:
                st.error(f"Error al generar reporte: {str(e)}")
                st.info("Nota: La generación de reportes PDF requiere que las bibliotecas ReportLab y WeasyPrint estén instaladas correctamente.")