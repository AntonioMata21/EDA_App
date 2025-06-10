import streamlit as st
import pandas as pd # Asegúrate de importar pandas
import plotly.express as px
import uuid
# Las importaciones de pandas.api.types siguen siendo útiles si las usas en otro lado,
# pero no son críticas para los cambios directos en esta función.

def show_summary_page(df: pd.DataFrame, summary: dict, filename: str):
    """
    Muestra la página de resumen general del dataset utilizando Pandas.
    
    Args:
        df: DataFrame de Pandas
        summary: Diccionario con el resumen del DataFrame (generado desde Pandas)
        filename: Nombre del archivo cargado
    """
    # Asegurarse de que df sea un DataFrame de Pandas
    if not isinstance(df, pd.DataFrame):
        st.error("Error: El DataFrame proporcionado no es un DataFrame de Pandas. Por favor, asegure la conversión.")
        # Opcionalmente, podrías intentar convertirlo aquí si sabes que podría ser Polars:
        # if hasattr(df, "to_pandas"): # Comprueba si tiene el método to_pandas (indicativo de Polars)
        #     try:
        #         df = df.to_pandas()
        #         st.info("DataFrame convertido de Polars a Pandas para la visualización de resumen.")
        #     except Exception as e:
        #         st.error(f"Error al convertir DataFrame a Pandas: {e}")
        #         return
        # else:
        #     st.error("Error: El DataFrame proporcionado no es un DataFrame de Pandas.")
        #     return # Salir si no es Pandas y no se pudo convertir

    if not isinstance(summary, dict):
        st.error("Error: El objeto de resumen no es válido. Por favor, cargue un archivo de datos.")
        return

    st.markdown("<div class='section-header'>📊 Resumen General</div>", unsafe_allow_html=True)
    st.markdown(f"**Archivo:** {filename}")
    
    # Información general
    st.markdown("<div class='subsection-header'>📋 Información General</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", summary.get("n_rows", len(df))) # Usar len(df) como fallback
    with col2:
        st.metric("Columnas", summary.get("n_cols", len(df.columns))) # Usar len(df.columns) como fallback
    with col3:
        # memory_usage en Pandas: df.memory_usage(deep=True).sum() / (1024 * 1024) para MB
        mem_usage_mb = summary.get("memory_usage") 
        if mem_usage_mb is None: # Calcular si no está en summary
            mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memoria Estimada", f"{mem_usage_mb:.2f} MB")
    
    # Vista previa de los datos
    st.markdown("<div class='subsection-header'>👁️ Vista Previa</div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True) # df.head() es igual en Pandas
    
    # Tipos de datos
    st.markdown("<div class='subsection-header'>🔢 Tipos de Datos</div>", unsafe_allow_html=True)
    
    # df.dtypes en Pandas devuelve una Serie (índice=nombre_columna, valor=tipo_dato)
    pandas_dtypes_series = summary.get("dtypes") 
    if pandas_dtypes_series is None: # Calcular si no está en summary
        pandas_dtypes_series = df.dtypes
    
    # Debugging (opcional)
    # st.write(f"Debug: pandas_dtypes_series type: {type(pandas_dtypes_series)}, value: {pandas_dtypes_series}")

    col1, col2 = st.columns([3, 2])
    
    with col1:
        if isinstance(pandas_dtypes_series, pd.Series):
            types_data = {
                "Columna": pandas_dtypes_series.index.tolist(),
                "Tipo": [str(dtype) for dtype in pandas_dtypes_series.values.tolist()] # Convertir tipos a string
            }
            types_df = pd.DataFrame(types_data)
            st.dataframe(types_df, use_container_width=True)
        else:
            st.warning("No se pudo generar la tabla de tipos de datos (summary['dtypes'] no es una Serie de Pandas).")
            
    with col2:
        if isinstance(pandas_dtypes_series, pd.Series):
            type_counts = {}
            for dtype_str in [str(dtype) for dtype in pandas_dtypes_series.values]:
                # Para Pandas, los dtypes son como 'int64', 'float64', 'object', 'bool', 'datetime64[ns]'
                # Puedes agruparlos si quieres (ej. 'int64', 'int32' -> 'integer')
                # Por ahora, usaremos el tipo exacto.
                base_type = dtype_str 
                type_counts[base_type] = type_counts.get(base_type, 0) + 1
            
            if type_counts:
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Distribución de Tipos de Datos",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True, key="dtype_distribution_pie_chart")
            else:
                st.info("No hay datos de tipos para mostrar en el gráfico.")
        else:
            st.warning("No se pudo generar el gráfico de tipos de datos.")

    # Valores faltantes
    st.markdown("<div class='subsection-header'>🔍 Valores Faltantes</div>", unsafe_allow_html=True)
    # En Pandas, df.isnull().sum() da una Serie con conteos de nulos por columna
    missing_counts_series = df.isnull().sum()
    cols_with_nulls_df = missing_counts_series[missing_counts_series > 0]

    if not cols_with_nulls_df.empty:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            missing_df = pd.DataFrame({
                "Columna": cols_with_nulls_df.index,
                "Valores Faltantes": cols_with_nulls_df.values
            })
            if len(df) > 0:
                 missing_df["Porcentaje"] = (missing_df["Valores Faltantes"] / len(df)) * 100
            else:
                 missing_df["Porcentaje"] = 0.0
            missing_df = missing_df.sort_values("Valores Faltantes", ascending=False)
            missing_df["Porcentaje"] = missing_df["Porcentaje"].round(2)
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            fig = px.bar(
                missing_df.head(10),
                x="Columna",
                y="Porcentaje",
                title="Top 10 Columnas con Valores Faltantes (%)",
                color="Porcentaje",
                color_continuous_scale="Reds"
            )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True, key="missing_values_bar_chart")
    else:
        st.info("No se encontraron valores faltantes en el dataset.")
    
    # Valores duplicados
    st.markdown("<div class='subsection-header'>🔄 Filas Duplicadas</div>", unsafe_allow_html=True)
    # n_duplicates y n_rows vienen de 'summary'
    n_duplicates = summary.get("n_duplicates")
    n_rows = summary.get("n_rows", len(df)) # fallback a len(df)

    if n_duplicates is None: # Calcular si no está en summary
        n_duplicates = df.duplicated().sum()
        
    if n_duplicates > 0 and n_rows > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Filas Duplicadas", n_duplicates)
        with col2:
            percentage = (n_duplicates / n_rows) * 100
            st.metric("Porcentaje", f"{percentage:.2f}%")
        
        # ---- CORRECCIÓN AQUÍ ----
        st.write(f"DEBUG - Button Key: show_duplicates_btn_pd_{st.session_state.get('filename')}")
        unique_suffix = st.session_state.get("filename") or str(uuid.uuid4())
        button_key = f"show_duplicates_btn_pd_{unique_suffix}"
        st.write(f"DEBUG - Button Key: {button_key}")  # Para monitorear qué está pasando

        if st.button("Mostrar Filas Duplicadas", key=button_key):
            duplicated_df = df[df.duplicated(keep=False)].sort_index()
            st.dataframe(duplicated_df.head(100), use_container_width=True)
            if len(duplicated_df) > 100:
                st.info(f"Mostrando 100 de {len(duplicated_df)} filas duplicadas.")
    elif n_duplicates == 0:
        st.info("No se encontraron filas duplicadas en el dataset.")
    else: # n_rows podría ser 0 o n_duplicates no ser un número
        st.info("No se pudo determinar la información de duplicados.")

    # Distribución de datos por tipo
    st.markdown("<div class='subsection-header'>📊 Distribución de Datos</div>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Numéricos", "Categóricos", "Fechas", "Booleanos"])
    
    # Pestaña de datos numéricos
    with tabs[0]:
        numeric_cols = summary.get("numeric_cols")
        if numeric_cols is None: # Inferir si no está en summary
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if numeric_cols:
            numeric_col = st.selectbox("Selecciona una columna numérica", numeric_cols, key="num_col_select_pd")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df, x=numeric_col,
                    title=f"Distribución de {numeric_col}",
                    nbins=30,
                    marginal="box"
                )
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True, key=f"numeric_histogram_pd_{numeric_col}")
            
            with col2:
                # Pandas: df[col].mean(), .median(), .std(), .min(), .max(), .quantile()
                stats_data = {
                    "Estadística": ["Media", "Mediana", "Desv. Estándar", "Mínimo", "Máximo", "Q1 (25%)", "Q3 (75%)"],
                    "Valor": [
                        df[numeric_col].mean(),
                        df[numeric_col].median(),
                        df[numeric_col].std(),
                        df[numeric_col].min(),
                        df[numeric_col].max(),
                        df[numeric_col].quantile(0.25),
                        df[numeric_col].quantile(0.75)
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df["Valor"] = stats_df["Valor"].round(3) # Redondear para mejor visualización
                st.dataframe(stats_df, use_container_width=True)
                
                fig_box = px.box(df, y=numeric_col, title=f"Boxplot de {numeric_col}")
                st.plotly_chart(fig_box, use_container_width=True, key=f"numeric_boxplot_pd_{numeric_col}")
        else:
            st.info("No se encontraron columnas numéricas en el dataset.")
    
    # Pestaña de datos categóricos
    with tabs[1]:
        categorical_cols = summary.get("categorical_cols")
        if categorical_cols is None: # Inferir si no está en summary
             categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        if categorical_cols:
            categorical_col = st.selectbox("Selecciona una columna categórica", categorical_cols, key="cat_col_select_pd")
            
            # Pandas: df[col].value_counts() devuelve una Serie
            value_counts_serie = df[categorical_col].value_counts(dropna=False)
            
            # Convertir Serie a DataFrame para facilitar el manejo
            value_counts_df = value_counts_serie.reset_index()
            value_counts_df.columns = [categorical_col, 'count'] # Renombrar columnas
            
            if len(df) > 0:
                value_counts_df["percentage"] = (value_counts_df["count"] / len(df)) * 100
            else:
                value_counts_df["percentage"] = 0.0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(value_counts_df, use_container_width=True)
            
            with col2:
                if len(value_counts_df) > 15:
                    plot_data = value_counts_df.head(15)
                    title = f"Top 15 valores de {categorical_col}"
                else:
                    plot_data = value_counts_df
                    title = f"Distribución de {categorical_col}"
                
                fig_bar_cat = px.bar(
                    plot_data,
                    x=categorical_col,
                    y="count",
                    title=title,
                    color="percentage",
                    color_continuous_scale="Blues"
                )
                fig_bar_cat.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar_cat, use_container_width=True, key=f"categorical_barchart_pd_{categorical_col}")
        else:
            st.info("No se encontraron columnas categóricas en el dataset.")
    
    # Pestaña de datos de fecha
    with tabs[2]:
        datetime_cols = summary.get("datetime_cols")
        if datetime_cols is None: # Inferir si no está en summary
            datetime_cols = df.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns.tolist()

        if datetime_cols:
            datetime_col = st.selectbox("Selecciona una columna de fecha", datetime_cols, key="date_col_select_pd")
            
            # La columna ya debería ser de tipo datetime si se cargó correctamente
            date_series = df[datetime_col]
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_date = date_series.min()
                max_date = date_series.max()
                range_days = "N/A"
                if pd.notna(min_date) and pd.notna(max_date):
                    range_days = (max_date - min_date).days
                
                stats_date_data = {
                    "Estadística": ["Fecha Mínima", "Fecha Máxima", "Rango (días)"],
                    "Valor": [min_date, max_date, range_days]
                }
                stats_date_df = pd.DataFrame(stats_date_data)
                st.dataframe(stats_date_df, use_container_width=True)
            
            with col2:
                fig_hist_date = px.histogram(
                    df, # Pasamos el df completo, plotly seleccionará la columna
                    x=datetime_col,
                    title=f"Distribución de {datetime_col}",
                    nbins=30
                )
                fig_hist_date.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist_date, use_container_width=True, key=f"datetime_histogram_pd_{datetime_col}")
        else:
            st.info("No se encontraron columnas de fecha/hora en el dataset.")
    
    # Pestaña de datos booleanos
    with tabs[3]:
        boolean_cols = summary.get("boolean_cols")
        if boolean_cols is None: # Inferir si no está en summary
            boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()

        if boolean_cols:
            boolean_col = st.selectbox("Selecciona una columna booleana", boolean_cols, key="bool_col_select_pd")
            
            # Pandas: df[col].value_counts(dropna=False)
            bool_counts = df[boolean_col].value_counts(dropna=False)
            true_count = bool_counts.get(True, 0)
            false_count = bool_counts.get(False, 0)
            # Los nulos se cuentan si dropna=False y existen en la Serie
            # Para ser explícitos:
            null_count = df[boolean_col].isnull().sum()
            
            total_valid_for_bool_pct = true_count + false_count + null_count # O len(df) si nulos son categoría
            
            col1, col2 = st.columns(2)
            
            with col1:
                stats_bool_data = {
                    "Valor": ["True", "False", "Nulos"],
                    "Conteo": [true_count, false_count, null_count],
                }
                if len(df) > 0: # Usar len(df) para el porcentaje total
                    stats_bool_data["Porcentaje"] = [
                        (true_count / len(df)) * 100 if len(df) > 0 else 0,
                        (false_count / len(df)) * 100 if len(df) > 0 else 0,
                        (null_count / len(df)) * 100 if len(df) > 0 else 0
                    ]
                else:
                    stats_bool_data["Porcentaje"] = [0.0, 0.0, 0.0]

                stats_bool_df = pd.DataFrame(stats_bool_data)
                stats_bool_df["Porcentaje"] = stats_bool_df["Porcentaje"].round(2)
                st.dataframe(stats_bool_df, use_container_width=True)
            
            with col2:
                # Asegurarse de que los valores para el pie chart no sean todos cero si df está vacío
                pie_values = [true_count, false_count, null_count]
                pie_names = ["True", "False", "Nulos"]

                if any(v > 0 for v in pie_values): # Solo mostrar gráfico si hay datos
                    fig_pie_bool = px.pie(
                        values=pie_values,
                        names=pie_names,
                        title=f"Distribución de {boolean_col}",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_pie_bool.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie_bool, use_container_width=True, key=f"boolean_pie_chart_pd_{boolean_col}")
                else:
                    st.info(f"No hay datos para graficar para la columna {boolean_col}.")
        else:
            st.info("No se encontraron columnas booleanas en el dataset.")