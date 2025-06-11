import streamlit as st
import pandas as pd
import numpy as np

def show_filters_page(df):
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        st.session_state.filtered_data = df
    """
    Muestra la página de filtros y transformaciones del dataset.
    
    Args:
        df: DataFrame de Pandas
    """
    st.markdown("<div class='section-header'>🔄 Filtros y Transformaciones</div>", unsafe_allow_html=True)
    
    # Inicializar DataFrame filtrado en session_state si no existe
    if "filtered_data" not in st.session_state:
        st.session_state.filtered_data = df.copy()
    
    # Crear pestañas para diferentes tipos de operaciones
    tabs = st.tabs(["Filtros", "Transformaciones", "Selección de Columnas", "Ordenamiento"])
    
    # Pestaña de filtros
    with tabs[0]:
        st.markdown("<div class='subsection-header'>🔍 Filtros</div>", unsafe_allow_html=True)
        
        # Crear contenedor para filtros
        filter_container = st.container()
        
        with filter_container:
            # Seleccionar columna para filtrar
            filter_col = st.selectbox(
                "Selecciona una columna para filtrar",
                df.columns
            )
            
            if filter_col:
                # Obtener tipo de columna
                col_type = df[filter_col].dtype
                
                # Filtros para columnas numéricas
                if pd.api.types.is_numeric_dtype(col_type):
                    # Obtener valores mínimo y máximo
                    min_val = float(df[filter_col].min())
                    max_val = float(df[filter_col].max())
                    
                    # Crear slider para rango
                    filter_range = st.slider(
                        f"Rango de valores para {filter_col}",
                        min_val, max_val,
                        (min_val, max_val)
                    )
                    
                    # Botón para aplicar filtro
                    if st.button("Aplicar Filtro Numérico"):
                        # Filtrar DataFrame
                        filtered_df = st.session_state.filtered_data[
                            (st.session_state.filtered_data[filter_col] >= filter_range[0]) & 
                            (st.session_state.filtered_data[filter_col] <= filter_range[1])
                        ]
                        
                        # Actualizar DataFrame filtrado
                        st.session_state.filtered_data = filtered_df
                        
                        # Mostrar mensaje
                        st.success(f"Filtro aplicado: {filter_col} entre {filter_range[0]} y {filter_range[1]}")
                
                # Filtros para columnas categóricas
                elif pd.api.types.is_categorical_dtype(col_type) or pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
                    # Obtener valores únicos
                    unique_values = df[filter_col].unique().tolist()
                    
                    # Crear multiselect para valores
                    selected_values = st.multiselect(
                        f"Selecciona valores para {filter_col}",
                        unique_values,
                        default=unique_values
                    )
                    
                    # Botón para aplicar filtro
                    if st.button("Aplicar Filtro Categórico"):
                        if selected_values:
                            # Filtrar DataFrame
                            filtered_df = st.session_state.filtered_data[
                                st.session_state.filtered_data[filter_col].isin(selected_values)
                            ]
                            
                            # Actualizar DataFrame filtrado
                            st.session_state.filtered_data = filtered_df
                            
                            # Mostrar mensaje
                            st.success(f"Filtro aplicado: {filter_col} en {', '.join(map(str, selected_values[:5]))}" + 
                                      ("..." if len(selected_values) > 5 else ""))
                        else:
                            st.warning("Debes seleccionar al menos un valor.")
                
                # Filtros para columnas booleanas
                elif pd.api.types.is_bool_dtype(col_type):
                    # Crear radio para valores
                    bool_value = st.radio(
                        f"Selecciona valor para {filter_col}",
                        ["Todos", "True", "False", "Nulos"]
                    )
                    
                    # Botón para aplicar filtro
                    if st.button("Aplicar Filtro Booleano"):
                        if bool_value == "True":
                            # Filtrar DataFrame
                            filtered_df = st.session_state.filtered_data[
                                st.session_state.filtered_data[filter_col] == True
                            ]
                        elif bool_value == "False":
                            # Filtrar DataFrame
                            filtered_df = st.session_state.filtered_data[
                                st.session_state.filtered_data[filter_col] == False
                            ]
                        elif bool_value == "Nulos":
                            # Filtrar DataFrame
                            filtered_df = st.session_state.filtered_data[
                                st.session_state.filtered_data[filter_col].isna()
                            ]
                        else:  # Todos
                            filtered_df = st.session_state.filtered_data
                        
                        # Actualizar DataFrame filtrado
                        st.session_state.filtered_data = filtered_df
                        
                        # Mostrar mensaje
                        st.success(f"Filtro aplicado: {filter_col} = {bool_value}")
                
                # Filtros para columnas de fecha/hora
                elif pd.api.types.is_datetime64_any_dtype(col_type):
                    # Obtener fechas mínima y máxima
                    min_date = df[filter_col].min()
                    max_date = df[filter_col].max()
                    
                    # Crear date_input para rango
                    date_range = st.date_input(
                        f"Rango de fechas para {filter_col}",
                        [min_date.date(), max_date.date()]
                    )
                    
                    # Botón para aplicar filtro
                    if st.button("Aplicar Filtro de Fecha"):
                        if len(date_range) == 2:
                            # Convertir a datetime
                            start_date = pd.Timestamp(date_range[0])
                            end_date = pd.Timestamp(date_range[1])
                            
                            # Filtrar DataFrame
                            filtered_df = st.session_state.filtered_data[
                                (st.session_state.filtered_data[filter_col] >= start_date) & 
                                (st.session_state.filtered_data[filter_col] <= end_date)
                            ]
                            
                            # Actualizar DataFrame filtrado
                            st.session_state.filtered_data = filtered_df
                            
                            # Mostrar mensaje
                            st.success(f"Filtro aplicado: {filter_col} entre {start_date.date()} y {end_date.date()}")
                        else:
                            st.warning("Debes seleccionar un rango de fechas.")
        
        # Botón para restablecer filtros
        if st.button("Restablecer Filtros"):
            st.session_state.filtered_data = df.copy()
            st.success("Filtros restablecidos.")
    
    # Pestaña de transformaciones
    with tabs[1]:
        st.markdown("<div class='subsection-header'>🔄 Transformaciones</div>", unsafe_allow_html=True)
        
        # Crear contenedor para transformaciones
        transform_container = st.container()
        
        with transform_container:
            # Seleccionar tipo de transformación
            transform_type = st.selectbox(
                "Selecciona tipo de transformación",
                ["Conversión de Tipos", "Renombrar Columnas", "Limpieza de Texto", "Normalización", "Discretización"]
            )
            
            # Transformación: Conversión de Tipos
            if transform_type == "Conversión de Tipos":
                # Seleccionar columna
                col_to_convert = st.selectbox(
                    "Selecciona columna para convertir",
                    df.columns
                )
                
                if col_to_convert:
                    # Obtener tipo actual
                    current_type = df[col_to_convert].dtype
                    
                    # Seleccionar nuevo tipo
                    new_type = st.selectbox(
                        "Selecciona nuevo tipo",
                        ["Int64", "Float64", "String", "Boolean", "Date", "Datetime"]
                    )
                    
                    # Botón para aplicar transformación
                    if st.button("Convertir Tipo"):
                        try:
                            # Clonar DataFrame filtrado
                            transformed_df = st.session_state.filtered_data.copy()
                            
                            # Aplicar conversión según el tipo seleccionado
                            if new_type == "Int64":
                                transformed_df[col_to_convert] = transformed_df[col_to_convert].astype('Int64')
                            elif new_type == "Float64":
                                transformed_df[col_to_convert] = transformed_df[col_to_convert].astype('float64')
                            elif new_type == "String":
                                transformed_df[col_to_convert] = transformed_df[col_to_convert].astype('string')
                            elif new_type == "Boolean":
                                transformed_df[col_to_convert] = transformed_df[col_to_convert].astype('boolean')
                            elif new_type == "Date":
                                transformed_df[col_to_convert] = pd.to_datetime(transformed_df[col_to_convert]).dt.date
                            elif new_type == "Datetime":
                                transformed_df[col_to_convert] = pd.to_datetime(transformed_df[col_to_convert])
                            
                            # Actualizar DataFrame filtrado
                            st.session_state.filtered_data = transformed_df
                            
                            # Mostrar mensaje
                            st.success(f"Columna {col_to_convert} convertida a {new_type}")
                        except Exception as e:
                            st.error(f"Error al convertir tipo: {str(e)}")
            
            # Transformación: Renombrar Columnas
            elif transform_type == "Renombrar Columnas":
                # Seleccionar columna
                col_to_rename = st.selectbox(
                    "Selecciona columna para renombrar",
                    df.columns
                )
                
                if col_to_rename:
                    # Ingresar nuevo nombre
                    new_name = st.text_input("Nuevo nombre", col_to_rename)
                    
                    # Botón para aplicar transformación
                    if st.button("Renombrar Columna"):
                        if new_name and new_name != col_to_rename:
                            try:
                                # Clonar DataFrame filtrado
                                transformed_df = st.session_state.filtered_data.copy()
                                
                                # Renombrar columna
                                transformed_df = transformed_df.rename(columns={col_to_rename: new_name})
                                
                                # Actualizar DataFrame filtrado
                                st.session_state.filtered_data = transformed_df
                                
                                # Mostrar mensaje
                                st.success(f"Columna {col_to_rename} renombrada a {new_name}")
                            except Exception as e:
                                st.error(f"Error al renombrar columna: {str(e)}")
                        else:
                            st.warning("Debes ingresar un nombre diferente.")
            
            # Transformación: Limpieza de Texto
            elif transform_type == "Limpieza de Texto":
                # Filtrar columnas de texto
                text_cols = [col for col in df.columns 
                           if pd.api.types.is_string_dtype(df[col].dtype) or pd.api.types.is_object_dtype(df[col].dtype) or pd.api.types.is_categorical_dtype(df[col].dtype)]
                
                if text_cols:
                    # Seleccionar columna
                    col_to_clean = st.selectbox(
                        "Selecciona columna de texto para limpiar",
                        text_cols
                    )
                    
                    if col_to_clean:
                        # Seleccionar operaciones de limpieza
                        clean_ops = st.multiselect(
                            "Selecciona operaciones de limpieza",
                            ["Eliminar espacios en blanco", "Convertir a minúsculas", "Convertir a mayúsculas", 
                             "Eliminar caracteres especiales", "Eliminar números"]
                        )
                        
                        # Botón para aplicar transformación
                        if st.button("Limpiar Texto"):
                            if clean_ops:
                                try:
                                    # Clonar DataFrame filtrado
                                    transformed_df = st.session_state.filtered_data.copy()
                                    
                                    # Aplicar operaciones de limpieza
                                    for op in clean_ops:
                                        if op == "Eliminar espacios en blanco":
                                            transformed_df[col_to_clean] = transformed_df[col_to_clean].str.strip()
                                        elif op == "Convertir a minúsculas":
                                            transformed_df[col_to_clean] = transformed_df[col_to_clean].str.lower()
                                        elif op == "Convertir a mayúsculas":
                                            transformed_df[col_to_clean] = transformed_df[col_to_clean].str.upper()
                                        elif op == "Eliminar caracteres especiales":
                                            transformed_df[col_to_clean] = transformed_df[col_to_clean].str.replace(r'[^\w\s]', "", regex=True)
                                        elif op == "Eliminar números":
                                            transformed_df[col_to_clean] = transformed_df[col_to_clean].str.replace(r'\d', "", regex=True)
                                    
                                    # Actualizar DataFrame filtrado
                                    st.session_state.filtered_data = transformed_df
                                    
                                    # Mostrar mensaje
                                    st.success(f"Columna {col_to_clean} limpiada con operaciones: {', '.join(clean_ops)}")
                                except Exception as e:
                                    st.error(f"Error al limpiar texto: {str(e)}")
                            else:
                                st.warning("Debes seleccionar al menos una operación de limpieza.")
                else:
                    st.info("No se encontraron columnas de texto en el dataset.")
            
            # Transformación: Normalización
            elif transform_type == "Normalización":
                # Filtrar columnas numéricas
                numeric_cols = [col for col in df.columns 
                              if pd.api.types.is_numeric_dtype(df[col].dtype)]
                
                if numeric_cols:
                    # Seleccionar columna
                    col_to_normalize = st.selectbox(
                        "Selecciona columna numérica para normalizar",
                        numeric_cols
                    )
                    
                    if col_to_normalize:
                        # Seleccionar método de normalización
                        norm_method = st.radio(
                            "Selecciona método de normalización",
                            ["Min-Max (0-1)", "Z-Score (Media 0, Desv. 1)", "Robust (Mediana, IQR)"]
                        )
                        
                        # Botón para aplicar transformación
                        if st.button("Normalizar Columna"):
                            try:
                                # Clonar DataFrame filtrado
                                transformed_df = st.session_state.filtered_data.copy()
                                
                                # Aplicar normalización según el método seleccionado
                                if norm_method == "Min-Max (0-1)":
                                    # Normalización Min-Max
                                    min_val = float(transformed_df[col_to_normalize].min())
                                    max_val = float(transformed_df[col_to_normalize].max())
                                    
                                    if max_val > min_val:  # Evitar división por cero
                                        transformed_df[f"{col_to_normalize}_norm"] = (transformed_df[col_to_normalize] - min_val) / (max_val - min_val)
                                    else:
                                        transformed_df[f"{col_to_normalize}_norm"] = transformed_df[col_to_normalize] * 0
                                
                                elif norm_method == "Z-Score (Media 0, Desv. 1)":
                                    # Normalización Z-Score
                                    mean_val = float(transformed_df[col_to_normalize].mean())
                                    std_val = float(transformed_df[col_to_normalize].std())
                                    
                                    if std_val > 0:  # Evitar división por cero
                                        transformed_df[f"{col_to_normalize}_norm"] = (transformed_df[col_to_normalize] - mean_val) / std_val
                                    else:
                                        transformed_df[f"{col_to_normalize}_norm"] = transformed_df[col_to_normalize] - mean_val
                                
                                elif norm_method == "Robust (Mediana, IQR)":
                                    # Normalización Robusta
                                    median_val = float(transformed_df[col_to_normalize].median())
                                    q1 = float(transformed_df[col_to_normalize].quantile(0.25))
                                    q3 = float(transformed_df[col_to_normalize].quantile(0.75))
                                    iqr = q3 - q1
                                    
                                    if iqr > 0:  # Evitar división por cero
                                        transformed_df[f"{col_to_normalize}_norm"] = (transformed_df[col_to_normalize] - median_val) / iqr
                                    else:
                                        transformed_df[f"{col_to_normalize}_norm"] = transformed_df[col_to_normalize] - median_val
                                
                                # Actualizar DataFrame filtrado
                                st.session_state.filtered_data = transformed_df
                                
                                # Mostrar mensaje
                                st.success(f"Columna {col_to_normalize} normalizada con método {norm_method}")
                            except Exception as e:
                                st.error(f"Error al normalizar columna: {str(e)}")
                else:
                    st.info("No se encontraron columnas numéricas en el dataset.")
            
            # Transformación: Discretización
            elif transform_type == "Discretización":
                # Filtrar columnas numéricas
                numeric_cols = [col for col in df.columns 
                              if pd.api.types.is_numeric_dtype(df[col].dtype)]
                
                if numeric_cols:
                    # Seleccionar columna
                    col_to_discretize = st.selectbox(
                        "Selecciona columna numérica para discretizar",
                        numeric_cols
                    )
                    
                    if col_to_discretize:
                        # Seleccionar método de discretización
                        disc_method = st.radio(
                            "Selecciona método de discretización",
                            ["Igual Ancho", "Igual Frecuencia", "Personalizado"]
                        )
                        
                        # Número de bins
                        n_bins = st.slider("Número de bins", 2, 20, 5)
                        
                        # Etiquetas personalizadas
                        use_custom_labels = st.checkbox("Usar etiquetas personalizadas")
                        custom_labels = None
                        
                        if use_custom_labels:
                            labels_input = st.text_input(
                                "Ingresa etiquetas separadas por coma",
                                value="Bajo, Medio, Alto, Muy Alto, Extremo"[:n_bins*2-1]
                            )
                            if labels_input:
                                custom_labels = [label.strip() for label in labels_input.split(",")]
                                if len(custom_labels) != n_bins:
                                    st.warning(f"El número de etiquetas ({len(custom_labels)}) no coincide con el número de bins ({n_bins}).")
                        
                        # Botón para aplicar transformación
                        if st.button("Discretizar Columna"):
                            try:
                                # Usar directamente pandas para discretización
                                transformed_df = st.session_state.filtered_data.copy()
                                
                                # Aplicar discretización según el método seleccionado
                                if disc_method == "Igual Ancho":
                                    # Discretización de igual ancho
                                    transformed_df[f"{col_to_discretize}_disc"] = pd.cut(
                                        transformed_df[col_to_discretize],
                                        bins=n_bins,
                                        labels=custom_labels if use_custom_labels else False
                                    )
                                
                                elif disc_method == "Igual Frecuencia":
                                    # Discretización de igual frecuencia
                                    transformed_df[f"{col_to_discretize}_disc"] = pd.qcut(
                                        transformed_df[col_to_discretize],
                                        q=n_bins,
                                        labels=custom_labels if use_custom_labels else False,
                                        duplicates='drop'
                                    )
                                
                                elif disc_method == "Personalizado":
                                    # Obtener valores mínimo y máximo
                                    min_val = float(transformed_df[col_to_discretize].min())
                                    max_val = float(transformed_df[col_to_discretize].max())
                                    
                                    # Crear bins personalizados
                                    custom_bins = st.slider(
                                        "Selecciona puntos de corte",
                                        min_val, max_val,
                                        (min_val, (min_val + max_val) / 2, max_val)
                                    )
                                    
                                    # Discretización personalizada
                                    transformed_df[f"{col_to_discretize}_disc"] = pd.cut(
                                        transformed_df[col_to_discretize],
                                        bins=custom_bins,
                                        labels=custom_labels if use_custom_labels else False
                                    )
                                
                                # Actualizar DataFrame filtrado
                                st.session_state.filtered_data = transformed_df
                                
                                # Mostrar mensaje
                                st.success(f"Columna {col_to_discretize} discretizada con método {disc_method}")
                            except Exception as e:
                                st.error(f"Error al discretizar columna: {str(e)}")
                else:
                    st.info("No se encontraron columnas numéricas en el dataset.")
    
    # Pestaña de selección de columnas
    with tabs[2]:
        st.markdown("<div class='subsection-header'>📋 Selección de Columnas</div>", unsafe_allow_html=True)
        
        # Seleccionar columnas a mantener
        selected_cols = st.multiselect(
            "Selecciona columnas a mantener",
            st.session_state.filtered_data.columns,
            default=st.session_state.filtered_data.columns
        )
        
        # Botón para aplicar selección
        if st.button("Aplicar Selección"):
            if selected_cols:
                # Filtrar columnas
                filtered_df = st.session_state.filtered_data[selected_cols]
                
                # Actualizar DataFrame filtrado
                st.session_state.filtered_data = filtered_df
                
                # Mostrar mensaje
                st.success(f"Seleccionadas {len(selected_cols)} columnas")
            else:
                st.warning("Debes seleccionar al menos una columna.")
    
    # Pestaña de ordenamiento
    with tabs[3]:
        st.markdown("<div class='subsection-header'>🔢 Ordenamiento</div>", unsafe_allow_html=True)
        
        # Seleccionar columna para ordenar
        sort_col = st.selectbox(
            "Selecciona columna para ordenar",
            st.session_state.filtered_data.columns
        )
        
        if sort_col:
            # Seleccionar dirección de ordenamiento
            sort_dir = st.radio(
                "Dirección",
                ["Ascendente", "Descendente"]
            )
            
            # Botón para aplicar ordenamiento
            if st.button("Aplicar Ordenamiento"):
                # Ordenar DataFrame
                filtered_df = st.session_state.filtered_data.sort_values(
                    by=sort_col,
                    ascending=(sort_dir != "Descendente")
                )
                
                # Actualizar DataFrame filtrado
                st.session_state.filtered_data = filtered_df
                
                # Mostrar mensaje
                st.success(f"Dataset ordenado por {sort_col} ({sort_dir})")
    
    # Mostrar vista previa del DataFrame filtrado
    st.markdown("<div class='subsection-header'>👁️ Vista Previa de Datos Filtrados</div>", unsafe_allow_html=True)
    
    # Información sobre el DataFrame filtrado
    st.write(f"**Dimensiones:** {st.session_state.filtered_data.shape[0]} filas × {st.session_state.filtered_data.shape[1]} columnas")
    
    # Mostrar vista previa
    st.dataframe(st.session_state.filtered_data.head(10), use_container_width=True)
    
    # Botón para guardar subconjunto filtrado
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Guardar Subconjunto"):
            # Guardar DataFrame filtrado en session_state
            st.session_state.filtered_data_saved = st.session_state.filtered_data.copy()
            st.success("Subconjunto guardado para exportación.")
    
    with col2:
        if st.button("Restablecer Todo"):
            # Restablecer DataFrame filtrado al original
            st.session_state.filtered_data = df.copy()
            st.success("Datos restablecidos al original.")
    
    # Mostrar información sobre el subconjunto guardado
    if "filtered_data_saved" in st.session_state:
        st.info(f"Subconjunto guardado: {st.session_state.filtered_data_saved.shape[0]} filas × {st.session_state.filtered_data_saved.shape[1]} columnas")