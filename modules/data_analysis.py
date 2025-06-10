import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def get_data_summary(df):
    """
    Genera un resumen general del DataFrame.
    
    Args:
        df: DataFrame de Pandas
    
    Returns:
        Diccionario con el resumen del DataFrame
    """
    if df is None or df.empty:
        return get_empty_summary()

    # Obtener información básica
    n_rows, n_cols = df.shape
    
    # Obtener tipos de datos
    dtypes = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    
    # Clasificar columnas por tipo
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    other_cols = [col for col in df.columns 
                 if col not in numeric_cols + categorical_cols + datetime_cols + boolean_cols]
    
    # Calcular valores nulos
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    # Calcular duplicados
    n_duplicates = df.shape[0] - df.drop_duplicates().shape[0]
    
    # Estimar uso de memoria
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # Crear resumen
    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtypes": dtypes,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "boolean_cols": boolean_cols,
        "other_cols": other_cols,
        "cols_with_nulls": cols_with_nulls,
        "n_duplicates": n_duplicates,
        "memory_usage": round(memory_usage, 2)  # En MB, redondeado a 2 decimales
    }
    
    return summary

@st.cache_data
def get_empty_summary():
    """
    Retorna un diccionario de resumen vacío o predeterminado.
    """
    return {
        "n_rows": 0,
        "n_cols": 0,
        "dtypes": {},
        "numeric_cols": [],
        "categorical_cols": [],
        "datetime_cols": [],
        "boolean_cols": [],
        "other_cols": [],
        "cols_with_nulls": [],
        "n_duplicates": 0,
        "memory_usage": 0.0
    }

@st.cache_data
def get_descriptive_stats(df, columns=None, include_percentiles=True):
    """
    Calcula estadísticas descriptivas para las columnas seleccionadas.
    
    Args:
        df: DataFrame de Pandas
        columns: Lista de columnas para calcular estadísticas (None = todas)
        include_percentiles: Si se incluyen percentiles en las estadísticas
    
    Returns:
        DataFrame con estadísticas descriptivas
    """
    if columns is None:
        columns = df.columns.tolist()
    
    # Filtrar solo las columnas existentes
    columns = [col for col in columns if col in df.columns]
    
    # Crear un DataFrame vacío para almacenar los resultados
    stats_data = {"statistic": []}
    for col in columns:
        stats_data[col] = []
    
    # Función para agregar una estadística al diccionario
    def add_stat(name, func, cols):
        stats_data["statistic"].append(name)
        for col in columns:
            if col in cols:
                try:
                    value = func(df, col)
                    stats_data[col].append(value)
                except Exception:
                    stats_data[col].append(None)
            else:
                stats_data[col].append(None)
    
    # Clasificar columnas por tipo
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in columns if col in numeric_cols]
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in columns if col in categorical_cols]
    
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    boolean_cols = [col for col in columns if col in boolean_cols]
    
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    datetime_cols = [col for col in columns if col in datetime_cols]
    
    # Estadísticas para columnas numéricas
    if numeric_cols:
        add_stat("count", lambda df, col: df[col].count(), numeric_cols)
        add_stat("mean", lambda df, col: float(df[col].mean()), numeric_cols)
        add_stat("std", lambda df, col: float(df[col].std()), numeric_cols)
        add_stat("min", lambda df, col: float(df[col].min()), numeric_cols)
        add_stat("max", lambda df, col: float(df[col].max()), numeric_cols)
        
        if include_percentiles:
            add_stat("25%", lambda df, col: float(df[col].quantile(0.25)), numeric_cols)
            add_stat("50%", lambda df, col: float(df[col].quantile(0.5)), numeric_cols)
            add_stat("75%", lambda df, col: float(df[col].quantile(0.75)), numeric_cols)
            add_stat("90%", lambda df, col: float(df[col].quantile(0.9)), numeric_cols)
            add_stat("95%", lambda df, col: float(df[col].quantile(0.95)), numeric_cols)
        
        add_stat("skewness", lambda df, col: float(df[col].skew()), numeric_cols)
        add_stat("kurtosis", lambda df, col: float(df[col].kurtosis()), numeric_cols)
    
    # Estadísticas para columnas categóricas
    if categorical_cols:
        add_stat("unique_values", lambda df, col: df[col].nunique(), categorical_cols)
        add_stat("mode", lambda df, col: df[col].mode()[0] if not df[col].mode().empty else None, categorical_cols)
        add_stat("most_frequent_%", 
                lambda df, col: (df[df[col] == df[col].mode()[0]].shape[0] / df.shape[0]) * 100 if not df[col].mode().empty else 0, 
                categorical_cols)
    
    # Estadísticas para columnas booleanas
    if boolean_cols:
        add_stat("true_count", lambda df, col: df[df[col] == True].shape[0], boolean_cols)
        add_stat("true_%", 
                lambda df, col: (df[df[col] == True].shape[0] / df.shape[0]) * 100, 
                boolean_cols)
        add_stat("false_count", lambda df, col: df[df[col] == False].shape[0], boolean_cols)
        add_stat("false_%", 
                lambda df, col: (df[df[col] == False].shape[0] / df.shape[0]) * 100, 
                boolean_cols)
    
    # Estadísticas para columnas de fecha/hora
    if datetime_cols:
        add_stat("min_date", lambda df, col: df[col].min(), datetime_cols)
        add_stat("max_date", lambda df, col: df[col].max(), datetime_cols)
        add_stat("range_days", 
                lambda df, col: (df[col].max() - df[col].min()).days 
                if hasattr((df[col].max() - df[col].min()), 'days') else None, 
                datetime_cols)
    
    # Valores nulos para todas las columnas
    add_stat("null_count", lambda df, col: df[col].isnull().sum(), columns)
    add_stat("null_%", 
            lambda df, col: (df[col].isnull().sum() / df.shape[0]) * 100, 
            columns)
    
    # Convertir a DataFrame de Pandas
    return pd.DataFrame(stats_data)

def analyze_column_distribution(df, column):
    """
    Analiza la distribución de una columna específica.
    
    Args:
        df: DataFrame de Pandas
        column: Nombre de la columna a analizar
    
    Returns:
        Diccionario con información sobre la distribución
    """
    if column not in df.columns:
        return {"error": f"La columna {column} no existe en el DataFrame"}
    
    col_type = df[column].dtype
    result = {"column": column, "type": str(col_type)}
    
    # Análisis para columnas numéricas
    if pd.api.types.is_numeric_dtype(col_type):
        # Estadísticas básicas
        result["count"] = df[column].count()
        result["null_count"] = df[column].isnull().sum()
        result["mean"] = float(df[column].mean())
        result["median"] = float(df[column].median())
        result["std"] = float(df[column].std())
        result["min"] = float(df[column].min())
        result["max"] = float(df[column].max())
        
        # Percentiles
        result["percentiles"] = {
            "25%": float(df[column].quantile(0.25)),
            "50%": float(df[column].quantile(0.5)),
            "75%": float(df[column].quantile(0.75)),
            "90%": float(df[column].quantile(0.9)),
            "95%": float(df[column].quantile(0.95)),
            "99%": float(df[column].quantile(0.99))
        }
        
        # Asimetría y curtosis
        result["skewness"] = float(df[column].skew())
        result["kurtosis"] = float(df[column].kurtosis())
        
        # Valores atípicos (outliers) usando IQR
        q1 = float(df[column].quantile(0.25))
        q3 = float(df[column].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        result["outliers_count"] = outliers.shape[0]
        result["outliers_percent"] = (outliers.shape[0] / df.shape[0]) * 100
    
    # Análisis para columnas categóricas
    elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_categorical_dtype(col_type):
        result["count"] = df[column].count()
        result["null_count"] = df[column].isnull().sum()
        result["unique_values"] = df[column].nunique()
        
        # Valores más frecuentes
        value_counts = df[column].value_counts().sort_values(ascending=False)
        
        # Convertir a diccionario para facilitar el acceso
        if len(value_counts) > 0:
            top_values = value_counts.head(10)
            result["top_values"] = {
                str(idx): int(val) for idx, val in top_values.items()
            }
            result["top_values_percent"] = {
                str(idx): (int(val) / df.shape[0]) * 100 for idx, val in top_values.items()
            }
    
    # Análisis para columnas booleanas
    elif pd.api.types.is_bool_dtype(col_type):
        result["count"] = df[column].count()
        result["null_count"] = df[column].isnull().sum()
        
        true_count = df[df[column] == True].shape[0]
        false_count = df[df[column] == False].shape[0]
        
        result["true_count"] = true_count
        result["false_count"] = false_count
        result["true_percent"] = (true_count / df.shape[0]) * 100
        result["false_percent"] = (false_count / df.shape[0]) * 100
    
    # Análisis para columnas de fecha/hora
    elif pd.api.types.is_datetime64_dtype(col_type):
        result["count"] = df[column].count()
        result["null_count"] = df[column].isnull().sum()
        result["min"] = df[column].min()
        result["max"] = df[column].max()
        
        # Calcular rango en días si es posible
        try:
            date_range = df[column].max() - df[column].min()
            if hasattr(date_range, 'days'):
                result["range_days"] = date_range.days
        except Exception:
            pass
    
    return result

def get_correlation_matrix(df, method="pearson"):
    """
    Calcula la matriz de correlación para columnas numéricas.
    
    Args:
        df: DataFrame de Pandas
        method: Método de correlación ('pearson' o 'spearman')
    
    Returns:
        DataFrame con la matriz de correlación
    """
    # Filtrar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None  # No hay suficientes columnas numéricas para correlación
    
    # Calcular matriz de correlación directamente con pandas
    if method.lower() in ["pearson", "spearman"]:
        corr_df = df[numeric_cols].corr(method=method.lower())
    else:
        raise ValueError(f"Método de correlación no soportado: {method}")
    
    return corr_df