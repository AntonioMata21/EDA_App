import pandas as pd
import io
import os
import streamlit as st

@st.cache_data
def load_data(file, file_type="csv", delimiter=",", encoding="utf-8", has_header=True):
    """
    Carga datos desde archivos CSV o Excel utilizando Pandas, con manejo robusto de errores comunes.

    Args:
        file: Archivo cargado por st.file_uploader
        file_type: Tipo de archivo ('csv' o 'excel')
        delimiter: Delimitador para archivos CSV
        encoding: Codificación del archivo
        has_header: Indica si el archivo tiene encabezado

    Returns:
        DataFrame de Pandas con los datos cargados
    """
    try:
        # Leer bytes del archivo
        file_bytes = file.read()
        buffer = io.BytesIO(file_bytes)

        if file_type.lower() == "csv":
            read_options = {
                "sep": delimiter,
                "encoding": encoding,
                "header": 0 if has_header else None,
            }

            # Primer intento con los read_options dados
            df = pd.read_csv(buffer, **read_options)

            # Diagnóstico: si solo hay 1 columna, quizás el separador está mal
            if df.shape[1] == 1:
                st.warning("⚠️ El archivo parece tener una sola columna. Probando con separador alternativo ';'")
                buffer.seek(0)
                read_options_alt = read_options.copy()
                read_options_alt["sep"] = ";"
                df = pd.read_csv(buffer, **read_options_alt)
                st.success("✔️ CSV recargado con separador ';'. Columnas detectadas:")
                st.write(df.columns.tolist())

        elif file_type.lower() == "excel":
            df = pd.read_excel(buffer, header=0 if has_header else None)

        else:
            raise ValueError(f"Tipo de archivo no soportado: {file_type}")

        # Generar nombres genéricos si no hay encabezado
        if not has_header:
            df.columns = [f"column_{i}" for i in range(len(df.columns))]

        return df

    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {str(e)}")
        return pd.DataFrame()

def validate_data(df):
    """
    Valida la estructura y contenido básico del DataFrame.
    
    Args:
        df: DataFrame de Pandas a validar
    
    Returns:
        Diccionario con el resultado de la validación
    """
    result = {"valid": True, "message": ""}

    # Verificar si el DataFrame está vacío
    if df.empty:
        result["valid"] = False
        result["message"] = "El archivo no contiene datos."
        return result

    # Verificar si hay columnas
    if len(df.columns) == 0:
        result["valid"] = False
        result["message"] = "El archivo no contiene columnas."
        return result

    # Verificar consistencia de filas/columnas
    try:
        if df.shape[0] > 0:
            _ = df.iloc[0]
    except Exception as e:
        result["valid"] = False
        result["message"] = f"Estructura de datos inconsistente: {str(e)}"
        return result

    # Verificar tipos de datos
    try:
        _ = df.dtypes
    except Exception as e:
        result["valid"] = False
        result["message"] = f"Error al inferir tipos de datos: {str(e)}"
        return result

    return result

def get_file_preview(df, max_rows=5):
    """
    Genera una vista previa del DataFrame.
    
    Args:
        df: DataFrame de Polars
        max_rows: Número máximo de filas a mostrar
    
    Returns:
        DataFrame de Polars con la vista previa
    """
    return df.head(max_rows)

def get_column_types(df):
    """
    Obtiene los tipos de datos de cada columna.
    
    Args:
        df: DataFrame de Polars
    
    Returns:
        Diccionario con los tipos de datos de cada columna
    """
    return {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

def get_missing_values(df):
    """
    Calcula el número y porcentaje de valores faltantes por columna.
    
    Args:
        df: DataFrame de Polars
    
    Returns:
        DataFrame con información de valores faltantes
    """
    # Calcular valores nulos por columna
    null_counts = df.null_count()
    
    # Crear un nuevo DataFrame con la información de valores faltantes
    missing_data = {
        "columna": df.columns,
        "valores_faltantes": [null_counts.get_column(col)[0] for col in df.columns],
        "porcentaje": [null_counts.get_column(col)[0] / df.height * 100 for col in df.columns]
    }
    
    return pd.DataFrame(missing_data)

def get_duplicate_info(df):
    """
    Calcula información sobre filas duplicadas.
    
    Args:
        df: DataFrame de Polars
    
    Returns:
        Diccionario con información sobre duplicados
    """
    # Contar duplicados
    duplicated_rows = df.shape[0] - df.unique().shape[0]
    
    return {
        "total_duplicados": duplicated_rows,
        "porcentaje": (duplicated_rows / df.shape[0] * 100) if df.shape[0] > 0 else 0
    }

def get_memory_usage(df):
    """
    Calcula el uso de memoria del DataFrame.
    
    Args:
        df: DataFrame de Polars
    
    Returns:
        Uso de memoria en MB
    """
    # Polars no tiene un método directo para esto, estimamos basado en el tamaño de los datos
    # Esta es una aproximación
    memory_bytes = sum(df.estimated_size())
    memory_mb = memory_bytes / (1024 * 1024)
    
    return round(memory_mb, 2)