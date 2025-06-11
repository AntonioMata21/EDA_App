import pandas as pd
import streamlit as st
import io
import base64
from datetime import datetime
import plotly
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

def export_dataframe_to_csv(df: pd.DataFrame, delimiter=',', include_header=True, encoding='utf-8') -> bytes:
    """
    Convierte un DataFrame de Pandas a bytes CSV.

    Args:
        df: DataFrame de Pandas
        delimiter: Delimitador a usar en el CSV.
        include_header: Booleano, si se incluye o no el encabezado.
        encoding: Codificación del archivo CSV.

    Returns:
        bytes con los datos CSV
    """
    buffer = io.BytesIO()
    # Cuando se escribe a un buffer de bytes, pandas puede manejar la codificación directamente.
    # El método to_csv escribirá bytes codificados en el buffer.
    df.to_csv(buffer, index=False, sep=delimiter, encoding=encoding, header=include_header)
    csv_bytes = buffer.getvalue() # Esto ya son bytes
    buffer.close()
    return csv_bytes
def export_dataframe_to_excel(df, sheet_name="Dataset", include_index=False):
    """
    Convierte un DataFrame de Pandas a un archivo Excel en formato de texto.
    Args:
        df: DataFrame de Pandas
    Returns:
        BytesIO con los datos Excel
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=include_index, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()
def export_dataframe_to_parquet(df):
    """Exporta DataFrame a bytes de Parquet."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False) # Parquet usualmente no guarda el índice de pandas
    buffer.seek(0)
    return buffer.getvalue() # Devuelve bytes

def export_dataframe_to_json(df, orient="records", indent=None):
    """Exporta DataFrame a una cadena JSON."""
    # Pandas to_json devuelve una cadena
    json_string = df.to_json(orient=orient, indent=indent, force_ascii=False)
    return json_string

# Funciones para estadísticas
def export_statistics_to_csv(stats_df: pd.DataFrame) -> str: # Anotación de tipo -> str
    """Exporta DataFrame de estadísticas a una cadena CSV."""
    buffer = io.StringIO() # Correcto para strings
    stats_df.to_csv(buffer, index=True, encoding='utf-8') # El índice puede ser útil aquí (nombre de la estadística)
    csv_string = buffer.getvalue()
    buffer.close()
    return csv_string 

def export_statistics_to_excel(stats_df, sheet_name="Statistics"): # Asume stats_df es un DataFrame
    """Exporta DataFrame de estadísticas a bytes de Excel."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        stats_df.to_excel(writer, index=True, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()

# Funciones para visualizaciones (Plotly)
def export_plotly_fig_to_image(fig, format="png", width=800, height=600, scale=2):
    """Exporta una figura Plotly a bytes de imagen (png, jpeg, webp, svg, pdf)."""
    if format.lower() == "pdf" or format.lower() == "svg": # Formatos vectoriales
         return fig.to_image(format=format.lower()) # No necesitan width/height/scale de la misma manera
    return fig.to_image(format=format.lower(), width=width, height=height, scale=scale)

def export_plotly_fig_to_html(fig):
    """Exporta una figura Plotly a una cadena HTML."""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def export_data(df, format="csv", filename=None):
    """
    Exporta un DataFrame a un archivo en el formato especificado.
    
    Args:
        df: DataFrame de Pandas
        format: Formato de exportación ('csv' o 'excel')
        filename: Nombre del archivo sin extensión
    
    Returns:
        BytesIO con los datos exportados
    """
    if filename is None:
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_export_{timestamp}"
    
    # Exportar según el formato
    if format.lower() == "csv":
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, encoding="utf-8")
        buffer.seek(0)
        return buffer, f"{filename}.csv", "text/csv"
    
    elif format.lower() == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")
        buffer.seek(0)
        return buffer, f"{filename}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    else:
        st.error(f"Formato no soportado: {format}")
        return None, None, None

def export_stats(stats_df, format="csv", filename=None):
    """
    Exporta estadísticas descriptivas a un archivo.
    
    Args:
        stats_df: DataFrame con estadísticas
        format: Formato de exportación ('csv' o 'excel')
        filename: Nombre del archivo sin extensión
    
    Returns:
        BytesIO con los datos exportados
    """
    if filename is None:
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats_export_{timestamp}"
    
    # Exportar según el formato
    if format.lower() == "csv":
        buffer = io.BytesIO()
        stats_df.to_csv(buffer, index=False, encoding="utf-8")
        buffer.seek(0)
        return buffer, f"{filename}.csv", "text/csv"
    
    elif format.lower() == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            stats_df.to_excel(writer, index=False, sheet_name="Statistics")
        buffer.seek(0)
        return buffer, f"{filename}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    else:
        st.error(f"Formato no soportado: {format}")
        return None, None, None

def export_visualization(fig, format="png", filename=None, width=800, height=600, scale=2):
    """
    Exporta una visualización a un archivo.
    
    Args:
        fig: Figura de Plotly o Matplotlib
        format: Formato de exportación ('png' o 'pdf')
        filename: Nombre del archivo sin extensión
        width, height: Dimensiones de la imagen
        scale: Factor de escala para la resolución
    
    Returns:
        BytesIO con la imagen exportada
    """
    if filename is None:
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{timestamp}"
    
    if format.lower() not in ["png", "pdf"]:
        st.error(f"Formato no soportado: {format}")
        return None, None, None
    
    # Verificar tipo de figura
    if "plotly" in str(type(fig)).lower():
        # Exportar figura de Plotly
        buffer = io.BytesIO()
        if format.lower() == "png":
            fig.write_image(buffer, format="png", width=width, height=height, scale=scale)
            mime_type = "image/png"
        else:  # pdf
            fig.write_image(buffer, format="pdf", width=width, height=height, scale=scale)
            mime_type = "application/pdf"
    elif "matplotlib" in str(type(fig)).lower():
        # Exportar figura de Matplotlib
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format.lower(), dpi=100*scale, bbox_inches="tight")
        if format.lower() == "png":
            mime_type = "image/png"
        else:  # pdf
            mime_type = "application/pdf"
    else:
        st.error("Tipo de figura no soportado")
        return None, None, None
    
    buffer.seek(0)
    return buffer, f"{filename}.{format.lower()}", mime_type

def generate_pdf_report(df, summary, stats, visualizations, filename=None):
    """
    Genera un informe en PDF con los resultados del análisis.
    
    Args:
        df: DataFrame de Pandas
        summary: Diccionario con el resumen del DataFrame
        stats: DataFrame con estadísticas descriptivas
        visualizations: Lista de tuplas (título, figura)
        filename: Nombre del archivo sin extensión
    
    Returns:
        BytesIO con el informe en PDF
    """
    if filename is None:
        # Generar nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eda_report_{timestamp}"
    
    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Configurar fuente
    pdf.set_font("Arial", "B", 16)
    
    # Título
    pdf.cell(0, 10, "Informe de Análisis Exploratorio de Datos", ln=True, align="C")
    pdf.ln(10)
    
    # Información general
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Información General", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Filas: {summary['n_rows']}, Columnas: {summary['n_cols']}", ln=True)
    pdf.ln(5)
    
    # Resumen de tipos de datos
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Tipos de Datos", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Columnas numéricas
    if summary['numeric_cols']:
        pdf.cell(0, 10, f"Columnas numéricas: {len(summary['numeric_cols'])}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, ", ".join(summary['numeric_cols'][:10]) + 
                     ("..." if len(summary['numeric_cols']) > 10 else ""))
        pdf.set_font("Arial", "", 12)
    
    # Columnas categóricas
    if summary['categorical_cols']:
        pdf.cell(0, 10, f"Columnas categóricas: {len(summary['categorical_cols'])}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, ", ".join(summary['categorical_cols'][:10]) + 
                     ("..." if len(summary['categorical_cols']) > 10 else ""))
        pdf.set_font("Arial", "", 12)
    
    # Columnas de fecha/hora
    if summary['datetime_cols']:
        pdf.cell(0, 10, f"Columnas de fecha/hora: {len(summary['datetime_cols'])}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, ", ".join(summary['datetime_cols']))
        pdf.set_font("Arial", "", 12)
    
    # Valores faltantes
    if summary['cols_with_nulls']:
        pdf.cell(0, 10, f"Columnas con valores faltantes: {len(summary['cols_with_nulls'])}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, ", ".join(summary['cols_with_nulls'][:10]) + 
                     ("..." if len(summary['cols_with_nulls']) > 10 else ""))
        pdf.set_font("Arial", "", 12)
    
    pdf.ln(5)
    
    # Estadísticas descriptivas
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Estadísticas Descriptivas", ln=True)
    
    # Convertir estadísticas a pandas para facilitar el manejo
    if isinstance(stats, pd.DataFrame):
        stats_pd = stats
    else:
        stats_pd = stats
    
    # Mostrar estadísticas para columnas numéricas (limitado a 5 columnas)
    numeric_stats = stats_pd[stats_pd["statistic"].isin(["count", "mean", "std", "min", "25%", "50%", "75%", "max"])]
    
    if not numeric_stats.empty:
        # Seleccionar hasta 5 columnas numéricas (excluyendo 'statistic')
        cols_to_show = ["statistic"] + list(numeric_stats.columns[1:6])
        numeric_stats_subset = numeric_stats[cols_to_show]
        
        # Crear tabla
        pdf.set_font("Arial", "B", 10)
        pdf.cell(40, 10, "Estadística", border=1)
        for col in cols_to_show[1:]:
            pdf.cell(30, 10, col[:10] + ("..." if len(col) > 10 else ""), border=1)
        pdf.ln()
        
        pdf.set_font("Arial", "", 10)
        for _, row in numeric_stats_subset.iterrows():
            pdf.cell(40, 10, str(row["statistic"]), border=1)
            for col in cols_to_show[1:]:
                value = row[col]
                if isinstance(value, (int, float)):
                    pdf.cell(30, 10, f"{value:.2f}" if abs(value) < 1000 else f"{value:.1e}", border=1)
                else:
                    pdf.cell(30, 10, str(value)[:10], border=1)
            pdf.ln()
    
    pdf.ln(10)
    
    # Visualizaciones
    if visualizations:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Visualizaciones", ln=True)
        pdf.ln(5)
        
        for i, (title, fig) in enumerate(visualizations):
            # Añadir nueva página si es necesario
            if i > 0:
                pdf.add_page()
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, title, ln=True)
            
            # Guardar figura en archivo temporal
            temp_img = f"temp_viz_{i}.png"
            
            if "plotly" in str(type(fig)).lower():
                fig.write_image(temp_img, width=700, height=400, scale=1)
            elif "matplotlib" in str(type(fig)).lower():
                fig.savefig(temp_img, format="png", dpi=100, bbox_inches="tight")
            
            # Añadir imagen al PDF
            pdf.image(temp_img, x=10, y=None, w=190)
            
            # Eliminar archivo temporal
            try:
                os.remove(temp_img)
            except Exception:
                pass
    
    # Guardar PDF en buffer
    buffer = io.BytesIO()
    buffer.write(pdf.output(dest="S").encode("latin1"))
    buffer.seek(0)
    
    
    return buffer.getvalue(), f"{filename}.pdf", "application/pdf"