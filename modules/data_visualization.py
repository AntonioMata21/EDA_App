import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Configuración de estilo para matplotlib y seaborn
plt.style.use('ggplot')
sns.set_style('whitegrid')

def create_visualization(df, viz_type, x=None, y=None, color=None, facet=None, **kwargs):
    """
    Crea una visualización basada en los parámetros proporcionados.
    
    Args:
        df: DataFrame de Pandas
        viz_type: Tipo de visualización
        x, y, color, facet: Variables para la visualización
        **kwargs: Argumentos adicionales específicos para cada tipo de gráfico
    
    Returns:
        Figura de Plotly o Matplotlib
    """
    
    # Crear visualización según el tipo
    if viz_type == "histogram":
        return create_histogram(df, x, **kwargs)
    elif viz_type == "boxplot":
        return create_boxplot(df, x, y, **kwargs)
    elif viz_type == "scatter":
        return create_scatter(df, x, y, color, **kwargs)
    elif viz_type == "bar":
        return create_bar(df, x, y, color, **kwargs)
    elif viz_type == "line":
        return create_line(df, x, y, color, **kwargs)
    elif viz_type == "pie":
        return create_pie(df, x, y, **kwargs)
    elif viz_type == "heatmap":
        return create_heatmap(df, **kwargs)
    elif viz_type == "violin":
        return create_violin(df, x, y, **kwargs)
    elif viz_type == "pairplot":
        return create_pairplot(df, **kwargs)
    elif viz_type == "timeseries":
        return create_timeseries(df, x, y, color, **kwargs)
    else:
        st.error(f"Tipo de visualización no soportado: {viz_type}")
        return None

def create_histogram(df, x, bins=30, kde=False, title=None, color=None, facet=None, height=500, width=700):
    """
    Crea un histograma con Plotly.
    """
    if x not in df.columns:
        st.error(f"La columna {x} no existe en el DataFrame")
        return None
    
    # Título por defecto
    if title is None:
        title = f"Histograma de {x}"
    
    # Crear histograma con Plotly
    fig = px.histogram(
        df, x=x, color=color, facet_col=facet,
        title=title, nbins=bins,
        height=height, width=width,
        marginal=None, # Remove marginal boxplot
        opacity=0.7,
    )

    if kde:
        from scipy.stats import gaussian_kde
        hist_data = df[x].dropna()
        if not hist_data.empty:
            kde_model = gaussian_kde(hist_data)
            kde_x = np.linspace(hist_data.min(), hist_data.max(), 500)
            kde_y = kde_model(kde_x)

            total_samples = len(hist_data)
            bin_width = (hist_data.max() - hist_data.min()) / bins if bins > 0 else 1
            scale_factor = total_samples * bin_width

            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y * scale_factor,
                    mode='lines',
                    name='KDE',
                    line=dict(color='red', width=2)
                )
            )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title="Frecuencia",
        legend_title_text=color,
        template="plotly_white"
    )
    
    return fig

def create_boxplot(df, x=None, y=None, title=None, color=None, points="outliers", height=500, width=700): # x e y ahora son opcionales
    """
    Crea un boxplot con Plotly. Puede manejar boxplot de una variable (solo y)
    o agrupado (x e y).
    """
    # Verificar que al menos una columna de valor (x o y para px.box) esté presente
    if y is None and x is None: # Si no se provee ni x ni y para graficar
        st.error("Debes proporcionar al menos una columna 'x' (para boxplot horizontal de una variable) o 'y' (para boxplot vertical de una variable).")
        return None
    
    # Verificar columnas si se proporcionan
    if y is not None and y not in df.columns:
        st.error(f"La columna '{y}' (para el eje Y) no existe en el DataFrame.")
        return None
    if x is not None and x not in df.columns:
        st.error(f"La columna '{x}' (para el eje X o categoría) no existe en el DataFrame.")
        return None
    
    # Título por defecto
    if title is None:
        if x is not None and y is not None: # Caso agrupado o scatter con box
            title = f"Boxplot de {y} por {x}"
        elif y is not None: # Boxplot de una variable vertical
            title = f"Boxplot de {y}"
        elif x is not None: # Boxplot de una variable horizontal
            title = f"Boxplot de {x}"
        else:
            title = "Boxplot" # No debería llegar aquí si la verificación anterior funciona
    
    # Crear boxplot con Plotly
    # px.box es flexible:
    # - Si solo 'y' se da, es un boxplot vertical de una variable.
    # - Si solo 'x' se da (y es numérico), es un boxplot horizontal de una variable.
    # - Si 'x' (categórico) e 'y' (numérico) se dan, es un boxplot agrupado.
    fig = px.box(
        df, x=x, y=y, color=color, # Pasamos x e y como vienen, px.box decide
        title=title,
        height=height, width=width,
        points=points
    )
    
    fig.update_layout(
        template="plotly_white",
        # Ajuste de boxmode si se usa color para agrupar en un boxplot de una sola variable
        boxmode="group" if (x is not None and color is not None) or (y is not None and color is not None and x is None) else "overlay"
    )
    
    return fig

def create_scatter(df, x, y, color=None, size=None, title=None, trendline=None, height=500, width=700):
    """
    Crea un gráfico de dispersión con Plotly.
    """
    # Verificar columnas
    if x not in df.columns:
        st.error(f"La columna {x} no existe en el DataFrame")
        return None
    if y not in df.columns:
        st.error(f"La columna {y} no existe en el DataFrame")
        return None
    
    # Título por defecto
    if title is None:
        title = f"Gráfico de dispersión: {y} vs {x}"
    
    # Crear scatter plot con Plotly
    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        title=title,
        height=height, width=width,
        trendline=trendline,  # "ols", "lowess", None
        opacity=0.7,
        hover_data=df.columns[:5]  # Mostrar primeras 5 columnas en hover
    )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text=color,
        template="plotly_white"
    )
    
    return fig

def create_bar(df, x, y, color=None, title=None, orientation="v", height=500, width=700, aggfunc="sum"):
    """
    Crea un gráfico de barras con Plotly.
    """
    # Verificar columnas
    if x not in df.columns:
        st.error(f"La columna {x} no existe en el DataFrame")
        return None
    if y is not None and y not in df.columns:
        st.error(f"La columna {y} no existe en el DataFrame")
        return None
    
    # Título por defecto
    if title is None:
        if y is not None:
            title = f"Gráfico de barras: {y} por {x}"
        else:
            title = f"Gráfico de barras: Conteo de {x}"
    
    # Si y es None, hacemos un conteo
    if y is None:
        # Crear gráfico de conteo
        fig = px.histogram(
            df, x=x, color=color,
            title=title,
            height=height, width=width,
            barmode="group" if color is not None else None
        )
    else:
        # Crear gráfico de barras con agregación
        fig = px.bar(
            df, x=x, y=y, color=color,
            title=title,
            height=height, width=width,
            barmode="group" if color is not None else None,
            orientation=orientation
        )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y if y is not None else "Conteo",
        legend_title_text=color,
        template="plotly_white",
        opacity=0.7
    )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title="Frecuencia",
        legend_title_text=color,
        template="plotly_white"
    )
    
    return fig



def create_scatter(df, x, y, color=None, size=None, title=None, trendline=None, height=500, width=700):
    """
    Crea un gráfico de dispersión con Plotly.
    """
    # Verificar columnas
    if x not in df.columns:
        st.error(f"La columna {x} no existe en el DataFrame")
        return None
    if y not in df.columns:
        st.error(f"La columna {y} no existe en el DataFrame")
        return None
    
    # Título por defecto
    if title is None:
        title = f"Gráfico de dispersión: {y} vs {x}"
    
    # Crear scatter plot con Plotly
    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        title=title,
        height=height, width=width,
        trendline=trendline,  # "ols", "lowess", None
        opacity=0.7,
        hover_data=df.columns[:5]  # Mostrar primeras 5 columnas en hover
    )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text=color,
        template="plotly_white"
    )
    
    return fig

def create_bar(df, x, y, color=None, title=None, orientation="v", height=500, width=700, aggfunc="sum"):
    """
    Crea un gráfico de barras con Plotly.
    """
    # Verificar columnas
    if x not in df.columns:
        st.error(f"La columna {x} no existe en el DataFrame")
        return None
    if y is not None and y not in df.columns:
        st.error(f"La columna {y} no existe en el DataFrame")
        return None
    
    # Título por defecto
    if title is None:
        if y is not None:
            title = f"Gráfico de barras: {y} por {x}"
        else:
            title = f"Gráfico de barras: Conteo de {x}"
    
    # Si y es None, hacemos un conteo
    if y is None:
        # Crear gráfico de conteo
        fig = px.histogram(
            df, x=x, color=color,
            title=title,
            height=height, width=width,
            barmode="group" if color is not None else None
        )
    else:
        # Crear gráfico de barras con agregación
        fig = px.bar(
            df, x=x, y=y, color=color,
            title=title,
            height=height, width=width,
            barmode="group" if color is not None else None,
            orientation=orientation
        )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y if y is not None else "Conteo",
        legend_title_text=color,
        template="plotly_white"
    )
    
    return fig

def create_line(df, x, y, color=None, title=None, height=500, width=700, markers=True):
    """
    Crea un gráfico de líneas con Plotly.
    """
    # Verificar columnas
    if x not in df.columns:
        st.error(f"La columna {x} no existe en el DataFrame")
        return None
    if y not in df.columns:
        st.error(f"La columna {y} no existe en el DataFrame")
        return None
    
    # Título por defecto
    if title is None:
        title = f"Gráfico de líneas: {y} vs {x}"
    
    # Crear gráfico de líneas con Plotly
    fig = px.line(
        df, x=x, y=y, color=color,
        title=title,
        height=height, width=width,
        markers=markers
    )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text=color,
        template="plotly_white"
    )
    
    return fig

def create_pie(df, names, values, title=None, height=500, width=700, hole=0):
    """
    Crea un gráfico de torta con Plotly.
    """
    # Verificar columnas
    if names not in df.columns:
        st.error(f"La columna {names} no existe en el DataFrame")
        return None
    if values not in df.columns:
        st.error(f"La columna {values} no existe en el DataFrame")
        return None
    
    # Título por defecto
    if title is None:
        title = f"Gráfico de torta: {values} por {names}"
    
    # Agregar datos si hay muchas categorías
    if df[names].nunique() > 15:
        # Tomar las 14 categorías principales y agrupar el resto
        top_categories = df.groupby(names)[values].sum().nlargest(14).index
        df_plot = df.copy()
        df_plot[names] = df_plot[names].apply(lambda x: x if x in top_categories else "Otros")
        df_agg = df_plot.groupby(names)[values].sum().reset_index()
    else:
        df_agg = df.groupby(names)[values].sum().reset_index()
    
    # Crear gráfico de torta con Plotly
    fig = px.pie(
        df_agg, names=names, values=values,
        title=title,
        height=height, width=width,
        hole=hole  # 0 para pie, >0 para donut
    )
    
    # Personalizar diseño
    fig.update_layout(
        template="plotly_white",
        legend_title_text=names
    )
    
    # Mostrar porcentajes y valores
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    
    return fig

def create_heatmap(df, columns=None, title=None, height=600, width=800, colorscale="RdBu_r", method="pearson"):
    """
    Crea un mapa de calor de correlación con Plotly.
    """
    # Si no se especifican columnas, usar todas las numéricas
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filtrar solo columnas numéricas existentes
        columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    
    if len(columns) < 2:
        st.error("Se necesitan al menos 2 columnas numéricas para crear un mapa de calor")
        return None
    
    # Calcular matriz de correlación
    corr_matrix = df[columns].corr(method=method)
    
    # Título por defecto
    if title is None:
        title = f"Mapa de calor de correlación ({method})"
    
    # Crear mapa de calor con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size":10},
        hoverongaps=False
    ))
    
    # Personalizar diseño
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template="plotly_white",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_tickangle=-45
    )
    
    return fig

def create_violin(df, x=None, y=None, title=None, color=None, height=500, width=700, box=True, points="outliers"): # x e y ahora son opcionales
    """
    Crea un gráfico de violín con Plotly. Puede manejar violín de una variable (solo y)
    o agrupado (x e y).
    """
    # Verificar que al menos una columna de valor (y para px.violin) esté presente
    # o una columna x para agrupar si y se da.
    if y is None and x is None: # Necesitamos al menos algo para graficar
        st.error("Debes proporcionar al menos una columna 'y' (para el valor numérico) o 'x' (para categoría si 'y' está presente).")
        return None
    if y is None and x is not None: # Solo x no tiene sentido para px.violin sin y
        st.error("Para un gráfico de violín, la columna 'y' (valor numérico) es requerida.")
        return None

    # Verificar columnas si se proporcionan
    if y is not None and y not in df.columns: # y siempre es requerido si se llama a esta función
        st.error(f"La columna '{y}' (para el eje Y) no existe en el DataFrame.")
        return None
    if x is not None and x not in df.columns:
        st.error(f"La columna '{x}' (para el eje X o categoría) no existe en el DataFrame.")
        return None
    
    # Título por defecto
    if title is None:
        if x is not None and y is not None: # Caso agrupado
            title = f"Gráfico de violín de {y} por {x}"
        elif y is not None: # Violín de una variable (y es la variable numérica)
            title = f"Gráfico de violín de {y}"
        # No deberíamos llegar a un caso 'else' sin título si las verificaciones anteriores son correctas
    
    # Crear gráfico de violín con Plotly
    # px.violin necesita 'y' para los valores. 'x' es para la categoría de agrupación.
    fig = px.violin(
        df, x=x, y=y, color=color, # Pasamos x (puede ser None) e y
        title=title,
        height=height, width=width,
        box=box,
        points=points
    )
    
    # Personalizar diseño
    fig.update_layout(
        template="plotly_white",
        # Si x (categoría) y color (subcategoría) están presentes, usa group. Sino, overlay.
        violinmode="group" if x is not None and color is not None else "overlay"
    )
    
    return fig

def create_pairplot(df, columns=None, title="Matriz de dispersión", height=800, width=800, color=None):
    """
    Crea una matriz de dispersión con Seaborn.
    """
    # Si no se especifican columnas, usar todas las numéricas (máximo 6)
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    else:
        # Filtrar solo columnas numéricas existentes
        columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)][:6]
    
    if len(columns) < 2:
        st.error("Se necesitan al menos 2 columnas numéricas para crear una matriz de dispersión")
        return None
    
    # Crear figura de matplotlib
    plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # Crear pairplot con Seaborn
    g = sns.pairplot(
        df[columns + ([color] if color is not None else [])],
        hue=color,
        diag_kind="kde",
        plot_kws={"alpha": 0.6},
        corner=True  # Solo mostrar la mitad inferior
    )
    
    # Ajustar diseño
    g.fig.suptitle(title, y=1.02)
    plt.tight_layout()
    
    # Convertir a imagen
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    
    # Mostrar imagen en Streamlit
    st.image(buf, width=width)
    
    return g.fig

def create_timeseries(df, x, y, color=None, title=None, height=500, width=700, markers=True):
    """
    Crea un gráfico de series temporales con Plotly.
    """
    # Verificar columnas
    if x not in df.columns:
        st.error(f"La columna {x} no existe en el DataFrame")
        return None
    if y not in df.columns:
        st.error(f"La columna {y} no existe en el DataFrame")
        return None
    
    # Verificar si x es una columna de fecha/hora
    if not pd.api.types.is_datetime64_any_dtype(df[x]):
        try:
            # Intentar convertir a datetime
            df = df.copy()
            df[x] = pd.to_datetime(df[x])
        except Exception:
            st.error(f"La columna {x} no es una fecha válida y no se puede convertir")
            return None
    
    # Título por defecto
    if title is None:
        title = f"Serie temporal: {y} vs {x}"
    
    # Crear gráfico de líneas con Plotly
    fig = px.line(
        df, x=x, y=y, color=color,
        title=title,
        height=height, width=width,
        markers=markers
    )
    
    # Personalizar diseño
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text=color,
        template="plotly_white"
    )
    
    # Mejorar formato de fechas en el eje x
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

def export_visualization(fig, format="png", filename="visualization", width=800, height=600, scale=2):
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
    if format.lower() not in ["png", "pdf"]:
        st.error(f"Formato no soportado: {format}")
        return None
    
    # Verificar tipo de figura
    if "plotly" in str(type(fig)).lower():
        # Exportar figura de Plotly
        buf = BytesIO()
        if format.lower() == "png":
            fig.write_image(buf, format="png", width=width, height=height, scale=scale)
        else:  # pdf
            fig.write_image(buf, format="pdf", width=width, height=height, scale=scale)
    elif "matplotlib" in str(type(fig)).lower():
        # Exportar figura de Matplotlib
        buf = BytesIO()
        fig.savefig(buf, format=format.lower(), dpi=100*scale, bbox_inches="tight")
    else:
        st.error("Tipo de figura no soportado")
        return None
    
    buf.seek(0)
    return buf