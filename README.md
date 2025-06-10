# Aplicación de Análisis Exploratorio de Datos (EDA)

Una aplicación web local, ligera y modular para análisis exploratorio de datos (EDA) desarrollada con Streamlit, Polars y herramientas de visualización modernas.

## Características Principales

### 1. Carga de Datos
- Soporte para formatos CSV y Excel
- Lectura eficiente usando Polars
- Validación automática de estructura y tipos
- Detección de valores faltantes y duplicados

### 2. Análisis y Exploración
- Resumen general del dataset (filas, columnas, tipos)
- Estadísticas descriptivas completas
- Análisis de valores nulos y duplicados
- Optimización de tipos y memoria

### 3. Filtros y Transformaciones
- Filtrado interactivo por tipo de columna
- Transformaciones de datos (conversión, limpieza, etc.)
- Guardado de subconjuntos filtrados

### 4. Visualización Interactiva
- Integración con Plotly, Seaborn y Matplotlib
- Múltiples tipos de gráficos (histogramas, boxplots, etc.)
- Herramientas interactivas (zoom, hover, selección)

### 5. Exportación de Resultados
- Descarga de datasets en múltiples formatos
- Exportación de estadísticas y métricas
- Guardado de visualizaciones
- Generación de reportes automáticos

## Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clona o descarga este repositorio

2. Crea un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   ```

3. Activa el entorno virtual:
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Inicia la aplicación:
   ```bash
   streamlit run app.py
   ```

2. Abre tu navegador en `http://localhost:8501`

3. Carga un archivo CSV o Excel desde el menú lateral

4. Explora las diferentes secciones de la aplicación:
   - Resumen General
   - Análisis Detallado
   - Filtros y Transformaciones
   - Visualización
   - Exportación

## Estructura del Proyecto

```
EDA-app/
├── app.py                  # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── modules/                # Módulos funcionales
│   ├── data_loader.py      # Carga y validación de datos
│   ├── data_analysis.py    # Funciones de análisis
│   ├── data_visualization.py # Funciones de visualización
│   └── data_export.py      # Exportación de resultados
└── pages/                  # Páginas de la aplicación
    ├── summary.py          # Página de resumen general
    ├── analysis.py         # Página de análisis detallado
    ├── filters.py          # Página de filtros y transformaciones
    ├── visualization.py    # Página de visualización
    └── export.py           # Página de exportación
```

## Extensiones Futuras

La aplicación está diseñada para ser fácilmente extendida con funcionalidades adicionales como:

- Análisis de componentes principales (PCA)
- Clustering (KMeans, DBSCAN)
- Modelos predictivos simples
- Detección de anomalías
- Análisis de texto y NLP

## Contribuciones

Las contribuciones son bienvenidas. Por favor, siente libre de abrir un issue o enviar un pull request con mejoras o correcciones.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.