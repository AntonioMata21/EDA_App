import streamlit as st
import os

# Configuración de la página
st.set_page_config(
    page_title="EDA App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar módulos
from modules.data_loader import load_data, validate_data
from modules.data_analysis import get_data_summary, get_descriptive_stats
from modules.data_visualization import create_visualization
from modules.data_export import export_data, export_stats, export_visualization

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 0.5rem 0.5rem 0 0;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar variables de sesión
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
from modules.data_analysis import get_empty_summary

if 'summary' not in st.session_state:
    st.session_state.summary = get_empty_summary()
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

# Barra lateral
with st.sidebar:
    st.markdown("<div class='main-header'>📊 EDA App</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sección de carga de datos
    st.markdown("<div class='subsection-header'>📁 Carga de Datos</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Selecciona un archivo CSV o Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Opciones de carga según el tipo de archivo
        with st.expander("Opciones de carga"):
            if file_extension == ".csv":
                delimiter = st.selectbox("Delimitador", [",", ";", "\t", "|"], index=0)
                encoding = st.selectbox("Codificación", ["utf-8", "latin-1", "iso-8859-1"], index=0)
            has_header = st.checkbox("El archivo tiene encabezado", value=True)
        
        # Botón para cargar datos
        if st.button("Cargar Datos"):
            try:
                # Cargar datos según el tipo de archivo
                if file_extension == ".csv":
                    data = load_data(uploaded_file, file_type="csv", delimiter=delimiter, encoding=encoding, has_header=has_header)
                else:  # .xlsx
                    data = load_data(uploaded_file, file_type="excel", has_header=has_header)
                
                # Validar datos
                validation_result = validate_data(data)
                
                if validation_result["valid"]:
                    st.session_state.data = data
                    st.session_state.summary = get_data_summary(data)
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.file_size = uploaded_file.size
                    st.session_state.file_type = file_type
                    st.session_state.delimiter = delimiter
                    st.session_state.encoding = encoding
                    st.session_state.has_header = has_header
                
                    st.write(f"DEBUG: Type of st.session_state.data after load: {type(st.session_state.data)}")
                    st.write(f"DEBUG: Type of st.session_state.summary after get_data_summary: {type(st.session_state.summary)}")
                    st.success(f"Datos cargados correctamente: {uploaded_file.name}")
                else:
                    st.error(f"Error en los datos: {validation_result['message']}")
                    st.session_state.summary = get_empty_summary()
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")
    
    # Navegación (solo visible si hay datos cargados)
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("<div class='subsection-header'>🧭 Navegación</div>", unsafe_allow_html=True)
        page = st.radio(
            "Ir a:",
            ["📊 Resumen General", "🔍 Análisis Detallado", "🔄 Filtros y Transformaciones", "📈 Visualización", "💾 Exportación"],
            key="main_navigation_radio"
        )

# Contenido principal
if st.session_state.data is None:
    # Pantalla de inicio cuando no hay datos cargados
    st.markdown("<div class='main-header'>Bienvenido a la App de Análisis Exploratorio de Datos</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h3>📊 Explora tus datos de manera sencilla e interactiva</h3>
        <p>Esta aplicación te permite realizar un análisis exploratorio completo de tus datos sin necesidad de programar.</p>
        <p>Características principales:</p>
        <ul>
            <li>Carga de archivos CSV y Excel</li>
            <li>Análisis estadístico automático</li>
            <li>Visualizaciones interactivas</li>
            <li>Filtrado y transformación de datos</li>
            <li>Exportación de resultados</li>
        </ul>
        </div>
        
        <div class='info-box'>
        <h3>🚀 Cómo empezar</h3>
        <p>1. Utiliza el panel lateral para cargar tu archivo de datos</p>
        <p>2. Explora las diferentes secciones de análisis</p>
        <p>3. Interactúa con los gráficos y filtros</p>
        <p>4. Exporta tus resultados</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
        <h3>📁 Formatos soportados</h3>
        <ul>
            <li>CSV (.csv)</li>
            <li>Excel (.xlsx)</li>
        </ul>
        </div>
        
        <div class='info-box'>
        <h3>⚙️ Funcionalidades</h3>
        <ul>
            <li>Resumen general del dataset</li>
            <li>Estadísticas descriptivas</li>
            <li>Filtros interactivos</li>
            <li>Visualizaciones personalizables</li>
            <li>Exportación de resultados</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    # Mostrar contenido según la navegación seleccionada
    if page == "📊 Resumen General":
        from pages.summary import show_summary_page

        show_summary_page(st.session_state.data, st.session_state.summary, st.session_state.filename)
    
    elif page == "🔍 Análisis Detallado":
        from pages.analysis import show_analysis_page
        show_analysis_page(st.session_state.data)
    
    elif page == "🔄 Filtros y Transformaciones":
        from pages.filters import show_filters_page
        show_filters_page(st.session_state.data)
    
    elif page == "📈 Visualización":
        from pages.visualization import show_visualization_page
        show_visualization_page(st.session_state.data)
    
    elif page == "💾 Exportación":
        from pages.export import show_export_page
        show_export_page(st.session_state.data)

