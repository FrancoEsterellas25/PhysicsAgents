import streamlit as st
import numpy as np
import polars as pl
import sys
from pathlib import Path

# Add project root to sys.path to resolve core imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from continuous_simulation import ContinuousSEIRSDSimulation
from plotly_animacion import generar_dashboard

# Set streamlit page config
st.set_page_config(
    page_title="Simulador de Epidemias SEIRS-D",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description with premium styling
st.markdown("""
<div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #DA70D6; margin-bottom: 25px;">
    <h1 style="color: #FFFFFF; margin: 0;">🦠 Simulador Epidemiológico SEIRS-D Continuo</h1>
    <p style="color: #B0B0B0; margin: 10px 0 0 0; font-size: 1.1rem;">
        Modelado físico espacial de transmisión por aerosol lagrangiano, agendas sociales de co-presencia, higiene personal y uso de barbijo.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### 🎛️ Configuración de Escenario")

# Section 1: Población
with st.sidebar.expander("👥 Demografía y Simulación", expanded=True):
    N_agentes = st.slider("Tamaño Población (N)", min_value=100, max_value=1000, value=500, step=50)
    L_espacio = st.slider("Dimensión del Espacio (L)", min_value=20.0, max_value=100.0, value=50.0, step=5.0)
    dias_simulacion = st.slider("Duración (Días)", min_value=10, max_value=45, value=25, step=5)
    seed_I = st.slider("Infectados Iniciales", min_value=1, max_value=20, value=5, step=1)

# Section 2: Dinámica de Movimiento
with st.sidebar.expander("🚶 Movilidad y Contactos", expanded=True):
    mov_tipo = st.selectbox(
        "Patrón de Movimiento",
        options=["Hubs Cerrados/Abiertos (Escuela, Oficina, Súper, Centro)", "Movimiento Browniano Libre (Toda la ciudad)"]
    )
    hubs_activos = (mov_tipo == "Hubs Cerrados/Abiertos (Escuela, Oficina, Súper, Centro)")
    mov_libre = not hubs_activos
    
    C_DS = st.slider("Distanciamiento Social (Cumplimiento C_DS)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)

# Section 3: Cuarentena y Aislamiento
with st.sidebar.expander("🛡️ Medidas de Intervención", expanded=True):
    enable_quarantine = st.checkbox("Activar Cuarentenas Domésticas", value=True)
    quarantine_trigger_pct = st.slider(
        "Gatillo Cuarentena (% Infectados Activos)",
        min_value=0.01, max_value=0.30, value=0.05, step=0.01,
        format="%.2f"
    )

# Section 4: Protección e Higiene
with st.sidebar.expander("😷 Medidas de Protección Personal", expanded=True):
    barbijo_cumplimiento = st.slider("Cumplimiento Mascarilla (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    barbijo_eficacia = st.slider("Eficacia Filtración Mascarilla (%)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    higiene_factor = st.slider("Nivel de Higiene Personal (Eleva umbral infeccioso)", min_value=1.0, max_value=3.0, value=1.0, step=0.1)

# Section 5: Parámetros Clínicos/Virus
with st.sidebar.expander("☣️ Parámetros Físicos del Virus", expanded=False):
    tau_max = st.slider("Dosis Tolerancia Máxima (T_inf)", min_value=0.1, max_value=1.5, value=0.5, step=0.05)
    ell = st.slider("Radio Dispersión Aerosol (ell)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    delta_ext = st.slider("Decaimiento Aerosol en Calles", min_value=0.2, max_value=3.0, value=1.0, step=0.1)

# Run Simulation Button
if st.button("🚀 Ejecutar Simulación", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Instantiate simulation
    status_text.text("Inicializando entorno y perfiles inmunes...")
    sim = ContinuousSEIRSDSimulation(N=N_agentes, L=L_espacio, t_max=dias_simulacion * 96)
    
    # Apply custom sidebar variables
    sim.dt = 1/96
    sim.hubs_activos = hubs_activos
    sim.movimiento_libre = mov_libre
    sim.c_DS = C_DS
    sim.enable_quarantine = enable_quarantine
    sim.quarantine_trigger_pct = quarantine_trigger_pct
    sim.higiene_factor = higiene_factor
    sim.barbijo_eficacia = barbijo_eficacia
    sim.barbijo_cumplimiento = barbijo_cumplimiento
    sim.tau_max = tau_max
    sim.ell = ell
    sim.delta_ext = delta_ext
    
    # 2. Run simulation steps (with custom step progression updates in Streamlit)
    status_text.text("Simulando trayectorias físicas y dosificación viral...")
    base_dir = Path(__file__).resolve().parent
    
    # Run the main ABM loop
    sim.run(output_dir=base_dir, n_seed=seed_I, seed=42)
    progress_bar.progress(50)
    
    # 3. Generate the interactive Plotly dashboard
    status_text.text("Generando animación e informes epidemiológicos...")
    generar_dashboard(base_dir)
    progress_bar.progress(100)
    status_text.text("¡Simulación completada con éxito!")
    
    # 4. Read generated HTML dashboard and embed it
    html_path = base_dir / "plotly_animacion.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        st.markdown("### 📊 Animación del Espacio Físico y Curvas SEIRS-D")
        st.components.v1.html(html_content, height=750, scrolling=True)
    else:
        st.error("No se pudo compilar el archivo visual interactivo.")
else:
    st.info("Ajusta los parámetros en la barra lateral y presiona 'Ejecutar Simulación' para visualizar los resultados.")
