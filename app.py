import streamlit as st
import numpy as np
import polars as pl
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add project root directory to sys.path to resolve folder imports
root_path = Path(__file__).resolve().parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "Continuo"))
sys.path.append(str(root_path / "Discreto"))

# Import simulations and animators
from Continuo.continuous_simulation import ContinuousSEIRSDSimulation
from Discreto.discrete_simulation import DiscreteSEIRSDSimulation
import Continuo.plotly_animacion as plotly_continuo
import Discreto.plotly_animacion as plotly_discreto

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
    <h1 style="color: #FFFFFF; margin: 0;">🦠 Simulador Epidemiológico SEIRS-D</h1>
    <p style="color: #B0B0B0; margin: 10px 0 0 0; font-size: 1.1rem;">
        Modelado estocástico multinivel de propagación viral. Elija entre el enfoque discreto en grilla o el enfoque físico espacial continuo.
    </p>
</div>
""", unsafe_allow_html=True)

# Main selector for the model approach
enfoque = st.sidebar.selectbox(
    "🔍 Enfoque del Modelo",
    options=["Continuo (Espacio Físico / Langevin)", "Discreto (Grilla / Autómata Celular)"]
)

st.sidebar.markdown("### 🎛️ Configuración del Escenario")

# Dynamic configuration based on selected approach
if enfoque == "Continuo (Espacio Físico / Langevin)":
    with st.sidebar.expander("👥 Demografía y Simulación", expanded=True):
        N_agentes = st.slider("Tamaño Población (N)", min_value=100, max_value=1000, value=500, step=50)
        L_espacio = st.slider("Dimensión del Espacio (L)", min_value=20.0, max_value=100.0, value=50.0, step=5.0)
        dias_simulacion = st.slider("Duración (Días)", min_value=10, max_value=45, value=25, step=5)
        seed_I = st.slider("Infectados Iniciales", min_value=1, max_value=20, value=5, step=1)

    with st.sidebar.expander("🚶 Movilidad y Contactos", expanded=True):
        mov_tipo = st.selectbox(
            "Patrón de Movimiento",
            options=["Hubs Cerrados/Abiertos (Escuela, Oficina, Súper, Centro)", "Movimiento Browniano Libre (Toda la ciudad)"]
        )
        hubs_activos = (mov_tipo == "Hubs Cerrados/Abiertos (Escuela, Oficina, Súper, Centro)")
        mov_libre = not hubs_activos
        C_DS = st.slider("Distanciamiento Social (Cumplimiento C_DS)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)

    with st.sidebar.expander("🛡️ Medidas de Intervención", expanded=True):
        enable_quarantine = st.checkbox("Activar Cuarentenas Domésticas", value=True)
        quarantine_trigger_pct = st.slider(
            "Gatillo Cuarentena (% Infectados Activos)",
            min_value=0.01, max_value=0.30, value=0.05, step=0.01,
            format="%.2f"
        )

    with st.sidebar.expander("😷 Medidas de Protección Personal", expanded=True):
        barbijo_cumplimiento = st.slider("Cumplimiento Mascarilla (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        eta_em = st.slider("Eficiencia Filtración Emisión (eta_em)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        eta_rec = st.slider("Eficiencia Filtración Recepción (eta_rec)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        eta_hig = st.slider("Higiene Personal (eta_hig)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    with st.sidebar.expander("☣️ Parámetros Físicos del Virus", expanded=False):
        tau_max = st.slider("Dosis Tolerancia Máxima (T_inf)", min_value=0.1, max_value=1.5, value=0.5, step=0.05)
        ell = st.slider("Radio Dispersión Aerosol (ell)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        delta_ext = st.slider("Decaimiento Aerosol en Calles", min_value=0.2, max_value=3.0, value=1.0, step=0.1)

else:
    with st.sidebar.expander("👥 Dimensiones de la Grilla", expanded=True):
        grid_side = st.slider("Lado de la Grilla (Dimensión)", min_value=20, max_value=50, value=30, step=5)
        N_agentes = grid_side * grid_side
        st.info(f"Población total resultante: {N_agentes} agentes")
        topology = st.selectbox("Topología de Vecindad", options=["moore", "von_neumann", "hexagonal"])
        dias_simulacion = st.slider("Duración (Días)", min_value=10, max_value=100, value=45, step=5)
        seed_I = st.slider("Infectados Iniciales", min_value=1, max_value=20, value=5, step=1)

    with st.sidebar.expander("🛡️ Medidas de Intervención", expanded=True):
        C_DS = st.slider("Cumplimiento Distanciamiento Social (C_DS)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        enable_quarantine = st.checkbox("Activar Cuarentenas Domésticas", value=True)
        quarantine_trigger_pct = st.slider(
            "Gatillo Cuarentena (% Infectados Activos)",
            min_value=0.01, max_value=0.30, value=0.05, step=0.01,
            format="%.2f"
        )
        eta_hig = st.slider("Higiene Personal (eta_hig - eleva umbral)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)


# Helper analysis functions
def calcular_kaplan_meier(t_inf_arr, t_max):
    tiempos = np.where(t_inf_arr >= 0, t_inf_arr, t_max)
    eventos = np.where(t_inf_arr >= 0, 1, 0)
    tiempos_unicos = np.unique(tiempos)
    tiempos_unicos = tiempos_unicos[tiempos_unicos >= 0]
    tiempos_unicos.sort()
    
    surv_prob = 1.0
    times = [0.0]
    probs = [1.0]
    for t in tiempos_unicos:
        d_j = np.sum((tiempos == t) & (eventos == 1))
        n_j = np.sum(tiempos >= t)
        if n_j > 0:
            surv_prob *= (1.0 - d_j / n_j)
        times.append(float(t))
        probs.append(float(surv_prob))
    return np.array(times), np.array(probs)

def estimar_r0_exacto(tiempo, infectados, tg):
    ventana = (tiempo >= 1) & (tiempo <= 10) & (infectados > 0)
    x = tiempo[ventana]
    y = np.log(infectados[ventana])
    if len(x) < 2:
        return 1.0, 0.0
    slope, _ = np.polyfit(x, y, 1)
    r = max(0.0, slope)
    r0 = np.exp(r * tg)
    return r0, r

def ajustar_cox_tiempo_variable(df_dinamico, df_estatico):
    df_first_transition = df_dinamico.filter(pl.col("estado").is_in([1, 2])).group_by("id_agente").agg(pl.col("tiempo").min().alias("t_event"))
    
    df_panel = df_dinamico.join(df_first_transition, on="id_agente", how="left")
    t_max = df_dinamico["tiempo"].max()
    df_panel = df_panel.with_columns(
        pl.col("t_event").fill_null(t_max)
    )
    
    df_panel = df_panel.filter(pl.col("tiempo") <= pl.col("t_event"))
    df_panel = df_panel.with_columns(
        pl.when((pl.col("tiempo") == pl.col("t_event")) & (pl.col("estado").is_in([1, 2])))
        .then(1)
        .otherwise(0)
        .alias("evento_infeccion")
    )
    
    df_model = df_panel.join(df_estatico.select(["id_agente", "omega_in"]), on="id_agente", how="left")
    
    X_cols = []
    if "dosis" in df_model.columns:
        X_cols.append("dosis")
    X_cols.append("omega_in")
    
    X = df_model.select(X_cols).to_numpy()
    y = df_model["evento_infeccion"].to_numpy()
    
    if len(np.unique(y)) > 1:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(penalty='l2', C=0.005, solver='lbfgs')
        clf.fit(X_scaled, y)
        coefs = clf.coef_[0]
        hrs = np.exp(coefs)
        return {col: hr for col, hr in zip(X_cols, hrs)}
    return None

# Run Simulation Button
if st.button("🚀 Ejecutar Simulación", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if enfoque == "Continuo (Espacio Físico / Langevin)":
        status_text.text("Inicializando simulación continua...")
        sim = ContinuousSEIRSDSimulation(N=N_agentes, L=L_espacio, t_max=int(dias_simulacion / 0.1))
        
        sim.dt = 0.1
        sim.hubs_activos = hubs_activos
        sim.movimiento_libre = mov_libre
        sim.c_DS = C_DS
        sim.enable_quarantine = enable_quarantine
        sim.quarantine_trigger_pct = quarantine_trigger_pct
        sim.eta_hig = eta_hig
        sim.eta_em = eta_em
        sim.eta_rec = eta_rec
        sim.barbijo_cumplimiento = barbijo_cumplimiento
        sim.tau_max = tau_max
        sim.ell = ell
        sim.delta_ext = delta_ext
        
        base_dir = root_path / "Continuo"
        status_text.text("Simulando movimiento continuo y contagios por aerosol...")
        sim.run(output_dir=base_dir, n_seed=seed_I, seed=42)
        progress_bar.progress(40)
        
        status_text.text("Generando animación espacial...")
        plotly_continuo.generar_dashboard(base_dir)
        html_path = base_dir / "plotly_animacion.html"
    else:
        status_text.text("Inicializando simulación en grilla discreta...")
        sim = DiscreteSEIRSDSimulation(grid_size=(grid_side, grid_side), topology=topology, t_max=dias_simulacion)
        
        sim.dt = 1.0
        sim.c_DS = C_DS
        sim.enable_quarantine = enable_quarantine
        sim.quarantine_trigger_pct = quarantine_trigger_pct
        sim.eta_hig = eta_hig
        
        base_dir = root_path / "Discreto"
        status_text.text("Simulando contagio discreto por autómatas celulares...")
        sim.run(output_dir=base_dir, n_seed=seed_I, seed=42)
        progress_bar.progress(40)
        
        status_text.text("Generando animación de celdas...")
        plotly_discreto.main()
        html_path = base_dir / "plotly_animacion.html"

    progress_bar.progress(70)
    
    # 4. Embed the HTML animation
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        st.markdown(f"### 📊 Visualización Interactiva ({enfoque})")
        st.components.v1.html(html_content, height=720, scrolling=True)
    else:
        st.error("No se pudo compilar el archivo visual interactivo.")
        
    # 5. Calculate and render Data-Oriented metrics
    status_text.text("Procesando métricas avanzadas y análisis de supervivencia...")
    
    df_dinamico = pl.read_parquet(base_dir / "telemetria_dinamica.parquet")
    df_estatico = pl.read_parquet(base_dir / "mapa_estatico.parquet")
    
    tiempos = df_dinamico["tiempo"].unique().sort().to_numpy()
    conteos_I = []
    for t in tiempos:
        df_t = df_dinamico.filter(pl.col("tiempo") == t)
        conteos_I.append(np.sum(df_t["estado"].to_numpy() == 2))
    
    # Generation time
    tg = 2.0 + (1.0 / 0.1)
    r0, r_rate = estimar_r0_exacto(tiempos, np.array(conteos_I), tg=tg)
    
    # Kaplan-Meier survival curves
    df_inf = df_dinamico.filter(pl.col("estado").is_in([1, 2]))
    df_first_inf = df_inf.group_by("id_agente").agg(pl.col("tiempo").min().alias("t_inf"))
    df_survival = df_estatico.join(df_first_inf, on="id_agente", how="left").fill_null(-1)
    
    median_omega_in = df_survival["omega_in"].median()
    df_high_inn = df_survival.filter(pl.col("omega_in") >= median_omega_in)
    df_low_inn = df_survival.filter(pl.col("omega_in") < median_omega_in)
    
    times_high, probs_high = calcular_kaplan_meier(df_high_inn["t_inf"].to_numpy(), tiempos[-1])
    times_low, probs_low = calcular_kaplan_meier(df_low_inn["t_inf"].to_numpy(), tiempos[-1])
    
    # 2-Column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Estimación del $R_0$")
        st.metric(label="Ritmo Básico de Reproducción (R0) Exacto", value=f"{r0:.2f}")
        st.metric(label="Tasa de Crecimiento Exponencial (r)", value=f"{r_rate:.4f} por día")
        st.write(f"**Tiempo de Generación Medio ($T_g$):** {tg:.2f} días")
        st.caption("Estimado usando ajuste exponencial exacto durante la fase de crecimiento exponencial temprano.")

    with col2:
        st.markdown("### 🔬 Modelo de Cox Proporcional")
        hrs = ajustar_cox_tiempo_variable(df_dinamico, df_estatico)
        if hrs:
            st.write("**Hazard Ratios (HR) por Desviación Estándar:**")
            sub_col1, sub_col2 = st.columns(2)
            for i, (cov, hr) in enumerate(hrs.items()):
                label_name = "Dosis Local de Aerosol" if cov == "dosis" else "Vulnerabilidad Inmune Basal"
                if i % 2 == 0:
                    sub_col1.metric(label=f"HR: {label_name}", value=f"{hr:.4f}")
                else:
                    sub_col2.metric(label=f"HR: {label_name}", value=f"{hr:.4f}")
                
            if "dosis" in hrs:
                pct_increase = (hrs["dosis"] - 1.0) * 100
                st.caption(
                    f"**Interpretación Física (Escalada):** Por cada aumento de **1 Desviación Estándar** "
                    f"en la dosis de aerosol respirada localmente, el riesgo instantáneo de contagio se multiplica por "
                    f"**{hrs['dosis']:.2f}** ({pct_increase:+.1f}%), controlando por la inmunidad genética basal del agente."
                )
            else:
                pct_decrease = (1.0 - hrs["omega_in"]) * 100
                st.caption(
                    f"**Interpretación Biológica (Escalada):** Un incremento de **1 Desviación Estándar** en la "
                    f"inmunidad innata del agente reduce el riesgo instantáneo de infección en un "
                    f"**{pct_decrease:.1f}%**, controlando por la vecindad espacial de contactos."
                )
        else:
            st.warning("Eventos insuficientes en la simulación para ajustar el modelo de Cox.")

    # Full-width section for Kaplan-Meier curves
    st.markdown("### ⏱️ Curvas de Supervivencia de Kaplan-Meier")
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor="#0F1117")
    ax.set_facecolor("#161B22")
    ax.step(times_high, probs_high, where='post', label=r'Inmunidad Innata Alta (>= Mediana)', color='teal', lw=2.5)
    ax.step(times_low, probs_low, where='post', label=r'Inmunidad Innata Baja (< Mediana)', color='crimson', lw=2.5)
    ax.set_title("Probabilidad de Permanecer Susceptible (S) a lo largo del Tiempo", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Tiempo (Días)", color="#CCCCCC", fontsize=10)
    ax.set_ylabel("Probabilidad de Supervivencia S(t)", color="#CCCCCC", fontsize=10)
    ax.tick_params(colors="#AAAAAA", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.legend(facecolor="#1A1F2E", edgecolor="#444444", labelcolor="white", fontsize=10, loc="upper right")
    ax.grid(True, color="#1E2530", linewidth=0.5, linestyle="--")
    st.pyplot(fig)
        
    # Export spatiotemporal panel dataset
    cols_to_select = [c for c in ["id_agente", "omega_in", "omega_ad", "edad"] if c in df_estatico.columns]
    df_multidimensional = df_dinamico.join(df_estatico.select(cols_to_select), on="id_agente", how="left")
    csv_data = df_multidimensional.to_pandas().to_csv(index=False).encode('utf-8')
    
    st.markdown("### 💾 Descarga de Datos de la Simulación")
    st.download_button(
        label=f"📥 Descargar Dataset Panel Multidimensional ({enfoque}) (CSV)",
        data=csv_data,
        file_name=f"trayectorias_y_biologia_{enfoque.split()[0].lower()}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    progress_bar.progress(100)
    status_text.text("¡Simulación y análisis completados con éxito!")
else:
    st.info("Ajusta los parámetros en la barra lateral y presiona 'Ejecutar Simulación' para visualizar los resultados.")

# Add full-width section for Sobol Sensitivity Analysis at the bottom of the page
st.markdown("---")
st.markdown("### 🔬 Análisis de Sensibilidad Global (Sobol)")
st.write(
    "El Análisis de Sobol evalúa la influencia de los **rangos completos** de los parámetros "
    "(inmunidad de la población, letalidad y decaimiento ambiental) sobre el impacto final de la epidemia (Tasa de Mortalidad). "
    "Al medir la sensibilidad global de los rangos, los índices **no cambian al mover un control de simulación individual**, "
    "sino que te indican cuáles de ellos son los más críticos de calibrar para el sistema."
)

# Display precalculated high-precision plot by default if it exists
sobol_img_path = root_path / "Data-Oriented" / "analisis_sobol.png"
if sobol_img_path.exists():
    st.markdown("#### 📊 Resultados Precalculados de Alta Precisión (Monte Carlo - N=250)")
    st.image(
        str(sobol_img_path),
        caption="Gráfico de Sensibilidad de Sobol Precalculado (1500 simulaciones). Las barras demuestran la influencia de los rangos de cada parámetro en la varianza de la tasa de ataque final.",
        use_container_width=True
    )
    
    # Premium explanation block for Sobol results
    st.markdown("""
    <div style="background-color: #1E2530; padding: 20px; border-radius: 8px; border-left: 4px solid #008080; margin-top: 15px;">
        <h4 style="color: #FFFFFF; margin-top: 0;">📖 Interpretación Científica de los Resultados de Sobol:</h4>
        <ul style="color: #DDDDDD; font-size: 0.95rem; line-height: 1.6;">
            <li><b>Efecto Directo (S_i - Barra Teal/Verde):</b> Mide la proporción de la varianza en la mortalidad explicada por cada parámetro de forma aislada.
                <ul>
                    <li>El <b>Decaimiento Viral</b> y la <b>Pendiente de Letalidad</b> tienen efectos directos detectables porque regulan directamente el volumen de contagio y la mortalidad básica.</li>
                </ul>
            </li>
            <li><b>Efecto Total (S_Ti - Barra Crimson/Roja):</b> Mide el impacto acumulado del parámetro incluyendo todas sus <b>interacciones no lineales</b> con los demás.</li>
            <li><b>Comportamiento de la Cópula (rho) (S_i ≈ 0, S_Ti ≈ 1.0):</b> 
                Representa una <b>interacción pura de acoplamiento</b>. Cambiar la correlación de la inmunidad de la población no tiene ningún impacto directo si el virus no se propaga o si no es letal. Su relevancia biológica solo despierta en escenarios combinados con alta carga de virus y alta letalidad.
            </li>
            <li><b>Dominancia de la Biología e Interacción:</b> Las variables de letalidad y ambiente saturan su efecto total porque en un modelo dinámico espacial continuo, la mortalidad resultante es un fenómeno emergente no lineal que depende críticamente de la superposición de todos los factores.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No se encontró una gráfica de Sobol precalculada.")
