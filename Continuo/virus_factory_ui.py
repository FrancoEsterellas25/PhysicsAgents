"""
================================================================================
virus_factory_ui.py  |  Fragmento Streamlit para Continuo/app.py
================================================================================
Inserta el bloque del selectbox "Virus Predefinido" DENTRO del expander
"Parametros Fisicos del Virus" de Continuo/app.py, ANTES de los sliders
existentes (lineas ~83-86).

INSTRUCCIONES DE INTEGRACION:
-------------------------------
En Continuo/app.py, reemplaza el bloque:

    # Section 5: Parametros Clinicos/Virus
    with st.sidebar.expander("Parametros Fisicos del Virus", expanded=False):
        tau_max = st.slider(...)
        ell     = st.slider(...)
        delta_ext = st.slider(...)

Por el siguiente bloque completo (copialo entero):
================================================================================
"""

# ── 1. Importar el catalogo al inicio del archivo (junto a los otros imports) ─

# from virus_factory import VIRUS_CATALOG, VirusProfile, apply_to_simulation, list_diseases, PARAMETER_REFERENCE


# ── 2. Reemplazar la Section 5 en Continuo/app.py por este bloque ──────────

# ============================================================================
# COPIAR DESDE AQUI
# ============================================================================

import streamlit as st
from virus_factory import (
    VIRUS_CATALOG,
    VirusProfile,
    apply_to_simulation,
    list_diseases,
    PARAMETER_REFERENCE,
)

# Mapa: clave interna -> etiqueta visible en el selectbox
_VIRUS_LABELS: dict[str, str] = {
    "custom"        : "Personalizado  (ajustar sliders manualmente)",
    "measles"       : "Sarampion  |  R0~15  |  aerosol fino  |  vitalicio",
    "covid19_delta" : "COVID-19 Delta  |  R0~5.5  |  aerosol mixto  |  waning 4m",
    "influenza"     : "Influenza Estacional  |  R0~1.3  |  gotas  |  CFR baja",
    "h1n1_2009" : "Gripe Pandémica 2009 (porcina)  |  R0~1.5  |  alta transmisión jóvenes  |  vacunación",
    "ebola"         : "Ebola Zaire  |  R0~2  |  contacto directo  |  CFR 50%+",
    "tuberculosis"  : "Tuberculosis  |  R0~10  |  aerosol persistente",
    "hiv" : "VIH/SIDA  |  R0~2-5  |  transmisión fluida  |  fase crónica",
    "sars_2003" : "SARS-CoV-1 (2003)  |  R0~2-3  |  clínico/hospitalario  |  letalidad alta",
    "junin" : "Fiebre Hemorrágica Arg.  |  vectorial/contacto  |  región pampeana  |  suero inmune"
}
_LABEL_TO_KEY: dict[str, str] = {v: k for k, v in _VIRUS_LABELS.items()}


with st.sidebar.expander("Parametros Fisicos del Virus", expanded=False):

    # --- Selector de virus predefinido ------------------------------------
    st.markdown("**Virus Predefinido**")
    selected_label: str = st.selectbox(
        label="Patogeno",
        options=list(_VIRUS_LABELS.values()),
        index=0,
        key="virus_preset",
        help=(
            "Carga automaticamente tau_max, ell, delta_ext, delta_cerrado, "
            "delta_abierto, lam, k_E, p_E, mu_R y M_R calibrados "
            "desde la literatura medica. Elige 'Personalizado' para "
            "ajustar los sliders manualmente."
        ),
    )

    selected_key: str = _LABEL_TO_KEY[selected_label]
    is_custom: bool   = (selected_key == "custom")

    # Perfil activo (None si es personalizado)
    active_profile: VirusProfile | None = (
        None if is_custom else VIRUS_CATALOG[selected_key]
    )

    # Tarjeta informativa del patogeno
    if active_profile is not None:
        inc_media = active_profile.mean_incubation_days
        st.markdown(
            f"**{active_profile.name}**  \n"
            f"R0 ref: `{active_profile.r0_ref}`  \n"
            f"CFR ref: `{active_profile.cfr_ref}`  \n"
            f"Incubacion media: `{inc_media:.1f} dias`  \n"
            f"Inmunidad: `{active_profile.mu_R:.0f} dias`"
        )
        st.markdown("---")

    # Helper: valor del perfil o fallback para slider
    def _val(attr: str, fallback: float) -> float:
        if active_profile is not None:
            return float(getattr(active_profile, attr))
        return fallback

    # --- Sliders (deshabilitados si hay preset activo) ---------------------
    tau_max: float = st.slider(
        "Dosis Tolerancia Maxima (tau_max)",
        min_value=0.5,
        max_value=50.0,
        value=_val("tau_max", 20.0),
        step=0.5,
        disabled=not is_custom,
        key="slider_tau_max",
        help="Umbral de dosis acumulada para infectarse. Bajo = muy contagioso.",
    )

    ell: float = st.slider(
        "Radio Aerosol (ell, metros)",
        min_value=0.1,
        max_value=6.0,
        value=_val("ell", 1.0),
        step=0.1,
        disabled=not is_custom,
        key="slider_ell",
        help="Radio del kernel gaussiano de dispersion viral.",
    )

    delta_ext: float = st.slider(
        "Decaimiento Aerosol — Transito/Calles (delta_ext, /dia)",
        min_value=0.01,
        max_value=15.0,
        value=_val("delta_ext", 1.0),
        step=0.05,
        disabled=not is_custom,
        key="slider_delta_ext",
        help="Fraccion de carga viral que decae por dia en exteriores/transito.",
    )

    delta_cerrado: float = st.slider(
        "Decaimiento Aerosol — Espacios Cerrados (delta_cerrado, /dia)",
        min_value=0.01,
        max_value=10.0,
        value=_val("delta_cerrado", 0.2),
        step=0.05,
        disabled=not is_custom,
        key="slider_delta_cerrado",
        help="Decaimiento en hogar, escuela, oficina (menor ventilacion).",
    )

    delta_abierto: float = st.slider(
        "Decaimiento Aerosol — Espacios Abiertos (delta_abierto, /dia)",
        min_value=0.1,
        max_value=20.0,
        value=_val("delta_abierto", 4.0),
        step=0.1,
        disabled=not is_custom,
        key="slider_delta_abierto",
        help="Decaimiento en plazas, mercados, parques (UV, viento).",
    )

    lam: float = st.slider(
        "Pendiente de Letalidad (lambda)",
        min_value=0.5,
        max_value=20.0,
        value=_val("lam", 5.0),
        step=0.5,
        disabled=not is_custom,
        key="slider_lam",
        help="Controla que tan rapido sube la mortalidad con el estres viral.",
    )

    st.markdown("**Incubacion NegBin(k_E, p_E)**")
    col_ke, col_pe = st.columns(2)
    with col_ke:
        k_E: int = int(st.number_input(
            "k_E", min_value=1, max_value=20,
            value=int(_val("k_E", 2)),
            step=1, disabled=not is_custom, key="slider_k_E",
        ))
    with col_pe:
        p_E: float = st.number_input(
            "p_E", min_value=0.05, max_value=0.99,
            value=_val("p_E", 0.5),
            step=0.01, format="%.2f",
            disabled=not is_custom, key="slider_p_E",
        )
    mean_inc_calc = k_E * (1.0 - p_E) / p_E
    st.caption(f"Incubacion media: {mean_inc_calc:.1f} dias")

    mu_R: float = st.slider(
        "Inmunidad Adquirida — Media (mu_R, dias)",
        min_value=1.0, max_value=36500.0,
        value=_val("mu_R", 180.0),
        step=1.0, format="%.0f",
        disabled=not is_custom, key="slider_mu_R",
    )

    M_R: float = st.slider(
        "Inmunidad Adquirida — Cap (M_R, dias)",
        min_value=1.0, max_value=36500.0,
        value=_val("M_R", 150.0),
        step=1.0, format="%.0f",
        disabled=not is_custom, key="slider_M_R",
    )

    # Tabla de referencia expandible
    with st.expander("Tabla de referencia paramétrica", expanded=False):
        st.code(PARAMETER_REFERENCE, language=None)

# ============================================================================
# FIN DEL BLOQUE A COPIAR
# ============================================================================


# ── 3. En el bloque "if st.button" (linea ~142 de app.py) ──────────────────
#    Consolida los parametros activos (preset o sliders) y llama a
#    apply_to_simulation ANTES de sim.run():
#
#    Reemplaza las lineas:
#        sim.tau_max   = tau_max
#        sim.ell       = ell
#        sim.delta_ext = delta_ext
#
#    Por:

def _aplicar_virus(sim, active_profile, tau_max, ell, delta_ext,
                   delta_cerrado, delta_abierto, lam, k_E, p_E, mu_R, M_R):
    """
    Inyecta parametros de virus en el simulador.
    Usa el perfil si hay preset activo; usa los sliders si es personalizado.
    """
    if active_profile is not None:
        apply_to_simulation(sim, active_profile)
    else:
        sim.tau_max        = tau_max
        sim.ell            = ell
        sim.delta_ext      = delta_ext
        sim.delta_cerrado  = delta_cerrado
        sim.delta_abierto  = delta_abierto
        sim.lam            = lam
        sim.k_E            = k_E
        sim.p_E            = p_E
        sim.mu_R           = mu_R
        sim.M_R            = M_R

# ── 4. En el bloque de ejecucion (linea ~148 en app.py), AGREGA esta llamada:
#
#    sim = ContinuousSEIRSDSimulation(N=N_agentes, L=L_espacio, t_max=...)
#    ...otros parametros de comportamiento...
#    _aplicar_virus(sim, active_profile, tau_max, ell, delta_ext,
#                   delta_cerrado, delta_abierto, lam, k_E, p_E, mu_R, M_R)
#    sim.run(output_dir=base_dir, n_seed=seed_I, seed=42)
