import os
import sys
import subprocess
from pathlib import Path

# Añadir la carpeta raíz al path para que Python encuentre la carpeta 'core'
sys.path.append(str(Path(__file__).parent.parent))

from continuous_simulation import ContinuousSEIRSDSimulation

# =====================================================================
# 🎛️ PANEL DE CONTROL CONTINUO - HIPERPARÁMETROS DE SIMULACIÓN Y RENDER
# =====================================================================

# 1. PARÁMETROS DEL ESPACIO FÍSICO Y SIMULACIÓN
N_AGENTES = 500
L_ESPACIO = 50.0          # ponytail: increased L to 50.0 to reduce density (sparse spatial contacts)
DIAS_SIMULACION = 30      # Duración física real de la simulación en días
SEMILLA_INICIAL = 42
AGENTES_INFECTADOS_INICIALES = 5 # ponytail: reduced initial seed to 1% of population
PASO_TIEMPO_DT = 1/96      # dt=0.1 representa 2.4 horas por paso de simulación

# 2. PARÁMETROS ESPECÍFICOS DEL ENFOQUE CONTINUO
DECAIMIENTO_DELTA = 0.5   # Tasa de degradación ambiental del virus
LONGITUD_ELL = 1.5        # Radio de propagación del aerosol
DIFUSIVIDAD_BASAL = 1.0   # Movilidad de agentes sanos
DIFUSIVIDAD_MIN = 0.05    # Movilidad mínima (agentes severamente enfermos)
V_SINTOMAS = 0.5          # Carga viral al 50% de inhibición motora
EXPO_N_MOV = 2.0          # Agudeza de inhibición cinemática

# 3. PARÁMETROS CLÍNICOS DEL NÚCLEO BIOLÓGICO
TAU_MAX = 0.5            # ponytail: reduced to 0.5 to compensate for lower density and allow spread
MU_R = 180.0             # Media de días de inmunidad tras recuperación
M_R = 150.0              # Moda de días de inmunidad
LAMBDA_LETALIDAD = 5.0   # Sensibilidad al estrés biológico neto
W1 = 0.3                 # Peso de la carga viral acumulada (AUC)
W2 = 0.7                 # Peso del tiempo crónico de enfermedad

# 4. CONFIGURACIÓN DE MANIM (RENDERIZADO VISUAL)
CALIDAD = 'l'            # Calidades: 'l' (Low), 'm' (Medium), 'h' (High)
ABRIR_VIDEO_AL_TERMINAR = True

# 5. PARÁMETROS DE LAS EXTENSIONES (PARTE VIII)
HUBS_ACTIVOS = True
T_TRIGGER_INTERVENCION = 10.0  # Día en que se activan cuarentenas/distanciamiento
EFICACIA_CUARENTENA_PQ = 0.8  # Eficacia del sistema de detección y aislamiento
UMBRAL_SINTOMAS_VSINT = 0.5   # Umbral sintomático (v_sint)
CUMPLIMIENTO_DISTANCIA_C_DS = 0.6 # Cumplimiento poblacional de distancia social (C_DS)
EFICACIA_DISTANCIA_ETA_MOV = 0.5 # Eficacia cinemática de la distancia social (eta_mov)

# =====================================================================
# MOTOR DE EJECUCIÓN (NO TOCAR SI NO SABES LO QUE HACES)
# =====================================================================
def main():
    base_dir = Path(__file__).parent
    
    print("===================================================")
    print(" INICIANDO SIMULACIÓN SEIRS-D CONTINUA")
    print("===================================================")
    
    # 1. Calcular número de pasos de simulación necesarios para simular los días deseados
    STEPS_SIMULACION = int(DIAS_SIMULACION / PASO_TIEMPO_DT)
    
    sim = ContinuousSEIRSDSimulation(
        N=N_AGENTES,
        L=L_ESPACIO,
        t_max=STEPS_SIMULACION
    )
    
    # Sobreescribir hiperparámetros de las físicas
    sim.dt = PASO_TIEMPO_DT
    sim.delta = DECAIMIENTO_DELTA
    sim.ell = LONGITUD_ELL
    sim.D_basal = DIFUSIVIDAD_BASAL
    sim.D_min = DIFUSIVIDAD_MIN
    sim.V_sint = V_SINTOMAS
    sim.n_mov = EXPO_N_MOV
    
    sim.tau_max = TAU_MAX
    sim.mu_R = MU_R
    sim.M_R = M_R
    sim.lam = LAMBDA_LETALIDAD
    sim.w1 = W1
    sim.w2 = W2
    
    # ponytail: configure hubs and clinical interventions (Part VIII)
    if HUBS_ACTIVOS:
        sim.H = 2
        import numpy as np
        sim.hubs_coords = np.array([
            [25.0, 25.0],  # Hub 1 (Escuela - Agenda)
            [10.0, 10.0]   # Hub 2 (Plaza - Gravitatorio)
        ], dtype=np.float32)
        sim.hubs_types = ["agenda", "gravitatorio"]
        sim.hubs_lambda = np.array([5.0/7.0, 0.0], dtype=np.float32)  # Frecuencia de visita (tasa: 5 días de 7)
        sim.hubs_alpha = np.array([12.0, 0.0], dtype=np.float32)      # Forma Gamma para estadía
        sim.hubs_beta = np.array([0.5 / 24.0, 0.0], dtype=np.float32)  # Escala Gamma (0.5 h convertida a días)
        sim.hubs_kappa = np.array([0.0, 2.0], dtype=np.float32)
        sim.hubs_ell = np.array([0.0, 8.0], dtype=np.float32)
        sim.hubs_rho = np.array([0.8, 1.0], dtype=np.float32)
        
    sim.t_trigger = T_TRIGGER_INTERVENCION
    sim.p_Q = EFICACIA_CUARENTENA_PQ
    sim.v_sint = UMBRAL_SINTOMAS_VSINT
    sim.C_DS = CUMPLIMIENTO_DISTANCIA_C_DS
    sim.eta_mov = EFICACIA_DISTANCIA_ETA_MOV
    
    print(f"Dimensión del espacio continuo: {L_ESPACIO}x{L_ESPACIO}")
    print(f"Población: {N_AGENTES} agentes (Siembra inicial: {AGENTES_INFECTADOS_INICIALES})...")
    print(f"Paso de tiempo continuo: dt = {PASO_TIEMPO_DT} días ({PASO_TIEMPO_DT * 24:.1f} horas)")
    print(f"Duración: {DIAS_SIMULACION} días físicos ({STEPS_SIMULACION} pasos de simulación / frames)")
    print(f"Paso de tiempo continuo: dt = {PASO_TIEMPO_DT} días ({PASO_TIEMPO_DT * 24:.1f} horas)")
    
    # Ejecutar motor estocástico
    sim.run(output_dir=base_dir, n_seed=AGENTES_INFECTADOS_INICIALES, seed=SEMILLA_INICIAL)
    print(" Archivos Telemetría (.parquet) generados con éxito.\n")
    
    # ponytail: Generar animación interactiva Plotly HTML
    print("===================================================")
    print(" INICIANDO GENERACIÓN DE ANIMACIÓN INTERACTIVA PLOTLY HTML")
    print("===================================================")
    plotly_script = str(base_dir / "plotly_animacion.py")
    try:
        subprocess.run(["python", plotly_script], check=True)
        print(" Archivo HTML de Plotly generado con éxito.\n")
    except Exception as e:
        print(f" Error generando animación Plotly: {e}\n")
    
    # 2. Configurar el renderizado visual de Manim
    print("===================================================")
    print(f" INICIANDO RENDERIZADO VISUAL EN CALIDAD '{CALIDAD.upper()}'")
    print("===================================================")
    
    animacion_script = str(base_dir / "animacion.py")
    flag_calidad = f"-pq{CALIDAD}" if ABRIR_VIDEO_AL_TERMINAR else f"-q{CALIDAD}"
    
    manim_cmd = [
        "python", "-m", "manim",
        animacion_script,
        "EscenaEpidemiologicaContinuo",
        flag_calidad,
        "--disable_caching",
        "--format", "mp4"
    ]
    
    try:
        subprocess.run(manim_cmd, check=True)
        print("\n Animación completada.")
    except subprocess.CalledProcessError as e:
        print(f"\n Error durante el renderizado de Manim: {e}")
    except FileNotFoundError:
        print("\n Manim no está disponible en la terminal. Instálalo con: pip install manim")

if __name__ == "__main__":
    main()
