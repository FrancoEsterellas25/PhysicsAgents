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

# 1. PARÁMETROS DEL ESPACIO FISICO Y SIMULACIÓN
N_AGENTES = 1600
L_ESPACIO = 100.0
DIAS_SIMULACION = 60
SEMILLA_INICIAL = 42
AGENTES_INFECTADOS_INICIALES = 10
PASO_TIEMPO_DT = 1/48     # Paso de tiempo fraccional (1/48 dias = 30 minutos)

# 2. PARÁMETROS ESPECÍFICOS DEL ENFOQUE CONTINUO
DECAIMIENTO_DELTA = 0.5   # Tasa de degradación ambiental del virus
LONGITUD_ELL = 1.0        # Radio de propagación del aerosol
DIFUSIVIDAD_BASAL = 1.0   # Movilidad de agentes sanos
DIFUSIVIDAD_MIN = 0.05    # Movilidad mínima (agentes severamente enfermos)
V_SINTOMAS = 0.5          # Carga viral al 50% de inhibición motora
EXPO_N_MOV = 2.0          # Agudeza de inhibición cinemática

# 3. PARÁMETROS CLÍNICOS DEL NÚCLEO BIOLÓGICO
MU_R = 180.0             # Media de días de inmunidad tras recuperación
M_R = 150.0              # Moda de días de inmunidad
LAMBDA_LETALIDAD = 5.0   # Sensibilidad al estrés biológico neto
W1 = 0.6                 # Peso de la carga viral acumulada (AUC)
W2 = 0.4                 # Peso del tiempo crónico de enfermedad

# 4. CONFIGURACIÓN DE MANIM (RENDERIZADO VISUAL)
CALIDAD = 'l'            # Calidades: 'l' (Low), 'm' (Medium), 'h' (High)
ABRIR_VIDEO_AL_TERMINAR = True

# =====================================================================
# MOTOR DE EJECUCIÓN (NO TOCAR SI NO SABES LO QUE HACES)
# =====================================================================
def main():
    base_dir = Path(__file__).parent
    
    print("===================================================")
    print(" INICIANDO SIMULACIÓN SEIRS-D CONTINUA")
    print("===================================================")
    
    # 1. Instanciar y configurar el simulador con los hiperparámetros
    sim = ContinuousSEIRSDSimulation(
        N=N_AGENTES,
        L=L_ESPACIO,
        t_max=DIAS_SIMULACION
    )
    
    # Sobreescribir hiperparámetros de las físicas
    sim.dt = PASO_TIEMPO_DT
    sim.delta = DECAIMIENTO_DELTA
    sim.ell = LONGITUD_ELL
    sim.D_basal = DIFUSIVIDAD_BASAL
    sim.D_min = DIFUSIVIDAD_MIN
    sim.V_sint = V_SINTOMAS
    sim.n_mov = EXPO_N_MOV
    
    sim.mu_R = MU_R
    sim.M_R = M_R
    sim.lam = LAMBDA_LETALIDAD
    sim.w1 = W1
    sim.w2 = W2
    
    print(f"Dimensión del espacio continuo: {L_ESPACIO}x{L_ESPACIO}")
    print(f"Población: {N_AGENTES} agentes (Siembra inicial: {AGENTES_INFECTADOS_INICIALES})...")
    print(f"Paso de tiempo continuo: dt = {PASO_TIEMPO_DT} días ({PASO_TIEMPO_DT * 24:.1f} horas)")
    
    # Ejecutar motor estocástico
    sim.run(output_dir=base_dir, n_seed=AGENTES_INFECTADOS_INICIALES, seed=SEMILLA_INICIAL)
    print(" Archivos Telemetría (.parquet) generados con éxito.\n")
    
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
