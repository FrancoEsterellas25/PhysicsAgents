import os
import sys
import subprocess
from pathlib import Path

# Añadir la carpeta raíz al path para que Python encuentre la carpeta 'core'
sys.path.append(str(Path(__file__).parent.parent))

from discrete_simulation import DiscreteSEIRSDSimulation

# =====================================================================
# 🎛️ PANEL DE CONTROL - HIPERPARÁMETROS DE SIMULACIÓN Y RENDERIZADO
# =====================================================================

# 1. PARÁMETROS DE LA GRILLA Y SIMULACIÓN
GRID_COLS = 40
GRID_ROWS = 40
TOPOLOGIA = 'hexagonal'  # Opciones: 'hexagonal', 'moore', 'von_neumann'
DIAS_SIMULACION = 60
SEMILLA_INICIAL = 42
AGENTES_INFECTADOS_INICIALES = 16 # ponytail: reduced initial seed to 1% of population (16 out of 1600)

# 2. PARÁMETROS DEL ENFOQUE DISCRETO (Mecanismo de Contagio - Función de Hill)
V_50_BASAL = 1.5         # ponytail: lowered from 3.0 to compensate for lower initial seed and allow transmission
GAMMA_HILL = 1.5         # Importancia de la barrera innata
N_HILL = 2.0             # Agudeza de la curva sigmoide de transmisión
K_E = 2                  # Etapas de incubación (Binomial Negativa)
P_E = 0.5                # Probabilidad geométrica en incubación

# 3. PARÁMETROS CLÍNICOS DEL NÚCLEO BIOLÓGICO
MU_R = 180.0             # Media de días de inmunidad tras recuperación (6 meses)
M_R = 150.0              # Moda de días de inmunidad
LAMBDA_LETALIDAD = 5.0   # Sensibilidad al estrés biológico neto
W1 = 0.6                 # Peso de la carga viral acumulada en el estrés (AUC)
W2 = 0.4                 # Peso del tiempo crónico de enfermedad en el estrés

# 4. CONFIGURACIÓN DE MANIM (RENDERIZADO VISUAL)
# Calidades de Manim:
# 'l' (Low - 480p) -> Muy rápido para testing
# 'm' (Medium - 720p) -> Ideal para ver bien
# 'h' (High - 1080p) -> Alta calidad HD
CALIDAD = 'l'            
ABRIR_VIDEO_AL_TERMINAR = True
RENDERIZAR_MANIM = False  # ponytail: set to False to skip slow Manim video render and only generate Plotly HTML

# 5. PARÁMETROS DE LAS EXTENSIONES (PARTE VIII)
T_TRIGGER_INTERVENCION = 10.0  # Día en que se activan cuarentenas/distanciamiento
EFICACIA_CUARENTENA_PQ = 0.8  # Eficacia del sistema de detección y aislamiento
UMBRAL_SINTOMAS_VSINT = 0.5   # Umbral sintomático (v_sint)
CUMPLIMIENTO_DISTANCIA_C_DS = 0.6 # Cumplimiento de distancia social
EFICACIA_DISTANCIA_ETA = 1.5   # Eficacia biológica de distancia social (eta)

# =====================================================================
# MOTOR DE EJECUCIÓN (NO TOCAR SI NO SABES LO QUE HACES)
# =====================================================================
def main():
    base_dir = Path(__file__).parent
    
    print("===================================================")
    print(" INICIANDO SIMULACIÓN SEIRS-D DISCRETA")
    print("===================================================")
    
    # 1. Aplicar parámetros y correr la simulación matemática
    sim = DiscreteSEIRSDSimulation(
        grid_size=(GRID_ROWS, GRID_COLS), 
        topology=TOPOLOGIA, 
        t_max=DIAS_SIMULACION
    )
    
    # Sobreescribir hiperparámetros de las físicas
    sim.V_50_basal = V_50_BASAL
    sim.gamma_hill = GAMMA_HILL
    sim.n_hill = N_HILL
    sim.k_E = K_E
    sim.p_E = P_E
    sim.mu_R = MU_R
    sim.M_R = M_R
    sim.lam = LAMBDA_LETALIDAD
    sim.w1 = W1
    sim.w2 = W2
    
    # ponytail: configure clinical interventions (Part VIII)
    sim.t_trigger = T_TRIGGER_INTERVENCION
    sim.p_Q = EFICACIA_CUARENTENA_PQ
    sim.v_sint = UMBRAL_SINTOMAS_VSINT
    sim.C_DS = CUMPLIMIENTO_DISTANCIA_C_DS
    sim.eta = EFICACIA_DISTANCIA_ETA
    
    print(f"Topología Geográfica: {TOPOLOGIA.capitalize()}")
    print(f"Resolviendo espacio: {GRID_ROWS}x{GRID_COLS} ({GRID_ROWS*GRID_COLS} agentes)...")
    
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
    if not RENDERIZAR_MANIM:
        print("\n[INFO] RENDERIZAR_MANIM está configurado en False. Saltando el renderizado de Manim...")
        print("Trayectorias de agentes generadas y dashboard interactivo de Plotly compilado con éxito.\n")
        return
        
    print("===================================================")
    print(f" INICIANDO RENDERIZADO VISUAL EN CALIDAD '{CALIDAD.upper()}'")
    print("===================================================")
    
    # Pasar la topología a animacion.py usando variable de entorno
    topo_visual = "hexagonal" if TOPOLOGIA == 'hexagonal' else "cuadrada"
    os.environ["SIM_TOPOLOGY"] = topo_visual
    
    # Construir el comando CLI para Manim
    animacion_script = str(base_dir / "animacion.py")
    flag_calidad = f"-pq{CALIDAD}" if ABRIR_VIDEO_AL_TERMINAR else f"-q{CALIDAD}"
    
    manim_cmd = [
        "python", "-m", "manim",
        animacion_script,
        "EscenaEpidemiologica",
        flag_calidad,
        "--disable_caching",
        "--format", "mp4"
    ]
    
    try:
        # Ejecutar Manim interactuando directamente con la consola
        subprocess.run(manim_cmd, check=True)
        print("\n Animación completada.")
    except subprocess.CalledProcessError as e:
        print(f"\n Error durante el renderizado de Manim: {e}")
    except FileNotFoundError:
        print("\n Manim no está disponible en la terminal. Instálalo con: pip install manim")

if __name__ == "__main__":
    main()
