import os
import sys
import time
import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root directory to path to resolve core imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Continuo.continuous_simulation import ContinuousSEIRSDSimulation

def saltelli_sample(problem, N):
    D = len(problem['names'])
    bounds = np.array(problem['bounds'])
    
    A = np.random.rand(N, D)
    B = np.random.rand(N, D)
    
    A_scaled = bounds[:, 0] + A * (bounds[:, 1] - bounds[:, 0])
    B_scaled = bounds[:, 0] + B * (bounds[:, 1] - bounds[:, 0])
    
    C_list = []
    for i in range(D):
        C = np.copy(B_scaled)
        C[:, i] = A_scaled[:, i]
        C_list.append(C)
        
    return A_scaled, B_scaled, C_list

def correr_simulacion(params, N_agentes=400, t_max=15):
    rho, lam, w1, delta_ext = params
    sim = ContinuousSEIRSDSimulation(N=N_agentes, t_max=t_max)
    
    # Configure simulation parameters to match Streamlit defaults
    sim.dt = 0.2
    sim.tau_max = 0.5       # Crucial threshold calibration
    sim.ell = 1.5           # Aerosol scale
    sim.eta_em = 0.6
    sim.eta_rec = 0.5
    sim.barbijo_cumplimiento = 0.0
    sim.eta_hig = 0.0
    
    # Sensitivity variables
    sim.rho = rho
    sim.lam = lam
    sim.w1 = w1
    sim.w2 = 1.0 - w1
    sim.delta_ext = delta_ext
    sim.hubs_activos = False
    sim.movimiento_libre = True
    
    sim.run(output_dir=None, n_seed=10, seed=42)
    
    # Measure mortality rate (D / N) to involve both infection and death biology
    muertos_final = np.sum(sim.state == sim.D)
    mortality_rate = muertos_final / sim.N
    return mortality_rate

def analizar_sobol(A_y, B_y, C_y_list):
    """
    Calcula los índices de Sobol utilizando el estimador robusto de Jansen (1999).
    """
    D = len(C_y_list)
    N = len(A_y)
    
    all_y = np.concatenate([A_y, B_y])
    var_total = np.var(all_y)
    if var_total == 0:
        var_total = 1e-10
        
    S_i = []
    S_Ti = []
    
    for i in range(D):
        # Estimador Jansen para Efecto Total (STi)
        v_ti = (1.0 / (2.0 * N)) * np.sum((A_y - C_y_list[i])**2)
        S_Ti.append(v_ti / var_total)
        
        # Estimador Jansen para Primer Orden (Si)
        v_i = var_total - (1.0 / (2.0 * N)) * np.sum((B_y - C_y_list[i])**2)
        S_i.append(v_i / var_total)
        
    return np.array(S_i), np.array(S_Ti)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15, help="Saltelli sample size N")
    args = parser.parse_args()
    
    N = args.N
    print(f"[Sobol] Iniciando Análisis de Sensibilidad de Mortalidad (Jansen) con N={N}...")
    t0 = time.time()
    
    problem = {
        'names': ['Cópula (rho)', 'Pendiente Letalidad (lam)', 'Peso Estrés (w1)', 'Decaimiento Viral (delta_ext)'],
        'bounds': [[0.2, 0.8], [3.0, 7.0], [0.3, 0.7], [0.5, 2.5]]
    }
    
    D = len(problem['names'])
    
    A, B, C_list = saltelli_sample(problem, N)
    
    A_y = np.zeros(N)
    B_y = np.zeros(N)
    C_y_list = [np.zeros(N) for _ in range(D)]
    
    for k in range(N):
        A_y[k] = correr_simulacion(A[k])
        
    for k in range(N):
        B_y[k] = correr_simulacion(B[k])
        
    for i in range(D):
        for k in range(N):
            C_y_list[i][k] = correr_simulacion(C_list[i][k])
            
    S_i, S_Ti = analizar_sobol(A_y, B_y, C_y_list)
    
    S_i = np.clip(S_i, 0.0, 1.0)
    S_Ti = np.clip(S_Ti, 0.0, 1.0)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0F1117")
    ax.set_facecolor("#161B22")
    
    x = np.arange(D)
    width = 0.35
    
    ax.bar(x - width/2, S_i, width, label='Efecto Directo (Primer Orden - S_i)', color='teal')
    ax.bar(x + width/2, S_Ti, width, label='Efecto Total (Interacciones - S_Ti)', color='crimson')
    
    ax.set_title(f"Sensibilidad Global de Sobol (Jansen N={N}) - Tasa de Mortalidad", color="white", fontsize=14, pad=15)
    ax.set_ylabel("Índices de Sensibilidad (Varianza Relativa)", color="#CCCCCC", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(problem['names'], color="white", fontsize=10, rotation=15)
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
        
    ax.legend(facecolor="#1A1F2E", edgecolor="#444444", labelcolor="white")
    ax.grid(True, color="#1E2530", linestyle="--", alpha=0.5)
    
    out_dir = Path(__file__).parent
    fig_path = out_dir / "analisis_sobol.png"
    plt.savefig(fig_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    
    elapsed = time.time() - t0
    print(f"[Sobol] Análisis completado en {elapsed:.2f} s. Gráfico guardado en {fig_path}")

if __name__ == "__main__":
    main()
