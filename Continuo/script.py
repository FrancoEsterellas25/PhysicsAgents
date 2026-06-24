import os
import sys
from pathlib import Path

# Añadir la carpeta raíz al path para que Python encuentre la carpeta 'core'
sys.path.append(str(Path(__file__).parent.parent))

from continuous_simulation import ContinuousSEIRSDSimulation

def main():
    base_dir = Path(__file__).parent
    print("===================================================")
    print(" INICIANDO SIMULACIÓN SEIRS-D CONTINUA")
    print("===================================================")
    
    # 1. Ejecutar motor estocástico continuo
    sim = ContinuousSEIRSDSimulation(
        N=1600, 
        L=100.0, 
        t_max=60
    )
    
    sim.run(output_dir=base_dir, n_seed=10, seed=42)
    print(" Archivos Telemetría (.parquet) generados con éxito.\n")

if __name__ == "__main__":
    main()
