import numpy as np
import polars as pl
from pathlib import Path
from core.base_simulation import BaseSEIRSDSimulation
from core.biology import generar_perfiles_inmunes

class ManimSEIRSDSimulation(BaseSEIRSDSimulation):
    def __init__(self, grid_size=(50, 50), topology='moore', t_max=30):
        """
        Motor de simulación estocástica ABM - SEIRS-D (Enfoque Discreto)
        Hereda el núcleo común de BaseSEIRSDSimulation y añade la lógica discreta en grilla.
        """
        self.grid_size = grid_size
        N = grid_size[0] * grid_size[1]
        super().__init__(N=N, t_max=t_max)
        
        self.topology = topology
        
        # Parámetros específicos del Enfoque Discreto
        self.V_50_basal = 3.0
        self.gamma_hill = 1.5
        self.n_hill = 2.0
        
        # Coordenadas de la grilla
        y_idx, x_idx = np.indices(grid_size)
        self.coord_x = x_idx.flatten().astype(np.int16)
        self.coord_y = y_idx.flatten().astype(np.int16)

    def _fase0_inicializacion(self, output_dir=None):
        """Cópula Gaussiana y Exportación del Mapa Estático con coordenadas."""
        self.omega_in, self.omega_ad = generar_perfiles_inmunes(
            self.N, self.rho,
            self.beta_in_a, self.beta_in_b,
            self.beta_ad_a, self.beta_ad_b
        )
        
        # Exportar Mapa Estático con Coordenadas (Específico de Discreto)
        base_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        df_estatico = pl.DataFrame({
            "id_agente": self.id_agente,
            "coord_x": self.coord_x,
            "coord_y": self.coord_y,
            "omega_in": self.omega_in,
            "omega_ad": self.omega_ad
        })
        df_estatico.write_parquet(base_dir / "mapa_estatico.parquet", compression="snappy")
        print(f"Exportado {base_dir / 'mapa_estatico.parquet'}")

    def _fase2_contagio(self):
        """Contagio S -> E basado en matrices desplazadas (vecindad)."""
        if self.topology == 'moore':
            dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        elif self.topology == 'von_neumann':
            dirs = [(-1,0), (1,0), (0,-1), (0,1)]
        else: # Hexagonal (Simplified Even-r representation)
            dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,1)]
            
        state_2d = self.state.reshape(self.grid_size)
        vl_2d = self.viral_load.reshape(self.grid_size)
        w_in_2d = self.omega_in.reshape(self.grid_size)
        
        prob_not_inf_2d = np.ones(self.grid_size, dtype=np.float32)
        
        # V50 dinámico
        v50_mat = self.V_50_basal * (1.0 + self.gamma_hill * w_in_2d)
        
        for dr, dc in dirs:
            shifted_state = np.roll(state_2d, shift=(dr, dc), axis=(0, 1))
            shifted_vl = np.roll(vl_2d, shift=(dr, dc), axis=(0, 1))
            
            mask_inf = (shifted_state == self.I)
            v_j = shifted_vl
            
            hill_sigma = (v_j**self.n_hill) / (v_j**self.n_hill + v50_mat**self.n_hill)
            
            # Solo aplica si el vecino estaba infectado
            prob_not_inf_2d[mask_inf] *= (1.0 - hill_sigma[mask_inf])
            
        p_not_inf = prob_not_inf_2d.flatten()
        
        mask_S = (self.state == self.S)
        becomes_E = mask_S & (np.random.rand(self.N) < (1.0 - p_not_inf))
        
        self.state[becomes_E] = self.E
        # X ~ NegBin(k=2, p=0.5)
        self.time_in_E[becomes_E] = np.random.negative_binomial(2, 0.5, size=np.sum(becomes_E))

    def run(self):
        """Bucle de ejecución adaptado para guardar los datos en la carpeta local de Discreto."""
        base_dir = Path(__file__).parent
        # Ejecutamos el pipeline común y guardamos telemetría en base_dir
        super().run(output_dir=base_dir)

if __name__ == '__main__':
    print("Iniciando Motor ABM-Manim (Discreto Modularizado)...")
    sim = ManimSEIRSDSimulation(grid_size=(40, 40), t_max=100)
    sim.run()
    print("Frames pre-calculados exitosamente para Manim.")

