import numpy as np
import polars as pl
from pathlib import Path
from core.base_simulation import BaseSEIRSDSimulation
from core.biology import generar_perfiles_inmunes

class DiscreteSEIRSDSimulation(BaseSEIRSDSimulation):
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
        self.v50 = None
        
        # Coordenadas de la grilla
        y_idx, x_idx = np.indices(grid_size)
        self.coord_x = x_idx.flatten().astype(np.int16)
        self.coord_y = y_idx.flatten().astype(np.int16)
        
        # Máscaras Estructurales para Topología Hexagonal (Row-Offset)
        self.even_row_mask = (y_idx % 2 == 0)
        self.odd_row_mask = (y_idx % 2 == 1)

    def _fase0_inicializacion(self, output_dir=None):
        """Cópula Gaussiana y Exportación del Mapa Estático con coordenadas."""
        super()._fase0_inicializacion(output_dir=output_dir)
        self.v50 = self.V_50_basal * (1.0 + self.gamma_hill * self.omega_in) * (1.0 + getattr(self, 'eta_hig', 0.0))
        
        # Exportar Mapa Estático con Coordenadas (Específico de Discreto)
        base_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        df_estatico = pl.DataFrame({
            "id_agente": self.id_agente,
            "coord_x": self.coord_x,
            "coord_y": self.coord_y,
            "omega_in": self.omega_in,
            "omega_ad": self.omega_ad,
            "v50": self.v50
        })
        df_estatico.write_parquet(base_dir / "mapa_estatico.parquet", compression="snappy")
        print(f"Exportado {base_dir / 'mapa_estatico.parquet'}")

    def _fase2_contagio(self):
        """Contagio S -> E basado en matrices desplazadas (vecindad) con soporte topológico estricto."""
        state_2d = self.state.reshape(self.grid_size)
        
        # Calcular prevalencia actual para gatillar contención
        pct_infected = np.sum(self.state == self.I) / self.N
        
        # Gatillo de cuarentena doméstica: los infectados reducen su emisión al 5% (95% aislamiento)
        quarantine_active = getattr(self, 'enable_quarantine', False) and (pct_infected >= getattr(self, 'quarantine_trigger_pct', 0.05))
        if quarantine_active:
            vl_effective = np.where(self.state == self.I, 0.05 * self.viral_load, self.viral_load)
        else:
            vl_effective = self.viral_load
            
        vl_2d = vl_effective.reshape(self.grid_size)
        
        # Gatillo de distanciamiento social: reduce la probabilidad de transmisión en base al cumplimiento c_DS
        ds_active = (getattr(self, 'c_DS', 0.0) > 0.0) and (pct_infected >= getattr(self, 'ds_trigger_pct', 0.05))
        ds_factor = (1.0 - getattr(self, 'c_DS', 0.0)) if ds_active else 1.0
        
        prob_not_inf_2d = np.ones(self.grid_size, dtype=np.float32)
        v50_mat = self.v50.reshape(self.grid_size)

        # Definir los desplazamientos a evaluar según topología
        # Estructura: (dr, dc, mascara_condicional)
        shifts = []
        if self.topology == 'moore':
            dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            shifts = [(dr, dc, None) for dr, dc in dirs]
        elif self.topology == 'von_neumann':
            dirs = [(-1,0), (1,0), (0,-1), (0,1)]
            shifts = [(dr, dc, None) for dr, dc in dirs]
        elif self.topology == 'hexagonal':
            # Vecinos Invariantes (Horizontales)
            shifts.append((0, -1, None))
            shifts.append((0, 1, None))
            # Vecinos Condicionales de Filas Pares
            for dr, dc in [(-1, 0), (1, 0), (-1, 1), (1, 1)]:
                shifts.append((dr, dc, self.even_row_mask))
            # Vecinos Condicionales de Filas Impares
            for dr, dc in [(-1, 0), (1, 0), (-1, -1), (1, -1)]:
                shifts.append((dr, dc, self.odd_row_mask))
        else:
            raise ValueError(f"Topología '{self.topology}' no soportada.")
        
        for dr, dc, apply_mask in shifts:
            shifted_state = np.roll(state_2d, shift=(dr, dc), axis=(0, 1))
            shifted_vl = np.roll(vl_2d, shift=(dr, dc), axis=(0, 1))
            
            # Confinamiento Físico: Evitar teletransporte toroidal de np.roll
            # Limpiar bordes envueltos (wrapped) según dirección del desplazamiento
            if dr > 0:
                shifted_state[:dr, :] = self.S
                shifted_vl[:dr, :] = 0.0
            elif dr < 0:
                shifted_state[dr:, :] = self.S
                shifted_vl[dr:, :] = 0.0
                
            if dc > 0:
                shifted_state[:, :dc] = self.S
                shifted_vl[:, :dc] = 0.0
            elif dc < 0:
                shifted_state[:, dc:] = self.S
                shifted_vl[:, dc:] = 0.0
            
            # Identificar celdas con vecinos infectados
            mask_inf = (shifted_state == self.I)
            
            # Aplicar máscara condicional de vecindad (útil en hex row-offset)
            if apply_mask is not None:
                mask_inf = mask_inf & apply_mask
                
            v_j = shifted_vl
            hill_sigma = (v_j**self.n_hill) / (v_j**self.n_hill + v50_mat**self.n_hill) * ds_factor
            
            # Multiplicar sobre la probabilidad de evadir infección
            prob_not_inf_2d[mask_inf] *= (1.0 - hill_sigma[mask_inf])
            
        p_not_inf = prob_not_inf_2d.flatten()
        
        mask_S = (self.state == self.S)
        becomes_E = mask_S & (np.random.rand(self.N) < (1.0 - p_not_inf))
        
        self.state[becomes_E] = self.E
        # X ~ NegBin parametrizada
        self.time_in_E[becomes_E] = np.random.negative_binomial(self.k_E, self.p_E, size=np.sum(becomes_E))

    def run(self, output_dir=None, n_seed=10, seed=None):
        """Bucle de ejecución adaptado para guardar los datos en la carpeta local de Discreto."""
        if output_dir is None:
            output_dir = Path(__file__).parent
        # Ejecutamos el pipeline común y guardamos telemetría
        super().run(output_dir=output_dir, n_seed=n_seed, seed=seed)

if __name__ == '__main__':
    print("Iniciando Motor ABM (Enfoque Discreto)...")
    sim = DiscreteSEIRSDSimulation(grid_size=(40, 40), t_max=100)
    sim.run()
    print("Simulación completada exitosamente.")
