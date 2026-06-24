import numpy as np
import polars as pl
import sys
from pathlib import Path

# Añadir el directorio padre al sys.path para poder importar 'core'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scipy.spatial import KDTree
from core.base_simulation import BaseSEIRSDSimulation

class ContinuousSEIRSDSimulation(BaseSEIRSDSimulation):
    def __init__(self, N=1600, L=100.0, t_max=30):
        """
        Simulación SEIRS-D en Espacio Continuo (Browniano).
        Implementa Langevin-Stratonovich, Kernel Gaussiano vía KDTree,
        Compensador de Itô en Emisión y Operator Splitting.
        """
        super().__init__(N=N, t_max=t_max)
        self.L = L
        
        # Parámetros específicos del Enfoque Continuo
        self.delta = 0.5         # Decaimiento ambiental
        self.ell = 1.0           # Longitud de escala del kernel gaussiano
        self.D_basal = 1.0       # Difusividad basal
        self.D_min = 0.05        # Difusividad mínima
        self.V_sint = 0.5        # Umbral de inhibición motora (carga viral)
        self.n_mov = 2.0         # Exponente de Hill para movimiento
        
        # Transición S -> E
        self.k_E = 2
        self.p_E = 0.5
        
        # Coordenadas iniciales continuas: x_i(0) ~ Uniforme([0, L]^2)
        self.coord_x = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        self.coord_y = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        
        # Dosis acumulada
        self.dosis = np.zeros(self.N, dtype=np.float32)
        self.tau_infection = None

    def _fase0_inicializacion(self, output_dir=None):
        """Inicializa perfiles inmunes en core y exporta el mapa estático del enfoque continuo."""
        super()._fase0_inicializacion(output_dir=output_dir)
        self.tau_infection = self.omega_in * self.tau_max
        
        base_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        
        df_estatico = pl.DataFrame({
            "id_agente": self.id_agente,
            "omega_in": self.omega_in,
            "omega_ad": self.omega_ad,
            "tau_infection": self.tau_infection
        })
        df_estatico.write_parquet(base_dir / "mapa_estatico.parquet", compression="snappy")
        print(f"Exportado {base_dir / 'mapa_estatico.parquet'} (Continuo)")

    def _fase2_contagio(self):
        """
        Contagio Continuo: Dosis ambiental acumulada vía campo gaussiano evaluado con KDTree,
        movimiento de Langevin-Stratonovich con inhibición motora (Hill), y operator splitting.
        """
        # --- 1. EVALUACIÓN DEL CAMPO (Dosis absorbida t_n) ---
        sigma = self.sigma_base * (1.0 - 0.5 * self.omega_ad)
        
        # Compensador de Itô en la emisión: promedio trapezoidal de V_t y V_t+dt
        emission_avg = 0.5 * (self.prev_viral_load + self.viral_load) * np.exp(0.5 * (sigma**2) * 1.0)
        
        coords_t = np.column_stack((self.coord_x, self.coord_y))
        tree_t = KDTree(coords_t)
        
        r_cut = 3.0 * self.ell
        pairs_t = tree_t.query_pairs(r_cut)
        
        R_t = np.zeros(self.N, dtype=np.float32)
        mask_I = (self.state == self.I)
        
        if len(pairs_t) > 0:
            pairs_arr = np.array(list(pairs_t))
            i_idx = pairs_arr[:, 0]
            j_idx = pairs_arr[:, 1]
            dists = np.linalg.norm(coords_t[i_idx] - coords_t[j_idx], axis=1)
            kernel_vals = np.exp(-(dists**2) / (2.0 * self.ell**2))
            
            # Suma de contribuciones bidireccionales
            np.add.at(R_t, i_idx, kernel_vals * emission_avg[j_idx] * mask_I[j_idx])
            np.add.at(R_t, j_idx, kernel_vals * emission_avg[i_idx] * mask_I[i_idx])

        # --- 2. PREDICCIÓN ESPACIAL (Stratonovich con difusividad dependiente de v_t+dt) ---
        # ponytail: evitamos division por cero si viral_load = 0 en el cálculo de Hill
        v_pow = self.viral_load**self.n_mov
        vsint_pow = self.V_sint**self.n_mov
        D_esp = np.where(
            self.viral_load > 0,
            self.D_basal * (1.0 - v_pow / (v_pow + vsint_pow)) + self.D_min,
            self.D_basal + self.D_min
        )
        
        # Paso de predicción espacial (ruido browniano en 2D)
        noise_x = np.random.normal(0.0, 1.0, self.N)
        noise_y = np.random.normal(0.0, 1.0, self.N)
        
        # Stratonovich: usamos el coeficiente en t_n+1 (difusividad evaluada con el V corregido)
        pred_x = np.clip(self.coord_x + np.sqrt(2.0 * D_esp * 1.0) * noise_x, 0.0, self.L)
        pred_y = np.clip(self.coord_y + np.sqrt(2.0 * D_esp * 1.0) * noise_y, 0.0, self.L)
        
        # --- 3. EVALUACIÓN DEL CAMPO PREDICHO ---
        coords_pred = np.column_stack((pred_x, pred_y))
        tree_pred = KDTree(coords_pred)
        pairs_pred = tree_pred.query_pairs(r_cut)
        
        R_pred = np.zeros(self.N, dtype=np.float32)
        if len(pairs_pred) > 0:
            pairs_pred_arr = np.array(list(pairs_pred))
            i_idx_p = pairs_pred_arr[:, 0]
            j_idx_p = pairs_pred_arr[:, 1]
            dists_p = np.linalg.norm(coords_pred[i_idx_p] - coords_pred[j_idx_p], axis=1)
            kernel_vals_p = np.exp(-(dists_p**2) / (2.0 * self.ell**2))
            
            np.add.at(R_pred, i_idx_p, kernel_vals_p * emission_avg[j_idx_p] * mask_I[j_idx_p])
            np.add.at(R_pred, j_idx_p, kernel_vals_p * emission_avg[i_idx_p] * mask_I[i_idx_p])
            
        # --- 4. CORRECCIÓN Y ACTUALIZACIÓN DE DOSIS ---
        # Decaimiento ambiental δ analíticamente estable + promedio de dosis actual y predicha
        self.dosis = self.dosis * np.exp(-self.delta * 1.0) + 0.5 * (R_t + R_pred) * 1.0
        
        # Consolidar posiciones predichas
        self.coord_x = pred_x
        self.coord_y = pred_y
        
        # Transición S -> E cuando se supera el umbral de tolerancia acumulada
        mask_S = (self.state == self.S)
        becomes_E = mask_S & (self.dosis >= self.tau_infection)
        
        self.state[becomes_E] = self.E
        self.time_in_E[becomes_E] = np.random.negative_binomial(self.k_E, self.p_E, size=np.sum(becomes_E))

    def _fase5_buffer(self, t):
        """Guarda el estado y añade las variables espaciales en la telemetría."""
        super()._fase5_buffer(t)
        if 'coord_x' not in self.telemetry:
            self.telemetry['coord_x'] = []
            self.telemetry['coord_y'] = []
            self.telemetry['dosis'] = []
        self.telemetry['coord_x'].append(self.coord_x.copy())
        self.telemetry['coord_y'].append(self.coord_y.copy())
        self.telemetry['dosis'].append(self.dosis.copy())

    def run(self, output_dir=None, n_seed=10, seed=None):
        """Bucle principal de la simulación adaptada al continuo."""
        if output_dir is None:
            output_dir = Path(__file__).parent
            
        self._fase0_inicializacion(output_dir=output_dir)
        self.seed_infection(n=n_seed, seed=seed)
        self._fase5_buffer(0)
        
        for t in range(1, self.t_max + 1):
            self._fase1_ou_y_auc()
            self._fase2_contagio()
            self._fase3_transiciones(t)
            self._fase4_congelamiento()
            self._fase5_buffer(t)
            
        # Compilación de la telemetría dinámica final con coordenadas continuas
        tiempo_arr = np.concatenate(self.telemetry['tiempo'])
        id_arr = np.concatenate(self.telemetry['id_agente'])
        estado_arr = np.concatenate(self.telemetry['estado'])
        carga_arr = np.concatenate(self.telemetry['carga_viral'])
        x_arr = np.concatenate(self.telemetry['coord_x'])
        y_arr = np.concatenate(self.telemetry['coord_y'])
        dosis_arr = np.concatenate(self.telemetry['dosis'])
        
        df_dinamico = pl.DataFrame({
            "tiempo": tiempo_arr,
            "id_agente": id_arr,
            "estado": estado_arr,
            "carga_viral": carga_arr,
            "coord_x": x_arr,
            "coord_y": y_arr,
            "dosis": dosis_arr
        })
        
        df_dinamico = df_dinamico.sort(["tiempo", "id_agente"])
        
        if output_dir is not None:
            path_dir = Path(output_dir)
            df_dinamico.write_parquet(path_dir / "telemetria_dinamica.parquet", compression="snappy")
            print(f"Exportado {path_dir / 'telemetria_dinamica.parquet'} (Continuo)")
            
        return df_dinamico

if __name__ == '__main__':
    print("Iniciando Motor ABM (Enfoque Continuo)...")
    sim = ContinuousSEIRSDSimulation(N=1000, L=100.0, t_max=10)
    sim.run()
    print("Simulación continua completada exitosamente.")
