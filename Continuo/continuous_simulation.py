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
        self.dt = 0.1            # Paso de tiempo del continuo (ej: 0.1 dias = 2.4 horas)
        self.L = L
        
        # Parámetros específicos del Enfoque Continuo
        self.delta = 0.5         # Decaimiento ambiental
        self.ell = 1.0           # Longitud de escala del kernel gaussiano
        self.D_basal = 1.0       # Difusividad basal
        self.D_min = 0.05        # Difusividad mínima
        self.V_sint = 0.5        # Umbral de inhibición motora (carga viral)
        self.n_mov = 2.0         # Exponente de Hill para movimiento
        
        # Coordenadas iniciales continuas: x_i(0) ~ Uniforme([0, L]^2)
        self.coord_x = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        self.coord_y = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        
        # Dosis acumulada
        self.dosis = np.zeros(self.N, dtype=np.float32)
        self.tau_infection = None
        
        # ponytail: Part VIII Hub parameters
        self.H = 0  # Number of hubs
        self.hubs_coords = np.zeros((0, 2), dtype=np.float32)
        self.hubs_types = []  # List of 'agenda' or 'gravitatorio' strings
        self.hubs_lambda = np.zeros(0, dtype=np.float32)  # Poisson rate for agenda hubs (visitas/dia)
        self.hubs_alpha = np.zeros(0, dtype=np.float32)   # Gamma stay shape
        self.hubs_beta = np.zeros(0, dtype=np.float32)    # Gamma stay scale
        self.hubs_kappa = np.zeros(0, dtype=np.float32)   # Gravity force
        self.hubs_ell = np.zeros(0, dtype=np.float32)     # Gravity influence radius
        self.hubs_rho = np.zeros(0, dtype=np.float32)     # Agenda emission scale factor
        
        # State tracking for visits (initialized in phase 0)
        self.remaining_visit_time = np.zeros((self.N, 0), dtype=np.float32)
        
        # Social distancing continuous effectiveness parameter
        self.eta_mov = 0.5

    def _fase0_inicializacion(self, output_dir=None):
        """Inicializa perfiles inmunes en core y exporta el mapa estático del enfoque continuo."""
        super()._fase0_inicializacion(output_dir=output_dir)
        self.tau_infection = self.omega_in * self.tau_max
        # ponytail: initialize remaining visit timer array
        self.remaining_visit_time = np.zeros((self.N, self.H), dtype=np.float32)
        
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
        # --- 0. OPERADOR DE AGENDA DE VISITAS (Poisson + Gamma) ---
        # ponytail: decrement remaining visit times for active visits
        if self.H > 0:
            self.remaining_visit_time = np.maximum(0.0, self.remaining_visit_time - self.dt)
            # Find agents who are not quarantined, not dead, and not visiting any hub
            visiting_any = np.any(self.remaining_visit_time > 0, axis=1)
            can_visit = ~visiting_any & ~self.quarantined & (self.state != self.D)
            
            # Check agenda hubs triggers
            for h in range(self.H):
                if self.hubs_types[h] == 'agenda' and self.hubs_lambda[h] > 0:
                    prob_start = 1.0 - np.exp(-self.hubs_lambda[h] * self.dt)
                    starts_visit = can_visit & (np.random.rand(self.N) < prob_start)
                    if np.any(starts_visit):
                        durations = np.random.gamma(self.hubs_alpha[h], self.hubs_beta[h], size=np.sum(starts_visit))
                        self.remaining_visit_time[starts_visit, h] = durations
                        # Disable them from starting other visits in this step
                        can_visit = can_visit & ~starts_visit
                        
            # Force coordinates of agents on active agenda visits to be at their hub
            visiting_any = np.any(self.remaining_visit_time > 0, axis=1)
            if np.any(visiting_any):
                visiting_hub_idx = np.argmax(self.remaining_visit_time > 0, axis=1)
                self.coord_x[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 0]
                self.coord_y[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 1]
        else:
            visiting_any = np.zeros(self.N, dtype=bool)
            visiting_hub_idx = np.zeros(self.N, dtype=np.int16)

        # --- 1. EVALUACIÓN DEL CAMPO (Dosis absorbida t_n) ---
        sigma = self.sigma_base * (1.0 - 0.5 * self.omega_ad)
        
        # Compensador de Itô en la emisión: promedio trapezoidal de V_t y V_t+dt
        emission_avg = 0.5 * (self.prev_viral_load + self.viral_load) * np.exp(0.5 * (sigma**2) * self.dt)
        
        # ponytail: scale emission by rho_h for agents in active agenda visits
        if self.H > 0 and np.any(visiting_any):
            scale_factors = np.ones(self.N, dtype=np.float32)
            scale_factors[visiting_any] = self.hubs_rho[visiting_hub_idx[visiting_any]]
            emission_avg *= scale_factors

        coords_t = np.column_stack((self.coord_x, self.coord_y))
        tree_t = KDTree(coords_t)
        
        r_cut = 3.0 * self.ell
        pairs_t = tree_t.query_pairs(r_cut)
        
        R_t = np.zeros(self.N, dtype=np.float32)
        # ponytail: mask_I excludes quarantined agents
        mask_I = (self.state == self.I) & ~self.quarantined
        
        if len(pairs_t) > 0:
            pairs_arr = np.array(list(pairs_t))
            i_idx = pairs_arr[:, 0]
            j_idx = pairs_arr[:, 1]
            dists = np.linalg.norm(coords_t[i_idx] - coords_t[j_idx], axis=1)
            kernel_vals = np.exp(-(dists**2) / (2.0 * self.ell**2))
            
            # Suma de contribuciones bidireccionales
            np.add.at(R_t, i_idx, kernel_vals * emission_avg[j_idx] * mask_I[j_idx])
            np.add.at(R_t, j_idx, kernel_vals * emission_avg[i_idx] * mask_I[i_idx])

        # --- 2. PREDICCIÓN ESPACIAL (Langevin-Stratonovich con deriva de hubs gravitatorios) ---
        # ponytail: social distancing basal diffusivity reduction (D_basal_DS = D_basal * (1 - c_DS * eta_mov))
        D_basal_agent = self.D_basal * (1.0 - self.c_DS * self.eta_mov)
        
        v_pow = self.viral_load**self.n_mov
        vsint_pow = self.V_sint**self.n_mov
        D_esp = np.where(
            self.viral_load > 0,
            D_basal_agent * (1.0 - v_pow / (v_pow + vsint_pow)) + self.D_min,
            D_basal_agent + self.D_min
        )
        # ponytail: deceased, quarantined, and agenda-visiting agents do not move
        D_esp = np.where(self.state == self.D, 0.0, D_esp)
        D_esp = np.where(self.quarantined, 0.0, D_esp)
        D_esp = np.where(visiting_any, 0.0, D_esp)
        
        # Calculate drift towards active gravitational hubs
        drift_x = np.zeros(self.N, dtype=np.float32)
        drift_y = np.zeros(self.N, dtype=np.float32)
        if self.H > 0:
            can_drift = ~visiting_any & ~self.quarantined & (self.state != self.D)
            if np.any(can_drift):
                for h in range(self.H):
                    if self.hubs_types[h] == 'gravitatorio' and self.hubs_kappa[h] > 0:
                        dx = self.hubs_coords[h, 0] - self.coord_x
                        dy = self.hubs_coords[h, 1] - self.coord_y
                        dists = np.sqrt(dx**2 + dy**2)
                        dir_x = dx / (dists + 1e-5)
                        dir_y = dy / (dists + 1e-5)
                        weight = np.exp(-(dists**2) / (2.0 * self.hubs_ell[h]**2))
                        drift_x[can_drift] += self.hubs_kappa[h] * dir_x[can_drift] * weight[can_drift]
                        drift_y[can_drift] += self.hubs_kappa[h] * dir_y[can_drift] * weight[can_drift]

        # Paso de predicción espacial (ruido browniano en 2D + deriva gravitatoria)
        noise_x = np.random.normal(0.0, 1.0, self.N)
        noise_y = np.random.normal(0.0, 1.0, self.N)
        
        pred_x = self.coord_x + drift_x * self.dt + np.sqrt(2.0 * D_esp * self.dt) * noise_x
        pred_y = self.coord_y + drift_y * self.dt + np.sqrt(2.0 * D_esp * self.dt) * noise_y
        
        # Condiciones de borde reflectantes
        # Eje X
        mask_low_x = (pred_x < 0.0)
        pred_x[mask_low_x] = -pred_x[mask_low_x]
        mask_high_x = (pred_x > self.L)
        pred_x[mask_high_x] = 2.0 * self.L - pred_x[mask_high_x]
        pred_x = np.clip(pred_x, 0.0, self.L)
        
        # Eje Y
        mask_low_y = (pred_y < 0.0)
        pred_y[mask_low_y] = -pred_y[mask_low_y]
        mask_high_y = (pred_y > self.L)
        pred_y[mask_high_y] = 2.0 * self.L - pred_y[mask_high_y]
        pred_y = np.clip(pred_y, 0.0, self.L)
        
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
        self.dosis = self.dosis * np.exp(-self.delta * self.dt) + 0.5 * (R_t + R_pred) * self.dt
        
        # Consolidar posiciones predichas
        self.coord_x = pred_x
        self.coord_y = pred_y
        
        # Re-fijar agentes que están en agenda visits
        if self.H > 0 and np.any(visiting_any):
            self.coord_x[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 0]
            self.coord_y[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 1]
        
        # Transición S -> E cuando se supera el umbral de tolerancia acumulada
        mask_S = (self.state == self.S)
        becomes_E = mask_S & (self.dosis >= self.tau_infection)
        
        # ponytail: trace infector generation tracking for R_ef
        if np.any(becomes_E) and np.any(mask_I):
            idx_I = np.where(mask_I)[0]
            coords_I = coords_t[idx_I]
            newly_infected = np.where(becomes_E)[0]
            for idx_new in newly_infected:
                pos_new = coords_t[idx_new]
                dists_to_I = np.linalg.norm(coords_I - pos_new, axis=1)
                if len(dists_to_I) > 0:
                    closest_idx = np.argmin(dists_to_I)
                    parent_id = idx_I[closest_idx]
                    self.infected_by[idx_new] = parent_id
                    self.infections_caused[parent_id] += 1
                    
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
        self._fase5_buffer(0.0)
        
        for step in range(1, self.t_max + 1):
            current_time = step * self.dt
            self._fase1_ou_y_auc(current_time)
            self._fase2_contagio()
            self._fase3_transiciones(current_time)
            self._fase4_congelamiento()
            self._fase5_buffer(current_time)
            
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
