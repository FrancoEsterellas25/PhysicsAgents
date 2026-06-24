import numpy as np
import polars as pl
from scipy.stats import norm, beta

class ManimSEIRSDSimulation:
    def __init__(self, grid_size=(50, 50), topology='moore', t_max=30):
        """
        Motor de simulación estocástica ABM - SEIRS-D (Enfoque Discreto)
        Optimizado de forma estricta para exportación a reproductor Manim.
        """
        self.grid_size = grid_size
        self.N = grid_size[0] * grid_size[1]
        self.topology = topology
        self.t_max = t_max
        
        # Parámetros Globales
        self.rho = 0.5           # Correlación sistémica (Cópula)
        self.beta_in_a = 2.0     # Forma Alpha Beta_in
        self.beta_in_b = 2.0     # Forma Beta Beta_in
        self.beta_ad_a = 2.0     # Forma Alpha Beta_ad
        self.beta_ad_b = 5.0     # Forma Beta Beta_ad
        
        # SDE-OU
        self.theta_low = 0.1
        self.theta_high = 0.5
        self.tau_peak = 5
        self.beta_ou = 0.5       # Sensibilidad theta a omega_ad
        self.v_base = 0.1
        self.v_peak_base = 10.0
        self.k_ou = 0.3
        self.sigma_base = 1.0
        self.eps = 0.01          # Ruido inicial viral
        
        # Contagio y Biología
        self.V_50_basal = 3.0
        self.gamma_hill = 1.5
        self.n_hill = 2.0
        self.alpha = 0.1         # Probabilidad base salida de I
        self.p_base = 1 - (0.992)**(1/365)
        
        # Letalidad (Logística)
        self.w1 = 0.6
        self.w2 = 0.4
        self.lam = 5.0
        self.tau_max = 20.0
        self.auc_norm_factor = 100.0 # Factor de normalización aproximado del AUC
        
        # Estados: 0=S, 1=E, 2=I, 3=R, 4=D
        self.S, self.E, self.I, self.R, self.D = 0, 1, 2, 3, 4
        
        # Memoria del sistema (Arrays 1D de tamaño N para eficiencia y Manim)
        self.id_agente = np.arange(self.N, dtype=np.int32)
        
        # Coordenadas
        y_idx, x_idx = np.indices(grid_size)
        self.coord_x = x_idx.flatten().astype(np.int16)
        self.coord_y = y_idx.flatten().astype(np.int16)
        
        self.state = np.zeros(self.N, dtype=np.int8)
        self.viral_load = np.zeros(self.N, dtype=np.float32)
        self.auc = np.zeros(self.N, dtype=np.float32)
        
        self.time_in_E = np.zeros(self.N, dtype=np.int16)
        self.time_in_I = np.zeros(self.N, dtype=np.int16)
        self.time_in_R = np.zeros(self.N, dtype=np.int16)
        
        self.telemetry = {
            'tiempo': [],
            'id_agente': [],
            'estado': [],
            'carga_viral': []
        }
        
    def _fase0_inicializacion(self):
        """Cópula Gaussiana y Exportación del Mapa Estático."""
        cov_matrix = [[1.0, self.rho], [self.rho, 1.0]]
        z = np.random.multivariate_normal([0, 0], cov_matrix, size=self.N)
        
        u1 = norm.cdf(z[:, 0])
        u2 = norm.cdf(z[:, 1])
        
        self.omega_in = beta.ppf(u1, self.beta_in_a, self.beta_in_b).astype(np.float32)
        self.omega_ad = beta.ppf(u2, self.beta_ad_a, self.beta_ad_b).astype(np.float32)
        
        # Exportar Mapa Estático
        from pathlib import Path
        base_dir = Path(__file__).parent
        df_estatico = pl.DataFrame({
            "id_agente": self.id_agente,
            "coord_x": self.coord_x,
            "coord_y": self.coord_y,
            "omega_in": self.omega_in,
            "omega_ad": self.omega_ad
        })
        df_estatico.write_parquet(base_dir / "mapa_estatico.parquet", compression="snappy")
        print(f"Exportado {base_dir / 'mapa_estatico.parquet'}")

    def seed_infection(self, n=5):
        idx = np.random.choice(self.N, n, replace=False)
        self.state[idx] = self.I
        self.viral_load[idx] = self.v_base + self.eps
        self.time_in_I[idx] = 1

    def _fase1_ou_y_auc(self):
        """Integración SDE-OU exacta y actualización del AUC para estado I."""
        mask_I = (self.state == self.I)
        if not np.any(mask_I):
            return
            
        tau = self.time_in_I[mask_I]
        w_ad = self.omega_ad[mask_I]
        v_t = self.viral_load[mask_I]
        
        # Theta
        theta_fase = np.where(tau < self.tau_peak, self.theta_low, self.theta_high)
        theta = theta_fase * (1.0 + self.beta_ou * w_ad)
        
        v_peak = self.v_peak_base * (1.0 - 0.5 * w_ad)
        sigma = self.sigma_base * (1.0 - 0.5 * w_ad)
        dt = 1.0
        term1 = v_t * np.exp(-theta * dt)

        # Integración exacta de la deriva con media dependiente del tiempo mu(s)
        diff = theta - self.k_ou
        term2 = np.where(
            np.abs(diff) > 1e-5,
            v_peak * (1.0 - np.exp(-theta * dt)) - (v_peak - self.v_base) * np.exp(-self.k_ou * tau) * (theta / diff) * (np.exp(-self.k_ou * dt) - np.exp(-theta * dt)),
            v_peak * (1.0 - np.exp(-theta * dt)) - (v_peak - self.v_base) * np.exp(-self.k_ou * tau) * (theta * dt * np.exp(-theta * dt))
        )
        std_dev = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))
        noise = np.random.normal(0, 1, size=len(v_t))
        
        v_next = term1 + term2 + std_dev * noise
        v_next = np.maximum(0.0, v_next) # Truncar a 0 mínimo
        
        # AUC Aproximación Trapezoidal
        self.auc[mask_I] += ((v_t + v_next) / 2.0) * dt
        self.viral_load[mask_I] = v_next

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

    def _fase3_transiciones(self):
        """E->I, I->R, I->D, R->S y p_base."""
        
        # Ruido de Fondo
        alive = (self.state != self.D)
        dies_base = alive & (np.random.rand(self.N) < self.p_base)
        self.state[dies_base] = self.D
        
        # E -> I
        mask_E = (self.state == self.E) & ~dies_base
        ready_to_I = mask_E & (self.time_in_E <= 0)
        self.state[ready_to_I] = self.I
        self.viral_load[ready_to_I] = self.v_base + self.eps
        self.time_in_I[ready_to_I] = 0
        self.auc[ready_to_I] = 0.0
        self.time_in_E[mask_E & ~ready_to_I] -= 1
        
        # I -> R o D
        mask_I = (self.state == self.I) & ~dies_base & ~ready_to_I
        self.time_in_I[mask_I] += 1
        
        p_exit = 1.0 - np.exp(-self.alpha * self.time_in_I[mask_I])
        exiting = (np.random.rand(np.sum(mask_I)) < p_exit)
        
        if np.any(exiting):
            idx_exiting = np.where(mask_I)[0][exiting]
            
            auc_norm = self.auc[idx_exiting] / self.auc_norm_factor
            tau_ratio = np.clip(self.time_in_I[idx_exiting] / self.tau_max, 0, 1)
            
            E_index = self.w1 * auc_norm + self.w2 * tau_ratio
            mu_v = 1.0 / (1.0 + np.exp(-self.lam * (E_index - self.omega_ad[idx_exiting])))
            
            dies_virus = (np.random.rand(len(idx_exiting)) <= mu_v)
            idx_D = idx_exiting[dies_virus]
            idx_R = idx_exiting[~dies_virus]
            
            self.state[idx_D] = self.D
            self.state[idx_R] = self.R
            # R ~ NegBin aproximado media/moda globales (ej. k=4, p=0.1)
            self.time_in_R[idx_R] = np.random.negative_binomial(4, 0.1, size=len(idx_R))
            
        # R -> S
        mask_R = (self.state == self.R) & ~dies_base
        ready_to_S = mask_R & (self.time_in_R <= 0)
        self.state[ready_to_S] = self.S
        self.time_in_R[mask_R & ~ready_to_S] -= 1

    def _fase4_congelamiento(self):
        """Fuerza invariantes sobre el estado D."""
        mask_D = (self.state == self.D)
        self.viral_load[mask_D] = 0.0
        self.auc[mask_D] = 0.0

    def _fase5_buffer(self, t):
        """Vuelca estado sincronizado. El indexado asegura pre-ordenamiento."""
        self.telemetry['tiempo'].append(np.full(self.N, t, dtype=np.int16))
        self.telemetry['id_agente'].append(self.id_agente)
        self.telemetry['estado'].append(self.state.copy())
        self.telemetry['carga_viral'].append(self.viral_load.copy())

    def run(self):
        self._fase0_inicializacion()
        self.seed_infection(n=10)
        
        for t in range(self.t_max):
            self._fase1_ou_y_auc()
            self._fase2_contagio()
            self._fase3_transiciones()
            self._fase4_congelamiento()
            self._fase5_buffer(t)
            
        # Compilación Polars Final
        tiempo_arr = np.concatenate(self.telemetry['tiempo'])
        id_arr = np.concatenate(self.telemetry['id_agente'])
        estado_arr = np.concatenate(self.telemetry['estado'])
        carga_arr = np.concatenate(self.telemetry['carga_viral'])
        
        df_dinamico = pl.DataFrame({
            "tiempo": tiempo_arr,
            "id_agente": id_arr,
            "estado": estado_arr,
            "carga_viral": carga_arr
        })
        
        # REGLA INNEGOCIABLE: ORDENAR POR TIEMPO Y LUEGO ID_AGENTE
        df_dinamico = df_dinamico.sort(["tiempo", "id_agente"])
        
        from pathlib import Path
        base_dir = Path(__file__).parent
        df_dinamico.write_parquet(base_dir / "telemetria_dinamica.parquet", compression="snappy")
        print(f"Exportado {base_dir / 'telemetria_dinamica.parquet'} (Orden estricto validado)")
        return df_dinamico

if __name__ == '__main__':
    print("Iniciando Motor ABM-Manim...")
    sim = ManimSEIRSDSimulation(grid_size=(40, 40), t_max=100)
    sim.run()
    print("Frames pre-calculados exitosamente para Manim.")
