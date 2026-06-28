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
        self.delta = 0.5         # Decaimiento ambiental default
        self.delta_cerrado = 0.2  # Decaimiento en espacios cerrados (Hogar, Escuela, Trabajo)
        self.delta_ext = 1.0     # Decaimiento en tránsito / exteriores
        self.delta_abierto = 4.0  # Decaimiento en espacios abiertos (Supermercado, Centro)
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
        
        # ponytail: Part VIII Household and Transit parameters
        self.N_hogar = 4         # Average household size
        self.T_transito = 0.5 / 24.0  # Transit duration in days (e.g. 30 minutes)
        
        # Motion states: 0 = Home, 1 = Transit, 2 = Hub
        self.motion_state = np.zeros(self.N, dtype=np.int8)
        self.remaining_transit_time = np.zeros(self.N, dtype=np.float32)
        self.transit_destination_hub = np.full(self.N, -1, dtype=np.int16)

    def _fase0_inicializacion(self, output_dir=None):
        """Inicializa perfiles inmunes en core y exporta el mapa estático del enfoque continuo."""
        super()._fase0_inicializacion(output_dir=output_dir)
        self.tau_infection = self.omega_in * self.tau_max
        
        # ponytail: initialize ages (edades)
        np.random.seed(42)
        self.edades = np.random.randint(5, 75, self.N)
        
        # ponytail: scale hubs dynamically based on population size N
        n_children = np.sum(self.edades < 18)
        n_adults = np.sum((self.edades >= 18) & (self.edades <= 65))
        
        S = max(1, int(n_children // 70))  # ~70 children per school
        W = max(1, int(n_adults // 60))    # ~60 adults per work center
        M = max(1, int(self.N // 400))     # ~400 agents per supermarket
        
        self.H = S + W + M + 1  # Schools + Work Centers + Supermarkets + El Centro
        
        # Allocate hub config arrays
        self.hubs_coords = np.zeros((self.H, 2), dtype=np.float32)
        self.hubs_types = []
        self.hubs_lambda = np.zeros(self.H, dtype=np.float32)
        self.hubs_alpha = np.zeros(self.H, dtype=np.float32)
        self.hubs_beta = np.zeros(self.H, dtype=np.float32)
        self.hubs_kappa = np.zeros(self.H, dtype=np.float32)
        self.hubs_ell = np.zeros(self.H, dtype=np.float32)
        self.hubs_rho = np.zeros(self.H, dtype=np.float32)
        self.hubs_is_closed = np.zeros(self.H, dtype=bool)
        hubs_names = []
        hubs_colors = []
        
        # Symmetrical Layout: Ring of agenda hubs + Centro in the middle
        center_x, center_y = self.L / 2.0, self.L / 2.0
        R_circle = self.L * 0.35
        
        # 1. Schools
        for h in range(S):
            self.hubs_types.append("agenda")
            self.hubs_lambda[h] = 5.0 / 7.0
            self.hubs_alpha[h] = 12.0
            self.hubs_beta[h] = 0.5 / 24.0
            self.hubs_kappa[h] = 0.0
            self.hubs_ell[h] = 0.0
            self.hubs_rho[h] = 0.8
            self.hubs_is_closed[h] = True
            hubs_names.append(f"Escuela {h+1}" if S > 1 else "Escuela")
            hubs_colors.append("Yellow")
            
        # 2. Work Centers
        for h in range(S, S + W):
            self.hubs_types.append("agenda")
            self.hubs_lambda[h] = 5.0 / 7.0
            self.hubs_alpha[h] = 16.0
            self.hubs_beta[h] = 0.5 / 24.0
            self.hubs_kappa[h] = 0.0
            self.hubs_ell[h] = 0.0
            self.hubs_rho[h] = 0.7
            self.hubs_is_closed[h] = True
            hubs_names.append(f"Trabajo {h-S+1}" if W > 1 else "Trabajo")
            hubs_colors.append("Orange")
            
        # 3. Supermarkets
        for h in range(S + W, S + W + M):
            self.hubs_types.append("agenda")
            self.hubs_lambda[h] = 0.5
            self.hubs_alpha[h] = 3.0
            self.hubs_beta[h] = 0.25 / 24.0
            self.hubs_kappa[h] = 0.0
            self.hubs_ell[h] = 0.0
            self.hubs_rho[h] = 0.9
            self.hubs_is_closed[h] = False
            hubs_names.append(f"Supermercado {h-S-W+1}" if M > 1 else "Supermercado")
            hubs_colors.append("Cyan")
            
        # 4. El Centro (Agenda - Open)
        h_centro = S + W + M
        self.hubs_coords[h_centro] = [center_x, center_y]
        self.hubs_types.append("agenda")
        self.hubs_lambda[h_centro] = 0.4                  # visits frequency (0.4 visits/day)
        self.hubs_alpha[h_centro] = 4.0                   # Gamma shape parameter for stay
        self.hubs_beta[h_centro] = 0.25 / 24.0            # Gamma scale parameter for stay (1.0 hour)
        self.hubs_kappa[h_centro] = 0.0
        self.hubs_ell[h_centro] = 0.0
        self.hubs_rho[h_centro] = 1.0
        self.hubs_is_closed[h_centro] = False
        hubs_names.append("El Centro")
        hubs_colors.append("Green")
        
        # Distribute agenda hubs coordinates in the ring
        for h in range(S + W + M):
            angle = 2.0 * np.pi * h / (S + W + M)
            self.hubs_coords[h, 0] = center_x + R_circle * np.cos(angle)
            self.hubs_coords[h, 1] = center_y + R_circle * np.sin(angle)
            
        # Initialize remaining visit timer array
        self.remaining_visit_time = np.zeros((self.N, self.H), dtype=np.float32)
        self.hub_group_id = np.full((self.N, self.H), -1, dtype=np.int32)
        
        # School assignment for children
        child_indices = np.where(self.edades < 18)[0]
        for idx_pos, idx in enumerate(child_indices):
            h_school = idx_pos % S
            self.hub_group_id[idx, h_school] = (idx_pos // S) // 20  # class size of 20
            
        # Work assignment for adults (ages 18 to 65 only, retirees are excluded!)
        adult_indices = np.where((self.edades >= 18) & (self.edades <= 65))[0]
        for idx_pos, idx in enumerate(adult_indices):
            h_work = S + (idx_pos % W)
            self.hub_group_id[idx, h_work] = (idx_pos // W) // 12  # office size of 12

        # ponytail: initialize household allocations and coordinates
        H_hogar = int(np.ceil(self.N / self.N_hogar))
        unique_home_coords = np.random.uniform(0.0, self.L, (H_hogar, 2)).astype(np.float32)
        self.household_id = (np.arange(self.N, dtype=np.int32) % H_hogar)
        np.random.shuffle(self.household_id)
        self.home_coords = unique_home_coords[self.household_id]
        
        # Initialize motion state and lock positions to homes initially
        self.motion_state = np.zeros(self.N, dtype=np.int8)
        self.coord_x = self.home_coords[:, 0].copy()
        self.coord_y = self.home_coords[:, 1].copy()
        self.remaining_transit_time = np.zeros(self.N, dtype=np.float32)
        self.transit_destination_hub = np.full(self.N, -1, dtype=np.int16)
        
        base_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        
        df_estatico = pl.DataFrame({
            "id_agente": self.id_agente,
            "omega_in": self.omega_in,
            "omega_ad": self.omega_ad,
            "tau_infection": self.tau_infection,
            "hogar_x": self.home_coords[:, 0],
            "hogar_y": self.home_coords[:, 1],
            "edad": self.edades
        })
        df_estatico.write_parquet(base_dir / "mapa_estatico.parquet", compression="snappy")
        print(f"Exportado {base_dir / 'mapa_estatico.parquet'} (Continuo)")
        
        # ponytail: export hubs.parquet if hubs exist (with name and color properties)
        if self.H > 0:
            df_hubs = pl.DataFrame({
                "x": self.hubs_coords[:, 0],
                "y": self.hubs_coords[:, 1],
                "tipo": self.hubs_types,
                "nombre": hubs_names,
                "color": hubs_colors
            })
            df_hubs.write_parquet(base_dir / "hubs.parquet", compression="snappy")
            print(f"Exportado {base_dir / 'hubs.parquet'}")

    def _fase2_contagio(self):
        """
        Contagio Continuo: Dosis ambiental acumulada vía campo gaussiano evaluado con KDTree,
        movimiento de Langevin-Stratonovich con inhibición motora (Hill), y operator splitting.
        Soporte para estados de movimiento (hogar, tránsito, hubs) y transmisión doméstica.
        """
        t = getattr(self, "current_time", 0.0)
        # 1. NIGHT STATUS CHECK
        # Noche: de 23:00 a 07:00 (fracción de día < 7/24 o > 23/24)
        is_night = (t % 1.0 < 7.0 / 24.0) or (t % 1.0 > 23.0 / 24.0)
        
        # 2. STATE MACHINE UPDATES
        # Force quarantined agents home
        self.motion_state[self.quarantined] = 0
        self.remaining_transit_time[self.quarantined] = 0.0
        self.transit_destination_hub[self.quarantined] = -1
        
        # Lock deceased agents home
        mask_dead = (self.state == self.D)
        self.motion_state[mask_dead] = 0
        
        # Decrement active visit times (motion_state == 2)
        mask_at_hub = (self.motion_state == 2)
        if np.any(mask_at_hub) and self.H > 0:
            self.remaining_visit_time[mask_at_hub] = np.maximum(0.0, self.remaining_visit_time[mask_at_hub] - self.dt)
            # If stay ends, initiate transit back home
            for i in np.where(mask_at_hub)[0]:
                hub_idx = np.argmax(self.remaining_visit_time[i] > 0) if np.any(self.remaining_visit_time[i] > 0) else 0
                if self.remaining_visit_time[i, hub_idx] <= 0:
                    self.motion_state[i] = 1
                    self.remaining_transit_time[i] = self.T_transito
                    self.transit_destination_hub[i] = -1

        # Decrement transit timers (motion_state == 1)
        mask_in_transit = (self.motion_state == 1)
        if np.any(mask_in_transit):
            self.remaining_transit_time[mask_in_transit] -= self.dt
            expired_transit = mask_in_transit & (self.remaining_transit_time <= 0.0)
            if np.any(expired_transit):
                for i in np.where(expired_transit)[0]:
                    dest = self.transit_destination_hub[i]
                    if dest == -1:
                        # Arrived home
                        self.motion_state[i] = 0
                        self.coord_x[i] = self.home_coords[i, 0]
                        self.coord_y[i] = self.home_coords[i, 1]
                    else:
                        # Arrived at hub
                        self.motion_state[i] = 2
                        self.coord_x[i] = self.hubs_coords[dest, 0]
                        self.coord_y[i] = self.hubs_coords[dest, 1]
                        # Sample stay duration
                        self.remaining_visit_time[i, dest] = np.random.gamma(self.hubs_alpha[dest], self.hubs_beta[dest])
            
            # Night abort transit: if heading to hub and it's night, redirect them back home
            if is_night:
                abort_mask = mask_in_transit & (self.transit_destination_hub >= 0)
                if np.any(abort_mask):
                    self.transit_destination_hub[abort_mask] = -1
                    self.remaining_transit_time[abort_mask] = self.T_transito

        # Visit triggers for agents at home (motion_state == 0) and not quarantined, not dead
        if not is_night and self.H > 0:
            mask_at_home = (self.motion_state == 0) & ~self.quarantined & ~mask_dead
            if np.any(mask_at_home):
                for h in range(self.H):
                    if self.hubs_types[h] == 'agenda' and self.hubs_lambda[h] > 0:
                        # Only allow visit if open hub OR if the agent has a group assigned in this closed hub
                        can_visit = np.ones(self.N, dtype=bool)
                        if self.hubs_is_closed[h]:
                            can_visit = (self.hub_group_id[:, h] != -1)
                        
                        prob_start = 1.0 - np.exp(-self.hubs_lambda[h] * self.dt)
                        starts_visit = mask_at_home & can_visit & (np.random.rand(self.N) < prob_start)
                        if np.any(starts_visit):
                            self.motion_state[starts_visit] = 1
                            self.remaining_transit_time[starts_visit] = self.T_transito
                            self.transit_destination_hub[starts_visit] = h
                            # Exclude from triggering multiple visits in same step
                            mask_at_home = mask_at_home & ~starts_visit

        # 3. FORCE POSITIONS (Home and Hubs)
        # ponytail: Allow agents to keep their local wandered positions.
        # Enforce local bounding radius clipping at the end of the step instead.
        mask_home = (self.motion_state == 0)
        mask_hub = (self.motion_state == 2)
        if np.any(mask_hub) and self.H > 0:
            visiting_hub_idx = np.argmax(self.remaining_visit_time > 0, axis=1)

        # 4. EMISSION SCALING (Domestic rho_hogar vs Hubs rho_h)
        sigma = self.sigma_base * (1.0 - 0.5 * self.omega_ad)
        emission_avg = 0.5 * (self.prev_viral_load + self.viral_load) * np.exp(0.5 * (sigma**2) * self.dt)
        
        scale_factors = np.ones(self.N, dtype=np.float32)
        # Domestic scale
        scale_factors[self.motion_state == 0] = self.rho_hogar
        # Hub scale
        if self.H > 0 and np.any(mask_hub):
            scale_factors[mask_hub] = self.hubs_rho[visiting_hub_idx[mask_hub]]
        emission_avg *= scale_factors

        # 5. DOSIS FIELD CALCULATIONS (t_n)
        coords_t = np.column_stack((self.coord_x, self.coord_y))
        tree_t = KDTree(coords_t)
        
        r_cut = 3.0 * self.ell
        pairs_t = tree_t.query_pairs(r_cut)
        
        R_t = np.zeros(self.N, dtype=np.float32)
        mask_I = (self.state == self.I) & ~self.quarantined
        
        if len(pairs_t) > 0:
            pairs_arr = np.array(list(pairs_t))
            i_idx = pairs_arr[:, 0]
            j_idx = pairs_arr[:, 1]
            
            # ponytail: Filter out pairs in the same closed hub that belong to different groups
            allowed_mask = np.ones(len(i_idx), dtype=bool)
            both_in_hub = (self.motion_state[i_idx] == 2) & (self.motion_state[j_idx] == 2)
            if np.any(both_in_hub):
                h_i = visiting_hub_idx[i_idx]
                h_j = visiting_hub_idx[j_idx]
                same_hub = both_in_hub & (h_i == h_j)
                if np.any(same_hub):
                    is_closed_hub = self.hubs_is_closed[h_i]
                    closed_same_hub = same_hub & is_closed_hub
                    if np.any(closed_same_hub):
                        group_i = self.hub_group_id[i_idx, h_i]
                        group_j = self.hub_group_id[j_idx, h_j]
                        different_group = (group_i != group_j)
                        allowed_mask[closed_same_hub & different_group] = False
            
            i_idx = i_idx[allowed_mask]
            j_idx = j_idx[allowed_mask]
            
            if len(i_idx) > 0:
                dists = np.linalg.norm(coords_t[i_idx] - coords_t[j_idx], axis=1)
                kernel_vals = np.exp(-(dists**2) / (2.0 * self.ell**2))
                
                # Sum bidirectionally
                np.add.at(R_t, i_idx, kernel_vals * emission_avg[j_idx] * mask_I[j_idx])
                np.add.at(R_t, j_idx, kernel_vals * emission_avg[i_idx] * mask_I[i_idx])

        # 6. SPATIAL PREDICTION (Free brownian motion only for agents in transit, local diffusion for home/hub)
        D_basal_agent = self.D_basal * (1.0 - self.c_DS * self.eta_mov)
        v_pow = self.viral_load**self.n_mov
        vsint_pow = self.V_sint**self.n_mov
        D_esp = np.where(
            self.viral_load > 0,
            D_basal_agent * (1.0 - v_pow / (v_pow + vsint_pow)) + self.D_min,
            D_basal_agent + self.D_min
        )
        
        # ponytail: assign non-zero diffusion coefficients for home and hub states to make the space look alive
        D_esp = np.where(
            self.motion_state == 1,
            D_esp,  # full speed in transit
            np.where(
                self.motion_state == 2,
                0.20 * D_basal_agent,  # moderate local movement in hubs (e.g. plaza, school)
                0.08 * D_basal_agent   # slow local movement within household bounds
            )
        )
        
        # Gravitational Drift (only applies to transit agents)
        drift_x = np.zeros(self.N, dtype=np.float32)
        drift_y = np.zeros(self.N, dtype=np.float32)
        if self.H > 0:
            can_drift = (self.motion_state == 1)
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

        noise_x = np.random.normal(0.0, 1.0, self.N)
        noise_y = np.random.normal(0.0, 1.0, self.N)
        
        pred_x = self.coord_x + drift_x * self.dt + np.sqrt(2.0 * D_esp * self.dt) * noise_x
        pred_y = self.coord_y + drift_y * self.dt + np.sqrt(2.0 * D_esp * self.dt) * noise_y
        
        # Boundary reflections for transit agents
        # X Axis
        mask_low_x = (pred_x < 0.0)
        pred_x[mask_low_x] = -pred_x[mask_low_x]
        mask_high_x = (pred_x > self.L)
        pred_x[mask_high_x] = 2.0 * self.L - pred_x[mask_high_x]
        pred_x = np.clip(pred_x, 0.0, self.L)
        
        # Y Axis
        mask_low_y = (pred_y < 0.0)
        pred_y[mask_low_y] = -pred_y[mask_low_y]
        mask_high_y = (pred_y > self.L)
        pred_y[mask_high_y] = 2.0 * self.L - pred_y[mask_high_y]
        pred_y = np.clip(pred_y, 0.0, self.L)
        
        # ponytail: enforce radial bounds around target centers instead of hard coordinate locking
        # Home radial bounds (max radius = 1.2 units)
        dx_home = pred_x - self.home_coords[:, 0]
        dy_home = pred_y - self.home_coords[:, 1]
        dist_home = np.sqrt(dx_home**2 + dy_home**2)
        too_far_home = mask_home & (dist_home > 1.2)
        pred_x[too_far_home] = self.home_coords[too_far_home, 0] + (dx_home[too_far_home] / dist_home[too_far_home]) * 1.2
        pred_y[too_far_home] = self.home_coords[too_far_home, 1] + (dy_home[too_far_home] / dist_home[too_far_home]) * 1.2
        
        # Hub radial bounds (max radius = 2.5 units)
        if np.any(mask_hub) and self.H > 0:
            hub_idx = visiting_hub_idx[mask_hub]
            h_cx = self.hubs_coords[hub_idx, 0]
            h_cy = self.hubs_coords[hub_idx, 1]
            dx_hub = pred_x[mask_hub] - h_cx
            dy_hub = pred_y[mask_hub] - h_cy
            dist_hub = np.sqrt(dx_hub**2 + dy_hub**2)
            too_far_hub = dist_hub > 2.5
            
            temp_x = pred_x[mask_hub]
            temp_y = pred_y[mask_hub]
            temp_x[too_far_hub] = h_cx[too_far_hub] + (dx_hub[too_far_hub] / dist_hub[too_far_hub]) * 2.5
            temp_y[too_far_hub] = h_cy[too_far_hub] + (dy_hub[too_far_hub] / dist_hub[too_far_hub]) * 2.5
            pred_x[mask_hub] = temp_x
            pred_y[mask_hub] = temp_y

        # 7. EVALUATE PREDICTED DOSIS FIELD (t_n+1)
        coords_pred = np.column_stack((pred_x, pred_y))
        tree_pred = KDTree(coords_pred)
        pairs_pred = tree_pred.query_pairs(r_cut)
        
        R_pred = np.zeros(self.N, dtype=np.float32)
        if len(pairs_pred) > 0:
            pairs_pred_arr = np.array(list(pairs_pred))
            i_idx_p = pairs_pred_arr[:, 0]
            j_idx_p = pairs_pred_arr[:, 1]
            
            # ponytail: Filter out pairs in the same closed hub that belong to different groups
            allowed_mask_p = np.ones(len(i_idx_p), dtype=bool)
            both_in_hub_p = (self.motion_state[i_idx_p] == 2) & (self.motion_state[j_idx_p] == 2)
            if np.any(both_in_hub_p):
                h_i_p = visiting_hub_idx[i_idx_p]
                h_j_p = visiting_hub_idx[j_idx_p]
                same_hub_p = both_in_hub_p & (h_i_p == h_j_p)
                if np.any(same_hub_p):
                    is_closed_hub_p = self.hubs_is_closed[h_i_p]
                    closed_same_hub_p = same_hub_p & is_closed_hub_p
                    if np.any(closed_same_hub_p):
                        group_i_p = self.hub_group_id[i_idx_p, h_i_p]
                        group_j_p = self.hub_group_id[j_idx_p, h_j_p]
                        different_group_p = (group_i_p != group_j_p)
                        allowed_mask_p[closed_same_hub_p & different_group_p] = False
            
            i_idx_p = i_idx_p[allowed_mask_p]
            j_idx_p = j_idx_p[allowed_mask_p]
            
            if len(i_idx_p) > 0:
                dists_p = np.linalg.norm(coords_pred[i_idx_p] - coords_pred[j_idx_p], axis=1)
                kernel_vals_p = np.exp(-(dists_p**2) / (2.0 * self.ell**2))
                
                np.add.at(R_pred, i_idx_p, kernel_vals_p * emission_avg[j_idx_p] * mask_I[j_idx_p])
                np.add.at(R_pred, j_idx_p, kernel_vals_p * emission_avg[i_idx_p] * mask_I[i_idx_p])
            
        # 8. UPDATE DOSIS AND CONSOLIDATE (using spatially heterogeneous decay)
        delta_agents = np.full(self.N, self.delta_ext, dtype=np.float32)
        # Closed spaces: Home (motion_state == 0)
        delta_agents[self.motion_state == 0] = self.delta_cerrado
        # Hubs (motion_state == 2)
        mask_hub_active = (self.motion_state == 2)
        if np.any(mask_hub_active) and self.H > 0:
            h_indices = visiting_hub_idx[mask_hub_active]
            is_closed_h = self.hubs_is_closed[h_indices]
            # Closed hubs get delta_cerrado, open hubs get delta_abierto
            delta_agents[mask_hub_active] = np.where(is_closed_h, self.delta_cerrado, self.delta_abierto)
            
        self.dosis = self.dosis * np.exp(-delta_agents * self.dt) + 0.5 * (R_t + R_pred) * self.dt
        self.coord_x = pred_x
        self.coord_y = pred_y
        
        # ponytail: Allow wandered coordinates to persist; no hard centers lock at step end.

        # 9. S -> E TRANSITIONS AND INFECTION TRACKING
        mask_S = (self.state == self.S)
        becomes_E = mask_S & (self.dosis >= self.tau_infection)
        
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
            self.telemetry['motion_state'] = []
        self.telemetry['coord_x'].append(self.coord_x.copy())
        self.telemetry['coord_y'].append(self.coord_y.copy())
        self.telemetry['dosis'].append(self.dosis.copy())
        self.telemetry['motion_state'].append(self.motion_state.copy())

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
        mstate_arr = np.concatenate(self.telemetry['motion_state'])
        
        df_dinamico = pl.DataFrame({
            "tiempo": tiempo_arr,
            "id_agente": id_arr,
            "estado": estado_arr,
            "carga_viral": carga_arr,
            "coord_x": x_arr,
            "coord_y": y_arr,
            "dosis": dosis_arr,
            "motion_state": mstate_arr
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
