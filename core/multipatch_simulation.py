import numpy as np
import polars as pl
from pathlib import Path
from core.base_simulation import BaseSEIRSDSimulation
from scipy.spatial import KDTree

# ponytail: high-performance multipatch simulation using single-array partitioning (no reallocation)

class MultipatchSEIRSDSimulation(BaseSEIRSDSimulation):
    def __init__(self, P=5, N_per_patch=200, L=100.0, t_max=30):
        self.P = P
        self.N_per_patch = N_per_patch
        N = P * N_per_patch
        super().__init__(N=N, t_max=t_max)
        
        self.L = L
        self.dt = 0.1  # Continuous step size
        self.delta = 0.5
        self.ell = 1.5
        self.D_basal = 1.0
        self.D_min = 0.05
        self.V_sint = 0.5
        self.n_mov = 2.0
        self.eta_mov = 0.5
        
        # Spatial coordinates (local to each patch [0, L]^2)
        self.coord_x = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        self.coord_y = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        self.dosis = np.zeros(self.N, dtype=np.float32)
        
        # Patch assignment
        self.patch_id = np.repeat(np.arange(P, dtype=np.int16), N_per_patch)
        
        # Migration parameters (P x P rate matrix)
        self.m_pq0 = np.zeros((P, P), dtype=np.float32)
        self.I_star = 20         # Threshold cases to trigger lock
        self.pi_min = 0.1        # Residual mobility factor
        
        # Hubs parameters (optional per patch, default empty)
        self.H = 0
        self.hubs_coords = np.zeros((0, 2), dtype=np.float32)
        self.hubs_types = []
        self.hubs_lambda = np.zeros(0, dtype=np.float32)
        self.hubs_alpha = np.zeros(0, dtype=np.float32)
        self.hubs_beta = np.zeros(0, dtype=np.float32)
        self.hubs_kappa = np.zeros(0, dtype=np.float32)
        self.hubs_ell = np.zeros(0, dtype=np.float32)
        self.hubs_rho = np.zeros(0, dtype=np.float32)
        
    def _fase0_inicializacion(self, output_dir=None):
        super()._fase0_inicializacion(output_dir=output_dir)
        self.tau_infection = self.omega_in * self.tau_max
        self.remaining_visit_time = np.zeros((self.N, self.H), dtype=np.float32)

    def _fase2_contagio(self):
        # 1. MIGRATION OPERATOR (VIII.A.2)
        # Calculate total active infections across all patches
        total_I = np.sum(self.state == self.I)
        pi_t = self.pi_min if total_I >= self.I_star else 1.0
        
        # Check migration for each agent
        for p in range(self.P):
            agents_in_p = (self.patch_id == p) & ~self.quarantined & (self.state != self.D)
            idx_in_p = np.where(agents_in_p)[0]
            if len(idx_in_p) == 0:
                continue
                
            for q in range(self.P):
                if p == q:
                    continue
                prob_migrate = self.m_pq0[p, q] * pi_t * self.dt
                migrators = np.random.rand(len(idx_in_p)) < prob_migrate
                if np.any(migrators):
                    mig_idx = idx_in_p[migrators]
                    self.patch_id[mig_idx] = q
                    # Re-sample coordinates in destination patch
                    self.coord_x[mig_idx] = np.random.uniform(0.0, self.L, len(mig_idx))
                    self.coord_y[mig_idx] = np.random.uniform(0.0, self.L, len(mig_idx))
                    self.dosis[mig_idx] = 0.0  # Reset local exposure dose

        # 2. LOCAL CONTAGION PER PATCH
        # Force coordinates of agents on active agenda visits (if any)
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
                        can_visit = can_visit & ~starts_visit
                        
            visiting_any = np.any(self.remaining_visit_time > 0, axis=1)
            if np.any(visiting_any):
                visiting_hub_idx = np.argmax(self.remaining_visit_time > 0, axis=1)
                self.coord_x[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 0]
                self.coord_y[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 1]
        else:
            visiting_any = np.zeros(self.N, dtype=bool)
            visiting_hub_idx = np.zeros(self.N, dtype=np.int16)

        sigma = self.sigma_base * (1.0 - 0.5 * self.omega_ad)
        emission_avg = 0.5 * (self.prev_viral_load + self.viral_load) * np.exp(0.5 * (sigma**2) * self.dt)
        
        # Scale emission by rho_h for agents in active agenda visits
        if self.H > 0 and np.any(visiting_any):
            scale_factors = np.ones(self.N, dtype=np.float32)
            scale_factors[visiting_any] = self.hubs_rho[visiting_hub_idx[visiting_any]]
            emission_avg *= scale_factors

        # Calculate dose absorption and Langevin updates per patch independently
        R_t = np.zeros(self.N, dtype=np.float32)
        R_pred = np.zeros(self.N, dtype=np.float32)
        mask_I = (self.state == self.I) & ~self.quarantined
        
        # Calculate new positions (Langevin + Gravitational Drift)
        pred_x = self.coord_x.copy()
        pred_y = self.coord_y.copy()
        
        D_basal_agent = self.D_basal * (1.0 - self.c_DS * self.eta_mov)
        v_pow = self.viral_load**self.n_mov
        vsint_pow = self.V_sint**self.n_mov
        D_esp = np.where(
            self.viral_load > 0,
            D_basal_agent * (1.0 - v_pow / (v_pow + vsint_pow)) + self.D_min,
            D_basal_agent + self.D_min
        )
        D_esp = np.where(self.state == self.D, 0.0, D_esp)
        D_esp = np.where(self.quarantined, 0.0, D_esp)
        D_esp = np.where(visiting_any, 0.0, D_esp)

        # Gravitational Drift
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

        # Langevin step
        noise_x = np.random.normal(0.0, 1.0, self.N)
        noise_y = np.random.normal(0.0, 1.0, self.N)
        
        pred_x = self.coord_x + drift_x * self.dt + np.sqrt(2.0 * D_esp * self.dt) * noise_x
        pred_y = self.coord_y + drift_y * self.dt + np.sqrt(2.0 * D_esp * self.dt) * noise_y

        # Reflective boundaries per patch
        for coord in [pred_x, pred_y]:
            mask_low = (coord < 0.0)
            coord[mask_low] = -coord[mask_low]
            mask_high = (coord > self.L)
            coord[mask_high] = 2.0 * self.L - coord[mask_high]
            np.clip(coord, 0.0, self.L, out=coord)

        # Evaluate contagion fields per patch
        for p in range(self.P):
            mask_p = (self.patch_id == p)
            idx_p = np.where(mask_p)[0]
            if len(idx_p) == 0:
                continue
                
            # Current positions inside patch p
            coords_t_p = np.column_stack((self.coord_x[idx_p], self.coord_y[idx_p]))
            tree_t_p = KDTree(coords_t_p)
            r_cut = 3.0 * self.ell
            pairs_t_p = tree_t_p.query_pairs(r_cut)
            
            # Predict positions inside patch p
            coords_pred_p = np.column_stack((pred_x[idx_p], pred_y[idx_p]))
            tree_pred_p = KDTree(coords_pred_p)
            pairs_pred_p = tree_pred_p.query_pairs(r_cut)
            
            # Evaluate t_n field
            R_t_p = np.zeros(len(idx_p), dtype=np.float32)
            mask_I_p = mask_I[idx_p]
            emission_p = emission_avg[idx_p]
            
            if len(pairs_t_p) > 0:
                pairs_arr = np.array(list(pairs_t_p))
                i_idx = pairs_arr[:, 0]
                j_idx = pairs_arr[:, 1]
                dists = np.linalg.norm(coords_t_p[i_idx] - coords_t_p[j_idx], axis=1)
                kernel_vals = np.exp(-(dists**2) / (2.0 * self.ell**2))
                np.add.at(R_t_p, i_idx, kernel_vals * emission_p[j_idx] * mask_I_p[j_idx])
                np.add.at(R_t_p, j_idx, kernel_vals * emission_p[i_idx] * mask_I_p[i_idx])
                
            R_t[idx_p] = R_t_p
            
            # Evaluate t_n+1 predicted field
            R_pred_p = np.zeros(len(idx_p), dtype=np.float32)
            if len(pairs_pred_p) > 0:
                pairs_pred_arr = np.array(list(pairs_pred_p))
                i_idx_p = pairs_pred_arr[:, 0]
                j_idx_p = pairs_pred_arr[:, 1]
                dists_p = np.linalg.norm(coords_pred_p[i_idx_p] - coords_pred_p[j_idx_p], axis=1)
                kernel_vals_p = np.exp(-(dists_p**2) / (2.0 * self.ell**2))
                np.add.at(R_pred_p, i_idx_p, kernel_vals_p * emission_p[j_idx_p] * mask_I_p[j_idx_p])
                np.add.at(R_pred_p, j_idx_p, kernel_vals_p * emission_p[i_idx_p] * mask_I_p[i_idx_p])
                
            R_pred[idx_p] = R_pred_p

        # Update cumulative doses
        self.dosis = self.dosis * np.exp(-self.delta * self.dt) + 0.5 * (R_t + R_pred) * self.dt
        
        # Consolidate positions
        self.coord_x = pred_x
        self.coord_y = pred_y
        
        if self.H > 0 and np.any(visiting_any):
            self.coord_x[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 0]
            self.coord_y[visiting_any] = self.hubs_coords[visiting_hub_idx[visiting_any], 1]

        # S -> E transitions
        mask_S = (self.state == self.S)
        becomes_E = mask_S & (self.dosis >= self.tau_infection)
        
        # Track infection source within patch
        if np.any(becomes_E) and np.any(mask_I):
            newly_infected = np.where(becomes_E)[0]
            for idx_new in newly_infected:
                p = self.patch_id[idx_new]
                idx_I_p = np.where(mask_I & (self.patch_id == p))[0]
                if len(idx_I_p) > 0:
                    pos_new = np.array([self.coord_x[idx_new], self.coord_y[idx_new]])
                    coords_I_p = np.column_stack((self.coord_x[idx_I_p], self.coord_y[idx_I_p]))
                    dists_to_I = np.linalg.norm(coords_I_p - pos_new, axis=1)
                    closest_idx = np.argmin(dists_to_I)
                    parent_id = idx_I_p[closest_idx]
                    self.infected_by[idx_new] = parent_id
                    self.infections_caused[parent_id] += 1

        self.state[becomes_E] = self.E
        self.time_in_E[becomes_E] = np.random.negative_binomial(self.k_E, self.p_E, size=np.sum(becomes_E))

    def _fase5_buffer(self, t):
        super()._fase5_buffer(t)
        if 'patch_id' not in self.telemetry:
            self.telemetry['patch_id'] = []
        self.telemetry['patch_id'].append(self.patch_id.copy())

    def run(self, output_dir=None, n_seed=10, seed=None):
        # Override run to include patch compilation in telemetry dataframe
        self._fase0_inicializacion(output_dir=output_dir)
        
        # Seeding strictly in patch 0
        if seed is not None:
            np.random.seed(seed)
        idx_patch0 = np.where(self.patch_id == 0)[0]
        seed_idx = np.random.choice(idx_patch0, n_seed, replace=False)
        self.state[seed_idx] = self.I
        self.viral_load[seed_idx] = self.v_base + self.eps
        self.time_in_I[seed_idx] = 0
        self.t_inf[seed_idx] = 0
        
        self._fase5_buffer(0.0)
        
        for step in range(1, self.t_max + 1):
            current_time = step * self.dt
            self._fase1_ou_y_auc(current_time)
            self._fase2_contagio()
            self._fase3_transiciones(current_time)
            self._fase4_congelamiento()
            self._fase5_buffer(current_time)
            
        tiempo_arr = np.concatenate(self.telemetry['tiempo'])
        id_arr = np.concatenate(self.telemetry['id_agente'])
        estado_arr = np.concatenate(self.telemetry['estado'])
        carga_arr = np.concatenate(self.telemetry['carga_viral'])
        patch_arr = np.concatenate(self.telemetry['patch_id'])
        
        df_dinamico = pl.DataFrame({
            "tiempo": tiempo_arr,
            "id_agente": id_arr,
            "estado": estado_arr,
            "carga_viral": carga_arr,
            "patch_id": patch_arr
        })
        
        df_dinamico = df_dinamico.sort(["tiempo", "id_agente"])
        
        if output_dir is not None:
            path_dir = Path(output_dir)
            df_dinamico.write_parquet(path_dir / "telemetria_dinamica.parquet", compression="snappy")
            print(f"Exportado {path_dir / 'telemetria_dinamica.parquet'} (Multiparche)")
            
        return df_dinamico
