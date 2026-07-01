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
        self.delta_ext = 1.0
        self.delta_cerrado = 0.2
        self.delta_abierto = 4.0
        self.ell = 1.0           # Longitud de escala del kernel gaussiano
        self.D_basal = 1.0       # Difusividad basal
        self.D_min = 0.05        # Difusividad mínima
        self.V_sint = 0.5        # Umbral de inhibición motora (carga viral)
        self.n_mov = 2.0         # Exponente de Hill para movimiento
        
        # Coordenadas iniciales continuas: x_i(0) ~ Uniforme([0, L]^2)
        self.coord_x = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        self.coord_y = np.random.uniform(0.0, self.L, self.N).astype(np.float32)
        self.hogar_x = self.coord_x.copy()
        self.hogar_y = self.coord_y.copy()
        
        # Dosis acumulada
        self.dosis = np.zeros(self.N, dtype=np.float32)
        self.tau_infection = None

        # Inicializar coordenadas de partículas de aerosol lagrangianas
        self.aerosol_coords = np.zeros((0, 2), dtype=np.float32)
        self.aerosol_dosis = np.zeros(0, dtype=np.float32)
        self.aerosol_decay = np.zeros(0, dtype=np.float32)
        
        # Telemetría de virus
        self.telemetry_virus = {
            'tiempo': [],
            'aerosol_x': [],
            'aerosol_y': [],
            'aerosol_dosis': []
        }


    def _fase0_inicializacion(self, output_dir=None):
        """Inicializa perfiles inmunes en core y exporta el mapa estático del enfoque continuo."""
        super()._fase0_inicializacion(output_dir=output_dir)
        self.tau_infection = self.omega_in * self.tau_max * (1.0 + getattr(self, 'eta_hig', 0.0))
        self.has_mask = np.random.rand(self.N) < getattr(self, 'barbijo_cumplimiento', 0.0)
        
        base_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        
        df_estatico = pl.DataFrame({
            "id_agente": self.id_agente,
            "omega_in": self.omega_in,
            "omega_ad": self.omega_ad,
            "tau_infection": self.tau_infection,
            "hogar_x": self.hogar_x,
            "hogar_y": self.hogar_y
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
        
        # Calcular prevalencia actual para gatillar intervenciones
        pct_infected = np.sum(self.state == self.I) / self.N
        
        # Gatillo de distanciamiento social (reduce el radio efectivo del aerosol ell)
        ds_active = (getattr(self, 'c_DS', 0.0) > 0.0) and (pct_infected >= getattr(self, 'ds_trigger_pct', 0.05))
        current_ell = self.ell * (1.0 - 0.5 * getattr(self, 'c_DS', 0.0)) if ds_active else self.ell
        
        # Reducir emisión si el emisor usa mascarilla
        mask_em_mult = np.where(self.has_mask, 1.0 - getattr(self, 'eta_em', 0.6), 1.0)
        # Compensador de Itô en la emisión: promedio trapezoidal de V_t y V_t+dt
        emission_avg = 0.5 * (self.prev_viral_load + self.viral_load) * np.exp(0.5 * (sigma**2) * self.dt) * mask_em_mult
        
        coords_t = np.column_stack((self.coord_x, self.coord_y))
        tree_t = KDTree(coords_t)
        
        r_cut = 3.0 * current_ell
        pairs_t = tree_t.query_pairs(r_cut)
        
        R_t = np.zeros(self.N, dtype=np.float32)
        mask_I = (self.state == self.I)
        
        if len(pairs_t) > 0:
            pairs_arr = np.array(list(pairs_t))
            i_idx = pairs_arr[:, 0]
            j_idx = pairs_arr[:, 1]
            dists = np.linalg.norm(coords_t[i_idx] - coords_t[j_idx], axis=1)
            kernel_vals = np.exp(-(dists**2) / (2.0 * current_ell**2))
            
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
        
        # Gatillo de cuarentena doméstica (los infectados y expuestos reducen su movilidad a 0)
        quarantine_active = getattr(self, 'enable_quarantine', False) and (pct_infected >= getattr(self, 'quarantine_trigger_pct', 0.05))
        if quarantine_active:
            D_esp[self.state == self.I] = 0.0
            D_esp[self.state == self.E] = 0.0
        
        # Paso de predicción espacial (ruido browniano en 2D)
        noise_x = np.random.normal(0.0, 1.0, self.N)
        noise_y = np.random.normal(0.0, 1.0, self.N)
        
        # Stratonovich: usamos el coeficiente en t_n+1 (difusividad evaluada con el V corregido)
        pred_x = self.coord_x + np.sqrt(2.0 * D_esp * self.dt) * noise_x
        pred_y = self.coord_y + np.sqrt(2.0 * D_esp * self.dt) * noise_y
        
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
        # Determinar decaimiento delta por agente dependiendo de si está dentro de un hub o en tránsito
        if getattr(self, 'hubs_activos', False):
            # Centros de los 4 hubs principales del espacio [0, L]^2
            hubs = np.array([
                [0.25 * self.L, 0.25 * self.L],
                [0.75 * self.L, 0.25 * self.L],
                [0.25 * self.L, 0.75 * self.L],
                [0.75 * self.L, 0.75 * self.L]
            ])
            r_hub = 0.1 * self.L  # Radio de influencia del hub (10% del tamaño del mapa)
            coords = np.column_stack((self.coord_x, self.coord_y))
            
            # Identificar agentes dentro de la zona de cualquier hub
            in_hub = np.any([np.linalg.norm(coords - hub, axis=1) < r_hub for hub in hubs], axis=0)
            
            # delta_cerrado en hubs, delta_abierto en exteriores/tránsito
            delta_step = np.where(in_hub, getattr(self, 'delta_cerrado', 0.2), getattr(self, 'delta_abierto', 4.0))
        else:
            # Movimiento libre / calles sin hubs activos
            delta_step = getattr(self, 'delta_ext', 1.0)

        # Reducir dosis absorbida si el receptor usa mascarilla
        mask_rec_mult = np.where(self.has_mask, 1.0 - getattr(self, 'eta_rec', 0.5), 1.0)
        # Decaimiento ambiental δ por agente analíticamente estable + promedio de dosis
        self.dosis = self.dosis * np.exp(-delta_step * self.dt) + 0.5 * (R_t + R_pred) * self.dt * mask_rec_mult
        
        # --- 4.1 SIMULACIÓN DE PARTÍCULAS DE AEROSOL PARA LA ANIMACIÓN ---
        # Decaimiento y actualización de partículas existentes
        if len(self.aerosol_coords) > 0:
            self.aerosol_dosis *= np.exp(-self.aerosol_decay * self.dt)
            keep_mask = (self.aerosol_dosis > 0.05)
            self.aerosol_coords = self.aerosol_coords[keep_mask]
            self.aerosol_dosis = self.aerosol_dosis[keep_mask]
            self.aerosol_decay = self.aerosol_decay[keep_mask]
            
            if len(self.aerosol_coords) > 0:
                # Deriva leve de los aerosoles en el aire
                self.aerosol_coords += np.random.normal(0.0, 0.08, size=self.aerosol_coords.shape)
                
        # Emisión de nuevas partículas por parte de agentes infectados (I)
        if np.any(mask_I):
            new_coords = []
            new_dosis = []
            new_decay = []
            for idx in np.where(mask_I)[0]:
                x = self.coord_x[idx]
                y = self.coord_y[idx]
                
                # Obtener la tasa de decaimiento en el punto del agente
                if getattr(self, 'hubs_activos', False):
                    in_any_hub = np.any([np.linalg.norm(np.array([x, y]) - hub) < r_hub for hub in hubs])
                    dec = getattr(self, 'delta_cerrado', 0.2) if in_any_hub else getattr(self, 'delta_abierto', 4.0)
                else:
                    dec = getattr(self, 'delta_ext', 1.0)
                
                # Emitir 2 partículas de aerosol
                for _ in range(2):
                    new_coords.append([x + np.random.normal(0.0, 0.15), y + np.random.normal(0.0, 0.15)])
                    new_dosis.append(1.0)
                    new_decay.append(dec)
            
            if len(new_coords) > 0:
                if len(self.aerosol_coords) > 0:
                    self.aerosol_coords = np.vstack((self.aerosol_coords, np.array(new_coords, dtype=np.float32)))
                else:
                    self.aerosol_coords = np.array(new_coords, dtype=np.float32)
                self.aerosol_dosis = np.concatenate((self.aerosol_dosis, np.array(new_dosis, dtype=np.float32)))
                self.aerosol_decay = np.concatenate((self.aerosol_decay, np.array(new_decay, dtype=np.float32)))
                
        # Limitar número máximo de partículas a 1200 por rendimiento
        if len(self.aerosol_coords) > 1200:
            idx_sorted = np.argsort(self.aerosol_dosis)[::-1][:1200]
            self.aerosol_coords = self.aerosol_coords[idx_sorted]
            self.aerosol_dosis = self.aerosol_dosis[idx_sorted]
            self.aerosol_decay = self.aerosol_decay[idx_sorted]

        
        
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

        self.telemetry_virus['tiempo'].append(t)
        if len(self.aerosol_coords) > 0:
            self.telemetry_virus['aerosol_x'].append(self.aerosol_coords[:, 0].copy())
            self.telemetry_virus['aerosol_y'].append(self.aerosol_coords[:, 1].copy())
            self.telemetry_virus['aerosol_dosis'].append(self.aerosol_dosis.copy())
        else:
            self.telemetry_virus['aerosol_x'].append(np.zeros(0, dtype=np.float32))
            self.telemetry_virus['aerosol_y'].append(np.zeros(0, dtype=np.float32))
            self.telemetry_virus['aerosol_dosis'].append(np.zeros(0, dtype=np.float32))


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

            # Exportar historial dinámico de partículas de aerosol
            df_virus = pl.DataFrame({
                "tiempo": self.telemetry_virus['tiempo'],
                "aerosol_x": [list(x) for x in self.telemetry_virus['aerosol_x']],
                "aerosol_y": [list(y) for y in self.telemetry_virus['aerosol_y']],
                "aerosol_dosis": [list(d) for d in self.telemetry_virus['aerosol_dosis']]
            })
            df_virus.write_parquet(path_dir / "telemetria_virus.parquet", compression="snappy")
            print(f"Exportado {path_dir / 'telemetria_virus.parquet'} (Continuo)")

            
        return df_dinamico

if __name__ == '__main__':
    print("Iniciando Motor ABM (Enfoque Continuo)...")
    sim = ContinuousSEIRSDSimulation(N=1000, L=100.0, t_max=10)
    sim.run()
    print("Simulación continua completada exitosamente.")
