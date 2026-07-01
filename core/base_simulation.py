import numpy as np
import polars as pl
from pathlib import Path
from core.biology import generar_perfiles_inmunes, integrar_sde_ou_exacto, resolver_negbin_params

class BaseSEIRSDSimulation:
    def __init__(self, N, t_max=30):
        """
        Clase base unificada para la simulación del Modelo Epidemiológico SEIRS-D.
        Contiene el Núcleo Biológico Universal y las transiciones de compartimentos comunes.
        """
        self.N = N
        self.t_max = t_max
        
        # Parámetros Globales (Por defecto del Núcleo Biológico)
        self.rho = 0.5           # Correlación sistémica (Cópula)
        self.beta_in_a = 2.0     # Forma Alpha Beta_in
        self.beta_in_b = 2.0     # Forma Beta Beta_in
        self.beta_ad_a = 2.0     # Forma Alpha Beta_ad
        self.beta_ad_b = 5.0     # Forma Beta Beta_ad
        
        # Parámetros SDE-OU (Dinámica Viral)
        self.theta_low = 0.1
        self.theta_high = 0.5
        self.tau_peak = 5
        self.beta_ou = 0.5       # Sensibilidad theta a omega_ad
        self.v_base = 0.1
        self.v_peak_base = 10.0
        self.k_ou = 0.3
        self.sigma_base = 1.0
        self.eps = 0.01          # Ruido inicial viral
        
        # Contagio y Biología Común
        self.alpha = 0.1         # Probabilidad base salida de I
        self.p_base = 1.0 - (0.992)**(1.0/365.0) # Mortalidad demográfica base diaria
        self.k_E = 2             # Etapas de incubación (Binomial Negativa)
        self.p_E = 0.5           # Probabilidad geométrica en incubación
        
        # Letalidad (Logística)
        self.w1 = 0.6
        self.w2 = 0.4
        self.lam = 5.0
        self.tau_max = 20.0
        self.ell = 1.0
        self.delta_ext = 1.0
        self.delta_cerrado = 0.2
        self.delta_abierto = 4.0
        # Normalización analítica del AUC (se calcula en Fase 0 tras posibles cambios de hiperparámetros)
        self.auc_norm_factor = None
        
        # Parámetros clínicos para pérdida de inmunidad (R -> S)
        self.mu_R = 180.0       # Media de días en estado R (ej: ~6 meses)
        self.M_R = 150.0        # Moda de días en estado R
        self.k_R, self.p_R = None, None
        
        # Estados: 0=S, 1=E, 2=I, 3=R, 4=D
        self.S, self.E, self.I, self.R, self.D = 0, 1, 2, 3, 4
        
        # Arrays de estado de los agentes
        self.id_agente = np.arange(self.N, dtype=np.int32)
        self.dt = 1.0            # Paso de tiempo de la simulación
        self.state = np.zeros(self.N, dtype=np.int8)
        self.viral_load = np.zeros(self.N, dtype=np.float32)
        self.prev_viral_load = np.zeros(self.N, dtype=np.float32)
        self.auc = np.zeros(self.N, dtype=np.float32)
        self.t_inf = np.full(self.N, -1.0, dtype=np.float32) # ponytail: -1.0 representa vacio (sin infeccion)
        
        # Tiempos acumulados por compartimento
        self.time_in_E = np.zeros(self.N, dtype=np.float32)
        self.time_in_I = np.zeros(self.N, dtype=np.float32)
        self.time_in_R = np.zeros(self.N, dtype=np.float32)
        
        # Perfiles inmunológicos (se inicializan en Fase 0)
        self.omega_in = None
        self.omega_ad = None
        
        # Buffer de Telemetría Dinámica
        self.telemetry = {
            'tiempo': [],
            'id_agente': [],
            'estado': [],
            'carga_viral': []
        }

    def _fase0_inicializacion(self, output_dir=None):
        """Inicializa perfiles inmunes correlacionados y exporta el mapa estático."""
        self.omega_in, self.omega_ad = generar_perfiles_inmunes(
            self.N, self.rho,
            self.beta_in_a, self.beta_in_b,
            self.beta_ad_a, self.beta_ad_b
        )
        self.auc_norm_factor = self.v_peak_base * self.tau_max
        self.k_R, self.p_R = resolver_negbin_params(self.mu_R, self.M_R)
        
        # Si se provee output_dir, exportamos mapa_estatico.parquet
        if output_dir is not None:
            path_dir = Path(output_dir)
            path_dir.mkdir(parents=True, exist_ok=True)
            
            df_estatico = pl.DataFrame({
                "id_agente": self.id_agente,
                "omega_in": self.omega_in,
                "omega_ad": self.omega_ad
            })
            df_estatico.write_parquet(path_dir / "mapa_estatico.parquet", compression="snappy")
            print(f"Exportado {path_dir / 'mapa_estatico.parquet'}")

    def seed_infection(self, n=5, seed=None):
        """
        Inyecta el patógeno en n agentes mediante una siembra aleatoria uniforme (Random Uniform Seeding)
        sin reemplazo sobre el total de la población N.
        Fija la semilla del generador para reproducibilidad exacta del subconjunto de pacientes cero I₀.
        """
        if seed is not None:
            np.random.seed(seed)
        idx = np.random.choice(self.N, n, replace=False)
        self.state[idx] = self.I
        self.viral_load[idx] = self.v_base + self.eps
        self.time_in_I[idx] = 0
        self.t_inf[idx] = 0

    def _fase1_ou_y_auc(self, current_time=None):
        """Integración SDE-OU exacta y actualización del AUC para el estado I."""
        self.prev_viral_load = self.viral_load.copy()
        mask_I = (self.state == self.I)
        if not np.any(mask_I):
            return
            
        if current_time is not None:
            tau = (current_time - self.dt) - self.t_inf[mask_I]
        else:
            tau = self.time_in_I[mask_I]
            
        w_ad = self.omega_ad[mask_I]
        v_t = self.viral_load[mask_I]
        
        # Integración exacta del proceso OU y cálculo exacto de AUC
        v_next, auc_inc = integrar_sde_ou_exacto(
            v_t, tau, w_ad,
            self.theta_low, self.theta_high, self.tau_peak, self.beta_ou,
            self.v_peak_base, self.v_base, self.k_ou, self.sigma_base, dt=self.dt
        )
        
        # Acumulación exacta de AUC
        self.auc[mask_I] += auc_inc
        self.viral_load[mask_I] = v_next

    def _fase2_contagio(self):
        """Mecanismo de contagio. Debe ser implementado por la subclase específica."""
        raise NotImplementedError("El mecanismo de contagio de la fase 2 debe implementarse por el enfoque específico (Discreto/Continuo)")

    def _fase3_transiciones(self, t=None):
        """E->I, I->R, I->D, R->S y p_base."""
        # 1. Ruido de Fondo (Mortalidad demográfica base escalada con dt)
        alive = (self.state != self.D)
        p_base_step = 1.0 - (1.0 - self.p_base) ** self.dt
        dies_base = alive & (np.random.rand(self.N) < p_base_step)
        self.state[dies_base] = self.D
        
        # 2. E -> I (Transición tras incubación en NegBin(2, 0.5) medida en días)
        mask_E = (self.state == self.E) & ~dies_base
        ready_to_I = mask_E & (self.time_in_E <= 0.0)
        self.state[ready_to_I] = self.I
        self.viral_load[ready_to_I] = self.v_base + self.eps
        self.time_in_I[ready_to_I] = 0.0
        if t is not None:
            self.t_inf[ready_to_I] = t
        self.auc[ready_to_I] = 0.0
        self.time_in_E[mask_E & ~ready_to_I] -= self.dt
        
        # 3. I -> R o D (Letalidad basada en estrés biológico neto y capacidad adaptativa)
        mask_I = (self.state == self.I) & ~dies_base & ~ready_to_I
        if t is not None:
            self.time_in_I[mask_I] = t - self.t_inf[mask_I]
        else:
            self.time_in_I[mask_I] += self.dt
        
        # Salida exponencial con media 1/alpha de días
        p_exit = 1.0 - np.exp(-self.alpha * self.dt)
        exiting = (np.random.rand(np.sum(mask_I)) < p_exit)
        
        if np.any(exiting):
            idx_exiting = np.where(mask_I)[0][exiting]
            
            auc_norm = np.clip(self.auc[idx_exiting] / self.auc_norm_factor, 0.0, 1.0) # ponytail: evitar que fluctuaciones estocasticas superen 1.0
            tau_ratio = np.clip(self.time_in_I[idx_exiting] / self.tau_max, 0, 1)
            
            # Índice de estrés biológico neto
            E_index = self.w1 * auc_norm + self.w2 * tau_ratio
            # Umbral de letalidad individualizado
            mu_v = 1.0 / (1.0 + np.exp(-self.lam * (E_index - self.omega_ad[idx_exiting])))
            
            dies_virus = (np.random.rand(len(idx_exiting)) <= mu_v)
            idx_D = idx_exiting[dies_virus]
            idx_R = idx_exiting[~dies_virus]
            
            self.state[idx_D] = self.D
            self.state[idx_R] = self.R
            # Pérdida de inmunidad modelada vía NegBin matemática (días)
            self.time_in_R[idx_R] = np.random.negative_binomial(self.k_R, self.p_R, size=len(idx_R))
            
        # 4. R -> S (Pérdida de inmunidad en días)
        mask_R = (self.state == self.R) & ~dies_base
        ready_to_S = mask_R & (self.time_in_R <= 0.0)
        self.state[ready_to_S] = self.S
        self.time_in_R[mask_R & ~ready_to_S] -= self.dt

    def _fase4_congelamiento(self):
        """Fuerza invariantes físicas sobre el estado D."""
        mask_D = (self.state == self.D)
        self.viral_load[mask_D] = 0.0
        self.auc[mask_D] = 0.0

    def _fase5_buffer(self, t):
        """Guarda el estado actual en el buffer de telemetría."""
        self.telemetry['tiempo'].append(np.full(self.N, t, dtype=np.float32)) # ponytail: usar float32 para admitir pasos de tiempo fraccionales
        self.telemetry['id_agente'].append(self.id_agente.copy())
        self.telemetry['estado'].append(self.state.copy())
        self.telemetry['carga_viral'].append(self.viral_load.copy())

    def run(self, output_dir=None, n_seed=10, seed=None):
        """Bucle principal de la simulación."""
        self._fase0_inicializacion(output_dir=output_dir)
        self.seed_infection(n=n_seed, seed=seed)
        
        # Guardar la foto exacta del Día 0 (con los pacientes cero recién inyectados)
        self._fase5_buffer(0.0)
        
        for step in range(1, self.t_max + 1):
            current_time = step * self.dt
            self._fase1_ou_y_auc(current_time)
            self._fase2_contagio()
            self._fase3_transiciones(current_time)
            self._fase4_congelamiento()
            self._fase5_buffer(current_time)
            
        # Compilación de la telemetría dinámica final
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
        
        # ORDENAR POR TIEMPO Y LUEGO ID_AGENTE
        df_dinamico = df_dinamico.sort(["tiempo", "id_agente"])
        
        if output_dir is not None:
            path_dir = Path(output_dir)
            df_dinamico.write_parquet(path_dir / "telemetria_dinamica.parquet", compression="snappy")
            print(f"Exportado {path_dir / 'telemetria_dinamica.parquet'} (Orden estricto validado)")
            
        return df_dinamico
