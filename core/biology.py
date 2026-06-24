import numpy as np
from scipy.stats import norm, beta

def generar_perfiles_inmunes(N, rho, beta_in_a, beta_in_b, beta_ad_a, beta_ad_b):
    """
    Genera el vector de vulnerabilidad correlacionado Omega_i = [omega_in, omega_ad]^T
    usando una Cópula Gaussiana para correlacionar la barrera innata con la capacidad adaptativa.
    """
    cov_matrix = [[1.0, rho], [rho, 1.0]]
    z = np.random.multivariate_normal([0, 0], cov_matrix, size=N)
    
    u1 = norm.cdf(z[:, 0])
    u2 = norm.cdf(z[:, 1])
    
    omega_in = beta.ppf(u1, beta_in_a, beta_in_b).astype(np.float32)
    omega_ad = beta.ppf(u2, beta_ad_a, beta_ad_b).astype(np.float32)
    
    return omega_in, omega_ad

def _ou_step(v_t, tau, theta, v_peak, v_base, k_ou, sigma, dt):
    """Evolución analítica de un paso individual del proceso de Ornstein-Uhlenbeck."""
    term1 = v_t * np.exp(-theta * dt)
    diff = theta - k_ou
    tau_mid = tau + dt / 2.0
    term2 = np.where(
        np.abs(diff) > 1e-5,
        v_peak * (1.0 - np.exp(-theta * dt)) - (v_peak - v_base) * np.exp(-k_ou * tau_mid) * (theta / diff) * (np.exp(-k_ou * dt) - np.exp(-theta * dt)),
        v_peak * (1.0 - np.exp(-theta * dt)) - (v_peak - v_base) * np.exp(-k_ou * tau_mid) * (theta * dt * np.exp(-theta * dt))
    )
    std_dev = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))
    noise = np.random.normal(0, 1, size=len(v_t))
    return term1 + term2 + std_dev * noise

def integrar_sde_ou_exacto(v_t, tau, w_ad, theta_low, theta_high, tau_peak, beta_ou, v_peak_base, v_base, k_ou, sigma_base, dt=1.0):
    """
    Integración exacta del proceso de Ornstein-Uhlenbeck con atractor dinámico dependiente de la fase.
    Evita errores de discretización de Euler-Maruyama y aplica step-splitting en la frontera tau_peak.
    """
    v_peak = v_peak_base * (1.0 - 0.5 * w_ad)
    sigma = sigma_base * (1.0 - 0.5 * w_ad)
    
    theta_low_arr = theta_low * (1.0 + beta_ou * w_ad)
    theta_high_arr = theta_high * (1.0 + beta_ou * w_ad)
    
    dt_pico = tau_peak - tau
    
    # ponytail: Máscara vectorial para detectar el cruce exacto de frontera en el intervalo
    mask_split = (dt_pico > 0.0) & (dt_pico < dt)
    mask_no_split = ~mask_split
    
    v_next = np.empty_like(v_t)
    auc_inc = np.empty_like(v_t)
    
    if np.any(mask_no_split):
        theta_fase = np.where(tau[mask_no_split] < tau_peak, theta_low_arr[mask_no_split], theta_high_arr[mask_no_split])
        v_next_no_split = _ou_step(
            v_t[mask_no_split],
            tau[mask_no_split],
            theta_fase,
            v_peak[mask_no_split],
            v_base,
            k_ou,
            sigma[mask_no_split],
            dt
        )
        v_next_no_split_clipped = np.maximum(0.0, v_next_no_split)
        v_next[mask_no_split] = v_next_no_split_clipped
        auc_inc[mask_no_split] = 0.5 * (v_t[mask_no_split] + v_next_no_split_clipped) * dt
        
    if np.any(mask_split):
        dt1 = dt_pico[mask_split]
        dt2 = dt - dt1
        
        # Sub-paso 1: Hasta el pico con theta_low
        v_star = _ou_step(
            v_t[mask_split],
            tau[mask_split],
            theta_low_arr[mask_split],
            v_peak[mask_split],
            v_base,
            k_ou,
            sigma[mask_split],
            dt1
        )
        v_star_clipped = np.maximum(0.0, v_star)
        
        # Sub-paso 2: Desde el pico en adelante con theta_high
        tau2 = np.full_like(dt1, tau_peak)
        v_next_split = _ou_step(
            v_star_clipped,
            tau2,
            theta_high_arr[mask_split],
            v_peak[mask_split],
            v_base,
            k_ou,
            sigma[mask_split],
            dt2
        )
        v_next_split_clipped = np.maximum(0.0, v_next_split)
        v_next[mask_split] = v_next_split_clipped
        
        # ponytail: integracion trapezoidal de dos etapas ponderada por la fraccion del paso
        auc_inc[mask_split] = 0.5 * (v_t[mask_split] + v_star_clipped) * dt1 + 0.5 * (v_star_clipped + v_next_split_clipped) * dt2
        
    return v_next, auc_inc

def resolver_negbin_params(mu, M):
    """
    Resuelve los parámetros r (o k) y p de una distribución binomial negativa
    dados la media mu y la moda M.
    mu = r(1-p)/p
    M = (r-1)(1-p)/p  (si r > 1, si no M = 0)
    """
    if mu <= 0:
        raise ValueError("La media mu debe ser mayor que 0")
    if M < 0:
        raise ValueError("La moda M debe ser no negativa")
        
    if M >= mu:
        # En una binomial negativa real con p in (0, 1), M < mu siempre se cumple.
        # Si la moda especificada es inválida, usamos una aproximación por defecto.
        M = max(0.0, mu - 1.0)
        
    if M == 0:
        # Caso límite o r <= 1
        p = 0.5
        r = mu
    else:
        # mu = r*(1-p)/p
        # M = (r-1)*(1-p)/p
        # mu - M = (1-p)/p = 1/p - 1 => 1/p = mu - M + 1 => p = 1 / (mu - M + 1)
        p = 1.0 / (mu - M + 1.0)
        r = mu * p / (1.0 - p)
        
    return r, p
