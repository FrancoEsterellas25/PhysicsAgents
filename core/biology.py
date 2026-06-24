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

def integrar_sde_ou_exacto(v_t, tau, w_ad, theta_low, theta_high, tau_peak, beta_ou, v_peak_base, v_base, k_ou, sigma_base, dt=1.0):
    """
    Integración exacta del proceso de Ornstein-Uhlenbeck con atractor dinámico dependiente de la fase.
    Evita errores de discretización de Euler-Maruyama.
    """
    # Theta dependiente del tiempo de infección tau y capacidad adaptativa
    theta_fase = np.where(tau < tau_peak, theta_low, theta_high)
    theta = theta_fase * (1.0 + beta_ou * w_ad)
    
    v_peak = v_peak_base * (1.0 - 0.5 * w_ad)
    sigma = sigma_base * (1.0 - 0.5 * w_ad)
    
    term1 = v_t * np.exp(-theta * dt)
    
    # Solución analítica del atractor dinámico mu(s) = v_base + (v_peak - v_base)*(1 - e^{-k*s})
    # Se evalúa en el punto medio (tau + dt/2) para mayor precisión en la fase ascendente.
    diff = theta - k_ou
    tau_mid = tau + dt / 2.0
    term2 = np.where(
        np.abs(diff) > 1e-5,
        v_peak * (1.0 - np.exp(-theta * dt)) - (v_peak - v_base) * np.exp(-k_ou * tau_mid) * (theta / diff) * (np.exp(-k_ou * dt) - np.exp(-theta * dt)),
        v_peak * (1.0 - np.exp(-theta * dt)) - (v_peak - v_base) * np.exp(-k_ou * tau_mid) * (theta * dt * np.exp(-theta * dt))
    )
    
    std_dev = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))
    noise = np.random.normal(0, 1, size=len(v_t))
    
    v_next = term1 + term2 + std_dev * noise
    return np.maximum(0.0, v_next)

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
