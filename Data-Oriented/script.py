import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def calcular_kaplan_meier(t_inf_arr, t_max):
    """
    Calcula la curva de supervivencia de Kaplan-Meier para el tiempo hasta la infección.
    t_inf_arr: array con el tiempo de infección de cada agente (o -1 si no se infectó).
    t_max: tiempo máximo de la simulación.
    """
    N = len(t_inf_arr)
    tiempos = np.where(t_inf_arr >= 0, t_inf_arr, t_max)
    eventos = np.where(t_inf_arr >= 0, 1, 0)  # 1 = Infección, 0 = Censura
    
    tiempos_unicos = np.unique(tiempos)
    tiempos_unicos = tiempos_unicos[tiempos_unicos >= 0]
    tiempos_unicos.sort()
    
    surv_prob = 1.0
    times = [0.0]
    probs = [1.0]
    
    for t in tiempos_unicos:
        d_j = np.sum((tiempos == t) & (eventos == 1))
        n_j = np.sum(tiempos >= t)
        
        if n_j > 0:
            surv_prob *= (1.0 - d_j / n_j)
            
        times.append(float(t))
        probs.append(float(surv_prob))
        
    return np.array(times), np.array(probs)

def estimar_r0_exacto(tiempo, infectados, tg):
    """
    Estima el R0 inicial ajustando la fase de crecimiento exponencial temprano
    y aplicando la relación exponencial exacta: R0 = exp(r * Tg).
    """
    ventana = (tiempo >= 1) & (tiempo <= 10) & (infectados > 0)
    x = tiempo[ventana]
    y = np.log(infectados[ventana])
    
    if len(x) < 2:
        return 1.0, 0.0
        
    slope, _ = np.polyfit(x, y, 1)
    r = max(0.0, slope)
    r0 = np.exp(r * tg)
    return r0, r

def ajustar_cox_tiempo_variable(df_dinamico, df_estatico):
    """
    Reestructura el dataset dinámico en un formato person-period panel
    y entrena un Modelo de Cox en Tiempo Discreto con alta regularización L2.
    """
    df_first_transition = df_dinamico.filter(pl.col("estado").is_in([1, 2])).group_by("id_agente").agg(pl.col("tiempo").min().alias("t_event"))
    
    df_panel = df_dinamico.join(df_first_transition, on="id_agente", how="left")
    t_max = df_dinamico["tiempo"].max()
    df_panel = df_panel.with_columns(
        pl.col("t_event").fill_null(t_max)
    )
    
    df_panel = df_panel.filter(pl.col("tiempo") <= pl.col("t_event"))
    df_panel = df_panel.with_columns(
        pl.when((pl.col("tiempo") == pl.col("t_event")) & (pl.col("estado").is_in([1, 2])))
        .then(1)
        .otherwise(0)
        .alias("evento_infeccion")
    )
    
    df_model = df_panel.join(df_estatico.select(["id_agente", "omega_in"]), on="id_agente", how="left")
    
    X_cols = []
    if "dosis" in df_model.columns:
        X_cols.append("dosis")
    X_cols.append("omega_in")
    
    X = df_model.select(X_cols).to_numpy()
    y = df_model["evento_infeccion"].to_numpy()
    
    if len(np.unique(y)) > 1:
        # Escalar covariables para estabilizar coeficientes
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ajustar regresión logística con alta regularización L2 (C=0.005) para suavizar la separación completa
        clf = LogisticRegression(penalty='l2', C=0.005, solver='lbfgs')
        clf.fit(X_scaled, y)
        coefs = clf.coef_[0]
        hrs = np.exp(coefs)
        
        return {col: hr for col, hr in zip(X_cols, hrs)}
    return None

def analizar_enfoque(nombre, path_dir, output_img_dir):
    mapa_path = path_dir / "mapa_estatico.parquet"
    telemetria_path = path_dir / "telemetria_dinamica.parquet"
    
    if not mapa_path.exists() or not telemetria_path.exists():
        print(f"[-] Archivos de {nombre} no encontrados en {path_dir}")
        return
        
    print(f"\n[+] Analizando Enfoque {nombre}...")
    df_estatico = pl.read_parquet(mapa_path)
    df_dinamico = pl.read_parquet(telemetria_path)
    
    alpha = 0.1
    tg = 2.0 + (1.0 / alpha)
    
    tiempos = df_dinamico["tiempo"].unique().sort().to_numpy()
    conteos = {
        'S': [], 'E': [], 'I': [], 'R': [], 'D': []
    }
    
    for t in tiempos:
        df_t = df_dinamico.filter(pl.col("tiempo") == t)
        estados = df_t["estado"].to_numpy()
        conteos['S'].append(np.sum(estados == 0))
        conteos['E'].append(np.sum(estados == 1))
        conteos['I'].append(np.sum(estados == 2))
        conteos['R'].append(np.sum(estados == 3))
        conteos['D'].append(np.sum(estados == 4))
        
    for k in conteos:
        conteos[k] = np.array(conteos[k])
        
    r0, r_rate = estimar_r0_exacto(tiempos, conteos['I'], tg=tg)
    print(f"    R0 estimado exacto (Tg = {tg:.2f}d): {r0:.4f} (tasa r = {r_rate:.4f}/día)")
    
    hrs = ajustar_cox_tiempo_variable(df_dinamico, df_estatico)
    if hrs:
        print("    Análisis de Cox en Tiempo Discreto (Riesgos Proporcionales, Regularizado por D.E.):")
        for cov, hr in hrs.items():
            print(f"      - Hazard Ratio para '{cov}': {hr:.4f} (e^beta)")
    
    df_inf = df_dinamico.filter(pl.col("estado").is_in([1, 2]))
    df_first_inf = df_inf.group_by("id_agente").agg(pl.col("tiempo").min().alias("t_inf"))
    df_survival = df_estatico.join(df_first_inf, on="id_agente", how="left").fill_null(-1)
    
    median_omega_in = df_survival["omega_in"].median()
    df_high_inn = df_survival.filter(pl.col("omega_in") >= median_omega_in)
    df_low_inn = df_survival.filter(pl.col("omega_in") < median_omega_in)
    
    times_km_high, probs_km_high = calcular_kaplan_meier(df_high_inn["t_inf"].to_numpy(), tiempos[-1])
    times_km_low, probs_km_low = calcular_kaplan_meier(df_low_inn["t_inf"].to_numpy(), tiempos[-1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(tiempos, conteos['S'], label='S (Susceptibles)', color='blue')
    ax1.plot(tiempos, conteos['E'], label='E (Expuestos)', color='orange')
    ax1.plot(tiempos, conteos['I'], label='I (Infectados)', color='red')
    ax1.plot(tiempos, conteos['R'], label='R (Recuperados)', color='green')
    ax1.plot(tiempos, conteos['D'], label='D (Muertos)', color='black')
    ax1.set_title(f"Evolución Epidemiológica SEIRS-D ({nombre})\n$R_0 \\approx {r0:.2f}$ (Exacto)")
    ax1.set_xlabel("Tiempo (Días)")
    ax1.set_ylabel("Agentes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.step(times_km_high, probs_km_high, where='post', label=r'Inmunidad Innata Alta ($\geq$ Mediana)', color='teal', lw=2)
    ax2.step(times_km_low, probs_km_low, where='post', label=r'Inmunidad Innata Baja (< Mediana)', color='crimson', lw=2)
    ax2.set_title(f"Supervivencia S(t) Estratificada - Tiempo a Infección ({nombre})")
    ax2.set_xlabel("Tiempo (Días)")
    ax2.set_ylabel("Probabilidad de permanecer S")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    img_path = output_img_dir / f"analisis_{nombre.lower()}.png"
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"    Gráfico guardado en {img_path}")
    
    cols_to_select = [c for c in ["id_agente", "omega_in", "omega_ad", "edad"] if c in df_estatico.columns]
    df_multidimensional = df_dinamico.join(df_estatico.select(cols_to_select), on="id_agente", how="left")
    
    dataset_path = path_dir / "dataset_agentes_analisis.parquet"
    df_multidimensional.write_parquet(dataset_path, compression="snappy")
    print(f"    Dataset multidimensional (panel) guardado en {dataset_path}")

def main():
    root_dir = Path(__file__).parent.parent
    output_img_dir = Path(__file__).parent
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    analizar_enfoque("Discreto", root_dir / "Discreto", output_img_dir)
    analizar_enfoque("Continuo", root_dir / "Continuo", output_img_dir)

if __name__ == "__main__":
    main()
