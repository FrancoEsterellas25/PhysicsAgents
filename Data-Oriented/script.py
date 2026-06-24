import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

def calcular_kaplan_meier(t_inf_arr, t_max):
    """
    Calcula la curva de supervivencia de Kaplan-Meier para el tiempo hasta la infección.
    t_inf_arr: array con el tiempo de infección de cada agente (o -1 si no se infectó).
    t_max: tiempo máximo de la simulación.
    """
    N = len(t_inf_arr)
    # Tiempos de evento: si se infectó, es t_inf; si no, es censura al final (t_max)
    tiempos = np.where(t_inf_arr >= 0, t_inf_arr, t_max)
    eventos = np.where(t_inf_arr >= 0, 1, 0)  # 1 = Infección, 0 = Censura
    
    tiempos_unicos = np.unique(tiempos)
    tiempos_unicos = tiempos_unicos[tiempos_unicos >= 0]
    tiempos_unicos.sort()
    
    surv_prob = 1.0
    times = [0.0]
    probs = [1.0]
    
    for t in tiempos_unicos:
        # Número de eventos en t
        d_j = np.sum((tiempos == t) & (eventos == 1))
        # Número de agentes en riesgo justo antes de t (tiempos >= t)
        n_j = np.sum(tiempos >= t)
        
        if n_j > 0:
            surv_prob *= (1.0 - d_j / n_j)
            
        times.append(float(t))
        probs.append(float(surv_prob))
        
    return np.array(times), np.array(probs)

def estimar_r0(tiempo, infectados, tg=5.0):
    """
    Estima el R0 inicial ajustando la fase de crecimiento exponencial temprano.
    R0 = 1 + r * Tg
    donde r es la tasa de crecimiento exponencial y Tg es el tiempo generacional.
    """
    # Tomar la ventana inicial (por ejemplo, los primeros 10 días o hasta que alcance un pico temprano)
    # Filtramos donde los infectados sean mayores a 0 y estén en la fase inicial de crecimiento
    ventana = (tiempo >= 1) & (tiempo <= 10) & (infectados > 0)
    x = tiempo[ventana]
    y = np.log(infectados[ventana])
    
    if len(x) < 2:
        return 0.0, 0.0
        
    # Ajuste lineal simple: log(I) = r * t + C
    slope, _ = np.polyfit(x, y, 1)
    r = max(0.0, slope)
    r0 = 1.0 + r * tg
    return r0, r

def analizar_enfoque(nombre, path_dir, output_img_dir):
    mapa_path = path_dir / "mapa_estatico.parquet"
    telemetria_path = path_dir / "telemetria_dinamica.parquet"
    
    if not mapa_path.exists() or not telemetria_path.exists():
        print(f"[-] Archivos de {nombre} no encontrados en {path_dir}")
        return
        
    print(f"\n[+] Analizando Enfoque {nombre}...")
    df_estatico = pl.read_parquet(mapa_path)
    df_dinamico = pl.read_parquet(telemetria_path)
    
    # 1. Curvas SEIRS-D Agregadas
    # Estados: 0=S, 1=E, 2=I, 3=R, 4=D
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
        
    # 2. Estimación de R0
    r0, r_rate = estimar_r0(tiempos, conteos['I'], tg=5.0)
    print(f"    R0 estimado en fase exponencial: {r0:.4f} (tasa r = {r_rate:.4f}/día)")
    
    # 3. Kaplan-Meier (Tiempo hasta la infección)
    # Encontrar primer tiempo t donde el estado es E o I para cada agente
    # Si nunca cambia de S, se considera censura.
    df_inf = df_dinamico.filter(pl.col("estado").is_in([1, 2]))
    # Agrupar por id_agente y tomar el tiempo mínimo
    df_first_inf = df_inf.group_by("id_agente").agg(pl.col("tiempo").min().alias("t_inf"))
    
    # Unir con el mapa estático para tener todos los agentes
    df_survival = df_estatico.join(df_first_inf, on="id_agente", how="left").fill_null(-1)
    t_inf_arr = df_survival["t_inf"].to_numpy()
    t_max = tiempos[-1]
    
    times_km, probs_km = calcular_kaplan_meier(t_inf_arr, t_max)
    
    # 4. Generación de Gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de Compartimentos
    ax1.plot(tiempos, conteos['S'], label='S (Susceptibles)', color='blue')
    ax1.plot(tiempos, conteos['E'], label='E (Expuestos)', color='orange')
    ax1.plot(tiempos, conteos['I'], label='I (Infectados)', color='red')
    ax1.plot(tiempos, conteos['R'], label='R (Recuperados)', color='green')
    ax1.plot(tiempos, conteos['D'], label='D (Muertos)', color='black')
    ax1.set_title(f"Evolución Epidemiológica SEIRS-D ({nombre})\n$R_0 \\approx {r0:.2f}$")
    ax1.set_xlabel("Tiempo (Días)")
    ax1.set_ylabel("Agentes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico Kaplan-Meier
    ax2.step(times_km, probs_km, where='post', label='Curva Kaplan-Meier (S)', color='purple', lw=2)
    ax2.set_title(f"Función de Supervivencia S(t) - Tiempo hasta Infección ({nombre})")
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
    
    # 5. Dataset Multidimensional por Agente
    # Para cada agente, recolectamos: id, omega_in, omega_ad, t_inf, carga_viral promedio, dosis final, auc final
    df_viral_stats = df_dinamico.group_by("id_agente").agg([
        pl.col("carga_viral").mean().alias("v_promedio"),
        pl.col("carga_viral").max().alias("v_max")
    ])
    
    df_agentes_final = df_survival.join(df_viral_stats, on="id_agente", how="left")
    
    # Guardar dataset del agente
    dataset_path = path_dir / "dataset_agentes_analisis.parquet"
    df_agentes_final.write_parquet(dataset_path, compression="snappy")
    print(f"    Dataset compilado guardado en {dataset_path}")

def main():
    root_dir = Path(__file__).parent.parent
    output_img_dir = Path(__file__).parent
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Analizar Enfoque Discreto
    analizar_enfoque("Discreto", root_dir / "Discreto", output_img_dir)
    
    # Analizar Enfoque Continuo
    analizar_enfoque("Continuo", root_dir / "Continuo", output_img_dir)

if __name__ == "__main__":
    main()
