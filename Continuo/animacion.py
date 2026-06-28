from manim import *
import polars as pl
import numpy as np
from pathlib import Path

# ponytail: allow Manim to use default/configured frame rates (15/30/60) for smooth playback

class EscenaEpidemiologicaContinuo(Scene):
    def construct(self):
        # 1. CARGA DE DATOS (Polars)
        base_dir = Path(__file__).parent
        try:
            df_estatico = pl.read_parquet(base_dir / "mapa_estatico.parquet")
            df_dinamico = pl.read_parquet(base_dir / "telemetria_dinamica.parquet")
        except Exception as e:
            print(f"Error cargando los archivos parquet: {e}")
            print("Asegúrate de ejecutar continuous_simulation.py primero para generarlos.")
            return

        N = len(df_estatico)
        tiempos = df_dinamico["tiempo"].unique(maintain_order=True).to_numpy()
        T_max = len(tiempos)
        
        # Encontrar límites espaciales (L se asume por los datos)
        max_x_val = df_dinamico["coord_x"].max()
        max_y_val = df_dinamico["coord_y"].max()
        L = float(max(max_x_val, max_y_val))

        # 2. DEFINICIÓN DE COLORES
        COLOR_MAP = {
            0: "#1F77B4",  # S: Azul
            1: "#FF7F0E",  # E: Naranja Claro
            2: "#D62728",  # I: Rojo Intenso
            3: "#2CA02C",  # R: Verde
            4: "#7F7F7F"   # D: Gris Oscuro
        }

        # 3. DISEÑO DE LA ESCENA (Layout)
        box_width = 7.0
        box_height = 7.0
        offset_x = -3.5
        offset_y = 0.0

        # Dibujar marco del espacio físico continuo
        marco = Rectangle(width=box_width, height=box_height, color=WHITE, stroke_width=2)
        marco.move_to(np.array([offset_x, offset_y, 0]))
        self.add(marco)

        # Mapeo de coordenadas continuas [0, L]^2 a coordenadas de pantalla Manim
        def mapear_posicion(cx, cy):
            px = offset_x - (box_width / 2.0) + (cx / L) * box_width
            py = offset_y - (box_height / 2.0) + (cy / L) * box_height
            return np.array([px, py, 0])

        # ponytail: Draw hubs if hubs.parquet exists
        try:
            df_hubs = pl.read_parquet(base_dir / "hubs.parquet")
            for idx, row in enumerate(df_hubs.iter_rows(named=True)):
                hx, hy, htipo = row["x"], row["y"], row["tipo"]
                pos = mapear_posicion(hx, hy)
                if htipo == "agenda":
                    if idx == 0:
                        name = "Escuela"
                        color = YELLOW
                    elif idx == 1:
                        name = "Trabajo"
                        color = ORANGE
                    else:
                        name = "Supermercado"
                        color = BLUE_D
                        
                    hub_mob = Square(side_length=0.4, color=color, fill_opacity=0.3, stroke_width=2, stroke_color=color)
                    hub_mob.move_to(pos)
                    label = Text(name, font_size=10, color=color).next_to(hub_mob, UP, buff=0.05)
                    self.add(hub_mob, label)
                else:
                    # ponytail: use standard Circle and manually configure dashed pattern or style
                    hub_mob = Circle(radius=0.4, color=GREEN, fill_opacity=0.1, stroke_width=2, stroke_color=GREEN)
                    hub_mob.move_to(pos)
                    label = Text("Plaza", font_size=10, color=GREEN).next_to(hub_mob, UP, buff=0.05)
                    self.add(hub_mob, label)
        except FileNotFoundError:
            pass

        # ponytail: generate static visual jitter offset per agent (in screen units)
        # to prevent cohabitants from stacking exactly on top of each other at home
        np.random.seed(42)
        jitter_angle = np.random.uniform(0, 2*np.pi, N)
        jitter_radius = np.random.uniform(0.015, 0.08, N)
        jitter_x = jitter_radius * np.cos(jitter_angle)
        jitter_y = jitter_radius * np.sin(jitter_angle)

        # Crear agentes como pequeños círculos
        agentes_mobjects = []
        df_t0 = df_dinamico.filter(pl.col("tiempo") == 0)
        x_t0 = df_t0["coord_x"].to_numpy()
        y_t0 = df_t0["coord_y"].to_numpy()
        estado_t0 = df_t0["estado"].to_numpy()
        carga_t0 = df_t0["carga_viral"].to_numpy()
        mstate_t0 = df_t0["motion_state"].to_numpy() if "motion_state" in df_t0.columns else np.zeros(N)

        for i in range(N):
            pos = mapear_posicion(x_t0[i], y_t0[i])
            if mstate_t0[i] == 0:
                pos += np.array([jitter_x[i], jitter_y[i], 0])
            # Círculos muy pequeños para acomodar N=1600 agentes
            agente = Circle(radius=0.035, stroke_width=0)
            agente.move_to(pos)
            
            est = estado_t0[i]
            color = COLOR_MAP[est]
            opacidad = 1.0
            if est == 2:
                opacidad = float(np.clip(carga_t0[i] / 10.0, 0.7, 1.0))
            elif est == 4:
                opacidad = 0.3
                
            agente.set_fill(color, opacity=opacidad)
            agentes_mobjects.append(agente)
            self.add(agente)

        # 4. REGION DERECHA: DASHBOARD
        dash_x = 3.5
        titulo = Text("Dashboard Continuo", font_size=28).move_to([dash_x, 3.5, 0])
        self.add(titulo)
        
        dia_text = Text("Día: 0", font_size=24, color=YELLOW).move_to([dash_x, 3.0, 0])
        self.add(dia_text)
        
        labels = ["(S) Susceptibles", "(E) Expuestos", "(I) Infectados", "(R) Recuperados", "(D) Muertos"]
        y_pos_labels = [2.4, 2.1, 1.8, 1.5, 1.2]
        
        conteo_t0 = np.bincount(estado_t0, minlength=5)
        textos_conteo = []
        for idx in range(5):
            t = Text(f"{labels[idx]}: {conteo_t0[idx]}", font_size=18, color=COLOR_MAP[idx])
            t.move_to([dash_x, y_pos_labels[idx], 0], aligned_edge=LEFT)
            t.shift(LEFT * 1.5)
            textos_conteo.append(t)
            self.add(t)

        # Ejes para las curvas de evolución (eje X en días físicos reales)
        axes = Axes(
            x_range=[0, float(tiempos[-1]), max(1.0, float(tiempos[-1])/5.0)],
            y_range=[0, 100, 25],
            x_length=5.5,
            y_length=3.0,
            axis_config={"color": WHITE, "font_size": 14}
        ).move_to([dash_x, -1.5, 0])
        self.add(axes)
        y_label = Text("%", font_size=14).next_to(axes.y_axis, UP, buff=0.1)
        self.add(y_label)
        
        curvas = [VMobject(fill_color=COLOR_MAP[i], fill_opacity=0.8, stroke_width=0.5, stroke_color=COLOR_MAP[i]) for i in range(5)]
        for idx_c, c in enumerate(curvas):
            self.add(c)
            # ponytail: initialize closed polygon at origin
            c.set_points_as_corners([axes.c2p(0.0, 0.0), axes.c2p(0.0, 0.0), axes.c2p(0.0, 0.0)])

        # Precomputar conteos históricos para optimizar curvas (filtrando por valor real de tiempo)
        historico_conteos = np.zeros((T_max, 5))
        for t_idx in range(T_max):
            df_t = df_dinamico.filter(pl.col("tiempo") == tiempos[t_idx])
            estados_t = df_t["estado"].to_numpy()
            historico_conteos[t_idx] = np.bincount(estados_t, minlength=5)

        # 5. BUCLE DE ANIMACIÓN
        # ponytail: adapt sub-frames to target frame rate for smooth movement interpolation
        import os
        steps_per_second = float(os.environ.get("STEPS_PER_SECOND", 15.0))
        frames_per_step = max(1, int(config.frame_rate / steps_per_second))
        wait_time = 1.0 / config.frame_rate

        for t_idx in range(1, T_max):
            t_val = tiempos[t_idx]
            t_prev = tiempos[t_idx - 1]
            
            df_prev = df_dinamico.filter(pl.col("tiempo") == t_prev)
            x_prev = df_prev["coord_x"].to_numpy()
            y_prev = df_prev["coord_y"].to_numpy()
            estado_prev = df_prev["estado"].to_numpy()
            carga_prev = df_prev["carga_viral"].to_numpy()
            
            df_t = df_dinamico.filter(pl.col("tiempo") == t_val)
            x_t = df_t["coord_x"].to_numpy()
            y_t = df_t["coord_y"].to_numpy()
            estado_t = df_t["estado"].to_numpy()
            carga_t = df_t["carga_viral"].to_numpy()
            
            mstate_prev = df_prev["motion_state"].to_numpy() if "motion_state" in df_prev.columns else np.zeros(N)
            mstate_t = df_t["motion_state"].to_numpy() if "motion_state" in df_t.columns else np.zeros(N)
            
            conteo_actual = historico_conteos[t_idx]
            
            for f in range(1, frames_per_step + 1):
                alpha = f / frames_per_step
                x_interp = (1.0 - alpha) * x_prev + alpha * x_t
                y_interp = (1.0 - alpha) * y_prev + alpha * y_t
                current_mstate = np.where(alpha >= 0.5, mstate_t, mstate_prev)
                
                for i in range(N):
                    pos = mapear_posicion(x_interp[i], y_interp[i])
                    if current_mstate[i] == 0:
                        pos += np.array([jitter_x[i], jitter_y[i], 0])
                    agentes_mobjects[i].move_to(pos)
                    
                    est = estado_t[i] if alpha >= 0.5 else estado_prev[i]
                    carga = carga_t[i] if alpha >= 0.5 else carga_prev[i]
                    color = COLOR_MAP[est]
                    opacidad = 1.0
                    if est == 2:
                        opacidad = float(np.clip(carga / 10.0, 0.7, 1.0))
                    elif est == 4:
                        opacidad = 0.3
                        
                    agentes_mobjects[i].set_fill(color, opacity=opacidad)

                dia_text_val = t_prev + alpha * (t_val - t_prev)
                nuevo_dia = Text(f"Día: {dia_text_val:.1f}", font_size=24, color=YELLOW).move_to([dash_x, 3.0, 0])
                dia_text.become(nuevo_dia)
                
                if f == frames_per_step:
                    for idx in range(5):
                        nuevo_t = Text(f"{labels[idx]}: {int(conteo_actual[idx])}", font_size=18, color=COLOR_MAP[idx])
                        nuevo_t.move_to([dash_x, y_pos_labels[idx], 0], aligned_edge=LEFT)
                        nuevo_t.shift(LEFT * 1.5)
                        textos_conteo[idx].become(nuevo_t)
                        
                    for idx in range(5):
                        pts_lower = [axes.c2p(tiempos[time_step], 100.0 * np.sum(historico_conteos[time_step, :idx]) / N) for time_step in range(t_idx + 1)]
                        pts_upper = [axes.c2p(tiempos[time_step], 100.0 * np.sum(historico_conteos[time_step, :idx+1]) / N) for time_step in range(t_idx + 1)]
                        pts_upper.reverse()
                        vertices = pts_lower + pts_upper + [pts_lower[0]]
                        curvas[idx].set_points_as_corners(vertices)
                        
                self.wait(wait_time)

        self.wait(2.0)
