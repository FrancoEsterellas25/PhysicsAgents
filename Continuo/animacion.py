from manim import *
import polars as pl
import numpy as np
from pathlib import Path

# Configuración del renderizado (2 FPS para no perder transiciones)
config.frame_rate = 2

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
        L = max(100.0, float(max(max_x_val, max_y_val)))

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

        # Crear agentes como pequeños círculos
        agentes_mobjects = []
        df_t0 = df_dinamico.filter(pl.col("tiempo") == 0)
        x_t0 = df_t0["coord_x"].to_numpy()
        y_t0 = df_t0["coord_y"].to_numpy()
        estado_t0 = df_t0["estado"].to_numpy()
        carga_t0 = df_t0["carga_viral"].to_numpy()

        for i in range(N):
            pos = mapear_posicion(x_t0[i], y_t0[i])
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

        # Ejes para las curvas de evolución
        axes = Axes(
            x_range=[0, max(10, T_max), max(1, T_max//5)],
            y_range=[0, N, max(1, N//4)],
            x_length=5.5,
            y_length=3.0,
            axis_config={"color": WHITE, "font_size": 14}
        ).move_to([dash_x, -1.5, 0])
        self.add(axes)
        
        curvas = [VMobject(color=COLOR_MAP[i], stroke_width=3) for i in range(5)]
        for c in curvas:
            self.add(c)
            c.set_points_as_corners([axes.c2p(0, conteo_t0[i]), axes.c2p(0, conteo_t0[i])])

        # Precomputar conteos históricos para optimizar curvas
        historico_conteos = np.zeros((T_max, 5))
        for t in range(T_max):
            df_t = df_dinamico.filter(pl.col("tiempo") == t)
            estados_t = df_t["estado"].to_numpy()
            historico_conteos[t] = np.bincount(estados_t, minlength=5)

        # 5. BUCLE DE ANIMACIÓN
        for t_idx in range(1, T_max):
            df_t = df_dinamico.filter(pl.col("tiempo") == t_idx)
            x_t = df_t["coord_x"].to_numpy()
            y_t = df_t["coord_y"].to_numpy()
            estado_t = df_t["estado"].to_numpy()
            carga_t = df_t["carga_viral"].to_numpy()
            
            # Actualizar posiciones y colores de los agentes
            for i in range(N):
                pos = mapear_posicion(x_t[i], y_t[i])
                agentes_mobjects[i].move_to(pos)
                
                est = estado_t[i]
                color = COLOR_MAP[est]
                opacidad = 1.0
                if est == 2:
                    opacidad = float(np.clip(carga_t[i] / 10.0, 0.7, 1.0))
                elif est == 4:
                    opacidad = 0.3
                    
                agentes_mobjects[i].set_fill(color, opacity=opacidad)

            # Actualizar textos del Dashboard
            conteo_actual = historico_conteos[t_idx]
            nuevo_dia = Text(f"Día: {t_idx}", font_size=24, color=YELLOW).move_to([dash_x, 3.0, 0])
            dia_text.become(nuevo_dia)
            
            for idx in range(5):
                nuevo_t = Text(f"{labels[idx]}: {int(conteo_actual[idx])}", font_size=18, color=COLOR_MAP[idx])
                nuevo_t.move_to([dash_x, y_pos_labels[idx], 0], aligned_edge=LEFT)
                nuevo_t.shift(LEFT * 1.5)
                textos_conteo[idx].become(nuevo_t)
                
            # Actualizar Curvas
            for idx in range(5):
                pts = [axes.c2p(time_step, historico_conteos[time_step, idx]) for time_step in range(t_idx + 1)]
                curvas[idx].set_points_as_corners(pts)

            self.wait(0.5)

        self.wait(2.0)
