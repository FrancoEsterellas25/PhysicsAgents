from manim import *
import polars as pl
import numpy as np

# Forzar el renderizado a 2 FPS por defecto para no perder información en los saltos de 0.5s
config.frame_rate = 2

class EscenaEpidemiologica(Scene):
    def construct(self):
        # 1. CARGA DE DATOS (Polars)
        # ---------------------------------------------------------
        from pathlib import Path
        base_dir = Path(__file__).parent
        try:
            df_estatico = pl.read_parquet(base_dir / "mapa_estatico.parquet")
            df_dinamico = pl.read_parquet(base_dir / "telemetria_dinamica.parquet")
        except Exception as e:
            print(f"Error cargando los archivos parquet: {e}")
            print("Asegúrate de ejecutar discrete_simulation.py primero para generarlos.")
            return

        coord_x = df_estatico["coord_x"].to_numpy()
        coord_y = df_estatico["coord_y"].to_numpy()
        N = len(coord_x)
        
        max_x = int(np.max(coord_x))
        max_y = int(np.max(coord_y))
        cols = max_x + 1
        rows = max_y + 1

        tiempos = df_dinamico["tiempo"].unique(maintain_order=True).to_numpy()
        T_max = len(tiempos)
        
        estados_matrix = df_dinamico["estado"].to_numpy().reshape((T_max, N))
        viral_matrix = df_dinamico["carga_viral"].to_numpy().reshape((T_max, N))

        # 2. DEFINICIÓN DE COLORES
        COLOR_MAP = {
            0: "#1F77B4",  # S: Azul
            1: "#FF7F0E",  # E: Naranja Claro
            2: "#D62728",  # I: Rojo Intenso
            3: "#2CA02C",  # R: Verde
            4: "#7F7F7F"   # D: Gris Oscuro
        }

        # Configuración Visual de la Topología
        import os
        TOPOLOGY_VISUAL = os.getenv("SIM_TOPOLOGY", "hexagonal")
        if TOPOLOGY_VISUAL not in ["hexagonal", "cuadrada"]:
            TOPOLOGY_VISUAL = "cuadrada"

        # 3. CONSTRUCCIÓN DE LA ARQUITECTURA VISUAL (Layout)
        # ---------------------------------------------------------
        # 3.1 Región Izquierda: Grilla Adaptativa
        grid_group = VGroup()
        
        # Caja delimitadora fija (Bounding Box) para la grilla
        box_width = 7.0
        box_height = 7.0
        offset_x = -3.5
        offset_y = 0.0
        
        # Cálculo de escalado dinámico (achicar casillas según cantidad de filas/columnas)
        cell_w = box_width / cols
        cell_h = box_height / rows
        
        # El lado de la celda tomará el menor valor para mantener la proporción sin deformarse
        lado_celda = min(cell_w, cell_h) * 0.95  # 5% de margen
        espaciado_x = min(cell_w, cell_h)
        espaciado_y = min(cell_w, cell_h)
        
        if TOPOLOGY_VISUAL == "hexagonal":
            # Compactación vertical hexagonal
            espaciado_y = espaciado_x * np.sqrt(3) / 2

        # Centrar exactamente dentro del Bounding Box
        real_width = cols * espaciado_x
        real_height = rows * espaciado_y
        start_x = offset_x - (real_width / 2) + (espaciado_x / 2)
        start_y = offset_y - (real_height / 2) + (espaciado_y / 2)

        celdas = []
        estado_inicial = estados_matrix[0]
        carga_inicial = viral_matrix[0]
        
        for i in range(N):
            cx, cy = coord_x[i], coord_y[i]
            
            x_pos = start_x + cx * espaciado_x
            y_pos = start_y + cy * espaciado_y
            
            # Aplicar desfase de fila (shoved row) para Hexágonos
            if TOPOLOGY_VISUAL == "hexagonal" and cy % 2 == 1:
                x_pos += espaciado_x / 2.0
                
            # Elegir polígono
            if TOPOLOGY_VISUAL == "hexagonal":
                celda = RegularPolygon(n=6, radius=lado_celda/2)
                celda.rotate(PI/6)  # Rotación "Pointy-top" para encastre horizontal
            else:
                celda = Square(side_length=lado_celda)
                
            celda.set_stroke(width=0.5, color=WHITE)
            celda.move_to(np.array([x_pos, y_pos, 0]))
            
            est = estado_inicial[i]
            color_celda = COLOR_MAP[est]
            
            opacidad = 1.0
            if est == 2:
                opacidad = float(np.clip(carga_inicial[i] / 10.0, 0.8, 1.0))
            elif est == 4:
                opacidad = 0.3
                
            celda.set_fill(color_celda, opacity=opacidad)
            grid_group.add(celda)
            celdas.append(celda)

        self.add(grid_group)

        # 3.2 Región Derecha: Dashboard y Gráficas de Evolución
        dash_x = 3.5
        titulo = Text("Dashboard Epidémico", font_size=28).move_to([dash_x, 3.5, 0])
        self.add(titulo)
        
        dia_text = Text("Día: 0", font_size=24, color=YELLOW).move_to([dash_x, 3.0, 0])
        self.add(dia_text)
        
        # Textos de Población
        conteo_inicial = np.bincount(estado_inicial, minlength=5)
        textos_conteo = []
        labels = ["(S) Susceptibles", "(E) Expuestos", "(I) Infectados", "(R) Recuperados", "(D) Muertos"]
        y_pos_labels = [2.4, 2.1, 1.8, 1.5, 1.2]
        
        for idx in range(5):
            t = Text(f"{labels[idx]}: {conteo_inicial[idx]}", font_size=18, color=COLOR_MAP[idx])
            t.move_to([dash_x, y_pos_labels[idx], 0], aligned_edge=LEFT)
            t.shift(LEFT * 1.5)  # Alinear visualmente
            textos_conteo.append(t)
            self.add(t)

        # Ejes para Plotear curvas
        axes = Axes(
            x_range=[0, max(10, T_max), max(1, T_max//5)],
            y_range=[0, N, max(1, N//4)],
            x_length=5.5,
            y_length=3.0,
            axis_config={"color": WHITE, "font_size": 14}
        ).move_to([dash_x, -1.5, 0])
        self.add(axes)
        
        # Precomputar todo el historial para dibujar
        historico_conteos = np.zeros((T_max, 5))
        for t in range(T_max):
            historico_conteos[t] = np.bincount(estados_matrix[t], minlength=5)
            
        curvas = [VMobject(color=COLOR_MAP[i], stroke_width=3) for i in range(5)]
        for c in curvas:
            self.add(c)

        # 4. BUCLE DE ANIMACIÓN
        # ---------------------------------------------------------
        estado_previo = np.copy(estado_inicial)
        
        # Inicializar puntos en frame 0
        for idx_c in range(5):
            curvas[idx_c].set_points_as_corners([axes.c2p(0, historico_conteos[0, idx_c]), axes.c2p(0, historico_conteos[0, idx_c])])

        for t_idx in range(1, T_max):
            estado_actual = estados_matrix[t_idx]
            carga_actual = viral_matrix[t_idx]
            
            # Actualizar Grilla (Solo celdas modificadas para O(1))
            cambios_mask = (estado_actual != estado_previo)
            infectados_mask = (estado_actual == 2)
            
            indices_actualizar = np.where(cambios_mask | infectados_mask)[0]
            for i in indices_actualizar:
                est = estado_actual[i]
                color_celda = COLOR_MAP[est]
                
                opacidad = 1.0
                if est == 2:
                    opacidad = float(np.clip(carga_actual[i] / 10.0, 0.8, 1.0))
                elif est == 4:
                    opacidad = 0.3
                    
                celdas[i].set_fill(color_celda, opacity=opacidad)
                
            estado_previo = np.copy(estado_actual)

            # Actualizar Textos
            conteo_actual = historico_conteos[t_idx]
            
            nuevo_dia = Text(f"Día: {t_idx}", font_size=24, color=YELLOW).move_to([dash_x, 3.0, 0])
            dia_text.become(nuevo_dia)
            
            for idx in range(5):
                nuevo_t = Text(f"{labels[idx]}: {int(conteo_actual[idx])}", font_size=18, color=COLOR_MAP[idx])
                nuevo_t.move_to([dash_x, y_pos_labels[idx], 0], aligned_edge=LEFT)
                nuevo_t.shift(LEFT * 1.5)
                textos_conteo[idx].become(nuevo_t)
                
            # Actualizar Curvas del Gráfico Frame a Frame
            for idx in range(5):
                # Genera los vértices hasta el día actual
                pts = [axes.c2p(t, historico_conteos[t, idx]) for t in range(t_idx + 1)]
                curvas[idx].set_points_as_corners(pts)

            # Pausa para este fotograma
            self.wait(0.5)
            
        # Pausa final al terminar la simulación
        self.wait(2.0)
