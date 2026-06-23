from manim import *
import polars as pl
import numpy as np

# Forzar el renderizado a 2 FPS por defecto para no perder información en los saltos de 0.5s
config.frame_rate = 2

class EscenaEpidemiologica(Scene):
    def construct(self):
        # 1. CARGA DE DATOS (Polars)
        # ---------------------------------------------------------
        # Leer el mapa estático
        from pathlib import Path
        base_dir = Path(__file__).parent
        try:
            df_estatico = pl.read_parquet(base_dir / "mapa_estatico.parquet")
            df_dinamico = pl.read_parquet(base_dir / "telemetria_dinamica.parquet")
        except Exception as e:
            print(f"Error cargando los archivos parquet: {e}")
            print("Asegúrate de ejecutar draft_simulacion.py primero para generarlos.")
            return

        # Extraer coordenadas y número de agentes
        coord_x = df_estatico["coord_x"].to_numpy()
        coord_y = df_estatico["coord_y"].to_numpy()
        N = len(coord_x)
        
        # Determinar dimensiones de la grilla
        max_x = int(np.max(coord_x))
        max_y = int(np.max(coord_y))
        cols = max_x + 1
        rows = max_y + 1

        # Extraer bloques dinámicos por frame
        # Dado que el DataFrame dinámico está ordenado por (tiempo, id_agente)
        tiempos = df_dinamico["tiempo"].unique(maintain_order=True).to_numpy()
        T_max = len(tiempos)
        
        estados_matrix = df_dinamico["estado"].to_numpy().reshape((T_max, N))
        viral_matrix = df_dinamico["carga_viral"].to_numpy().reshape((T_max, N))

        # 2. DEFINICIÓN DE COLORES
        # ---------------------------------------------------------
        COLOR_MAP = {
            0: "#1F77B4",  # S: Azul
            1: "#FF7F0E",  # E: Naranja Claro
            2: "#D62728",  # I: Rojo Intenso
            3: "#2CA02C",  # R: Verde
            4: "#7F7F7F"   # D: Gris Oscuro
        }

        # 3. CONSTRUCCIÓN DE LA ARQUITECTURA VISUAL (Layout)
        # ---------------------------------------------------------
        # 3.1 Región Izquierda: Grilla (65% del ancho)
        grid_group = VGroup()
        
        # Parámetros de geometría para encajar la grilla en la izquierda
        lado_celda = 0.15
        espaciado = 0.02
        ancho_grilla = cols * (lado_celda + espaciado)
        alto_grilla = rows * (lado_celda + espaciado)
        
        # Centrar la grilla en x = -3 (Región Izquierda)
        offset_x = -3 - (ancho_grilla / 2)
        offset_y = - (alto_grilla / 2)
        
        # Crear celdas e indexarlas por ID para acceso O(1)
        celdas = []
        estado_inicial = estados_matrix[0]
        carga_inicial = viral_matrix[0]
        
        for i in range(N):
            cx, cy = coord_x[i], coord_y[i]
            x_pos = offset_x + cx * (lado_celda + espaciado)
            y_pos = offset_y + cy * (lado_celda + espaciado)
            
            # Usar Square para geometría Moore / Von Neumann
            celda = Square(side_length=lado_celda)
            celda.set_stroke(width=0.5, color=WHITE)
            celda.move_to(np.array([x_pos, y_pos, 0]))
            
            est = estado_inicial[i]
            color_celda = COLOR_MAP[est]
            
            # Ajuste dinámico de opacidad si está infectado
            opacidad = 1.0
            if est == 2:
                opacidad = np.clip(carga_inicial[i], 0.3, 1.0)
            elif est == 4:
                opacidad = 0.3  # D tiene menor opacidad
                
            celda.set_fill(color_celda, opacity=opacidad)
            
            grid_group.add(celda)
            celdas.append(celda)

        self.add(grid_group)

        # 3.2 Región Derecha: Dashboard (35% del ancho)
        dash_x = 4.0
        titulo = Text("Dashboard Epidémico", font_size=32).move_to([dash_x, 3.0, 0])
        self.add(titulo)
        
        dia_text = Text("Día: 0", font_size=24, color=YELLOW).move_to([dash_x, 2.0, 0])
        self.add(dia_text)
        
        # Variables de Tracker
        conteo_inicial = np.bincount(estado_inicial, minlength=5)
        textos_conteo = []
        labels = ["Susceptibles (S)", "Expuestos (E)", "Infectados (I)", "Recuperados (R)", "Muertos (D)"]
        y_pos_labels = [1.0, 0.5, 0.0, -0.5, -1.0]
        
        for idx in range(5):
            t = Text(f"{labels[idx]}: {conteo_inicial[idx]}", font_size=20, color=COLOR_MAP[idx])
            t.move_to([dash_x, y_pos_labels[idx], 0])
            textos_conteo.append(t)
            self.add(t)

        # Barra de progreso de infecciones
        barra_bg = Rectangle(width=4.0, height=0.2, color=WHITE).move_to([dash_x, -2.0, 0])
        barra_bg.set_fill(DARK_GRAY, opacity=0.5)
        self.add(barra_bg)
        
        barra_fill = Rectangle(width=0.01, height=0.2, color=COLOR_MAP[2])
        barra_fill.set_fill(COLOR_MAP[2], opacity=1.0)
        barra_fill.align_to(barra_bg, LEFT).match_y(barra_bg)
        self.add(barra_fill)

        # Retardo inicial
        self.wait(1.0)

        # 4. BUCLE DE ANIMACIÓN O(1) SÍNCRONO Y DIRECTO
        # ---------------------------------------------------------
        estado_previo = np.copy(estado_inicial)
        
        for t_idx in range(1, T_max):
            estado_actual = estados_matrix[t_idx]
            carga_actual = viral_matrix[t_idx]
            
            # Buscar índices que cambiaron de estado o que son I (requieren update de opacidad)
            cambios_mask = (estado_actual != estado_previo)
            infectados_mask = (estado_actual == 2)
            muertos_mask = (estado_actual == 4)
            
            # Actualizar celdas de forma manual (Modificación Directa de Atributos)
            indices_actualizar = np.where(cambios_mask | infectados_mask)[0]
            for i in indices_actualizar:
                est = estado_actual[i]
                color_celda = COLOR_MAP[est]
                
                opacidad = 1.0
                if est == 2:
                    opacidad = float(np.clip(carga_actual[i], 0.3, 1.0))
                elif est == 4:
                    opacidad = 0.3
                    
                # Aplicamos mutación in-place al fill, sin self.play() pesado
                celdas[i].set_fill(color_celda, opacity=opacidad)
                
            estado_previo = np.copy(estado_actual)

            # Actualizar Dashboard (Strings directo)
            conteo_actual = np.bincount(estado_actual, minlength=5)
            
            # Modificar strings in-place
            # Manim permite cambiar la textura del texto creando un nuevo objeto Text de forma barata o reescribiendo si es tex
            # Para mayor eficiencia visual sin crear subgrafos, reconstruimos los textos rápidamente o los escalamos
            
            # Nota: Debido a la inmutabilidad de VGroups, reemplazaremos el texto mediante sub-transformaciones
            # o podemos reasignar .become().
            nuevo_dia = Text(f"Día: {t_idx}", font_size=24, color=YELLOW).move_to([dash_x, 2.0, 0])
            dia_text.become(nuevo_dia)
            
            for idx in range(5):
                nuevo_t = Text(f"{labels[idx]}: {conteo_actual[idx]}", font_size=20, color=COLOR_MAP[idx])
                nuevo_t.move_to([dash_x, y_pos_labels[idx], 0])
                textos_conteo[idx].become(nuevo_t)
                
            # Actualizar ancho de la barra
            proporcion_inf = conteo_actual[2] / float(N)
            nuevo_ancho = max(0.01, 4.0 * proporcion_inf) # 4.0 es el ancho total del bg
            nueva_barra = Rectangle(width=nuevo_ancho, height=0.2, color=COLOR_MAP[2])
            nueva_barra.set_fill(COLOR_MAP[2], opacity=1.0)
            nueva_barra.align_to(barra_bg, LEFT).match_y(barra_bg)
            barra_fill.become(nueva_barra)

            # Ejecutar un único wait por frame para compilar el cambio visual
            # 2 FPS de simulación -> 0.5s de espera para no perder información
            self.wait(0.5)
            
        # Retardo final
        self.wait(2.0)

"""
========================================================================
COMANDOS DE EXPORTACIÓN CLI PARA MANIM
========================================================================

Para ejecutar este script en la terminal, sitúate en la raíz del repositorio
o dentro de la carpeta Discreto.
Asegúrate de haber corrido `draft_simulacion.py` primero para tener datos.

Debido a la naturaleza de la simulación discreta, TODAS las calidades están
forzadas y sincronizadas internamente a 2 FPS (1 día = 1 fotograma = 0.5s).

1. Renderizado de Prueba (Resolución Baja 480p, 2 FPS):
   python -m manim -pql Discreto/animacion.py EscenaEpidemiologica --format mp4

2. Renderizado Intermedio (Resolución Media 720p, 2 FPS):
   python -m manim -pqm Discreto/animacion.py EscenaEpidemiologica --format mp4

3. Renderizado de Producción (Resolución Alta 1080p, 2 FPS):
   python -m manim -pqh Discreto/animacion.py EscenaEpidemiologica --format mp4

Nota: El script fuerza el framerate a 2 FPS usando `config.frame_rate = 2`.
La bandera `--format mp4` garantiza que se compile un archivo de video y
evita problemas de generación de PNGs estáticos.
"""
