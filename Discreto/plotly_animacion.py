import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ponytail: unified agent scatter trace to prevent index-shift teleportation/swapping in Plotly

def main():
    base_dir = Path(__file__).parent
    
    # 1. Load data
    try:
        df_estatico = pl.read_parquet(base_dir / "mapa_estatico.parquet")
        df_dinamico = pl.read_parquet(base_dir / "telemetria_dinamica.parquet")
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        return

    # Join coordinates and ensure strict ordering by time, then agent ID
    df_joined = df_dinamico.join(df_estatico.select(["id_agente", "coord_x", "coord_y"]), on="id_agente")
    df_joined = df_joined.sort(["tiempo", "id_agente"])
    df_pd = df_joined.to_pandas()

    N = len(df_estatico)
    tiempos = df_dinamico["tiempo"].unique(maintain_order=True).to_numpy()
    tiempos = np.sort(tiempos)
    T_max = len(tiempos)
    
    max_x = int(df_pd["coord_x"].max())
    max_y = int(df_pd["coord_y"].max())
    
    # Define states, labels and colors
    states = [0, 1, 2, 3, 4]
    state_names = ["Susceptibles (S)", "Expuestos (E)", "Infectados (I)", "Recuperados (R)", "Muertos (D)"]
    COLOR_MAP = {
        0: "#1F77B4",  # S: Blue
        1: "#FF7F0E",  # E: Orange
        2: "#D62728",  # I: Red
        3: "#2CA02C",  # R: Green
        4: "#7F7F7F"   # D: Gray
    }

    # Precompute historical counts for the area plot
    historico_conteos = np.zeros((T_max, 5))
    for t_idx, t_val in enumerate(tiempos):
        df_t = df_dinamico.filter(pl.col("tiempo") == t_val)
        historico_conteos[t_idx] = np.bincount(df_t["estado"].to_numpy(), minlength=5)

    # 2. Build Subplot Layout
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Grilla Geográfica (Estados)", "Evolución Epidemiológica (%)"),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.08
    )

    # Initial data at t = 0
    df_t0 = df_pd[df_pd["tiempo"] == 0]
    colores_t0 = [COLOR_MAP[st] for st in df_t0["estado"].to_numpy()]
    
    # Trace 0-4: Empty dummy traces for the legend
    for s in states:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=state_names[s],
                marker=dict(symbol="square", color=COLOR_MAP[s], size=10),
                legendgroup=state_names[s],
                showlegend=True
            ),
            row=1, col=1
        )
        
    # Trace 5: The actual single Scatter grid trace (strictly sorted by id_agente)
    fig.add_trace(
        go.Scatter(
            x=df_t0["coord_x"].to_numpy(),
            y=df_t0["coord_y"].to_numpy(),
            mode="markers",
            marker=dict(symbol="square", color=colores_t0, size=9, opacity=0.9),
            showlegend=False
        ),
        row=1, col=1
    )
        
    # Trace 6-10: Stacked Area curves
    for s in states:
        y_val = 100.0 * historico_conteos[0:1, s] / N
        fig.add_trace(
            go.Scatter(
                x=tiempos[0:1],
                y=y_val,
                mode="lines",
                stackgroup="one",
                groupnorm="percent",
                name=state_names[s],
                fillcolor=COLOR_MAP[s],
                line=dict(color=COLOR_MAP[s], width=0.5),
                legendgroup=state_names[s],
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. Build Animation Frames
    frames = []
    for t_idx, t_val in enumerate(tiempos):
        df_t = df_pd[df_pd["tiempo"] == t_val]
        colores_t = [COLOR_MAP[st] for st in df_t["estado"].to_numpy()]
        frame_data = []
        
        # Legend dummies (Traces 0-4)
        for s in states:
            frame_data.append(go.Scatter())
            
        # Agent coordinates and updated state colors (Trace 5)
        frame_data.append(
            go.Scatter(
                x=df_t["coord_x"].to_numpy(),
                y=df_t["coord_y"].to_numpy(),
                marker=dict(symbol="square", color=colores_t, size=9, opacity=0.9)
            )
        )
            
        # Stacked curves history (Traces 6-10)
        for s in states:
            y_val = 100.0 * historico_conteos[0:t_idx+1, s] / N
            frame_data.append(
                go.Scatter(
                    x=tiempos[0:t_idx+1],
                    y=y_val
                )
            )
            
        frames.append(go.Frame(data=frame_data, name=f"dia_{t_val:.1f}"))

    fig.frames = frames

    # 4. Interactive controls
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Día de Simulación: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    for t_val in tiempos:
        slider_step = {
            "args": [
                [f"dia_{t_val:.1f}"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
            ],
            "label": f"{t_val:.1f}",
            "method": "animate"
        }
        sliders_dict["steps"].append(slider_step)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Dashboard Epidémico Interactivo SEIRS-D (Enfoque Discreto)", font=dict(size=24)),
        width=1100,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        xaxis=dict(range=[-1, max_x + 1], scaleanchor="y", scaleratio=1, title="Columnas"),
        yaxis=dict(range=[-1, max_y + 1], title="Filas"),
        xaxis2=dict(range=[0, tiempos[-1]], title="Tiempo (Días)"),
        yaxis2=dict(range=[0, 100], title="Porcentaje (%)"),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        sliders=[sliders_dict]
    )

    # Export to standalone html
    output_file = base_dir / "plotly_animacion.html"
    fig.write_html(output_file, auto_open=False)
    print(f"Interactive dashboard generated successfully at: {output_file}")

if __name__ == "__main__":
    main()
