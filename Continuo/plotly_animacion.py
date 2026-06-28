import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ponytail: unified agent scatter trace to prevent index-shift teleportation/swapping in Plotly

def generar_dashboard(base_dir=None):
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    # Resolve relative paths in base_dir
    base_dir = Path(base_dir).resolve()
    
    # 1. Load data
    try:
        df_dinamico = pl.read_parquet(base_dir / "telemetria_dinamica.parquet")
        df_virus = pl.read_parquet(base_dir / "telemetria_virus.parquet")
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        return

    # Ensure strictly sorted by tiempo, then id_agente
    df_dinamico = df_dinamico.sort(["tiempo", "id_agente"])
    
    N = df_dinamico.filter(pl.col("tiempo") == 0).height
    tiempos_all = df_dinamico["tiempo"].unique(maintain_order=True).to_numpy()
    tiempos_all = np.sort(tiempos_all)
    
    # ponytail: Subsample time steps to prevent browser memory exhaustion (capping at ~200 frames)
    step_modulo = max(1, len(tiempos_all) // 200)
    tiempos = tiempos_all[::step_modulo]
    T_max = len(tiempos)
    
    L = float(max(df_dinamico["coord_x"].max(), df_dinamico["coord_y"].max()))
    
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
        subplot_titles=("Espacio Físico (Movimiento Suave y Carga Viral)", "Evolución Epidemiológica (%)"),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.08
    )

    # Initial data at t = 0
    df_t0 = df_dinamico.filter(pl.col("tiempo") == 0)
    colores_t0 = [COLOR_MAP[st] for st in df_t0["estado"].to_numpy()]
    
    # Trace 0: Lagrangian Aerosol Particles (dynamic background)
    ax_t0 = df_virus["aerosol_x"][0]
    ay_t0 = df_virus["aerosol_y"][0]
    ad_t0 = df_virus["aerosol_dosis"][0]
    
    fig.add_trace(
        go.Scattergl(
            x=ax_t0,
            y=ay_t0,
            mode="markers",
            marker=dict(
                color=["rgba(238, 130, 238, " + str(min(0.5, op * 0.35)) + ")" for op in ad_t0],
                size=3
            ),
            name="Aerosol Viral (Gas)",
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Trace 1: Unique residences/homes (static background)
    df_estatico = pl.read_parquet(base_dir / "mapa_estatico.parquet")
    unique_homes = df_estatico.select(["hogar_x", "hogar_y"]).unique()
    fig.add_trace(
        go.Scatter(
            x=unique_homes["hogar_x"].to_numpy(),
            y=unique_homes["hogar_y"].to_numpy(),
            mode="markers",
            marker=dict(symbol="x", color="grey", size=4, opacity=0.3),
            name="Residencia (Hogar)",
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Trace 2-6: Empty dummy traces for the legend
    for s in states:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=state_names[s],
                marker=dict(color=COLOR_MAP[s], size=8),
                legendgroup=state_names[s],
                showlegend=True
            ),
            row=1, col=1
        )
        
    # Trace 7: The actual single Scatter trace with ALL agents (strictly sorted by id_agente)
    fig.add_trace(
        go.Scatter(
            x=df_t0["coord_x"].to_numpy(),
            y=df_t0["coord_y"].to_numpy(),
            mode="markers",
            marker=dict(color=colores_t0, size=5, opacity=0.8),
            showlegend=False
        ),
        row=1, col=1
    )
        
    # Trace 8-12: Stacked Area curves (one per state)
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
        df_t = df_dinamico.filter(pl.col("tiempo") == t_val)
        colores_t = [COLOR_MAP[st] for st in df_t["estado"].to_numpy()]
        
        # Load aerosol particles for this frame via integer index
        grid_idx = min(t_idx * step_modulo, len(df_virus) - 1)
        ax_t = df_virus["aerosol_x"][grid_idx]
        ay_t = df_virus["aerosol_y"][grid_idx]
        ad_t = df_virus["aerosol_dosis"][grid_idx]
        
        frame_data = [
            go.Scattergl(
                x=ax_t,
                y=ay_t,
                marker=dict(
                    color=["rgba(238, 130, 238, " + str(min(0.5, op * 0.35)) + ")" for op in ad_t],
                    size=3
                )
            ),
            # Agent coordinates (Trace 7)
            go.Scatter(
                x=df_t["coord_x"].to_numpy(),
                y=df_t["coord_y"].to_numpy(),
                marker=dict(color=colores_t, size=5, opacity=0.8)
            )
        ]
            
        # Stacked curves history (Traces 8-12)
        for s in states:
            y_val = 100.0 * historico_conteos[0:t_idx+1, s] / N
            frame_data.append(
                go.Scatter(
                    x=tiempos[0:t_idx+1],
                    y=y_val
                )
            )
            
        frames.append(go.Frame(data=frame_data, name=f"frame_{t_idx}", traces=[0, 7, 8, 9, 10, 11, 12]))

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

    for t_idx, t_val in enumerate(tiempos):
        slider_step = {
            "args": [
                [f"frame_{t_idx}"],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
            ],
            "label": f"{t_val:.1f}",
            "method": "animate"
        }
        sliders_dict["steps"].append(slider_step)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="Dashboard Epidémico Interactivo SEIRS-D", font=dict(size=24)),
        width=1100,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        xaxis=dict(range=[0, L], scaleanchor="y", scaleratio=1, title="Posición X"),
        yaxis=dict(range=[0, L], title="Posición Y"),
        xaxis2=dict(range=[0, tiempos[-1]], title="Tiempo (Días)"),
        yaxis2=dict(range=[0, 100], title="Porcentaje (%)"),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
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

    # ponytail: Add hubs as background shapes and annotations
    try:
        df_hubs = pl.read_parquet(base_dir / "hubs.parquet")
        for idx, row in enumerate(df_hubs.iter_rows(named=True)):
            hx, hy, htipo = row["x"], row["y"], row["tipo"]
            name = row.get("nombre", "Hub")
            color_str = row.get("color", "White")
            
            # Map manim / lowercase colors to Plotly compatible strings
            color = color_str.lower()
            if color == "yellow":
                color_val = "Yellow"
            elif color == "orange":
                color_val = "Orange"
            elif color == "cyan":
                color_val = "Cyan"
            elif color == "blue_d" or color == "blue":
                color_val = "DeepSkyBlue"
            elif color == "green":
                color_val = "Green"
            else:
                color_val = "White"
                
            ambiente = row.get("ambiente", "cerrado")
            
            if ambiente == "cerrado":
                fig.add_shape(
                    type="rect",
                    xref="x1", yref="y1",
                    x0=hx-1.5, y0=hy-1.5,
                    x1=hx+1.5, y1=hy+1.5,
                    fillcolor=color_val,
                    opacity=0.15,
                    line=dict(color=color_val, width=1.5)
                )
                fig.add_annotation(
                    x=hx, y=hy+2.2,
                    xref="x1", yref="y1",
                    text=name,
                    showarrow=False,
                    font=dict(color=color_val, size=10),
                    bgcolor="rgba(0,0,0,0.5)"
                )
            else:
                fig.add_shape(
                    type="circle",
                    xref="x1", yref="y1",
                    x0=hx-4.0, y0=hy-4.0,
                    x1=hx+4.0, y1=hy+4.0,
                    fillcolor=color_val,
                    opacity=0.10,
                    line=dict(color=color_val, width=1.5, dash="dash")
                )
                fig.add_annotation(
                    x=hx, y=hy+5.0,
                    xref="x1", yref="y1",
                    text=name,
                    showarrow=False,
                    font=dict(color=color_val, size=10),
                    bgcolor="rgba(0,0,0,0.5)"
                )
    except FileNotFoundError:
        pass

    # Export to standalone html
    output_file = base_dir / "plotly_animacion.html"
    fig.write_html(output_file, auto_open=False)
    print(f"Interactive dashboard generated successfully at: {output_file}")

def main():
    generar_dashboard()

if __name__ == "__main__":
    main()
