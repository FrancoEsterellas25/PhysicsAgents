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
        df_dinamico = pl.read_parquet(base_dir / "telemetria_dinamica.parquet")
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        return

    # Ensure strictly sorted by tiempo, then id_agente
    df_dinamico = df_dinamico.sort(["tiempo", "id_agente"])
    
    N = df_dinamico.filter(pl.col("tiempo") == 0).height
    tiempos = df_dinamico["tiempo"].unique(maintain_order=True).to_numpy()
    tiempos = np.sort(tiempos)
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
        subplot_titles=("Espacio Físico (Movimiento Suave)", "Evolución Epidemiológica (%)"),
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.08
    )

    # ponytail: generate static visual jitter offset per agent (in screen coordinate units)
    np.random.seed(42)
    jitter_angle = np.random.uniform(0, 2*np.pi, N)
    jitter_radius = np.random.uniform(0.15, 0.8, N)  # Offset fits the coordinate space scale visually
    jitter_x = jitter_radius * np.cos(jitter_angle)
    jitter_y = jitter_radius * np.sin(jitter_angle)

    # Initial data at t = 0
    df_t0 = df_dinamico.filter(pl.col("tiempo") == 0)
    colores_t0 = [COLOR_MAP[st] for st in df_t0["estado"].to_numpy()]
    mstate_t0 = df_t0["motion_state"].to_numpy() if "motion_state" in df_t0.columns else np.zeros(N)
    
    x_t0_plot = df_t0["coord_x"].to_numpy().copy()
    y_t0_plot = df_t0["coord_y"].to_numpy().copy()
    mask_home_t0 = (mstate_t0 == 0)
    x_t0_plot[mask_home_t0] += jitter_x[mask_home_t0]
    y_t0_plot[mask_home_t0] += jitter_y[mask_home_t0]
    
    # Trace 0-4: Empty dummy traces for the legend
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
        
    # Trace 5: The actual single Scatter trace with ALL agents (strictly sorted by id_agente)
    fig.add_trace(
        go.Scatter(
            x=x_t0_plot,
            y=y_t0_plot,
            mode="markers",
            marker=dict(color=colores_t0, size=5, opacity=0.8),
            showlegend=False
        ),
        row=1, col=1
    )
        
    # Trace 6-10: Stacked Area curves (one per state)
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
        mstate_t = df_t["motion_state"].to_numpy() if "motion_state" in df_t.columns else np.zeros(N)
        
        x_t_plot = df_t["coord_x"].to_numpy().copy()
        y_t_plot = df_t["coord_y"].to_numpy().copy()
        mask_home_t = (mstate_t == 0)
        x_t_plot[mask_home_t] += jitter_x[mask_home_t]
        y_t_plot[mask_home_t] += jitter_y[mask_home_t]
        
        frame_data = []
        
        # Legend dummies (Traces 0-4) - stay empty
        for s in states:
            frame_data.append(go.Scatter())
            
        # Agent coordinates and updated state colors (Trace 5)
        frame_data.append(
            go.Scatter(
                x=x_t_plot,
                y=y_t_plot,
                marker=dict(color=colores_t, size=5, opacity=0.8)
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
            
        # ponytail: specify traces index list explicitly to ensure subplot trace update binding
        frames.append(go.Frame(data=frame_data, name=f"dia_{t_val:.1f}", traces=list(range(11))))

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
                        "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
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
        for row in df_hubs.iter_rows(named=True):
            hx, hy, htipo = row["x"], row["y"], row["tipo"]
            if htipo == "agenda":
                fig.add_shape(
                    type="rect",
                    xref="x1", yref="y1",
                    x0=hx-1.5, y0=hy-1.5,
                    x1=hx+1.5, y1=hy+1.5,
                    fillcolor="Yellow",
                    opacity=0.15,
                    line=dict(color="Yellow", width=1.5)
                )
                fig.add_annotation(
                    x=hx, y=hy+2.2,
                    xref="x1", yref="y1",
                    text="Escuela",
                    showarrow=False,
                    font=dict(color="Yellow", size=10),
                    bgcolor="rgba(0,0,0,0.5)"
                )
            else:
                fig.add_shape(
                    type="circle",
                    xref="x1", yref="y1",
                    x0=hx-4.0, y0=hy-4.0,
                    x1=hx+4.0, y1=hy+4.0,
                    fillcolor="Green",
                    opacity=0.10,
                    line=dict(color="Green", width=1.5, dash="dash")
                )
                fig.add_annotation(
                    x=hx, y=hy+5.0,
                    xref="x1", yref="y1",
                    text="Plaza",
                    showarrow=False,
                    font=dict(color="Green", size=10),
                    bgcolor="rgba(0,0,0,0.5)"
                )
    except FileNotFoundError:
        pass

    # Export to standalone html
    output_file = base_dir / "plotly_animacion.html"
    fig.write_html(output_file, auto_open=False)
    print(f"Interactive dashboard generated successfully at: {output_file}")

if __name__ == "__main__":
    main()
