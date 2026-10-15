"""Interactive Plotly charts (hover, zoom, animation) for the web UI."""
import numpy as np
import plotly.graph_objects as go

from features import AMBER, AREAS, BLUE, BUILDINGS, GREEN, NAVY, RED, ZONE_POS
from ui import short

BASE = dict(template="plotly_white", font=dict(family="Inter, sans-serif", size=12, color="#334155"),
            margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white")
STATUS_COL = {"Parking Full": RED, "Limited Availability": AMBER, "Spaces Available": GREEN}


def _title(t):
    return dict(text=f"<b>{t}</b>", x=0, font=dict(size=15)) if t else None


def campus_map(snap, destination=None, best_zone=None):
    fig = go.Figure()
    for x0, y0, x1, y1 in [(50, 0, 50, 42), (14, 42, 86, 42)]:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#CBD5E1", width=16), layer="below")
    if destination and best_zone:
        (zx, zy), (bx, by) = ZONE_POS[best_zone], BUILDINGS[destination]
        fig.add_trace(go.Scatter(x=[zx, bx], y=[zy, by], mode="lines", hoverinfo="skip",
                                 line=dict(color=BLUE, width=3, dash="dash"), showlegend=False))
    for name, (x, y) in BUILDINGS.items():
        fig.add_annotation(x=x, y=y, text=f"<b>{name}</b>", showarrow=False, font=dict(color="white", size=11),
                           bgcolor=BLUE if name == destination else NAVY, borderpad=5, bordercolor="white")
    if best_zone:
        x, y = ZONE_POS[best_zone]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=82, color="rgba(0,0,0,0)", line=dict(color=BLUE, width=3))))
    xs = [ZONE_POS[a][0] for a in snap["Parking Area"]]
    ys = [ZONE_POS[a][1] for a in snap["Parking Area"]]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=snap["Available"].astype(str), textfont=dict(color="white", size=15),
        marker=dict(size=snap["Capacity"] * 0.42 + 22, color=[STATUS_COL[s] for s in snap["Status"]],
                    line=dict(color="white", width=3)),
        customdata=np.stack([snap["Parking Area"], snap["Available"], snap["Free Low"], snap["Free High"],
                             snap["Occupancy %"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Free: %{customdata[1]} (likely %{customdata[2]}-%{customdata[3]})"
                      "<br>Occupancy: %{customdata[4]}%<extra></extra>", showlegend=False))
    for a, x, y in zip(snap["Parking Area"], xs, ys):
        fig.add_annotation(x=x, y=y - 10, text=f"<b>{short(a)}</b>", showarrow=False, font=dict(size=11, color="#334155"))
    for lab, col in [("Available", GREEN), ("Limited", AMBER), ("Full", RED)]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=lab, marker=dict(size=10, color=col)))
    fig.update_xaxes(range=[0, 100], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0, 100], visible=False, scaleanchor="x", fixedrange=True)
    fig.update_layout(**{**BASE, "plot_bgcolor": "#EEF3F9"}, height=500, title=_title("Campus map (number = free spaces)"),
                      legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"))
    return fig


def free_curve(hours, free, lo, hi, selected, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=hours, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(37,99,235,.13)", name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hours, y=free, mode="lines+markers", name="Predicted free",
                             line=dict(color=BLUE, width=3), hovertemplate="%{x}:00 - %{y} free<extra></extra>"))
    fig.add_vline(x=selected, line=dict(color=RED, dash="dash", width=1.5))
    fig.update_xaxes(tickmode="array", tickvals=hours, ticktext=[f"{h:02d}" for h in hours], title="Hour of day")
    fig.update_yaxes(title="Free spaces", rangemode="tozero")
    fig.update_layout(**BASE, height=380, title=_title(title), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig


def stacked_bar(snap):
    x = [short(a) for a in snap["Parking Area"]]
    fig = go.Figure([go.Bar(x=x, y=snap["Occupied"], name="Occupied", marker_color=BLUE),
                     go.Bar(x=x, y=snap["Available"], name="Available", marker_color="#CBD5E1")])
    fig.update_layout(**BASE, barmode="stack", height=400, title=_title("Capacity distribution"),
                      legend=dict(orientation="h", y=1.1, x=1, xanchor="right"), yaxis_title="Parking spaces")
    return fig


def heatmap(grid, title, label, zmin, zmax, scale, height=330):
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"{h:02d}:00" for h in grid.columns], y=[short(i) for i in grid.index],
        zmin=zmin, zmax=zmax, colorscale=scale, colorbar=dict(title=label),
        hovertemplate="%{y} @ %{x}<br>%{z:.0f}<extra></extra>"))
    fig.update_layout(**BASE, height=height, title=_title(title))
    return fig


def scenarios(hours, curves, title):
    cols = {"Normal": GREEN, "Rain": BLUE, "Special event": AMBER, "Event + rain": RED}
    fig = go.Figure([go.Scatter(x=hours, y=v, mode="lines", name=k, line=dict(width=3, color=cols[k]))
                     for k, v in curves.items()])
    fig.update_xaxes(tickmode="array", tickvals=hours, ticktext=[f"{h:02d}" for h in hours])
    fig.update_layout(**BASE, height=380, title=_title(title), yaxis_title="Free spaces", hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig


def gauge(free, cap, title):
    col = RED if free <= 0 else AMBER if free / cap <= 0.2 else GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=free, title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
        number=dict(suffix=f" / {cap}", font=dict(size=24)),
        gauge=dict(axis=dict(range=[0, cap]), bar=dict(color=col), bgcolor="#E5EAF2", borderwidth=0)))
    fig.update_layout(height=210, margin=dict(l=25, r=25, t=55, b=5), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def live_lines(series):
    cols = [BLUE, GREEN, AMBER, RED]
    fig = go.Figure([go.Scatter(y=v, mode="lines", name=short(a), line=dict(width=3, color=c),
                                hovertemplate="%{y} free<extra></extra>")
                     for (a, v), c in zip(series.items(), cols)])
    fig.update_layout(**BASE, height=340, title=_title("Free spaces over recent readings"),
                      yaxis_title="Free spaces", xaxis_title="Recent readings", hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig


def replay(occ_by_area, hours):
    """Animated stacked bars: occupancy of every zone as the day progresses."""
    names = list(AREAS)

    def bars(i):
        occ = [int(occ_by_area[a][i]) for a in names]
        return [go.Bar(x=[short(a) for a in names], y=occ, name="Occupied", marker_color=BLUE,
                       text=occ, textposition="inside"),
                go.Bar(x=[short(a) for a in names], y=[AREAS[a] - o for a, o in zip(names, occ)],
                       name="Free", marker_color="#CBD5E1")]

    labels = [f"{h:02d}:00" for h in hours]
    fig = go.Figure(data=bars(0), frames=[go.Frame(data=bars(i), name=l) for i, l in enumerate(labels)])
    play = dict(label="▶ Play", method="animate",
                args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True, transition=dict(duration=300))])
    pause = dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])
    fig.update_layout(**BASE, barmode="stack", height=470, title=_title("Day replay - press Play"),
                      yaxis=dict(range=[0, max(AREAS.values()) + 5], title="Spaces"),
                      updatemenus=[dict(type="buttons", showactive=False, x=1, xanchor="right", y=1.18, buttons=[play, pause])],
                      sliders=[dict(x=0.05, len=0.9, currentvalue=dict(prefix="Time: "),
                                    steps=[dict(method="animate", label=l,
                                                args=[[l], dict(mode="immediate", frame=dict(duration=300, redraw=True))])
                                           for l in labels])],
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig
