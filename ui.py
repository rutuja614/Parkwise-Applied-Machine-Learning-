"""Styling and reusable UI / chart components."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from features import AMBER, BLUE, BUILDINGS, GREEN, NAVY, RED, ZONE_POS, OPEN_HOURS

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp {background:#F4F7FB; color:#172033; font-family:'Inter',sans-serif;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101C35 0%,#18345C 100%);}
[data-testid="stSidebar"] * {color:#F1F5F9;}
.brand {padding:10px 5px 18px; border-bottom:1px solid #344563; margin-bottom:18px;}
.brand-title {font-size:27px; font-weight:800; letter-spacing:-1px;}
.brand-sub {font-size:11px; color:#AFC3E3; letter-spacing:2px;}
.hero {background:linear-gradient(120deg,#11213F 0%,#174A85 65%,#2878C7 100%);
       padding:30px; border-radius:20px; margin-bottom:22px; box-shadow:0 12px 30px rgba(18,49,90,.12);}
.hero-badge {display:inline-block; padding:6px 13px; background:rgba(255,255,255,.13);
       border:1px solid rgba(255,255,255,.2); border-radius:30px; color:#DBEAFE;
       font-size:11px; font-weight:700; letter-spacing:1px; margin-bottom:12px;}
.hero h1 {color:white; font-size:32px; font-weight:800; margin:0 0 8px;}
.hero p {color:#DCEBFF; font-size:14px; line-height:1.7; max-width:760px; margin:0;}
.section-title {font-size:21px; font-weight:800; margin:16px 0 3px;}
.section-desc {color:#718096; font-size:13px; margin-bottom:14px;}
.metric-card {background:white; border:1px solid #E5EAF2; border-radius:16px; padding:18px 20px;
       min-height:118px; box-shadow:0 5px 18px rgba(30,50,80,.04); margin-bottom:8px;}
.metric-label {color:#718096; font-size:12px; font-weight:600; margin-bottom:10px;}
.metric-value {font-size:28px; font-weight:800;}
.metric-caption {color:#8A96A8; font-size:11px; margin-top:5px;}
.result {border-radius:18px; padding:22px; margin:16px 0; border:1px solid;}
.result.ok {background:#ECFDF5; border-color:#A7F3D0;}
.result.limited {background:#FFF7ED; border-color:#FED7AA;}
.result.full {background:#FFF1F2; border-color:#FECDD3;}
.result-title {font-size:21px; font-weight:800;}
.result-number {font-size:36px; font-weight:800; margin:6px 0;}
.result-desc {font-size:13px; color:#64748B;}
.pill {display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700;}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {background:#2563EB;
       color:white; border:none; border-radius:10px; font-weight:700; min-height:44px;}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover
       {background:#1D4ED8; color:white;}
.footer {text-align:center; color:#8793A5; font-size:11px; padding:25px 0 10px;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, text, badge):
    st.markdown(f'<div class="hero"><div class="hero-badge">{badge}</div><h1>{title}</h1>'
                f'<p>{text}</p></div>', unsafe_allow_html=True)


def metric(label, value, caption, icon="●"):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{icon} &nbsp;{label}</div>'
                f'<div class="metric-value">{value}</div><div class="metric-caption">{caption}</div></div>',
                unsafe_allow_html=True)


def section(title, desc=""):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-desc">{desc}</div>',
                unsafe_allow_html=True)


def result_card(css, title, big, desc):
    st.markdown(f'<div class="result {css}"><div class="result-title">{title}</div>'
                f'<div class="result-number">{big}</div><div class="result-desc">{desc}</div></div>',
                unsafe_allow_html=True)


def footer():
    st.markdown('<div class="footer">ParkWise | Smart Campus Parking Availability Predictor</div>',
                unsafe_allow_html=True)


def style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#DCE3ED")
    ax.tick_params(colors="#64748B", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def show_fig(fig):
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def short(area):
    return area.replace(" Parking", "")


def draw_map(snap, destination=None, best_zone=None):
    """Schematic campus map: circles = parking zones coloured by predicted status."""
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((1, 1), 98, 98, fc="#EEF3F9", ec="#CBD5E1", lw=1.5, zorder=0))
    ax.plot([50, 50], [0, 42], color="#CBD5E1", lw=11, solid_capstyle="round", zorder=1)
    ax.plot([14, 86], [42, 42], color="#CBD5E1", lw=11, solid_capstyle="round", zorder=1)

    for name, (x, y) in BUILDINGS.items():
        hi = name == destination
        ax.add_patch(FancyBboxPatch((x - 9.5, y - 3.2), 19, 6.4, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=BLUE if hi else NAVY, ec="white", lw=1.5, zorder=3))
        ax.text(x, y, name, color="white", ha="center", va="center", fontsize=7,
                fontweight="bold", zorder=4)

    if destination and best_zone:
        (zx, zy), (bx, by) = ZONE_POS[best_zone], BUILDINGS[destination]
        ax.plot([zx, bx], [zy, by], color=BLUE, lw=2.2, linestyle=(0, (4, 3)), zorder=2)

    for _, r in snap.iterrows():
        x, y = ZONE_POS[r["Parking Area"]]
        col = {"Parking Full": RED, "Limited Availability": AMBER}.get(r["Status"], GREEN)
        ax.scatter(x, y, s=r["Capacity"] * 17, color=col, alpha=0.9, edgecolor="white",
                   linewidth=2.5, zorder=5)
        if r["Parking Area"] == best_zone:
            ax.scatter(x, y, s=r["Capacity"] * 17 + 900, facecolors="none", edgecolors=BLUE,
                       linewidth=2.5, zorder=4)
        ax.text(x, y, str(int(r["Available"])), color="white", ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=6)
        ax.text(x, y - 9.6, short(r["Parking Area"]), ha="center", fontsize=7.5,
                color="#334155", fontweight="bold", zorder=6)

    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=l)
                       for c, l in [(GREEN, "Available"), (AMBER, "Limited"), (RED, "Full")]],
              loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Campus parking map (number = free spaces)", loc="left", fontsize=11, fontweight="bold")
    return fig


def draw_heat(grid, title="", cbar="% occupied", vmin=0, vmax=100, cmap="RdYlGn_r", figsize=(10, 2.8)):
    """grid: DataFrame rows=labels, columns=hours."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(grid.index))); ax.set_yticklabels([short(i) for i in grid.index])
    ax.set_xticks(range(len(grid.columns))); ax.set_xticklabels([f"{h:02d}" for h in grid.columns])
    ax.tick_params(labelsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label=cbar)
    return fig
