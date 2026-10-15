"""ParkWise - Smart Campus Parking Availability Predictor.  Run:  streamlit run app.py"""
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

import charts
import engine as eng
import ui
from features import AREAS, BUILDINGS, DAYS, OPEN_HOURS
from train_model import train

st.set_page_config(page_title="ParkWise | Smart Campus Parking", page_icon="🅿️",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
bundle = eng.load_bundle()

# Admin password: set the PARKWISE_ADMIN_PASSWORD environment variable to change it.
ADMIN_PASSWORD = os.environ.get("PARKWISE_ADMIN_PASSWORD", "parkwise@admin")
hfmt = lambda h: f"{h:02d}:00"


def now_slot():
    n = datetime.now()
    return min(max(n.hour, OPEN_HOURS[0]), OPEN_HOURS[-1]), n.weekday()


def conditions(cols):
    e = cols[0].selectbox("Special Event", ["No", "Yes"])
    w = cols[1].selectbox("Weather", ["Clear", "Rainy"])
    return int(e == "Yes"), int(w == "Rainy")


# ================================================================ OVERVIEW
def page_overview():
    ui.hero("Smarter Parking Starts Here",
            "Know before you go. Live-updating forecasts, smart parking recommendations and an "
            "interactive campus map.", "PARKWISE • CAMPUS MOBILITY")

    @st.fragment(run_every=30)
    def live():
        hour, dow = now_slot()
        snap = eng.snapshot(bundle, hour, dow, 0, 0)
        hist = eng.read_history()
        busiest = snap.loc[snap["Occupancy %"].idxmax()]
        st.caption(f"🟢 Live view • updated {datetime.now():%H:%M:%S} • refreshes every 30 seconds")
        c = st.columns(4)
        with c[0]: ui.metric("Total Capacity", int(snap["Capacity"].sum()), f"{len(AREAS)} parking zones", "▦")
        with c[1]: ui.metric("Free Spaces", int(snap["Available"].sum()), f"Expected at {hfmt(hour)}", "✓")
        with c[2]: ui.metric("Busiest Zone", ui.short(busiest["Parking Area"]), f"{busiest['Occupancy %']}% full", "⌖")
        with c[3]: ui.metric("Saved Predictions", len(hist), "In history", "↗")
        for _, r in snap[snap["Status"] != "Spaces Available"].iterrows():
            (st.error if r["Status"] == "Parking Full" else st.warning)(
                f"**{r['Parking Area']}** is expected to be *{r['Status'].lower()}* around {hfmt(hour)}.")
        left, right = st.columns([1.4, 0.9])
        with left:
            st.plotly_chart(charts.campus_map(snap), width="stretch", config={"displayModeBar": False})
        with right:
            ui.section("Today's Outlook", "Busiest and quietest hours per zone.")
            rows = []
            for a, cap in AREAS.items():
                occ = eng.predict(bundle, OPEN_HOURS, dow, a, 0, 0)
                rows.append({"Zone": ui.short(a), "Busiest": hfmt(OPEN_HOURS[int(occ.argmax())]),
                             "Quietest": hfmt(OPEN_HOURS[int(occ.argmin())]),
                             "Peak %": round(100 * occ.max() / cap)})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            for _, r in snap.iterrows():
                st.progress(float(r["Available"] / r["Capacity"]),
                            text=f"{ui.short(r['Parking Area'])}: {r['Available']} of {r['Capacity']} free")
    live()
    st.info("Figures are estimates based on usual campus patterns, not live sensor readings.")
    ui.footer()


# ================================================================ FIND PARKING
def page_find():
    ui.hero("Find Me Parking", "Tell us where you're going. We rank every zone by walking distance "
            "and expected availability.", "SMART RECOMMENDATION")
    c = st.columns(5)
    dest = c[0].selectbox("Destination", list(BUILDINGS))
    day = c[1].selectbox("Day", DAYS, index=datetime.now().weekday())
    hour = c[2].selectbox("Arrival Time", OPEN_HOURS, index=3, format_func=hfmt)
    event, rain = conditions(c[3:5])
    dow = DAYS.index(day)
    rec = eng.recommend(bundle, dest, hour, dow, event, rain)
    snap = eng.snapshot(bundle, hour, dow, event, rain)
    if rec.empty:
        st.error("All zones are expected to be full at that time. Try a different hour.")
        st.plotly_chart(charts.campus_map(snap, dest), width="stretch", config={"displayModeBar": False}); return
    top = rec.iloc[0]
    status, icon, css = eng.status_of(top["Available"], top["Capacity"])
    ui.result_card(css, f"{icon} Best choice: {top['Parking Area']}", f"{int(top['Walk (min)'])} min walk",
                   f"{top['Walk (m)']} m to {dest} &nbsp;|&nbsp; ~{top['Available']} spaces free "
                   f"(likely {top['Free Low']}-{top['Free High']}) &nbsp;|&nbsp; {status}")
    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(charts.campus_map(snap, dest, top["Parking Area"]), width="stretch",
                        config={"displayModeBar": False})
    with right:
        ui.section("All Options Ranked", "Lower score = better (walking time + crowding penalty).")
        st.dataframe(rec[["Rank", "Parking Area", "Walk (min)", "Available", "Status", "Score"]],
                     width="stretch", hide_index=True)
        if len(rec) > 1:
            alt = rec.iloc[1]
            st.caption(f"Backup: **{alt['Parking Area']}** - {alt['Walk (min)']} min walk, {alt['Available']} free.")
    ui.footer()


# ================================================================ PREDICTOR
def page_predictor():
    ui.hero("Parking Availability Predictor",
            "Choose when and where you plan to park and get an estimate with a likely range.",
            "PREDICTION WORKSPACE")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Your Name", placeholder="Enter your name")
            d = st.date_input("Date", value=date.today())
        with c2:
            hour = st.selectbox("Arrival Time", OPEN_HOURS, index=3, format_func=hfmt)
            area = st.selectbox("Parking Area", list(AREAS))
        with c3:
            ev_t = st.selectbox("Special Event on Campus", ["No", "Yes"])
            rn_t = st.selectbox("Weather", ["Clear", "Rainy"])
        use_live = st.checkbox("Compare with the occupancy I can see right now")
        live = st.number_input("Current occupied spaces (used only if ticked)", 0, max(AREAS.values()), 0)
        go = st.form_submit_button("✦ Generate Parking Prediction", width="stretch")

    if go:
        if not name.strip():
            st.warning("Please enter your name before generating a prediction.")
        else:
            cap, ev, rn = AREAS[area], int(ev_t == "Yes"), int(rn_t == "Rainy")
            occ = int(eng.predict(bundle, [hour], d.weekday(), area, ev, rn)[0])
            hw, free = bundle["metrics"]["interval"], cap - occ
            st.session_state["latest"] = {
                "User Name": name.strip(), "Date": str(d), "Time": hfmt(hour), "Parking Area": area,
                "Day": DAYS[d.weekday()], "Special Event": ev_t, "Weather": rn_t, "Capacity": cap,
                "Predicted Occupied": occ, "Estimated Available": free,
                "Free Range": f"{max(0, free - hw)}-{min(cap, free + hw)}",
                "Live Occupied": int(live) if use_live else "",
                "Status": eng.status_of(free, cap)[0],
                "_hour": hour, "_dow": d.weekday(), "_ev": ev, "_rn": rn}
            st.session_state["saved"] = False

    r = st.session_state.get("latest")
    if not r:
        ui.footer(); return
    ui.section("Prediction Results", "Your parking estimate is ready.")
    status, icon, css = eng.status_of(r["Estimated Available"], r["Capacity"])
    c = st.columns(4)
    with c[0]: ui.metric("Total Capacity", r["Capacity"], r["Parking Area"], "▦")
    with c[1]: ui.metric("Expected Occupied", r["Predicted Occupied"], f"At {r['Time']}", "◉")
    with c[2]: ui.metric("Estimated Available", r["Estimated Available"], f"Likely range {r['Free Range']}", "✦")
    with c[3]: ui.metric("Expected Occupancy", f"{round(100 * r['Predicted Occupied'] / r['Capacity'])}%", "Of capacity", "◔")
    ui.result_card(css, f"{icon} {status}", f"{r['Estimated Available']} <span style='font-size:16px'>spaces estimated</span>",
                   f"{r['Parking Area']} &nbsp;|&nbsp; {r['Day']}, {r['Date']} &nbsp;|&nbsp; {r['Time']}")
    if r["Live Occupied"] != "":
        diff = int(r["Live Occupied"]) - r["Predicted Occupied"]
        if abs(diff) <= 3:
            st.info(f"Your live count ({r['Live Occupied']}) matches what is usual ({r['Predicted Occupied']}). A normal day.")
        elif diff > 0:
            st.warning(f"Your live count is {diff} higher than usual for this time - expect it to be busier than normal.")
        else:
            st.success(f"Your live count is {-diff} lower than usual for this time - quieter than normal.")

    occ_curve = eng.predict(bundle, OPEN_HOURS, r["_dow"], r["Parking Area"], r["_ev"], r["_rn"])
    free = r["Capacity"] - occ_curve
    hw = bundle["metrics"]["interval"]
    best_h, best_f = max([(h, f) for h, f in zip(OPEN_HOURS, free) if h >= r["_hour"]], key=lambda x: x[1])
    left, right = st.columns([1.3, 0.7])
    with left:
        st.plotly_chart(charts.free_curve(OPEN_HOURS, free, np.clip(free - hw, 0, None),
                                          np.clip(free + hw, 0, r["Capacity"]), r["_hour"],
                                          f"Expected free spaces - {r['Parking Area']}"),
                        width="stretch", config={"displayModeBar": False})
    with right:
        ui.metric("Best Time to Arrive", hfmt(best_h), f"≈ {best_f} free (from selected time onward)", "⌚")
        st.write(f"**Day:** {r['Day']}  \n**Event:** {r['Special Event']}  \n**Weather:** {r['Weather']}")
    if st.button("💾 Save Prediction to History", width="stretch"):
        if st.session_state.get("saved"):
            st.info("This prediction has already been saved.")
        else:
            eng.save_record({k: v for k, v in r.items() if not k.startswith("_")})
            st.session_state["saved"] = True
            st.success("Prediction saved successfully!")


# ================================================================ PLANNER
def page_planner():
    ui.hero("Weekly Planner & What-If", "Find the quietest slots of the week and see how rain or events "
            "change availability.", "PLAN AHEAD")
    c = st.columns(3)
    area = c[0].selectbox("Parking Area", list(AREAS))
    event, rain = conditions(c[1:3])
    cap = AREAS[area]
    grid = eng.week_grid(bundle, area, event, rain)
    t1, t2, t3 = st.tabs(["📅 Weekly Heatmap", "🔀 What-If Scenarios", "⭐ Best Slots"])
    with t1:
        st.plotly_chart(charts.heatmap(grid, f"Expected free spaces - {area}", "free", 0, cap, "RdYlGn", 380),
                        width="stretch", config={"displayModeBar": False})
    with t2:
        day = st.selectbox("Day for comparison", DAYS, index=datetime.now().weekday(), key="wi_day")
        dow = DAYS.index(day)
        curves = {lab: cap - eng.predict(bundle, OPEN_HOURS, dow, area, e, r_) for lab, (e, r_) in
                  {"Normal": (0, 0), "Rain": (0, 1), "Special event": (1, 0), "Event + rain": (1, 1)}.items()}
        st.plotly_chart(charts.scenarios(OPEN_HOURS, curves, f"{area} - {day}"), width="stretch",
                        config={"displayModeBar": False})
        i = OPEN_HOURS.index(10)
        st.info(f"At 10:00 an event with rain adds about **{int(curves['Normal'][i] - curves['Event + rain'][i])}** "
                "extra parked vehicles compared to a normal day.")
    with t3:
        sub = grid[[h for h in grid.columns if 8 <= h <= 17]].stack().reset_index()
        sub.columns = ["Day", "Hour", "Free"]
        top = sub.sort_values("Free", ascending=False).head(5).copy()
        top["Hour"] = top["Hour"].map(hfmt); top["Free"] = top["Free"].astype(int)
        st.write("**Top 5 quietest slots (campus hours 08:00-17:00)**")
        st.dataframe(top.rename(columns={"Free": "Free spaces"}), width="stretch", hide_index=True)
        worst = sub.sort_values("Free").iloc[0]
        st.warning(f"Avoid **{worst['Day']} {hfmt(int(worst['Hour']))}** - the most crowded slot ({int(worst['Free'])} free).")
    ui.footer()


# ================================================================ LIVE MONITOR
def page_live():
    ui.hero("Live Monitor", "Animated gauges and alerts for every zone. This is a simulated demo feed - "
            "connect real sensors or gate data to go fully live.", "DEMO FEED • SIMULATED")
    c = st.columns(4)
    auto = c[0].toggle("Auto-refresh (3 s)", value=True)
    event = int(c[1].toggle("Simulate special event"))
    rain = int(c[2].toggle("Simulate rain"))
    if c[3].button("↺ Reset", width="stretch"):
        st.session_state.pop("sim", None)
    if not auto:
        st.button("📡 Take new reading")

    @st.fragment(run_every=3 if auto else None)
    def panel():
        hour, dow = now_slot()
        sim = st.session_state.setdefault("sim", {a: [] for a in AREAS})
        rng = np.random.default_rng()
        for a, cap in AREAS.items():
            target = int(eng.predict(bundle, [hour], dow, a, event, rain)[0])
            for _ in range(12 if not sim[a] else 1):
                prev = sim[a][-1] if sim[a] else target
                sim[a].append(int(np.clip(prev + round(0.3 * (target - prev) + rng.normal(0, 1.5)), 0, cap)))
            del sim[a][:-40]
        st.caption(f"Last reading {datetime.now():%H:%M:%S}")
        cols = st.columns(len(AREAS))
        for col, (a, cap) in zip(cols, AREAS.items()):
            free = cap - sim[a][-1]
            with col:
                st.plotly_chart(charts.gauge(free, cap, ui.short(a)), width="stretch",
                                config={"displayModeBar": False}, key=f"g_{a}")
                css = eng.status_of(free, cap)[2]
                if css == "full": st.error("FULL - redirect drivers")
                elif css == "limited": st.warning(f"Almost full ({free} left)")
                else: st.success("Spaces available")
        st.plotly_chart(charts.live_lines({a: [AREAS[a] - v for v in sim[a]] for a in AREAS}),
                        width="stretch", config={"displayModeBar": False}, key="live_lines")
    panel()
    ui.footer()


# ================================================================ DASHBOARD
def page_dashboard():
    ui.hero("Parking Operations Dashboard", "Compare zones, replay the day and export a PDF report.",
            "CAMPUS ANALYTICS")
    c = st.columns(4)
    day = c[0].selectbox("Day", DAYS, index=datetime.now().weekday())
    hour = c[1].selectbox("Time", OPEN_HOURS, index=3, format_func=hfmt)
    event, rain = conditions(c[2:4])
    dow = DAYS.index(day)
    snap = eng.snapshot(bundle, hour, dow, event, rain)
    peak = snap.loc[snap["Occupancy %"].idxmax()]
    k = st.columns(4)
    with k[0]: ui.metric("Total Capacity", int(snap["Capacity"].sum()), "All zones", "▦")
    with k[1]: ui.metric("Expected Occupied", int(snap["Occupied"].sum()), f"{day} {hfmt(hour)}", "◉")
    with k[2]: ui.metric("Expected Free", int(snap["Available"].sum()), "All zones", "✓")
    with k[3]: ui.metric("Busiest Zone", ui.short(peak["Parking Area"]), f"{peak['Occupancy %']}% full", "⌖")

    left, right = st.columns([0.9, 1.1])
    with left:
        st.plotly_chart(charts.stacked_bar(snap), width="stretch", config={"displayModeBar": False})
    with right:
        st.plotly_chart(charts.campus_map(snap), width="stretch", config={"displayModeBar": False})

    ui.section("▶ Day Replay", f"Watch the zones fill up and empty through {day}.")
    occ = {a: eng.predict(bundle, OPEN_HOURS, dow, a, event, rain) for a in AREAS}
    st.plotly_chart(charts.replay(occ, OPEN_HOURS), width="stretch", config={"displayModeBar": False})

    ui.section("Occupancy Heatmap", f"Expected occupancy % through {day} (hover for values).")
    st.plotly_chart(charts.heatmap(eng.day_grid(bundle, dow, event, rain), "", "% occupied", 0, 100, "RdYlGn_r"),
                    width="stretch", config={"displayModeBar": False})
    ui.section("Parking Statistics")
    st.dataframe(snap.drop(columns=["Free Low", "Free High"]), width="stretch", hide_index=True)
    st.download_button("📄 Download PDF Report", eng.make_pdf(bundle, dow, hour, event, rain),
                       f"parkwise_report_{day.lower()}_{hour:02d}00.pdf", "application/pdf", width="stretch")
    ui.footer()


# ================================================================ HISTORY
def page_history():
    ui.hero("Prediction History", "Review saved estimates, search records and export them.", "RECORDS & REPORTING")
    h = eng.read_history()
    if h.empty:
        st.info("No saved predictions yet. Generate one on the Parking Predictor page and save it."); return
    c = st.columns(3)
    with c[0]: ui.metric("Total Records", len(h), "Saved predictions", "▤")
    with c[1]: ui.metric("Zones Used", h["Parking Area"].nunique(), "Unique areas", "⌖")
    with c[2]: ui.metric("Latest Record", str(h.iloc[-1]["Date"]), "Most recent date", "◷")
    q = st.text_input("Search Records", placeholder="Search by user name or parking area...")
    shown = h
    if q.strip():
        m = pd.Series(False, index=h.index)
        for col in ("User Name", "Parking Area"):
            if col in h.columns:
                m |= h[col].astype(str).str.contains(q, case=False, na=False)
        shown = h[m]
    st.dataframe(shown, width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.download_button("⬇ Download as CSV", shown.to_csv(index=False).encode(), "parking_history.csv",
                      "text/csv", width="stretch")
    if b.button("🗑 Clear All History", width="stretch"):
        eng.HISTORY_PATH.unlink(missing_ok=True); st.rerun()


# ================================================================ ADMIN: DATA LAB
def page_datalab():
    ui.hero("Data Lab (Admin)", "Log real car counts or import a CSV, then update the prediction engine.",
            "ADMIN • REAL DATA")
    if msg := st.session_state.pop("retrained", None):
        st.success(msg)
    obs = eng.read_observed()
    c = st.columns(2)
    with c[0]: ui.metric("Real Observations", len(obs), "Logged so far", "▤")
    with c[1]: ui.metric("Last Update", bundle["trained_at"], "Prediction engine", "⚙")

    ui.section("1. Log a real observation", "Count the cars in a zone and record it.")
    with st.form("obs_form"):
        f = st.columns(6)
        d = f[0].date_input("Date", value=date.today())
        h = f[1].selectbox("Hour", OPEN_HOURS, index=3, format_func=hfmt)
        a = f[2].selectbox("Area", list(AREAS))
        e = f[3].selectbox("Event", ["No", "Yes"])
        w = f[4].selectbox("Weather", ["Clear", "Rainy"])
        o = f[5].number_input("Cars counted", 0, max(AREAS.values()), 0)
        add = st.form_submit_button("➕ Add observation", width="stretch")
    if add:
        if o > AREAS[a]:
            st.warning(f"{a} holds at most {AREAS[a]} vehicles.")
        else:
            n = eng.append_observed(pd.DataFrame([{"date": str(d), "hour": h, "area": a,
                                                   "special_event": int(e == "Yes"), "rain": int(w == "Rainy"),
                                                   "occupied": int(o)}]))
            st.success(f"Saved. You now have {n} real observation(s).")
            obs = eng.read_observed()

    ui.section("2. Import a CSV", "Columns: date, hour, area, special_event, rain, occupied")
    up = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
    if up is not None and st.button("Import uploaded rows", width="stretch"):
        clean, problem = eng.validate_upload(pd.read_csv(up))
        if clean is None:
            st.error(problem)
        else:
            n = eng.append_observed(clean)
            st.success(f"Imported {len(clean)} row(s). Total observations: {n}. {problem or ''}")
            obs = eng.read_observed()
    st.download_button("⬇ Download CSV template",
                       "date,hour,area,special_event,rain,occupied\n2026-10-12,10,Main Gate Parking,0,0,64\n",
                       "observation_template.csv", "text/csv")

    ui.section("3. Update the prediction engine", "Real observations carry much more weight than the sample data.")
    if not obs.empty:
        st.dataframe(obs.tail(10), width="stretch", hide_index=True)
    b1, b2 = st.columns(2)
    if b1.button("🔁 Update predictions now", width="stretch"):
        with st.spinner("Updating..."):
            nb = train()
        eng.load_bundle.clear()
        st.session_state["retrained"] = f"Prediction engine updated using {nb['real_rows']} real observation(s)."
        st.rerun()
    if b2.button("🗑 Delete all observations", width="stretch"):
        eng.OBSERVED_PATH.unlink(missing_ok=True); st.rerun()


# ================================================================ ABOUT
def page_about():
    ui.hero("About ParkWise", "A machine-learning powered system for smarter campus parking.", "PROJECT INFORMATION")
    l, r = st.columns([1.1, 0.9])
    with l:
        st.markdown("### The Problem")
        st.write("Drivers waste time and fuel circling full lots, and campus staff have no forecast of peak demand.")
        st.markdown("### The Solution")
        st.write("ParkWise forecasts occupancy for every parking zone from the time, weekday, weather and campus "
                 "events, then turns the forecast into recommendations, alerts and reports.")
        st.markdown("### Key Features")
        st.write("• *Find Me Parking* with walking distance and an interactive campus map\n\n"
                 "• Forecasts with a likely range and best arrival time\n\n"
                 "• Weekly heatmap and what-if scenarios for rain and events\n\n"
                 "• Live monitor with animated gauges and alerts\n\n"
                 "• Day replay and one-click PDF reports\n\n"
                 "• Admin tools to add real campus data")
        st.markdown("### Future Scope")
        st.write("• Sensor and gate-log integration  • Mobile app with alerts  • Space reservation")
    with r:
        st.markdown("### How It Works")
        st.code("Campus data\n   ↓ Prediction engine\n   ↓ Recommendations & alerts\n   ↓ Interactive web dashboard",
                language="text")
        st.markdown("### Technology")
        for t, d in [("Python", "Core language"), ("Streamlit", "Web interface"), ("Plotly", "Interactive charts"),
                     ("Pandas / NumPy", "Data handling"), ("scikit-learn", "Machine learning")]:
            st.markdown(f"**{t}** — {d}")
    ui.footer()


# ================================================================ NAVIGATION
PUBLIC = {"🏠 Overview": page_overview, "🧭 Find Me Parking": page_find, "🔮 Parking Predictor": page_predictor,
          "📅 Weekly Planner": page_planner, "📡 Live Monitor": page_live, "📊 Dashboard": page_dashboard,
          "🗂 History": page_history, "ℹ️ About": page_about}
is_admin = st.session_state.get("admin", False)
PAGES = {**PUBLIC, "🧪 Data Lab": page_datalab} if is_admin else PUBLIC

with st.sidebar:
    st.markdown('<div class="brand"><div style="font-size:36px">🅿️</div><div class="brand-title">ParkWise</div>'
                '<div class="brand-sub">SMART CAMPUS MOBILITY</div></div>', unsafe_allow_html=True)
    st.caption("MAIN MENU")
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    with st.expander("🔐 Admin"):
        if is_admin:
            st.success("Admin mode on")
            if st.button("Log out"):
                st.session_state["admin"] = False; st.rerun()
        else:
            pw = st.text_input("Password", type="password", key="admin_pw")
            if st.button("Log in"):
                if pw == ADMIN_PASSWORD:
                    st.session_state["admin"] = True; st.rerun()
                else:
                    st.error("Incorrect password")
    st.caption("Smart Campus Project")

PAGES[choice]()
