# ParkWise – Smart Campus Parking Availability Predictor

Interactive web app (Python, Streamlit, Plotly, scikit-learn) that forecasts campus parking
availability, recommends where to park, and learns from real campus counts.

## Public pages (what visitors see)
Overview (auto-refreshing) • Find Me Parking • Parking Predictor • Weekly Planner •
Live Monitor (animated gauges) • Dashboard (day replay, heatmap, PDF report) • History • About

## Admin
Sidebar → **🔐 Admin** → password (default `parkwise@admin`; change it by setting the
`PARKWISE_ADMIN_PASSWORD` environment variable). Unlocks **Data Lab**: log/import real counts
and update the prediction engine.

## Model details are developer-only
Model names, accuracy and comparisons are NOT shown in the website. They appear only when you
run `python train_model.py` (terminal) and are saved to `models/training_report.txt`.

## Structure
```
app.py          pages + navigation        charts.py   interactive Plotly charts
engine.py       predict/recommend/PDF     ui.py       styling, PDF chart helpers
train_model.py  data, model comparison    features.py campus config + features
```

## Run
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python train_model.py --regenerate
streamlit run app.py
```
Start with `streamlit run app.py`, not `python app.py`.

## Data note
The bundled dataset is synthetic. Add real counts in Data Lab (weighted 20x) for genuine results.
