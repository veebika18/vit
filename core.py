import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dash import ctx

# ---------------------------
# 1. Train Congestion Predictor (XGBoost)
# ---------------------------
def generate_training_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    demand = rng.uniform(5, 40, size=n)
    users = rng.integers(10, 100, size=n)
    voip_fraction = rng.uniform(0.05, 0.5, size=n)
    latency = rng.uniform(20, 150, size=n)
    interference = rng.uniform(0, 1, size=n)

    y = ((demand/users > 0.4) & (latency > 80)) | (interference > 0.7)
    X = np.c_[demand, users, voip_fraction, latency, interference]
    return X, y.astype(int)

X, y = generate_training_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'objective': 'binary:logistic', 'max_depth': 3, 'eval_metric': 'logloss'}
xgb_model = xgb.train(params, dtrain, num_boost_round=30)

y_pred_prob = xgb_model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost Predictor Accuracy: {acc*100:.2f}%")

# ---------------------------
# 2. Dash App Setup
# ---------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Spectrum Optimization for Crowded Cities", style={'textAlign': 'center'}),
    html.Div(id="status-panel", style={"margin": "20px", "fontWeight": "bold"}),
    html.Div(id="summary-panel", style={"margin": "20px", "fontWeight": "bold", "color": "#333"}),
    html.Label("Number of Users:"),
    dcc.Slider(id="user-slider", min=30, max=80, step=1, value=50),
    dcc.Graph(id="latency-graph"),
    dcc.Graph(id="throughput-graph"),
    dcc.Graph(id="bandload-graph"),
    dcc.Interval(id="interval", interval=1000, n_intervals=0),  # update every 1s
    html.Button("Download Data", id="download-btn", n_clicks=0),
    dcc.Download(id="download-data")
])

# ---------------------------
# 3. Simulation State
# ---------------------------
steps = 100
t = 0
history = {"time": [], "voip_latency": [], "bulk_tp": [], "bandA": [], "bandB": [], "predicted": []}

def qos(demand, cap, voip_frac):
    if demand <= cap:
        latency = 40 + np.random.uniform(0, 10)
        voip_latency = latency
        bulk_tp = demand * (1 - voip_frac)
    else:
        overload = demand / cap
        latency = 50 * overload + np.random.uniform(20, 40)
        voip_latency = latency
        bulk_tp = cap * (1 - voip_frac) * 0.7
    return voip_latency, bulk_tp

# ---------------------------
# SDR Sensing Simulation & Regulatory Compliance
# ---------------------------
def sdr_sense():
    bands = [
        {"name": "Band A", "freq": 1850, "capacity": 20.0},
        {"name": "Band B", "freq": 1950, "capacity": 20.0}
    ]
    interference = np.random.uniform(0, 1, size=len(bands))
    available = [b for b, i in zip(bands, interference) if i < 0.8]
    return available, interference

# ---------------------------
# Fair Allocation Function
# ---------------------------
def fair_allocation(total_demand, bands):
    # Divide demand fairly among available bands
    if len(bands) == 0:
        return [0, 0]
    elif len(bands) == 1:
        return [total_demand, 0]
    else:
        # Split demand equally
        split = total_demand / len(bands)
        return [split, split]

# ---------------------------
# 4. Update Callback
# ---------------------------
@app.callback(
    [Output("latency-graph", "figure"),
     Output("throughput-graph", "figure"),
     Output("bandload-graph", "figure"),
     Output("status-panel", "children"),
     Output("summary-panel", "children"),
     Output("download-data", "data")],
    [Input("interval", "n_intervals"),
     Input("user-slider", "value"),
     Input("download-btn", "n_clicks")]
)
def update_graph(n, n_users, download_clicks):
    global t

    if t >= steps:
        t = 0
        for k in history:
            history[k].clear()

    # SDR Sensing & Regulatory Compliance
    bands, interference = sdr_sense()
    # Remove band filtering based on user selection
    if not bands:
        status = "No spectrum available (all bands congested per DoT/TRAI norms)."
        return go.Figure(), go.Figure(), go.Figure(), status, "", None

    voip_fraction = np.random.uniform(0.1, 0.4)
    total_demand = np.random.uniform(0.2, 0.7) * n_users  # demand scales with users

    # ML prediction (XGBoost)
    latency = np.random.uniform(30, 120)
    X_live = np.array([[total_demand, n_users, voip_fraction, latency, max(interference)]])
    d_live = xgb.DMatrix(X_live)
    p_cong = xgb_model.predict(d_live)[0]
    congested_next = p_cong > 0.6

    # Fair allocation
    allocations = fair_allocation(total_demand, bands)
    demand_A = allocations[0]
    demand_B = allocations[1]

    voipA, bulkA = qos(demand_A, bands[0]["capacity"] if len(bands) > 0 else 20.0, voip_fraction)
    voipB, bulkB = qos(demand_B, bands[1]["capacity"] if len(bands) > 1 else 20.0, voip_fraction)

    avg_voip_latency = np.mean([voipA, voipB])
    total_bulk_tp = bulkA + bulkB

    # --- Force throughput drop and latency increase if congestion predicted ---
    if congested_next:
        avg_voip_latency *= 1.5  # Increase latency by 50%
        total_bulk_tp *= 0.5     # Decrease throughput by 50%
    # -------------------------------------------------------------------------

    history["time"].append(t)
    history["voip_latency"].append(avg_voip_latency)
    history["bulk_tp"].append(total_bulk_tp)
    history["bandA"].append(demand_A)
    history["bandB"].append(demand_B)
    history["predicted"].append(congested_next)
    t += 1

    # Graphs
    colors = ["red" if c else "green" for c in history["predicted"]]
    latency_fig = go.Figure()
    latency_fig.add_trace(go.Scatter(
        x=history["time"], y=history["voip_latency"], mode="lines+markers",
        marker=dict(color=colors),
        name="VoIP Latency"
    ))
    latency_fig.add_hline(y=100, line_dash="dash", annotation_text="QoS Threshold")
    latency_fig.update_layout(title="VoIP Latency (Red = Congestion Predicted)", yaxis_title="Latency (ms)")

    tp_fig = go.Figure()
    tp_fig.add_trace(go.Scatter(
        x=history["time"], y=history["bulk_tp"], mode="lines+markers", name="Bulk Throughput",
        marker=dict(color="orange")
    ))
    tp_fig.update_layout(title="Bulk Throughput", yaxis_title="Mbps")

    band_fig = go.Figure()
    band_fig.add_trace(go.Scatter(x=history["time"], y=history["bandA"], mode="lines", name="Band A Load"))
    band_fig.add_trace(go.Scatter(x=history["time"], y=history["bandB"], mode="lines", name="Band B Load"))
    band_fig.update_layout(title="Dynamic Frequency Reallocation", yaxis_title="Load (Mbps)", xaxis_title="Time")

    bands_used = ", ".join([f"{b['name']} ({b['freq']} MHz)" for b in bands])
    status = f"Step {t}: {'Congestion detected, reallocating...' if congested_next else 'Normal operation'} | Bands used: {bands_used if bands_used else 'None'}"

    # Summary statistics (last 20 steps or all if less)
    N = 20
    latency_vals = history["voip_latency"][-N:] if len(history["voip_latency"]) >= N else history["voip_latency"]
    tp_vals = history["bulk_tp"][-N:] if len(history["bulk_tp"]) >= N else history["bulk_tp"]
    cong_vals = history["predicted"][-N:] if len(history["predicted"]) >= N else history["predicted"]

    avg_latency = np.mean(latency_vals) if latency_vals else 0
    avg_tp = np.mean(tp_vals) if tp_vals else 0
    cong_count = sum(cong_vals)
    bands_used = ", ".join([f"{b['name']} ({b['freq']} MHz)" for b in bands])

    # Previous step latency
    prev_latency = history["voip_latency"][-2] if len(history["voip_latency"]) > 1 else None
    # Next step latency (just calculated)
    next_latency = avg_voip_latency

    alert_msg = ""
    if next_latency > 100:
        alert_msg += f"⚠ QoS Violation: Latency ({next_latency:.2f} ms) exceeds threshold! "
    if avg_tp < 5:
        alert_msg += f"⚠ QoS Violation: Throughput ({avg_tp:.2f} Mbps) below threshold! "

    compliance_info = []
    for i, b in enumerate(bands):
        compliant = "Compliant" if 1800 <= b["freq"] <= 2000 else "Non-Compliant"
        interference_val = interference[i] if i < len(interference) else None
        interference_status = f"Interference: {interference_val:.2f}" if interference_val is not None else ""
        compliance_info.append(f"{b['name']} ({b['freq']} MHz): {compliant} {interference_status}")

    summary = html.Div([
        html.H4("Consolidated Network Summary"),
        html.P(f"Current Step: {t}"),
        html.P(f"Number of Users: {n_users}"),
        html.P(f"Bands Used: {bands_used if bands_used else 'None'}"),
        html.P(f"Previous Step Latency: {f'{prev_latency:.2f} ms' if prev_latency is not None else 'N/A'}"),
        html.P(f"Next Step Latency: {f'{next_latency:.2f} ms'}"),
        html.P(f"Average Latency (last {N} steps): {avg_latency:.2f} ms"),  # <-- Add this line
        html.P(f"Average Throughput (last {N} steps): {avg_tp:.2f} Mbps"),
        html.P(f"Congestion Events (last {N} steps): {cong_count}"),
        html.P(alert_msg, style={"color": "red", "fontWeight": "bold"}) if alert_msg else None,
        html.H5("Band Regulatory Compliance:"),
        html.Ul([html.Li(info) for info in compliance_info])
    ])

    # Prepare download data if button clicked
    download_data = None
    if ctx.triggered_id == "download-btn":
        import pandas as pd
        df = pd.DataFrame({
            "Step": history["time"],
            "Latency": history["voip_latency"],
            "Throughput": history["bulk_tp"],
            "BandA": history["bandA"],
            "BandB": history["bandB"],
            "Congestion": history["predicted"]
        })
        download_data = dcc.send_data_frame(df.to_csv, "simulation_history.csv")

    return latency_fig, tp_fig, band_fig, status, summary, download_data

# ---------------------------
# 5. Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)