import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# 1. Train Congestion Predictor
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

predictor = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
predictor.fit(X_train, y_train)

y_pred = predictor.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ ML Predictor Accuracy: {acc*100:.2f}%")

# ---------------------------
# 2. Dash App Setup
# ---------------------------
import shap

# Explainability: SHAP explainer for logistic regression
explainer = shap.Explainer(predictor.named_steps['clf'], predictor.named_steps['scaler'].transform(X_train))
feature_names = ["Demand", "Users", "VoIP Fraction", "Latency", "Interference"]

# RL Agent (Q-learning) for spectrum reallocation
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q = {}  # state tuple -> action -> value
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        qs = [self.get_q(state, a) for a in self.actions]
        return self.actions[np.argmax(qs)]

    def learn(self, state, action, reward, next_state):
        maxq_next = max([self.get_q(next_state, a) for a in self.actions])
        old_q = self.get_q(state, action)
        self.q[(state, action)] = old_q + self.alpha * (reward + self.gamma * maxq_next - old_q)

# RL actions: shift % (0, 10, 20, ..., 50)
rl_actions = [0, 10, 20, 30, 40, 50]
rl_agent = QLearningAgent(rl_actions)

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Spectrum Optimization for Crowded Cities", style={'textAlign': 'center'}),

    dcc.Graph(id="latency-graph"),
    dcc.Graph(id="throughput-graph"),
    dcc.Graph(id="bandload-graph"),

    html.Div(id="explanation", style={'textAlign': 'center', 'fontSize': '18px', 'margin': '20px'}),

    dcc.Interval(id="interval", interval=1000, n_intervals=0)  # update every 1s
])

# ---------------------------
# 3. Simulation State
# ---------------------------
steps = 100
t = 0
history = {"time": [], "voip_latency": [], "video_latency": [], "bulk_tp": [], "bandA": [], "bandB": [], "predicted": [], "explanation": []}
band_capacity = 20.0

# QoS for multiple traffic types
def qos(demand, cap, voip_frac, video_frac):
    if demand <= cap:
        latency = 40 + np.random.uniform(0, 10)
        voip_latency = latency
        video_latency = latency + np.random.uniform(5, 15)
        bulk_tp = demand * (1 - voip_frac - video_frac)
    else:
        overload = demand / cap
        latency = 50 * overload + np.random.uniform(20, 40)
        voip_latency = latency
        video_latency = latency + np.random.uniform(5, 15)
        bulk_tp = cap * (1 - voip_frac - video_frac) * 0.7
    return voip_latency, video_latency, bulk_tp

# ---------------------------
# 4. Update Callback
# ---------------------------
@app.callback(
    [Output("latency-graph", "figure"),
     Output("throughput-graph", "figure"),
     Output("bandload-graph", "figure"),
     Output("explanation", "children")],
    [Input("interval", "n_intervals")]
)
def update_graph(n):
    global t

    if t >= steps:
        t = 0
        for k in history:
            history[k].clear()

    # Simulate demand
    n_users = np.random.randint(30, 80)
    voip_fraction = np.random.uniform(0.1, 0.3)
    video_fraction = np.random.uniform(0.1, 0.3)
    if voip_fraction + video_fraction > 0.6:
        video_fraction = 0.6 - voip_fraction
    total_demand = np.random.uniform(10, 45)
    latency = np.random.uniform(30, 120)
    interference = np.random.uniform(0, 1)

    demand_A, demand_B = total_demand, 0

    # ML prediction
    X_live = np.array([[demand_A, n_users, voip_fraction, latency, interference]])
    p_cong = predictor.predict_proba(X_live)[0, 1]
    congested_next = p_cong > 0.6

    # RL state/action
    rl_state = (int(demand_A//5), int(n_users//10), int(interference*10), int(latency//20))
    if congested_next:
        action = rl_agent.choose_action(rl_state)
        shift = (action/100) * demand_A
        demand_A -= shift
        demand_B += shift
    else:
        action = 0

    voipA, videoA, bulkA = qos(demand_A, band_capacity, voip_fraction, video_fraction)
    voipB, videoB, bulkB = qos(demand_B, band_capacity, voip_fraction, video_fraction)

    avg_voip_latency = np.mean([voipA, voipB])
    avg_video_latency = np.mean([videoA, videoB])
    total_bulk_tp = bulkA + bulkB

    # RL reward: negative latency, positive throughput
    reward = -avg_voip_latency - 0.5*avg_video_latency + 0.5*total_bulk_tp
    next_state = (int(demand_A//5), int(n_users//10), int(interference*10), int(latency//20))
    rl_agent.learn(rl_state, action, reward, next_state)

    # SHAP/feature importance for explainability
    shap_vals = explainer(predictor.named_steps['scaler'].transform(X_live))
    shap_imp = shap_vals.values[0]
    top_idx = np.argsort(np.abs(shap_imp))[::-1]
    top_feat = feature_names[top_idx[0]]
    top_val = X_live[0, top_idx[0]]
    top_imp = shap_imp[top_idx[0]]

    # Rule-based explanation
    rules = []
    if latency > 80:
        rules.append("⚠️ Latency High")
    if demand_A/n_users > 0.4:
        rules.append("⚠️ High per-user demand")
    if interference > 0.7:
        rules.append("⚠️ Interference High")
    if voip_fraction > 0.3:
        rules.append("VoIP load is high")
    if video_fraction > 0.3:
        rules.append("Video load is high")
    if not rules:
        rules.append("No major congestion factors detected.")

    explanation = f"<b>Congestion Explanation:</b> {', '.join(rules)}<br>"
    explanation += f"<b>Top ML Factor:</b> {top_feat} ({top_val:.2f}), SHAP={top_imp:+.2f}"
    if congested_next:
        explanation += f"<br><b>RL Action:</b> Shift {action}% to Band B"
    else:
        explanation += f"<br><b>RL Action:</b> No shift needed"

    history["time"].append(t)
    history["voip_latency"].append(avg_voip_latency)
    history["video_latency"].append(avg_video_latency)
    history["bulk_tp"].append(total_bulk_tp)
    history["bandA"].append(demand_A)
    history["bandB"].append(demand_B)
    history["predicted"].append(congested_next)
    history["explanation"].append(explanation)
    t += 1

    # Graph 1: VoIP & Video Latency
    latency_fig = go.Figure()
    colors = ["red" if c else "green" for c in history["predicted"]]
    latency_fig.add_trace(go.Scatter(
        x=history["time"], y=history["voip_latency"], mode="lines+markers",
        marker=dict(color=colors),
        name="VoIP Latency"
    ))
    latency_fig.add_trace(go.Scatter(
        x=history["time"], y=history["video_latency"], mode="lines+markers",
        marker=dict(color="blue"),
        name="Video Latency"
    ))
    latency_fig.add_hline(y=100, line_dash="dash", annotation_text="QoS Threshold")
    latency_fig.update_layout(title="VoIP & Video Latency (Red = Congestion Predicted)",
                              yaxis_title="Latency (ms)")

    # Graph 2: Bulk Throughput
    tp_fig = go.Figure()
    tp_fig.add_trace(go.Scatter(
        x=history["time"], y=history["bulk_tp"], mode="lines+markers", name="Bulk Throughput",
        marker=dict(color="orange")
    ))
    tp_fig.update_layout(title="Bulk Throughput", yaxis_title="Mbps")

    # Graph 3: Band Load
    band_fig = go.Figure()
    band_fig.add_trace(go.Scatter(x=history["time"], y=history["bandA"], mode="lines", name="Band A Load"))
    band_fig.add_trace(go.Scatter(x=history["time"], y=history["bandB"], mode="lines", name="Band B Load"))
    band_fig.update_layout(title="Dynamic Frequency Reallocation", yaxis_title="Load (Mbps)", xaxis_title="Time")

    return latency_fig, tp_fig, band_fig, explanation

# ---------------------------
# 5. Run App
if __name__ == "__main__":
    app.run(debug=True)



