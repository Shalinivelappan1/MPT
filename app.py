import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

st.set_page_config(page_title="Markowitz Lab", layout="wide")

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload adjusted close CSV", type=["csv"])
rf = st.sidebar.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.5, value=0.06, step=0.005)
long_only = st.sidebar.checkbox("Long-only (no shorts)", value=True)
trading_days = st.sidebar.number_input("Trading days per year", min_value=200, max_value=365, value=252, step=1)
num_pts = st.sidebar.slider("Frontier points", 20, 150, 60, 5)
top_hold = st.sidebar.slider("Top holdings", 5, 30, 12, 1)

st.title("Mean–Variance Portfolio Lab (Markowitz)")
st.caption("Upload adjusted closing prices (columns=tickers, index=Date).")

@st.cache_data(show_spinner=True)
def load_prices(file) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df = df.dropna(axis=1, how="all")
    return df

if uploaded is None:
    st.warning("Upload a CSV of adjusted closes to begin.")
    st.stop()

prices = load_prices(uploaded)
returns = prices.pct_change().dropna(how="all").dropna(axis=1, how="all")
tickers = returns.columns.tolist()

mu_daily = returns.mean()
mu = mu_daily * trading_days

try:
    lw = LedoitWolf().fit(returns.fillna(0).values)
    cov_mat = lw.covariance_ * trading_days
except Exception:
    cov_mat = returns.cov().loc[tickers, tickers].values * trading_days

eigmin = np.linalg.eigvalsh(cov_mat).min()
if eigmin <= 1e-10:
    cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * (abs(eigmin) + 1e-8)

mu_vec = mu.loc[tickers].values
n = len(tickers)

bounds = None if not long_only else tuple((0.0, 1.0) for _ in range(n))
cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

def port_return(w): return float(w @ mu_vec)
def port_vol(w): return float(np.sqrt(w.T @ cov_mat @ w))

def min_var_portfolio():
    x0 = np.repeat(1.0/n, n)
    res = minimize(lambda w: float(w.T @ cov_mat @ w), x0, method='SLSQP',
                   bounds=bounds, constraints=(cons_sum,), options={'maxiter': 2000})
    if not res.success: st.stop()
    return res.x

def max_sharpe_portfolio(rf=0.0):
    def neg_sh(w):
        v = np.sqrt(w.T @ cov_mat @ w)
        if v == 0: return 1e9
        return - (float(w @ mu_vec) - rf) / v
    x0 = np.repeat(1.0/n, n)
    res = minimize(neg_sh, x0, method='SLSQP',
                   bounds=bounds, constraints=(cons_sum,), options={'maxiter': 2000})
    if not res.success: st.stop()
    return res.x

def min_var_for_target(target):
    cons = (cons_sum, {'type':'eq', 'fun': lambda w: float(w @ mu_vec) - target})
    x0 = np.repeat(1.0/n, n)
    res = minimize(lambda w: float(w.T @ cov_mat @ w), x0, method='SLSQP',
                   bounds=bounds, constraints=cons, options={'maxiter': 2000})
    return res.x if res.success else None

with st.spinner("Optimizing..."):
    w_mv = min_var_portfolio()
    w_ms = max_sharpe_portfolio(rf=rf)

r_mv, v_mv = port_return(w_mv), port_vol(w_mv)
r_ms, v_ms = port_return(w_ms), port_vol(w_ms)

targets = np.linspace(r_mv, float(mu_vec.max()), num_pts)
frontier_ws, frontier_rs, frontier_vs = [], [], []
for t in targets:
    w = min_var_for_target(t)
    if w is not None:
        frontier_ws.append(w)
        frontier_rs.append(port_return(w))
        frontier_vs.append(port_vol(w))

def weights_table(w, k=top_hold):
    s = pd.Series(w, index=tickers).sort_values(ascending=False).head(k)
    return s.to_frame("Weight").reset_index().rename(columns={"index":"Ticker"})

col1, col2 = st.columns(2)
with col1:
    st.subheader("Global Min-Variance")
    st.metric("Return", f"{r_mv:.2%}")
    st.metric("Volatility", f"{v_mv:.2%}")
    st.dataframe(weights_table(w_mv))
with col2:
    sh = (r_ms - rf)/v_ms if v_ms>0 else 0.0
    st.subheader("Max Sharpe")
    st.metric("Return", f"{r_ms:.2%}")
    st.metric("Volatility", f"{v_ms:.2%}")
    st.metric("Sharpe", f"{sh:.3f}")
    st.dataframe(weights_table(w_ms))

fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                    specs=[[{"type":"scatter"}, {"type":"bar"}]],
                    subplot_titles=("Efficient Frontier", "Top holdings (selected)"))

fig.add_trace(go.Scatter(x=frontier_vs, y=frontier_rs, mode='lines+markers', name='Frontier'), row=1, col=1)
fig.add_trace(go.Scatter(x=[v_mv], y=[r_mv], mode='markers', name='Min-Var'), row=1, col=1)
fig.add_trace(go.Scatter(x=[v_ms], y=[r_ms], mode='markers', name='Tangency'), row=1, col=1)

sel_idx = len(frontier_rs)//2
sel_top = weights_table(frontier_ws[sel_idx])
fig.add_trace(go.Scatter(x=[frontier_vs[sel_idx]], y=[frontier_rs[sel_idx]], mode='markers', name='Selected'), row=1, col=1)
fig.add_trace(go.Bar(x=sel_top['Ticker'], y=sel_top['Weight'], name='Weights'), row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

idx = st.slider("Select frontier point", 0, len(frontier_rs)-1, sel_idx, 1)
st.dataframe(weights_table(frontier_ws[idx]))
