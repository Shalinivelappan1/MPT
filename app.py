import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

st.set_page_config(page_title="Markowitz Portfolio Lab", layout="wide")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Inputs")

uploaded = st.sidebar.file_uploader("Upload adjusted close CSV", type=["csv"])
rf = st.sidebar.number_input("Risk-free rate (annual)", 0.0, 0.5, 0.06, 0.005)
long_only = st.sidebar.checkbox("Long-only (no short selling)", True)
num_pts = st.sidebar.slider("Frontier points", 20, 150, 60)
top_hold = st.sidebar.slider("Top holdings to display", 5, 30, 10)

st.title("📊 Mean–Variance Portfolio Lab (Markowitz)")
st.caption("Designed by Prof. Shalini Velappan, IIM Trichy")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_prices(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df = df.dropna(axis=1, how="all")
    return df

if uploaded is None:
    st.warning("Upload a CSV file to begin.")
    st.stop()

prices = load_prices(uploaded)
returns = prices.pct_change().dropna()
tickers = returns.columns.tolist()
n = len(tickers)

# =====================================================
# FREQUENCY
# =====================================================
date_diffs = prices.index.to_series().diff().dropna()
avg_days = date_diffs.dt.days.mean()

periods_per_year = 12 if avg_days > 25 else 252
st.sidebar.info("Monthly data detected" if avg_days > 25 else "Daily data detected")

# =====================================================
# RETURNS
# =====================================================
mu = returns.mean() * periods_per_year
mu_vec = mu.values

clean_prices = prices[tickers].dropna()
total_years = (clean_prices.index[-1] - clean_prices.index[0]).days / 365.25
cagr = (clean_prices.iloc[-1] / clean_prices.iloc[0]) ** (1 / total_years) - 1
geo_vec = cagr.values

# =====================================================
# COVARIANCE
# =====================================================
try:
    lw = LedoitWolf().fit(returns.values)
    cov_mat = lw.covariance_ * periods_per_year
except:
    cov_mat = returns.cov().values * periods_per_year

eigmin = np.linalg.eigvalsh(cov_mat).min()
if eigmin <= 1e-10:
    cov_mat += np.eye(n) * (abs(eigmin) + 1e-8)

# =====================================================
# OPTIMIZATION
# =====================================================
bounds = None if not long_only else tuple((0,1) for _ in range(n))
cons = {'type': 'eq','fun': lambda w: np.sum(w)-1}

def port_return(w): return float(w @ mu_vec)
def port_vol(w): return float(np.sqrt(w.T @ cov_mat @ w))

def min_var():
    res = minimize(lambda w: w.T@cov_mat@w,
                   np.repeat(1/n,n), bounds=bounds,constraints=cons)
    return res.x

def max_sharpe():
    def f(w):
        v = port_vol(w)
        return -(port_return(w)-rf)/v if v>0 else 1e9
    res = minimize(f,np.repeat(1/n,n),bounds=bounds,constraints=cons)
    return res.x

def frontier_for_target(target):
    cons2 = (cons,{'type':'eq','fun': lambda w: port_return(w)-target})
    res = minimize(lambda w: w.T@cov_mat@w,
                   np.repeat(1/n,n),bounds=bounds,constraints=cons2)
    return res.x if res.success else None

# =====================================================
# BASE PORTFOLIOS
# =====================================================
w_mv = min_var()
w_ms = max_sharpe()

r_mv,v_mv = port_return(w_mv),port_vol(w_mv)
r_ms,v_ms = port_return(w_ms),port_vol(w_ms)

geo_r_mv = float(w_mv@geo_vec)
geo_r_ms = float(w_ms@geo_vec)

# =====================================================
# FRONTIER
# =====================================================
targets = np.linspace(min(mu_vec), max(mu_vec), num_pts)

frontier_rs,frontier_vs,frontier_ws = [],[],[]
for t in targets:
    w = frontier_for_target(t)
    if w is not None:
        frontier_ws.append(w)
        frontier_rs.append(port_return(w))
        frontier_vs.append(port_vol(w))

# =====================================================
# INVESTOR TYPE
# =====================================================
st.sidebar.subheader("Select Portfolio on Frontier")

investor = st.sidebar.radio(
    "Investor Style",
    ["Manual slider","🛡 Conservative","⚖ Balanced","🚀 Aggressive","⭐ Max Sharpe","🌍 Minimum Variance"]
)

vol_array = np.array(frontier_vs)
sorted_idx = np.argsort(vol_array)

if investor=="Manual slider":
    sel_idx = st.sidebar.slider("Move along Efficient Frontier",0,len(frontier_ws)-1,len(frontier_ws)//2)
elif investor=="🛡 Conservative":
    sel_idx = sorted_idx[int(len(sorted_idx)*0.2)]
elif investor=="⚖ Balanced":
    sel_idx = sorted_idx[int(len(sorted_idx)*0.5)]
elif investor=="🚀 Aggressive":
    sel_idx = sorted_idx[int(len(sorted_idx)*0.8)]
elif investor=="⭐ Max Sharpe":
    sharpe_list=[(r-rf)/v if v>0 else 0 for r,v in zip(frontier_rs,frontier_vs)]
    sel_idx=int(np.argmax(sharpe_list))
elif investor=="🌍 Minimum Variance":
    sel_idx=int(np.argmin(frontier_vs))

w_selected = frontier_ws[sel_idx]
r_selected = frontier_rs[sel_idx]
v_selected = frontier_vs[sel_idx]

# =====================================================
# UTILITY
# =====================================================
st.sidebar.divider()
st.sidebar.subheader("Risk Preference")

A = st.sidebar.slider("Risk Aversion (A)",0.0,15.0,4.0,0.5)

utilities=[r-0.5*A*(v**2) for r,v in zip(frontier_rs,frontier_vs)]
opt_idx=int(np.argmax(utilities))

w_util = frontier_ws[opt_idx]
r_util = frontier_rs[opt_idx]
v_util = frontier_vs[opt_idx]

U_star = r_util - 0.5*A*(v_util**2)
sigma_curve=np.linspace(0,max(frontier_vs)*1.1,200)
indiff_curve=U_star + 0.5*A*(sigma_curve**2)

# =====================================================
# DISPLAY
# =====================================================
col1,col2=st.columns(2)

with col1:
    st.subheader("🌍 Global Minimum Variance")
    st.metric("Return",f"{r_mv:.2%}")
    st.metric("Volatility",f"{v_mv:.2%}")

with col2:
    sharpe=(r_ms-rf)/v_ms
    st.subheader("⭐ Tangency")
    st.metric("Return",f"{r_ms:.2%}")
    st.metric("Volatility",f"{v_ms:.2%}")
    st.metric("Sharpe",f"{sharpe:.2f}")

st.divider()
st.subheader("🎯 Selected Portfolio")

c1,c2,c3=st.columns(3)
c1.metric("Return",f"{r_selected:.2%}")
c2.metric("Volatility",f"{v_selected:.2%}")
c3.metric("Sharpe",f"{(r_selected-rf)/v_selected:.2f}")

st.subheader("⭐ Utility Optimal Portfolio")

u1,u2,u3=st.columns(3)
u1.metric("Return",f"{r_util:.2%}")
u2.metric("Volatility",f"{v_util:.2%}")
u3.metric("Sharpe",f"{(r_util-rf)/v_util:.2f}")

# =====================================================
# =====================================================
# PLOT (Enhanced Visual Version)
# =====================================================
fig = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.65, 0.35],
    specs=[[{"type": "scatter"}, {"type": "bar"}]]
)

# Efficient Frontier
fig.add_trace(
    go.Scatter(
        x=frontier_vs,
        y=frontier_rs,
        mode='lines',
        line=dict(width=3),
        name='Efficient Frontier'
    ),
    row=1, col=1
)

# Min Variance
fig.add_trace(
    go.Scatter(
        x=[v_mv],
        y=[r_mv],
        mode='markers',
        marker=dict(size=10),
        name='Minimum Variance'
    ),
    row=1, col=1
)

# Tangency
fig.add_trace(
    go.Scatter(
        x=[v_ms],
        y=[r_ms],
        mode='markers',
        marker=dict(size=10),
        name='Tangency Portfolio'
    ),
    row=1, col=1
)

# Selected Portfolio
fig.add_trace(
    go.Scatter(
        x=[v_selected],
        y=[r_selected],
        mode='markers',
        marker=dict(size=12),
        name='Selected Portfolio'
    ),
    row=1, col=1
)

# Utility Optimal (Highlighted Star)
fig.add_trace(
    go.Scatter(
        x=[v_util],
        y=[r_util],
        mode='markers',
        marker=dict(size=18, symbol="star"),
        name='Utility Optimal'
    ),
    row=1, col=1
)

# Indifference Curve
fig.add_trace(
    go.Scatter(
        x=sigma_curve,
        y=indiff_curve,
        mode='lines',
        line=dict(dash='dot', width=2),
        name='Indifference Curve'
    ),
    row=1, col=1
)

# Capital Market Line
sharpe = (r_ms - rf) / v_ms
cml_x = np.linspace(0, max(frontier_vs)*1.1, 100)
cml_y = rf + sharpe * cml_x

fig.add_trace(
    go.Scatter(
        x=cml_x,
        y=cml_y,
        line=dict(dash='dash'),
        name='Capital Market Line'
    ),
    row=1, col=1
)

# Bar Chart
weights = pd.Series(w_selected, index=tickers)\
    .sort_values(ascending=False)\
    .head(top_hold)

fig.add_trace(
    go.Bar(
        x=weights.index,
        y=weights.values
    ),
    row=1, col=2
)

# Axis labels
fig.update_xaxes(title_text="Volatility (σ)", row=1, col=1)
fig.update_yaxes(title_text="Expected Return (μ)", row=1, col=1)

# Smooth animation effect
fig.update_layout(
    transition_duration=500,
    hovermode="closest"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# LEARNING PANEL (FULL PRESERVED + ADDED)
# =====================================================
# =====================================================
# 📚 LEARNING PANEL — FULL THEORY + EXPLANATION
# =====================================================
st.divider()
st.header("📚 Understanding the Mathematics Behind the Portfolio")

st.write("""
This section explains how returns, risk, variance, covariance,
correlation, portfolio diversification, and investor preferences are calculated.
""")

# =====================================================
# RETURNS
# =====================================================
with st.expander("1️⃣ Returns — What are we measuring?"):

    st.write("""
A return measures how much an asset’s price changes from one period to the next.
It tells us how much we gained or lost relative to the previous price.
""")

    st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}}")

    st.write("""
Arithmetic return is the simple average return.
It is used in Markowitz optimisation.
""")

    st.latex(r"E[R] = \bar{R} \times \text{periods per year}")

    st.write("""
Geometric return (CAGR) measures compounded growth.
It reflects what investors actually earn over time.
""")

    st.latex(r"CAGR = \left(\frac{P_T}{P_0}\right)^{\frac{1}{T}} - 1")


# =====================================================
# VOLATILITY
# =====================================================
with st.expander("2️⃣ Volatility — How do we measure risk?"):

    st.write("""
Volatility measures how much returns fluctuate around their average.
Higher volatility means higher uncertainty and therefore higher risk.
""")

    st.latex(r"\sigma = \sqrt{Var(R)}")

    st.write("Annualised volatility scales risk to yearly terms:")

    st.latex(r"\sigma_{annual} = \sigma_{period} \times \sqrt{\text{periods per year}}")

    vol = returns.std() * np.sqrt(periods_per_year)
    st.dataframe(vol.to_frame("Annual Volatility"))


# =====================================================
# VARIANCE & COVARIANCE
# =====================================================
with st.expander("3️⃣ Variance & Covariance — How assets move together"):

    st.write("""
Variance measures the risk of a single asset.

Covariance measures how two assets move together.
It is crucial for diversification.
""")

    st.latex(r"Cov(i,j) = E[(R_i - \mu_i)(R_j - \mu_j)]")

    st.latex(r"\Sigma = Cov(R_i, R_j)")

    cov_df = pd.DataFrame(cov_mat, index=tickers, columns=tickers)
    st.dataframe(cov_df.style.format("{:.4f}"))


# =====================================================
# CORRELATION
# =====================================================
with st.expander("4️⃣ Correlation — Standardised relationship"):

    st.latex(r"\rho_{ij} = \frac{Cov(i,j)}{\sigma_i \sigma_j}")

    corr = returns.corr()
    st.dataframe(corr.style.format("{:.2f}"))


# =====================================================
# PORTFOLIO RETURN
# =====================================================
with st.expander("5️⃣ Portfolio Return — Weighted average"):

    st.latex(r"R_p = \sum_{i=1}^{n} w_i R_i")

    st.write("Min-variance portfolio weights:")
    st.dataframe(pd.Series(w_mv, index=tickers).to_frame("Weight"))

    st.write("Tangency portfolio weights:")
    st.dataframe(pd.Series(w_ms, index=tickers).to_frame("Weight"))


# =====================================================
# PORTFOLIO RISK
# =====================================================
with st.expander("6️⃣ Portfolio Risk — Why diversification works"):

    st.latex(r"\sigma_p^2 = w^T \Sigma w")

    st.latex(r"\sigma_p = \sqrt{w^T \Sigma w}")

    st.write("""
Portfolio risk depends not only on individual volatility,
but also on how assets move together.
""")


# =====================================================
# UTILITY & RISK AVERSION
# =====================================================
with st.expander("7️⃣ Investor Utility & Risk Aversion — Choosing the Optimal Portfolio"):

    st.latex(r"U = E(R_p) - \frac{1}{2}A\sigma_p^2")

    st.write("""
Risk aversion determines where an investor chooses on the efficient frontier.

Higher A → more conservative  
Lower A → more aggressive  

The optimal portfolio is the one that maximises utility.
This explains why different investors choose different portfolios.
""")
