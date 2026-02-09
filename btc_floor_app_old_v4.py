"""
================================================================================
BITCOIN FLOOR ANALYSIS - STREAMLIT APP
================================================================================
Power Law & Quantile Regression Floor Models
Data source: CryptoCompare API
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="BTC Floor Analysis",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #F7931A;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #F7931A;
    }
    .signal-buy {
        background: linear-gradient(135deg, #0f5132 0%, #198754 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .signal-watch {
        background: linear-gradient(135deg, #664d03 0%, #ffc107 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .signal-hold {
        background: linear-gradient(135deg, #495057 0%, #6c757d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .floor-table {
        font-size: 0.9rem;
    }
    .info-box {
        background-color: #262730;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
GENESIS_DATE = pd.Timestamp('2009-01-03')
START_DATE = '2013-01-01'

# ============================================================================
# DATA FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_btc_cryptocompare(start_date='2010-07-18'):
    """Download BTC data from CryptoCompare API"""
    all_data = []
    current_ts = int(datetime.now().timestamp())
    start_ts = int(pd.Timestamp(start_date).timestamp())
    
    while current_ts > start_ts:
        url = 'https://min-api.cryptocompare.com/data/v2/histoday'
        params = {'fsym': 'BTC', 'tsym': 'USD', 'limit': 2000, 'toTs': current_ts}
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data['Response'] != 'Success':
            break
        
        df_chunk = pd.DataFrame(data['Data']['Data'])
        all_data.append(df_chunk)
        oldest_ts = df_chunk['time'].min()
        current_ts = oldest_ts - 86400
        
        if oldest_ts <= start_ts:
            break
        time.sleep(0.2)
    
    df = pd.concat(all_data, ignore_index=True)
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df[['Date', 'close', 'high', 'low', 'open']]
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open']
    df = df[~df['Date'].duplicated(keep='first')]
    df = df[df['Date'] >= start_date]
    df = df[df['Close'] > 0].reset_index(drop=True)
    
    return df

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================
def fit_power_law_ols(log_days, log_prices):
    """Fit Power Law: P = a * D^b"""
    X = sm.add_constant(log_days)
    model = sm.OLS(log_prices, X).fit()
    log_a, b = model.params[0], model.params[1]
    a = 10 ** log_a
    residual_std = np.std(log_prices - model.predict(X))
    return {'a': a, 'b': b, 'log_a': log_a, 'r2': model.rsquared, 'residual_std': residual_std}

def fit_quantile_regression(df, quantile, price_col='LogLow'):
    """Quantile regression for floor estimation"""
    data = pd.DataFrame({'x': df['LogDays'], 'y': df[price_col]})
    model = smf.quantreg('y ~ x', data).fit(q=quantile)
    log_a, b = model.params['Intercept'], model.params['x']
    return {'quantile': quantile, 'a': 10**log_a, 'b': b, 'log_a': log_a}

def predict_pl(days, a, b):
    """Predict price from Power Law"""
    return a * (days ** b)

def predict_pl_band(days, log_a, b, residual_std, sigma=2):
    """Predict Power Law band"""
    log_days = np.log10(days)
    log_price = log_a + b * log_days
    return 10 ** (log_price - sigma * residual_std), 10 ** log_price, 10 ** (log_price + sigma * residual_std)

def compute_nlb(df):
    """Compute Never Look Back floor"""
    df = df.copy().sort_values('Date').reset_index(drop=True)
    n = len(df)
    future_min = np.zeros(n)
    future_min[-1] = df['Low'].iloc[-1]
    for i in range(n-2, -1, -1):
        future_min[i] = min(df['Low'].iloc[i], future_min[i+1])
    df['Future_Min'] = future_min
    df['NLB'] = df['Future_Min'].cummax()
    return df

def monte_carlo_min(df, floor_model, n_sims=5000, horizon=730):
    """Monte Carlo simulation for minimum price"""
    np.random.seed(42)
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()
    sigma_n = log_ret[log_ret.abs() < 1.5*sigma].std()
    sigma_s = log_ret[log_ret.abs() >= 1.5*sigma].std()
    p0, d0 = df['Close'].iloc[-1], df['Days'].iloc[-1]
    
    results = []
    for _ in range(n_sims):
        price, min_p, min_d = p0, p0, 0
        for d in range(1, horizon+1):
            vol = sigma_s if np.random.random() < 0.15 else sigma_n
            floor = predict_pl(d0+d, floor_model['a'], floor_model['b'])
            price = price * np.exp(np.random.normal(mu, vol))
            if price < floor * 0.90:
                price = floor * (0.90 + np.random.uniform(0, 0.15))
            if price < min_p:
                min_p, min_d = price, d
        results.append({'min_price': min_p, 'min_day': min_d})
    return pd.DataFrame(results)

def analyze_historical_crashes(df):
    """Analyze historical drawdowns and crashes"""
    df = df.copy()
    df['ATH'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] / df['ATH'] - 1) * 100
    
    # Find major crashes (local minima of drawdown)
    crashes = []
    in_crash = False
    crash_start = None
    peak_price = None
    
    for i in range(1, len(df)):
        if df['Drawdown'].iloc[i] < -20 and not in_crash:
            in_crash = True
            crash_start = df['Date'].iloc[i]
            peak_idx = df['Close'].iloc[:i].idxmax()
            peak_price = df['Close'].iloc[peak_idx]
            peak_date = df['Date'].iloc[peak_idx]
        
        if in_crash and (df['Drawdown'].iloc[i] > -10 or i == len(df)-1):
            crash_end = df['Date'].iloc[i]
            bottom_idx = df.loc[df['Date'] >= crash_start].loc[df['Date'] <= crash_end, 'Close'].idxmin()
            bottom_price = df['Close'].iloc[bottom_idx]
            bottom_date = df['Date'].iloc[bottom_idx]
            drawdown = (bottom_price / peak_price - 1) * 100
            
            # Recovery time
            recovery_df = df[df['Date'] > bottom_date]
            recovery_idx = recovery_df[recovery_df['Close'] >= peak_price].index
            if len(recovery_idx) > 0:
                recovery_date = df['Date'].iloc[recovery_idx[0]]
                recovery_days = (recovery_date - bottom_date).days
            else:
                recovery_date = None
                recovery_days = None
            
            if drawdown < -30:  # Only significant crashes
                crashes.append({
                    'peak_date': peak_date,
                    'peak_price': peak_price,
                    'bottom_date': bottom_date,
                    'bottom_price': bottom_price,
                    'drawdown': drawdown,
                    'crash_days': (bottom_date - peak_date).days,
                    'recovery_days': recovery_days
                })
            
            in_crash = False
    
    return pd.DataFrame(crashes)

def monte_carlo_black_swan(df, floor_model, n_sims=5000, horizon=730, 
                            black_swan_prob=0.05, black_swan_impact=-0.50):
    """
    Monte Carlo with Black Swan events.
    Black swan = sudden extreme drop that can violate floors.
    """
    np.random.seed(42)
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()
    sigma_n = log_ret[log_ret.abs() < 1.5*sigma].std()
    sigma_s = log_ret[log_ret.abs() >= 1.5*sigma].std()
    p0, d0 = df['Close'].iloc[-1], df['Days'].iloc[-1]
    
    results = []
    for sim in range(n_sims):
        price, min_p, min_d = p0, p0, 0
        black_swan_occurred = False
        black_swan_day = 0
        
        # Decide if black swan happens in this simulation
        has_black_swan = np.random.random() < black_swan_prob
        if has_black_swan:
            # Black swan occurs at random day in first half of horizon
            black_swan_day = np.random.randint(1, horizon // 2)
        
        for d in range(1, horizon+1):
            # Check for black swan event
            if has_black_swan and d == black_swan_day and not black_swan_occurred:
                # Black swan: sudden crash
                shock = black_swan_impact + np.random.uniform(-0.10, 0.05)
                price = price * (1 + shock)
                black_swan_occurred = True
            else:
                # Normal dynamics
                vol = sigma_s if np.random.random() < 0.15 else sigma_n
                price = price * np.exp(np.random.normal(mu, vol))
            
            # NO floor bounce during black swan recovery (first 30 days after)
            if not (black_swan_occurred and d < black_swan_day + 30):
                floor = predict_pl(d0+d, floor_model['a'], floor_model['b'])
                if price < floor * 0.90:
                    price = floor * (0.90 + np.random.uniform(0, 0.15))
            
            if price < min_p:
                min_p, min_d = price, d
        
        results.append({
            'min_price': min_p, 
            'min_day': min_d,
            'black_swan': black_swan_occurred
        })
    
    return pd.DataFrame(results)

def calculate_evt_floor(df, confidence=0.99):
    """
    Extreme Value Theory (EVT) floor estimation.
    Uses Generalized Pareto Distribution on extreme losses.
    """
    from scipy.stats import genpareto
    
    # Calculate daily returns
    returns = df['Close'].pct_change().dropna()
    
    # Focus on losses (negative returns)
    losses = -returns[returns < 0]
    
    # Threshold: 90th percentile of losses
    threshold = losses.quantile(0.90)
    exceedances = losses[losses > threshold] - threshold
    
    if len(exceedances) < 20:
        return None, None
    
    # Fit Generalized Pareto Distribution
    try:
        params = genpareto.fit(exceedances)
        c, loc, scale = params
        
        # VaR at confidence level
        n = len(returns)
        n_exceed = len(exceedances)
        
        # Expected maximum loss
        var_evt = threshold + (scale / c) * (((n / n_exceed) * (1 - confidence)) ** (-c) - 1)
        
        # Translate to price floor
        current_price = df['Close'].iloc[-1]
        evt_floor = current_price * (1 - var_evt)
        
        return evt_floor, var_evt
    except:
        return None, None

def get_signal(dist_q01, dist_q02, dist_q05, dist_q10):
    """Determine buy signal"""
    if dist_q01 <= 0:
        return 'EXTREME BUY', '🟢🟢🟢', 'SOTTO Q01 (1%)! Evento rarissimo!', 'buy'
    elif dist_q02 <= 0:
        return 'STRONG BUY', '🟢🟢', 'SOTTO Q02 (2%)! Opportunità rara!', 'buy'
    elif dist_q05 <= 0:
        return 'STRONG BUY', '🟢🟢', 'SOTTO Q05 (5%)', 'buy'
    elif dist_q05 <= 10:
        return 'BUY', '🟢', 'Vicino al floor Q05', 'buy'
    elif dist_q10 <= 0:
        return 'BUY', '🟢', 'SOTTO Q10 (10%)', 'buy'
    elif dist_q10 <= 15:
        return 'ACCUMULATE', '🟡', 'Vicino al floor Q10', 'watch'
    elif dist_q10 <= 30:
        return 'WATCH', '🟡', 'Zona di accumulo', 'watch'
    else:
        return 'HOLD', '⚪', 'Sopra i floor', 'hold'

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<p class="main-header">₿ Bitcoin Floor Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Power Law & Quantile Regression Models</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=80)
        st.title("⚙️ Settings")
        
        start_year = st.selectbox("Start Year", [2010, 2011, 2012, 2013, 2014, 2015], index=3)
        mc_sims = st.slider("Monte Carlo Simulations", 1000, 10000, 5000, 1000)
        projection_years = st.slider("Projection (years)", 1, 5, 2)
        
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        **Floor Models:**
        - **Q01 (1%)**: Extreme floor
        - **Q05 (5%)**: PlanC floor  
        - **Q10 (10%)**: Realistic floor
        - **NLB**: Never Look Back
        
        **Data Source:** CryptoCompare
        """)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("📥 Loading Bitcoin data..."):
        try:
            btc_raw = get_btc_cryptocompare(f'{start_year}-01-01')
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Prepare data
    df = btc_raw.copy()
    df['Days'] = (df['Date'] - GENESIS_DATE).dt.days
    df['LogDays'] = np.log10(df['Days'])
    df['LogPrice'] = np.log10(df['Close'])
    df['LogLow'] = np.log10(df['Low'])
    
    TODAY = df['Date'].iloc[-1]
    TODAY_DAYS = df['Days'].iloc[-1]
    CURRENT_PRICE = df['Close'].iloc[-1]
    
    # Fit models
    pl_standard = fit_power_law_ols(df['LogDays'].values, df['LogPrice'].values)
    
    quantiles = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    qr_models = {}
    for q in quantiles:
        qr_models[q] = fit_quantile_regression(df, q, 'LogLow')
    
    df = compute_nlb(df)
    current_nlb = df['NLB'].iloc[-1]
    
    # Calculate floors
    q01 = predict_pl(TODAY_DAYS, qr_models[0.01]['a'], qr_models[0.01]['b'])
    q02 = predict_pl(TODAY_DAYS, qr_models[0.02]['a'], qr_models[0.02]['b'])
    q05 = predict_pl(TODAY_DAYS, qr_models[0.05]['a'], qr_models[0.05]['b'])
    q10 = predict_pl(TODAY_DAYS, qr_models[0.10]['a'], qr_models[0.10]['b'])
    q15 = predict_pl(TODAY_DAYS, qr_models[0.15]['a'], qr_models[0.15]['b'])
    q20 = predict_pl(TODAY_DAYS, qr_models[0.20]['a'], qr_models[0.20]['b'])
    
    pl_lower, pl_fair, pl_upper = predict_pl_band(TODAY_DAYS, pl_standard['log_a'], pl_standard['b'], pl_standard['residual_std'])
    
    dist_q01 = (CURRENT_PRICE / q01 - 1) * 100
    dist_q02 = (CURRENT_PRICE / q02 - 1) * 100
    dist_q05 = (CURRENT_PRICE / q05 - 1) * 100
    dist_q10 = (CURRENT_PRICE / q10 - 1) * 100
    
    signal_name, signal_emoji, signal_desc, signal_type = get_signal(dist_q01, dist_q02, dist_q05, dist_q10)
    
    # ============================================================================
    # MAIN CONTENT
    # ============================================================================
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 BTC Price",
            f"${CURRENT_PRICE:,.0f}",
            f"{(CURRENT_PRICE/df['Close'].iloc[-2]-1)*100:+.2f}%"
        )
    
    with col2:
        st.metric(
            "📅 Data",
            TODAY.strftime('%Y-%m-%d'),
            f"{len(df):,} days"
        )
    
    with col3:
        st.metric(
            "📊 vs Q05 Floor",
            f"${q05:,.0f}",
            f"{dist_q05:+.1f}%"
        )
    
    with col4:
        st.metric(
            "📈 vs Fair Value",
            f"${pl_fair:,.0f}",
            f"{(CURRENT_PRICE/pl_fair-1)*100:+.1f}%"
        )
    
    st.markdown("---")
    
    # Signal Box
    signal_class = f"signal-{signal_type}"
    st.markdown(f"""
    <div class="{signal_class}">
        {signal_emoji} SIGNAL: {signal_name}<br>
        <span style="font-size: 1rem; font-weight: normal;">{signal_desc}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two columns: Chart and Tables
    chart_col, table_col = st.columns([2, 1])
    
    with chart_col:
        st.subheader("📈 Price & Floor Chart")
        
        # Calculate floors for entire dataframe
        df['Q01_Floor'] = predict_pl(df['Days'], qr_models[0.01]['a'], qr_models[0.01]['b'])
        df['Q05_Floor'] = predict_pl(df['Days'], qr_models[0.05]['a'], qr_models[0.05]['b'])
        df['Q10_Floor'] = predict_pl(df['Days'], qr_models[0.10]['a'], qr_models[0.10]['b'])
        df['Q20_Floor'] = predict_pl(df['Days'], qr_models[0.20]['a'], qr_models[0.20]['b'])
        df['PL_Fair'] = predict_pl(df['Days'], pl_standard['a'], pl_standard['b'])
        
        # Projection
        proj_days_count = projection_years * 365
        proj_dates = pd.date_range(start=TODAY + pd.Timedelta(days=1), periods=proj_days_count, freq='D')
        proj_days = [(d - GENESIS_DATE).days for d in proj_dates]
        proj = pd.DataFrame({'Date': proj_dates, 'Days': proj_days})
        proj['Q01'] = predict_pl(proj['Days'], qr_models[0.01]['a'], qr_models[0.01]['b'])
        proj['Q05'] = predict_pl(proj['Days'], qr_models[0.05]['a'], qr_models[0.05]['b'])
        proj['Q10'] = predict_pl(proj['Days'], qr_models[0.10]['a'], qr_models[0.10]['b'])
        
        # Create chart
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'],
            name='BTC Price', line=dict(color='#F7931A', width=2)
        ))
        
        # NLB
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['NLB'],
            name='NLB', line=dict(color='#dc3545', width=1.5, dash='dot'),
            fill='tozeroy', fillcolor='rgba(220,53,69,0.05)'
        ))
        
        # Floors
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Q01_Floor'],
            name='Q01 (1%)', line=dict(color='#8B0000', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Q05_Floor'],
            name='Q05 (5%)', line=dict(color='#dc3545', width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Q10_Floor'],
            name='Q10 (10%)', line=dict(color='#28a745', width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Q20_Floor'],
            name='Q20 (20%)', line=dict(color='#90EE90', width=1)
        ))
        
        # Fair value
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['PL_Fair'],
            name='PL Fair', line=dict(color='#6c757d', width=1, dash='dash')
        ))
        
        # Projections
        fig.add_trace(go.Scatter(
            x=proj['Date'], y=proj['Q01'],
            name='Q01 proj', line=dict(color='#8B0000', width=1, dash='dot'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=proj['Date'], y=proj['Q05'],
            name='Q05 proj', line=dict(color='#dc3545', width=1, dash='dot'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=proj['Date'], y=proj['Q10'],
            name='Q10 proj', line=dict(color='#28a745', width=1, dash='dot'),
            showlegend=False
        ))
        
        fig.update_layout(
            yaxis_type='log',
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            height=500,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template='plotly_dark',
            margin=dict(l=0, r=0, t=30, b=0)
        )
        fig.update_yaxes(range=[np.log10(100), np.log10(500000)])
        
        st.plotly_chart(fig, use_container_width=True)
    
    with table_col:
        st.subheader("📊 Floor Levels")
        
        # Floor comparison table
        floor_data = {
            'Floor': ['Q01 (1%)', 'Q02 (2%)', 'Q05 (5%)', 'Q10 (10%)', 'Q15 (15%)', 'Q20 (20%)', 'NLB', 'PL Fair'],
            'Value': [q01, q02, q05, q10, q15, q20, current_nlb, pl_fair],
            'vs Price': [dist_q01, dist_q02, dist_q05, dist_q10,
                        (CURRENT_PRICE/q15-1)*100, (CURRENT_PRICE/q20-1)*100,
                        (CURRENT_PRICE/current_nlb-1)*100, (CURRENT_PRICE/pl_fair-1)*100]
        }
        
        floor_df = pd.DataFrame(floor_data)
        floor_df['Value'] = floor_df['Value'].apply(lambda x: f"${x:,.0f}")
        floor_df['vs Price'] = floor_df['vs Price'].apply(lambda x: f"{x:+.1f}%")
        floor_df['Status'] = floor_data['vs Price']
        floor_df['Status'] = floor_df['Status'].apply(
            lambda x: '🔴 BELOW' if x < 0 else ('🟡 NEAR' if x < 15 else '🟢 ABOVE')
        )
        
        st.dataframe(floor_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🎯 Buy Levels")
        buy_levels = {
            'Signal': ['🟢 STRONG BUY', '🟢 BUY', '🟡 ACCUMULATE', '🟠 WATCH'],
            'Price': [f"< ${q05*1.00:,.0f}", f"< ${q05*1.10:,.0f}", f"< ${q05*1.20:,.0f}", f"< ${q05*1.35:,.0f}"],
            'Description': ['At Q05 floor', '+10% from Q05', '+20% from Q05', '+35% from Q05']
        }
        st.dataframe(pd.DataFrame(buy_levels), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Projection and Monte Carlo
    proj_col, mc_col = st.columns(2)
    
    with proj_col:
        st.subheader(f"🔮 {projection_years}Y Floor Projection")
        
        periods = {'Today': 0, '3M': 90, '6M': 180, '1Y': 365}
        if projection_years >= 2:
            periods['2Y'] = 730
        if projection_years >= 3:
            periods['3Y'] = 1095
        
        proj_data = {'Floor': []}
        for p in periods.keys():
            proj_data[p] = []
        
        for fname, q in [('Q01', 0.01), ('Q02', 0.02), ('Q05', 0.05), ('Q10', 0.10)]:
            proj_data['Floor'].append(fname)
            qm = qr_models[q]
            for pname, days in periods.items():
                val = predict_pl(TODAY_DAYS + days, qm['a'], qm['b'])
                proj_data[pname].append(f"${val:,.0f}")
        
        st.dataframe(pd.DataFrame(proj_data), use_container_width=True, hide_index=True)
    
    with mc_col:
        st.subheader("🎲 Monte Carlo Simulation")
        
        with st.spinner(f"Running {mc_sims:,} simulations..."):
            mc = monte_carlo_min(df, qr_models[0.05], n_sims=mc_sims, horizon=projection_years*365)
        
        mc_data = {
            'Percentile': ['1%', '5%', '10%', '25%', '50%', '75%'],
            'Min Price': [
                f"${mc['min_price'].quantile(0.01):,.0f}",
                f"${mc['min_price'].quantile(0.05):,.0f}",
                f"${mc['min_price'].quantile(0.10):,.0f}",
                f"${mc['min_price'].quantile(0.25):,.0f}",
                f"${mc['min_price'].quantile(0.50):,.0f}",
                f"${mc['min_price'].quantile(0.75):,.0f}",
            ],
            'Timing': [
                f"~{mc[mc['min_price'] <= mc['min_price'].quantile(0.05)]['min_day'].median()/30:.0f}m",
                f"~{mc[mc['min_price'] <= mc['min_price'].quantile(0.10)]['min_day'].median()/30:.0f}m",
                f"~{mc[mc['min_price'] <= mc['min_price'].quantile(0.15)]['min_day'].median()/30:.0f}m",
                f"~{mc[mc['min_price'] <= mc['min_price'].quantile(0.30)]['min_day'].median()/30:.0f}m",
                f"~{mc[mc['min_price'] <= mc['min_price'].quantile(0.55)]['min_day'].median()/30:.0f}m",
                f"~{mc[mc['min_price'] <= mc['min_price'].quantile(0.80)]['min_day'].median()/30:.0f}m",
            ]
        }
        st.dataframe(pd.DataFrame(mc_data), use_container_width=True, hide_index=True)
        
        st.info(f"""
        **Interpretation:**
        - 5% probability min < **${mc['min_price'].quantile(0.05):,.0f}**
        - 50% probability min < **${mc['min_price'].quantile(0.50):,.0f}**
        - Current price: **${CURRENT_PRICE:,.0f}**
        """)
    
    st.markdown("---")
    
    # Monte Carlo histogram
    st.subheader("📊 Monte Carlo Distribution")
    
    fig_mc = make_subplots(rows=1, cols=2, subplot_titles=('Minimum Price Distribution', 'Timing of Minimum'))
    
    fig_mc.add_trace(
        go.Histogram(x=mc['min_price'], nbinsx=50, marker_color='#dc3545', opacity=0.7),
        row=1, col=1
    )
    fig_mc.add_vline(x=mc['min_price'].quantile(0.05), line_dash='dash', line_color='red', row=1, col=1)
    fig_mc.add_vline(x=mc['min_price'].quantile(0.50), line_dash='dash', line_color='blue', row=1, col=1)
    fig_mc.add_vline(x=CURRENT_PRICE, line_dash='solid', line_color='#F7931A', row=1, col=1)
    
    fig_mc.add_trace(
        go.Histogram(x=mc['min_day']/30, nbinsx=50, marker_color='#17a2b8', opacity=0.7),
        row=1, col=2
    )
    
    fig_mc.update_layout(
        height=350,
        showlegend=False,
        template='plotly_dark',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig_mc.update_xaxes(title_text='Price ($)', row=1, col=1)
    fig_mc.update_xaxes(title_text='Months from now', row=1, col=2)
    
    st.plotly_chart(fig_mc, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================================================
    # BLACK SWAN ANALYSIS
    # ============================================================================
    st.header("🦢 Black Swan Analysis")
    st.markdown("""
    Un **Cigno Nero** è un evento raro, imprevedibile e ad alto impatto che può violare anche i floor più conservativi.
    Questa sezione analizza scenari estremi per stress-testing del portafoglio.
    """)
    
    # Black Swan Settings
    bs_col1, bs_col2, bs_col3 = st.columns(3)
    with bs_col1:
        bs_probability = st.slider("🎲 Probabilità Black Swan (%)", 1, 20, 5) / 100
    with bs_col2:
        bs_impact = st.slider("💥 Impatto Black Swan (%)", -80, -30, -50)
    with bs_col3:
        bs_sims = st.slider("🔄 Simulazioni", 1000, 10000, 5000, 1000)
    
    # Tabs for different analyses
    bs_tab1, bs_tab2, bs_tab3, bs_tab4 = st.tabs([
        "📉 Crash Storici", 
        "🎲 Monte Carlo + Black Swan",
        "📊 Stress Scenarios",
        "🔬 Extreme Value Theory"
    ])
    
    with bs_tab1:
        st.subheader("📉 Peggiori Crash Storici di Bitcoin")
        
        crashes_df = analyze_historical_crashes(df)
        
        if len(crashes_df) > 0:
            # Format the dataframe
            display_crashes = crashes_df.copy()
            display_crashes['peak_date'] = display_crashes['peak_date'].dt.strftime('%Y-%m-%d')
            display_crashes['bottom_date'] = display_crashes['bottom_date'].dt.strftime('%Y-%m-%d')
            display_crashes['peak_price'] = display_crashes['peak_price'].apply(lambda x: f"${x:,.0f}")
            display_crashes['bottom_price'] = display_crashes['bottom_price'].apply(lambda x: f"${x:,.0f}")
            display_crashes['drawdown'] = display_crashes['drawdown'].apply(lambda x: f"{x:.1f}%")
            display_crashes['crash_days'] = display_crashes['crash_days'].apply(lambda x: f"{x} giorni")
            display_crashes['recovery_days'] = display_crashes['recovery_days'].apply(
                lambda x: f"{x} giorni" if pd.notna(x) else "Non recuperato"
            )
            
            display_crashes.columns = ['Peak Date', 'Peak Price', 'Bottom Date', 'Bottom Price', 
                                       'Drawdown', 'Crash Duration', 'Recovery Time']
            
            st.dataframe(display_crashes.sort_values('Drawdown'), use_container_width=True, hide_index=True)
            
            # Stats
            st.markdown("#### 📊 Statistiche Crash")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                worst_dd = crashes_df['drawdown'].min()
                st.metric("Peggior Drawdown", f"{worst_dd:.1f}%")
            
            with stat_col2:
                avg_dd = crashes_df['drawdown'].mean()
                st.metric("Drawdown Medio", f"{avg_dd:.1f}%")
            
            with stat_col3:
                avg_crash_days = crashes_df['crash_days'].mean()
                st.metric("Durata Media Crash", f"{avg_crash_days:.0f} giorni")
            
            with stat_col4:
                recovered = crashes_df[crashes_df['recovery_days'].notna()]
                if len(recovered) > 0:
                    avg_recovery = recovered['recovery_days'].mean()
                    st.metric("Recovery Medio", f"{avg_recovery:.0f} giorni")
                else:
                    st.metric("Recovery Medio", "N/A")
            
            # Worst case scenario based on history
            st.markdown("#### 🎯 Scenario Worst Case (basato su storia)")
            worst_historical = crashes_df['drawdown'].min() / 100
            ath = df['Close'].max()
            worst_case_price = CURRENT_PRICE * (1 + worst_historical)
            
            st.error(f"""
            **Se si ripetesse il peggior crash storico ({worst_dd:.1f}%):**
            - Dal prezzo attuale (${CURRENT_PRICE:,.0f}): **${worst_case_price:,.0f}**
            - Dall'ATH (${ath:,.0f}): **${ath * (1 + worst_historical):,.0f}**
            """)
        else:
            st.info("Non abbastanza dati per analizzare i crash storici.")
    
    with bs_tab2:
        st.subheader("🎲 Monte Carlo con Eventi Black Swan")
        
        st.markdown(f"""
        Simulazione che include possibilità di **eventi catastrofici**:
        - **Probabilità Black Swan**: {bs_probability*100:.0f}% delle simulazioni
        - **Impatto**: {bs_impact}% drop improvviso
        - Durante un black swan, i floor Power Law **possono essere violati**
        """)
        
        with st.spinner(f"Running {bs_sims:,} simulazioni con Black Swan..."):
            mc_bs = monte_carlo_black_swan(
                df, qr_models[0.05], 
                n_sims=bs_sims, 
                horizon=projection_years*365,
                black_swan_prob=bs_probability,
                black_swan_impact=bs_impact/100
            )
        
        # Split results
        mc_normal = mc_bs[mc_bs['black_swan'] == False]
        mc_swan = mc_bs[mc_bs['black_swan'] == True]
        
        # Display results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("##### 📊 Scenario NORMALE (no black swan)")
            if len(mc_normal) > 0:
                normal_data = {
                    'Percentile': ['5%', '10%', '25%', '50%'],
                    'Min Price': [
                        f"${mc_normal['min_price'].quantile(0.05):,.0f}",
                        f"${mc_normal['min_price'].quantile(0.10):,.0f}",
                        f"${mc_normal['min_price'].quantile(0.25):,.0f}",
                        f"${mc_normal['min_price'].quantile(0.50):,.0f}",
                    ]
                }
                st.dataframe(pd.DataFrame(normal_data), use_container_width=True, hide_index=True)
            else:
                st.info("Tutte le simulazioni hanno avuto un black swan")
        
        with res_col2:
            st.markdown("##### 🦢 Scenario BLACK SWAN")
            if len(mc_swan) > 0:
                swan_data = {
                    'Percentile': ['5%', '10%', '25%', '50%'],
                    'Min Price': [
                        f"${mc_swan['min_price'].quantile(0.05):,.0f}",
                        f"${mc_swan['min_price'].quantile(0.10):,.0f}",
                        f"${mc_swan['min_price'].quantile(0.25):,.0f}",
                        f"${mc_swan['min_price'].quantile(0.50):,.0f}",
                    ]
                }
                st.dataframe(pd.DataFrame(swan_data), use_container_width=True, hide_index=True)
            else:
                st.info("Nessuna simulazione con black swan")
        
        # Combined histogram
        fig_bs = go.Figure()
        
        if len(mc_normal) > 0:
            fig_bs.add_trace(go.Histogram(
                x=mc_normal['min_price'], 
                nbinsx=50, 
                name='Normale',
                marker_color='#28a745',
                opacity=0.6
            ))
        
        if len(mc_swan) > 0:
            fig_bs.add_trace(go.Histogram(
                x=mc_swan['min_price'], 
                nbinsx=50, 
                name='Black Swan',
                marker_color='#dc3545',
                opacity=0.6
            ))
        
        fig_bs.add_vline(x=CURRENT_PRICE, line_dash='solid', line_color='#F7931A', 
                         annotation_text=f"Oggi: ${CURRENT_PRICE:,.0f}")
        fig_bs.add_vline(x=q05, line_dash='dash', line_color='white',
                         annotation_text=f"Q05: ${q05:,.0f}")
        
        fig_bs.update_layout(
            title='Distribuzione Prezzo Minimo: Normale vs Black Swan',
            xaxis_title='Prezzo ($)',
            yaxis_title='Frequenza',
            barmode='overlay',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_bs, use_container_width=True)
        
        # Key insight
        if len(mc_swan) > 0:
            pct_below_q05 = (mc_swan['min_price'] < q05).mean() * 100
            st.warning(f"""
            ⚠️ **Nel {pct_below_q05:.1f}% degli scenari Black Swan, il prezzo scende SOTTO il floor Q05!**
            
            Questo dimostra che durante eventi estremi i modelli Power Law possono essere temporaneamente violati.
            """)
    
    with bs_tab3:
        st.subheader("📊 Stress Scenarios")
        
        st.markdown("""
        Analisi di scenari predefiniti basati su eventi storici o ipotetici.
        """)
        
        ath = df['Close'].max()
        
        scenarios = {
            '🟡 Correzione Moderata': {'drawdown': -30, 'description': 'Correzione tipica in bull market'},
            '🟠 Bear Market': {'drawdown': -50, 'description': 'Bear market standard (2018, 2022)'},
            '🔴 Crash Severo': {'drawdown': -65, 'description': 'Crash tipo COVID marzo 2020'},
            '⚫ Cigno Nero': {'drawdown': -80, 'description': 'Evento catastrofico (tipo Mt.Gox 2014)'},
            '💀 Extinction Event': {'drawdown': -90, 'description': 'Scenario apocalittico'},
        }
        
        stress_data = []
        for name, params in scenarios.items():
            dd = params['drawdown']
            price_from_current = CURRENT_PRICE * (1 + dd/100)
            price_from_ath = ath * (1 + dd/100)
            vs_q05 = (price_from_current / q05 - 1) * 100
            vs_q01 = (price_from_current / q01 - 1) * 100
            
            stress_data.append({
                'Scenario': name,
                'Drawdown': f"{dd}%",
                'Da Oggi': f"${price_from_current:,.0f}",
                'Da ATH': f"${price_from_ath:,.0f}",
                'vs Q05': f"{vs_q05:+.1f}%",
                'vs Q01': f"{vs_q01:+.1f}%",
                'Note': params['description']
            })
        
        st.dataframe(pd.DataFrame(stress_data), use_container_width=True, hide_index=True)
        
        # Visual stress test
        st.markdown("#### 📈 Visualizzazione Stress Levels")
        
        fig_stress = go.Figure()
        
        # Current price line
        fig_stress.add_trace(go.Scatter(
            x=[0, 1], y=[CURRENT_PRICE, CURRENT_PRICE],
            mode='lines', name=f'Prezzo Attuale (${CURRENT_PRICE:,.0f})',
            line=dict(color='#F7931A', width=3)
        ))
        
        # Floors
        fig_stress.add_trace(go.Scatter(
            x=[0, 1], y=[q01, q01],
            mode='lines', name=f'Q01 Floor (${q01:,.0f})',
            line=dict(color='#8B0000', width=2, dash='dash')
        ))
        fig_stress.add_trace(go.Scatter(
            x=[0, 1], y=[q05, q05],
            mode='lines', name=f'Q05 Floor (${q05:,.0f})',
            line=dict(color='#dc3545', width=2, dash='dash')
        ))
        
        # Stress scenarios
        colors = ['#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#000000']
        for i, (name, params) in enumerate(scenarios.items()):
            price = CURRENT_PRICE * (1 + params['drawdown']/100)
            fig_stress.add_trace(go.Scatter(
                x=[0, 1], y=[price, price],
                mode='lines', name=f"{name} (${price:,.0f})",
                line=dict(color=colors[i], width=1, dash='dot')
            ))
        
        fig_stress.update_layout(
            title='Stress Scenarios vs Floor Levels',
            yaxis_title='Price ($)',
            yaxis_type='log',
            template='plotly_dark',
            height=400,
            showlegend=True,
            xaxis=dict(showticklabels=False)
        )
        
        st.plotly_chart(fig_stress, use_container_width=True)
    
    with bs_tab4:
        st.subheader("🔬 Extreme Value Theory (EVT)")
        
        st.markdown("""
        **EVT** è un framework statistico per modellare eventi estremi (code della distribuzione).
        Usa la **Generalized Pareto Distribution (GPD)** per stimare perdite estreme.
        """)
        
        evt_floor, evt_var = calculate_evt_floor(df, confidence=0.99)
        
        if evt_floor is not None:
            evt_col1, evt_col2, evt_col3 = st.columns(3)
            
            with evt_col1:
                st.metric("EVT Floor (99%)", f"${evt_floor:,.0f}")
            with evt_col2:
                st.metric("EVT VaR (1-day)", f"{evt_var*100:.1f}%")
            with evt_col3:
                vs_current = (CURRENT_PRICE / evt_floor - 1) * 100
                st.metric("vs Prezzo Attuale", f"{vs_current:+.1f}%")
            
            st.markdown("#### 📊 Confronto Floor Methods")
            
            floor_comparison = {
                'Method': ['Q01 (1%)', 'Q05 (5%)', 'EVT (99%)', 'NLB'],
                'Floor': [f"${q01:,.0f}", f"${q05:,.0f}", f"${evt_floor:,.0f}", f"${current_nlb:,.0f}"],
                'vs Price': [
                    f"{(CURRENT_PRICE/q01-1)*100:+.1f}%",
                    f"{(CURRENT_PRICE/q05-1)*100:+.1f}%",
                    f"{(CURRENT_PRICE/evt_floor-1)*100:+.1f}%",
                    f"{(CURRENT_PRICE/current_nlb-1)*100:+.1f}%"
                ],
                'Method Type': ['Quantile Regression', 'Quantile Regression', 'Extreme Value Theory', 'Historical']
            }
            st.dataframe(pd.DataFrame(floor_comparison), use_container_width=True, hide_index=True)
            
            st.info("""
            **Interpretazione EVT:**
            - Il floor EVT stima il prezzo sotto il quale c'è solo 1% di probabilità di scendere in un singolo giorno
            - È più conservativo per eventi estremi rispetto ai quantili standard
            - Utile per stress testing e risk management
            """)
        else:
            st.warning("Non è stato possibile calcolare il floor EVT. Servono più dati.")
    
    st.markdown("---")
    
    # Model Parameters (expandable)
    with st.expander("🔧 Model Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quantile Regression (on LOW prices)**")
            params_data = {
                'Model': ['Q01', 'Q02', 'Q05', 'Q10', 'Q15', 'Q20'],
                'a': [f"{qr_models[q]['a']:.4e}" for q in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]],
                'b': [f"{qr_models[q]['b']:.4f}" for q in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]]
            }
            st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Power Law (on CLOSE prices)**")
            st.markdown(f"""
            - **a** = {pl_standard['a']:.4e}
            - **b** = {pl_standard['b']:.4f}
            - **R²** = {pl_standard['r2']:.4f}
            - **Residual σ** = {pl_standard['residual_std']:.4f}
            
            Formula: `P = a × Days^b`
            """)
    
    # Methodology (expandable)
    with st.expander("📚 Methodology"):
        st.markdown("""
        ### Floor Models Explained
        
        | Model | Description | Use Case |
        |-------|-------------|----------|
        | **Q01 (1%)** | 1st percentile of LOW prices | Extreme floor - rarely violated |
        | **Q05 (5%)** | 5th percentile (PlanC method) | Conservative floor |
        | **Q10 (10%)** | 10th percentile | Realistic floor for trading |
        | **NLB** | Never Look Back - price never revisited | Historical absolute floor |
        | **PL Fair** | Power Law regression median | Fair value estimate |
        
        ### Quantile Regression
        
        Instead of fitting the mean (OLS), quantile regression fits specific percentiles.
        This provides more robust floor estimates because:
        
        1. **Resistant to outliers** - extreme highs don't affect floor estimates
        2. **Direct floor modeling** - we model the lower bound directly
        3. **Probabilistic interpretation** - Q05 means 5% of prices fall below
        
        ### Monte Carlo Simulation
        
        Simulates thousands of price paths using:
        - Historical return distribution
        - Regime switching (normal/stress volatility)
        - Power Law floor as soft support
        
        This gives a probabilistic estimate of the minimum price over the projection period.
        """)
    
    # Footer
    st.markdown("---")
    
    # Disclaimer
    st.subheader("⚠️ DISCLAIMER / AVVERTENZE")
    
    disc_col1, disc_col2 = st.columns(2)
    
    with disc_col1:
        st.markdown("#### 🇮🇹 Italiano")
        st.warning("""
        **Questo strumento è fornito esclusivamente a scopo informativo e didattico.**
        
        Le informazioni presentate **NON costituiscono consulenza finanziaria, di investimento, fiscale o legale**. 
        I modelli matematici utilizzati (Power Law, Quantile Regression, NLB) sono basati su dati storici e 
        **non garantiscono risultati futuri**.
        
        Il mercato delle criptovalute è altamente volatile e speculativo. 
        **Potresti perdere tutto il capitale investito.**
        
        Prima di prendere qualsiasi decisione di investimento, consulta un consulente finanziario qualificato e autorizzato. 
        L'autore di questo strumento **non si assume alcuna responsabilità** per eventuali perdite derivanti dall'uso delle informazioni qui contenute.
        """)
    
    with disc_col2:
        st.markdown("#### 🇬🇧 English")
        st.warning("""
        **This tool is provided for informational and educational purposes only.**
        
        The information presented **does NOT constitute financial, investment, tax, or legal advice**. 
        The mathematical models used (Power Law, Quantile Regression, NLB) are based on historical data and 
        **do not guarantee future results**.
        
        The cryptocurrency market is highly volatile and speculative. 
        **You may lose all of your invested capital.**
        
        Before making any investment decision, consult a qualified and authorized financial advisor. 
        The author of this tool **assumes no responsibility** for any losses arising from the use of the information contained herein.
        """)
    
    st.info("""
    📊 **Past performance is not indicative of future results.** | I rendimenti passati non sono indicativi di risultati futuri.
    
    🔬 **This is a research tool, not a trading system.** | Questo è uno strumento di ricerca, non un sistema di trading.
    
    💡 **Always do your own research (DYOR).** | Fai sempre le tue ricerche personali.
    """)
    
    st.markdown("---")
    st.caption(f"Data: CryptoCompare | Models: Power Law, Quantile Regression, NLB | Last update: {TODAY.strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
