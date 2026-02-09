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
from io import BytesIO
import io

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="BTC Floor Analysis",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE - Disclaimer acceptance
# ============================================================================
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False

# ============================================================================
# DISCLAIMER POPUP
# ============================================================================
def show_disclaimer_popup():
    """Show disclaimer popup that must be accepted to continue"""
    
    st.markdown("""
    <style>
        .disclaimer-popup {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #dc3545;
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem auto;
            max-width: 800px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">₿ Bitcoin Floor Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("## ⚠️ DISCLAIMER / AVVERTENZE")
        
        st.error("**LEGGERE ATTENTAMENTE PRIMA DI PROCEDERE**")
        
        st.markdown("### 🇮🇹 Italiano")
        st.markdown("""
        **Questo strumento è fornito esclusivamente a scopo informativo e didattico.**
        
        - Le informazioni presentate **NON costituiscono consulenza finanziaria, di investimento, fiscale o legale**
        - I modelli matematici utilizzati (Power Law, Quantile Regression, NLB) sono basati su dati storici e **non garantiscono risultati futuri**
        - Il mercato delle criptovalute è altamente volatile e speculativo
        - **Potresti perdere tutto il capitale investito**
        - Prima di prendere qualsiasi decisione di investimento, consulta un consulente finanziario qualificato e autorizzato
        - L'autore di questo strumento **non si assume alcuna responsabilità** per eventuali perdite derivanti dall'uso delle informazioni qui contenute
        """)
        
        st.markdown("### 🇬🇧 English")
        st.markdown("""
        **This tool is provided for informational and educational purposes only.**
        
        - The information presented **does NOT constitute financial, investment, tax, or legal advice**
        - The mathematical models used (Power Law, Quantile Regression, NLB) are based on historical data and **do not guarantee future results**
        - The cryptocurrency market is highly volatile and speculative
        - **You may lose all of your invested capital**
        - Before making any investment decision, consult a qualified and authorized financial advisor
        - The author of this tool **assumes no responsibility** for any losses arising from the use of the information contained herein
        """)
        
        st.markdown("---")
        
        st.info("""
        📊 **Past performance is not indicative of future results.** | I rendimenti passati non sono indicativi di risultati futuri.
        
        🔬 **This is a research tool, not a trading system.** | Questo è uno strumento di ricerca, non un sistema di trading.
        
        💡 **Always do your own research (DYOR).** | Fai sempre le tue ricerche personali.
        """)
        
        st.markdown("---")
        
        # Checkbox and button
        accept = st.checkbox("✅ **Ho letto, compreso e accetto le condizioni sopra indicate** / **I have read, understood and accept the conditions above**")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🚀 **ACCEDI / ENTER**", type="primary", disabled=not accept, width="stretch"):
                st.session_state.disclaimer_accepted = True
                st.rerun()
        
        if not accept:
            st.warning("⚠️ Devi accettare le condizioni per procedere / You must accept the conditions to proceed")

# Check if disclaimer was accepted
if not st.session_state.disclaimer_accepted:
    show_disclaimer_popup()
    st.stop()

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
    # Extract standard errors and p-values for diagnostics
    se_intercept = model.bse['Intercept']
    se_slope = model.bse['x']
    pval_slope = model.pvalues['x']
    return {
        'quantile': quantile, 'a': 10**log_a, 'b': b, 'log_a': log_a,
        'se_intercept': se_intercept, 'se_slope': se_slope, 'pval_slope': pval_slope
    }

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

def monte_carlo_min(df, floor_model, n_sims=5000, horizon=730, seed=42):
    """Monte Carlo simulation for minimum price (chunked vectorization for memory safety)"""
    rng = np.random.RandomState(seed)
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()
    sigma_n = log_ret[log_ret.abs() < 1.5*sigma].std()
    sigma_s = log_ret[log_ret.abs() >= 1.5*sigma].std()
    p0, d0 = df['Close'].iloc[-1], df['Days'].iloc[-1]
    a, b = floor_model['a'], floor_model['b']
    
    # Pre-compute floors for all days (horizon,)
    days_arr = np.arange(1, horizon + 1)
    floors = a * ((d0 + days_arr) ** b)
    
    # Process in chunks to limit memory (~50MB per chunk max)
    CHUNK = min(n_sims, 2000)
    all_min_prices = []
    all_min_days = []
    
    remaining = n_sims
    while remaining > 0:
        chunk_size = min(CHUNK, remaining)
        remaining -= chunk_size
        
        regime_draws = rng.random((chunk_size, horizon))
        vols = np.where(regime_draws < 0.15, sigma_s, sigma_n)
        shocks = rng.normal(mu, 1.0, (chunk_size, horizon)) * vols
        bounce_draws = rng.random((chunk_size, horizon))
        bounce_levels = rng.uniform(0.0, 0.20, (chunk_size, horizon))
        
        prices = np.full(chunk_size, p0)
        min_prices = np.full(chunk_size, p0)
        min_days = np.zeros(chunk_size, dtype=int)
        
        for t in range(horizon):
            prices = prices * np.exp(shocks[:, t])
            floor_t = floors[t]
            below_floor = prices < floor_t * 0.85
            bounces = below_floor & (bounce_draws[:, t] < 0.70)
            prices[bounces] = floor_t * (0.85 + bounce_levels[bounces, t])
            new_min = prices < min_prices
            min_prices[new_min] = prices[new_min]
            min_days[new_min] = t + 1
        
        all_min_prices.append(min_prices)
        all_min_days.append(min_days)
    
    return pd.DataFrame({
        'min_price': np.concatenate(all_min_prices),
        'min_day': np.concatenate(all_min_days)
    })

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
                            black_swan_prob=0.05, black_swan_impact=-0.50, seed=42):
    """
    Monte Carlo with Black Swan events (chunked vectorization for memory safety).
    """
    rng = np.random.RandomState(seed)
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    mu, sigma = log_ret.mean(), log_ret.std()
    sigma_n = log_ret[log_ret.abs() < 1.5*sigma].std()
    sigma_s = log_ret[log_ret.abs() >= 1.5*sigma].std()
    p0, d0 = df['Close'].iloc[-1], df['Days'].iloc[-1]
    a, b = floor_model['a'], floor_model['b']
    
    days_arr = np.arange(1, horizon + 1)
    floors = a * ((d0 + days_arr) ** b)
    
    CHUNK = min(n_sims, 2000)
    all_min_prices = []
    all_min_days = []
    all_bs = []
    
    remaining = n_sims
    while remaining > 0:
        chunk_size = min(CHUNK, remaining)
        remaining -= chunk_size
        
        # Black swan setup
        has_bs = rng.random(chunk_size) < black_swan_prob
        bs_days = np.where(has_bs, rng.randint(1, max(horizon // 2, 2), size=chunk_size), -1)
        bs_shocks = black_swan_impact + rng.uniform(-0.10, 0.05, chunk_size)
        
        regime_draws = rng.random((chunk_size, horizon))
        vols = np.where(regime_draws < 0.15, sigma_s, sigma_n)
        normal_shocks = rng.normal(mu, 1.0, (chunk_size, horizon)) * vols
        bounce_draws = rng.random((chunk_size, horizon))
        bounce_levels = rng.uniform(0.0, 0.20, (chunk_size, horizon))
        
        prices = np.full(chunk_size, p0)
        min_prices = np.full(chunk_size, p0)
        min_days = np.zeros(chunk_size, dtype=int)
        bs_occurred = np.zeros(chunk_size, dtype=bool)
        
        for t in range(horizon):
            d = t + 1
            bs_today = has_bs & (bs_days == d) & (~bs_occurred)
            if bs_today.any():
                prices[bs_today] = prices[bs_today] * (1 + bs_shocks[bs_today])
                bs_occurred[bs_today] = True
            
            normal_mask = ~bs_today
            prices[normal_mask] = prices[normal_mask] * np.exp(normal_shocks[normal_mask, t])
            
            in_recovery = bs_occurred & (d < bs_days + 30)
            can_bounce = ~in_recovery
            floor_t = floors[t]
            below_floor = can_bounce & (prices < floor_t * 0.85)
            bounces = below_floor & (bounce_draws[:, t] < 0.70)
            prices[bounces] = floor_t * (0.85 + bounce_levels[bounces, t])
            
            new_min = prices < min_prices
            min_prices[new_min] = prices[new_min]
            min_days[new_min] = d
        
        all_min_prices.append(min_prices)
        all_min_days.append(min_days)
        all_bs.append(bs_occurred)
    
    return pd.DataFrame({
        'min_price': np.concatenate(all_min_prices), 
        'min_day': np.concatenate(all_min_days),
        'black_swan': np.concatenate(all_bs)
    })

def calculate_evt_floor(df, confidence=0.99, horizon_days=30):
    """
    Extreme Value Theory (EVT) floor estimation.
    Uses Generalized Pareto Distribution on extreme losses.
    Returns multi-period floor using temporal scaling.
    
    Parameters:
        confidence: VaR confidence level (e.g. 0.99)
        horizon_days: risk horizon in days for multi-period scaling
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
        return None, None, None
    
    # Fit Generalized Pareto Distribution (loc=0 by definition for exceedances)
    try:
        c, loc, scale = genpareto.fit(exceedances, floc=0)
        
        # Shape parameter check
        if c <= -0.5 or c > 2.0:
            return None, None, None
        
        # VaR at confidence level
        n = len(returns)
        n_exceed = len(exceedances)
        
        # Daily VaR (GPD formula)
        if abs(c) < 1e-8:
            # c ≈ 0: exponential tail (use limit formula)
            var_daily = threshold + scale * np.log((n / n_exceed) * (1 - confidence))
        else:
            var_daily = threshold + (scale / c) * (((n / n_exceed) * (1 - confidence)) ** (-c) - 1)
        
        # Multi-period scaling (√t for VaR under iid assumption, conservative for BTC)
        var_horizon = var_daily * np.sqrt(horizon_days)
        # Cap at reasonable maximum (99% loss)
        var_horizon = min(var_horizon, 0.99)
        
        # Translate to price floor
        current_price = df['Close'].iloc[-1]
        evt_floor_daily = current_price * (1 - var_daily)
        evt_floor_horizon = current_price * (1 - var_horizon)
        
        return evt_floor_horizon, var_daily, var_horizon
    except Exception:
        return None, None, None

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
        mc_sims = st.slider("Monte Carlo Simulations", 1000, 50000, 5000, 1000,
                            help="5k per analisi rapida, 20k+ per percentili estremi accurati. >20k può essere lento.")
        projection_years = st.slider("Projection (years)", 1, 5, 2)
        
        st.markdown("---")
        st.markdown("### 🎲 Monte Carlo Seed")
        seed_input = st.text_input("Random Seed", value="42", help="Intero positivo per la riproducibilità. Cambia seed per verificare la robustezza.")
        # Validate seed input
        try:
            mc_seed = int(seed_input)
            if mc_seed < 0 or mc_seed > 2**31 - 1:
                st.warning("⚠️ Seed deve essere tra 0 e 2,147,483,647. Uso 42.")
                mc_seed = 42
        except ValueError:
            st.warning("⚠️ Seed non valido. Deve essere un intero. Uso 42.")
            mc_seed = 42
        
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
    
    # Upper quantile models on HIGH prices (for exit strategy)
    upper_quantiles = [0.80, 0.85, 0.90, 0.95, 0.99]
    qr_upper = {}
    df['LogHigh'] = np.log10(df['High'])
    for q in upper_quantiles:
        qr_upper[q] = fit_quantile_regression(df, q, 'LogHigh')
    
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
    
    # Halving Cycle Indicator
    HALVING_DATES = [
        pd.Timestamp('2012-11-28'),
        pd.Timestamp('2016-07-09'),
        pd.Timestamp('2020-05-11'),
        pd.Timestamp('2024-04-20'),
    ]
    # Estimate next halving (~4 years after last)
    NEXT_HALVING_EST = HALVING_DATES[-1] + pd.Timedelta(days=4*365)
    
    last_halving = HALVING_DATES[-1]
    days_since_halving = (TODAY - last_halving).days
    days_to_next_halving = (NEXT_HALVING_EST - TODAY).days
    cycle_progress = days_since_halving / (days_since_halving + max(days_to_next_halving, 1)) * 100
    
    # Cycle phase interpretation
    if cycle_progress < 25:
        cycle_phase = "🟢 Accumulo Post-Halving"
        cycle_note = "Fase storica favorevole"
    elif cycle_progress < 50:
        cycle_phase = "🟢 Bull Run"
        cycle_note = "Storicamente fase di crescita rapida"
    elif cycle_progress < 75:
        cycle_phase = "🟡 Maturità del Ciclo"
        cycle_note = "Possibili top, cautela crescente"
    else:
        cycle_phase = "🟠 Fine Ciclo / Pre-Halving"
        cycle_note = "Storicamente fase di consolidamento/correzione"
    
    cycle_col1, cycle_col2, cycle_col3 = st.columns(3)
    with cycle_col1:
        st.metric("⛏️ Ultimo Halving", last_halving.strftime('%Y-%m-%d'), f"{days_since_halving} giorni fa")
    with cycle_col2:
        st.metric("⏳ Prossimo Halving (est.)", NEXT_HALVING_EST.strftime('%Y-%m'), f"~{days_to_next_halving} giorni")
    with cycle_col3:
        st.metric("📍 Progresso Ciclo", f"{cycle_progress:.0f}%")
    
    # Phase shown as colored info box (not truncated in st.metric delta)
    if cycle_progress < 25:
        st.success(f"🟢 **Fase: Accumulo Post-Halving** — Storicamente fase favorevole per accumulazione ({cycle_progress:.0f}% del ciclo)")
    elif cycle_progress < 50:
        st.success(f"🟢 **Fase: Bull Run** — Storicamente fase di crescita rapida ({cycle_progress:.0f}% del ciclo)")
    elif cycle_progress < 75:
        st.warning(f"🟡 **Fase: Maturità del Ciclo** — Possibili top, cautela crescente ({cycle_progress:.0f}% del ciclo)")
    else:
        st.warning(f"🟠 **Fase: Fine Ciclo / Pre-Halving** — Storicamente fase di consolidamento o correzione ({cycle_progress:.0f}% del ciclo)")
    
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
        # Dynamic y-axis range
        y_max = max(df['Close'].max(), proj['Q10'].max()) * 2
        fig.update_yaxes(range=[np.log10(100), np.log10(y_max)])
        
        st.plotly_chart(fig, width="stretch")
    
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
        
        st.dataframe(floor_df, width="stretch", hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🎯 Buy Levels")
        buy_levels = {
            'Signal': ['🟢 STRONG BUY', '🟢 BUY', '🟡 ACCUMULATE', '🟠 WATCH'],
            'Price': [f"< ${q05*1.00:,.0f}", f"< ${q05*1.10:,.0f}", f"< ${q05*1.20:,.0f}", f"< ${q05*1.35:,.0f}"],
            'Description': ['At Q05 floor', '+10% from Q05', '+20% from Q05', '+35% from Q05']
        }
        st.dataframe(pd.DataFrame(buy_levels), width="stretch", hide_index=True)
    
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
        if projection_years >= 4:
            periods['4Y'] = 1461
        if projection_years >= 5:
            periods['5Y'] = 1826
        
        proj_data = {'Floor': []}
        for p in periods.keys():
            proj_data[p] = []
        
        for fname, q in [('Q01', 0.01), ('Q02', 0.02), ('Q05', 0.05), ('Q10', 0.10)]:
            proj_data['Floor'].append(fname)
            qm = qr_models[q]
            for pname, days in periods.items():
                val = predict_pl(TODAY_DAYS + days, qm['a'], qm['b'])
                proj_data[pname].append(f"${val:,.0f}")
        
        st.dataframe(pd.DataFrame(proj_data), width="stretch", hide_index=True)
    
    with mc_col:
        st.subheader("🎲 Monte Carlo Simulation")
        
        with st.spinner(f"Running {mc_sims:,} simulations..."):
            mc = monte_carlo_min(df, qr_models[0.05], n_sims=mc_sims, horizon=projection_years*365, seed=mc_seed)
        
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
        st.dataframe(pd.DataFrame(mc_data), width="stretch", hide_index=True)
        
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
    
    st.plotly_chart(fig_mc, width="stretch")
    
    st.markdown("---")
    
    # ============================================================================
    # BLACK SWAN ANALYSIS
    # ============================================================================
    st.header("🦢 Black Swan Analysis")
    st.markdown("""
    Un **Cigno Nero** è un evento raro, imprevedibile e ad alto impatto che può violare anche i floor più conservativi.
    Questa sezione analizza scenari estremi per stress-testing del portafoglio.
    
    📊 **Riferimenti storici**: Flash crash BTC >30% in 1-3 giorni: ~5-6 volte dal 2010 (~1 ogni 2.5 anni).
    Crash finanziari tradizionali (1929, 1987, 2008, COVID): ~3-4 per secolo (~3-4%/anno).
    """)
    
    # Black Swan Settings
    bs_col1, bs_col2, bs_col3 = st.columns(3)
    with bs_col1:
        bs_probability = st.slider("🎲 Probabilità Black Swan (%)", 1, 20, 5,
                                   help="BTC storico: ~35-40% su 2 anni. Mercati trad: 3-4%/anno. Default 5% = conservativo per 2Y.") / 100
    with bs_col2:
        bs_impact = st.slider("💥 Impatto Black Swan (%)", -80, -30, -50,
                              help="Flash crash BTC storici: -30% a -50% in giorni. Worst case: -80% (Mt.Gox 2014).")
    with bs_col3:
        bs_sims = st.slider("🔄 Simulazioni", 1000, 50000, 5000, 1000)
    
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
            
            st.dataframe(display_crashes.sort_values('Drawdown'), width="stretch", hide_index=True)
            
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
                black_swan_impact=bs_impact/100,
                seed=mc_seed
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
                st.dataframe(pd.DataFrame(normal_data), width="stretch", hide_index=True)
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
                st.dataframe(pd.DataFrame(swan_data), width="stretch", hide_index=True)
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
        
        st.plotly_chart(fig_bs, width="stretch")
        
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
        **Cosa succederebbe se BTC crollasse del X% da oggi?**
        """)
        
        ath = df['Close'].max()
        
        scenarios = {
            '🟡 Correzione Moderata': {'drawdown': -30, 'description': 'Correzione tipica in bull market', 'example': ''},
            '🟠 Bear Market': {'drawdown': -50, 'description': 'Bear market standard', 'example': '2018, 2022'},
            '🔴 Crash Severo': {'drawdown': -65, 'description': 'Crash violento', 'example': 'COVID Mar 2020'},
            '⚫ Cigno Nero': {'drawdown': -80, 'description': 'Evento catastrofico', 'example': 'Mt.Gox 2014'},
            '💀 Extinction Event': {'drawdown': -90, 'description': 'Scenario apocalittico', 'example': 'Ipotetico'},
        }
        
        st.markdown(f"### 💰 Prezzo attuale: **${CURRENT_PRICE:,.0f}**")
        st.markdown(f"### 🏔️ ATH: **${ath:,.0f}**")
        
        st.markdown("---")
        
        stress_data = []
        for name, params in scenarios.items():
            dd = params['drawdown']
            price_from_current = CURRENT_PRICE * (1 + dd/100)
            price_from_ath = ath * (1 + dd/100)
            
            # Determine if below floors
            below_q05 = "🔴 SÌ" if price_from_current < q05 else "🟢 NO"
            below_q01 = "🔴 SÌ" if price_from_current < q01 else "🟢 NO"
            below_nlb = "🔴 SÌ" if price_from_current < current_nlb else "🟢 NO"
            
            stress_data.append({
                'Scenario': name,
                'Drawdown': f"{dd}%",
                '📉 Prezzo Raggiunto': f"${price_from_current:,.0f}",
                'Da ATH': f"${price_from_ath:,.0f}",
                'Sotto Q05?': below_q05,
                'Sotto Q01?': below_q01,
                'Sotto NLB?': below_nlb,
                'Esempio Storico': params['example']
            })
        
        st.dataframe(pd.DataFrame(stress_data), width="stretch", hide_index=True)
        
        # Floor reference
        st.markdown("---")
        st.markdown("#### 📊 Riferimento Floor Attuali")
        
        floor_ref_col1, floor_ref_col2, floor_ref_col3 = st.columns(3)
        with floor_ref_col1:
            st.metric("Q05 Floor (5%)", f"${q05:,.0f}", f"{(CURRENT_PRICE/q05-1)*100:+.1f}% da oggi")
        with floor_ref_col2:
            st.metric("Q01 Floor (1%)", f"${q01:,.0f}", f"{(CURRENT_PRICE/q01-1)*100:+.1f}% da oggi")
        with floor_ref_col3:
            st.metric("NLB Floor", f"${current_nlb:,.0f}", f"{(CURRENT_PRICE/current_nlb-1)*100:+.1f}% da oggi")
        
        st.markdown("---")
        
        # Visual stress test
        st.markdown("#### 📈 Visualizzazione Livelli di Prezzo")
        
        fig_stress = go.Figure()
        
        # Current price
        fig_stress.add_hline(
            y=CURRENT_PRICE, 
            line_dash="solid", 
            line_color="#F7931A",
            line_width=3,
            annotation_text=f"💰 OGGI: ${CURRENT_PRICE:,.0f}",
            annotation_position="right"
        )
        
        # Floors
        fig_stress.add_hline(
            y=q05, 
            line_dash="dash", 
            line_color="#dc3545",
            line_width=2,
            annotation_text=f"Q05: ${q05:,.0f}",
            annotation_position="right"
        )
        fig_stress.add_hline(
            y=q01, 
            line_dash="dash", 
            line_color="#8B0000",
            line_width=2,
            annotation_text=f"Q01: ${q01:,.0f}",
            annotation_position="right"
        )
        fig_stress.add_hline(
            y=current_nlb, 
            line_dash="dot", 
            line_color="#ff6b6b",
            line_width=2,
            annotation_text=f"NLB: ${current_nlb:,.0f}",
            annotation_position="right"
        )
        
        # Stress scenario prices as bars
        scenario_names = []
        scenario_prices = []
        scenario_colors = []
        color_map = {'🟡': '#ffc107', '🟠': '#fd7e14', '🔴': '#dc3545', '⚫': '#6c757d', '💀': '#212529'}
        
        for name, params in scenarios.items():
            price = CURRENT_PRICE * (1 + params['drawdown']/100)
            scenario_names.append(f"{name}\n${price:,.0f}")
            scenario_prices.append(price)
            emoji = name.split()[0]
            scenario_colors.append(color_map.get(emoji, '#888888'))
        
        fig_stress.add_trace(go.Bar(
            x=scenario_names,
            y=scenario_prices,
            marker_color=scenario_colors,
            text=[f"${p:,.0f}" for p in scenario_prices],
            textposition='outside',
            name='Prezzo Scenario'
        ))
        
        fig_stress.update_layout(
            title='Prezzi Raggiunti in Ogni Scenario vs Floor',
            yaxis_title='Prezzo ($)',
            yaxis_type='log',
            template='plotly_dark',
            height=500,
            showlegend=False,
        )
        
        # Set y-axis range to show all levels
        min_price = min(scenario_prices) * 0.8
        max_price = CURRENT_PRICE * 1.2
        fig_stress.update_yaxes(range=[np.log10(min_price), np.log10(max_price)])
        
        st.plotly_chart(fig_stress, width="stretch")
        
        # Interpretation
        st.markdown("#### 🎯 Interpretazione")
        
        # Find which scenario breaks which floor
        breaks_q05 = None
        breaks_q01 = None
        breaks_nlb = None
        
        for name, params in scenarios.items():
            price = CURRENT_PRICE * (1 + params['drawdown']/100)
            if price < q05 and breaks_q05 is None:
                breaks_q05 = (name, params['drawdown'], price)
            if price < q01 and breaks_q01 is None:
                breaks_q01 = (name, params['drawdown'], price)
            if price < current_nlb and breaks_nlb is None:
                breaks_nlb = (name, params['drawdown'], price)
        
        if breaks_q05:
            st.warning(f"⚠️ **Q05 Floor (${q05:,.0f})** verrebbe violato da: **{breaks_q05[0]}** ({breaks_q05[1]}%) → ${breaks_q05[2]:,.0f}")
        else:
            st.success(f"✅ **Q05 Floor (${q05:,.0f})** regge in tutti gli scenari!")
            
        if breaks_q01:
            st.warning(f"⚠️ **Q01 Floor (${q01:,.0f})** verrebbe violato da: **{breaks_q01[0]}** ({breaks_q01[1]}%) → ${breaks_q01[2]:,.0f}")
        else:
            st.success(f"✅ **Q01 Floor (${q01:,.0f})** regge in tutti gli scenari!")
            
        if breaks_nlb:
            st.error(f"🚨 **NLB Floor (${current_nlb:,.0f})** verrebbe violato da: **{breaks_nlb[0]}** ({breaks_nlb[1]}%) → ${breaks_nlb[2]:,.0f}")
        else:
            st.success(f"✅ **NLB Floor (${current_nlb:,.0f})** regge in tutti gli scenari!")
    
    with bs_tab4:
        st.subheader("🔬 Extreme Value Theory (EVT)")
        
        st.markdown("""
        **EVT** è un framework statistico per modellare eventi estremi (code della distribuzione).
        Usa la **Generalized Pareto Distribution (GPD)** per stimare perdite estreme.
        """)
        
        evt_floor, evt_var_daily, evt_var_horizon = calculate_evt_floor(df, confidence=0.99, horizon_days=30)
        
        if evt_floor is not None:
            evt_col1, evt_col2, evt_col3, evt_col4 = st.columns(4)
            
            with evt_col1:
                st.metric("EVT Floor (30d, 99%)", f"${evt_floor:,.0f}")
            with evt_col2:
                st.metric("VaR Giornaliero (1-day)", f"{evt_var_daily*100:.1f}%")
            with evt_col3:
                st.metric("VaR 30-day (√t scaled)", f"{evt_var_horizon*100:.1f}%")
            with evt_col4:
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
            st.dataframe(pd.DataFrame(floor_comparison), width="stretch", hide_index=True)
            
            st.info("""
            **Interpretazione EVT:**
            - Il **VaR giornaliero** stima la massima perdita in un singolo giorno al 99% di confidenza
            - Il **VaR 30-day** scala il rischio su un orizzonte mensile (approx. √t, conservativo)
            - Il **floor EVT** è il prezzo raggiunto applicando il VaR 30-day al prezzo corrente
            - ⚠️ Lo scaling √t assume rendimenti iid — per BTC (autocorrelazione, fat tails) è un'approssimazione
            """)
        else:
            st.warning("Non è stato possibile calcolare il floor EVT. Servono più dati.")
    
    st.markdown("---")
    
    # ============================================================================
    # ENTRY STRATEGY CALCULATOR
    # ============================================================================
    st.header("🎯 Entry Strategy Calculator")
    st.markdown("""
    Confronta diverse strategie di ingresso: **comprare ora** vs **aspettare un dip** vs **tenere riserva per Black Swan**.
    Analisi basata su probabilità Monte Carlo e backtest storico.
    """)
    
    # Input parameters
    strat_param_col1, strat_param_col2, strat_param_col3 = st.columns(3)
    with strat_param_col1:
        budget = st.number_input("💰 Budget Totale (€)", min_value=100, max_value=10000000, value=10000, step=1000)
    with strat_param_col2:
        horizon_months = st.selectbox("📅 Orizzonte", [12, 24, 36], index=1, format_func=lambda x: f"{x} mesi ({x//12} anni)")
    with strat_param_col3:
        target_price_2y = st.number_input("🎯 Target Price (stima)", min_value=50000, max_value=1000000, value=int(pl_fair * 1.5), step=10000)
    
    # Reserve & Black Swan parameters
    st.markdown("#### 🦢 Parametri Riserva & Black Swan")
    st.caption("La **Riserva Opportunistica** è una quota del budget tenuta in cash per comprare durante dip estremi (Black Swan o flash crash). Aumenta il potenziale rendimento ma riduce l'esposizione immediata.")
    
    reserve_col1, reserve_col2, reserve_col3 = st.columns(3)
    with reserve_col1:
        reserve_pct = st.slider("💵 Riserva Opportunistica (%)", 0, 50, 20, 5,
                                help="% del budget tenuto in cash come riserva per cogliere occasioni. 0% = tutto investito subito.") / 100
    with reserve_col2:
        es_bs_prob = st.slider("🎲 Prob. Black Swan (%)", 1, 20, 5, key="es_bs_prob2",
                               help="Probabilità di un evento BS nell'orizzonte. BTC storico: ~35-40% su 2 anni.") / 100
    with reserve_col3:
        es_bs_impact = st.slider("💥 Impatto BS (%)", -80, -30, -50, key="es_bs_impact2") / 100
    
    budget_invest = budget * (1 - reserve_pct)
    budget_reserve = budget * reserve_pct
    
    if reserve_pct > 0:
        st.info(f"💰 **Budget investito subito**: €{budget_invest:,.0f} ({(1-reserve_pct)*100:.0f}%) | 🦢 **Riserva BS**: €{budget_reserve:,.0f} ({reserve_pct*100:.0f}%)")
    
    # Define entry levels
    entry_levels = {
        'Prezzo Attuale': CURRENT_PRICE,
        'Q20 Floor': predict_pl(TODAY_DAYS, qr_models[0.20]['a'], qr_models[0.20]['b']),
        'Q15 Floor': predict_pl(TODAY_DAYS, qr_models[0.15]['a'], qr_models[0.15]['b']),
        'Q10 Floor': predict_pl(TODAY_DAYS, qr_models[0.10]['a'], qr_models[0.10]['b']),
        'Q05 Floor': predict_pl(TODAY_DAYS, qr_models[0.05]['a'], qr_models[0.05]['b']),
        'Q01 Floor': predict_pl(TODAY_DAYS, qr_models[0.01]['a'], qr_models[0.01]['b']),
    }
    
    # Tabs
    strat_tab1, strat_tab2, strat_tab3 = st.tabs([
        "📊 Probabilità & Expected Value",
        "📈 Backtest Storico",
        "🏆 Raccomandazione"
    ])
    
    with strat_tab1:
        st.subheader("📊 Probabilità di Raggiungimento & Rendimento Atteso")
        
        st.markdown(f"""
        **Monte Carlo** ({horizon_months} mesi) — Include eventi Black Swan ({es_bs_prob*100:.0f}% prob., {es_bs_impact*100:.0f}% impatto).  
        Budget investito: €{budget_invest:,.0f} | Riserva BS: €{budget_reserve:,.0f}
        """)
        
        @st.cache_data(ttl=3600)
        def calc_entry_probs_v2(_df_close, current_price, floor_params_tuple,
                                n_sims=10000, horizon_days=730, seed=42,
                                bs_prob=0.05, bs_impact=-0.50):
            """MC with Black Swan: tracks probability + whether reached during BS window"""
            rng = np.random.RandomState(seed)
            df_temp = pd.DataFrame({'Close': _df_close})
            log_ret = np.log(df_temp['Close'] / df_temp['Close'].shift(1)).dropna()
            mu, sigma = log_ret.mean(), log_ret.std()
            sigma_n = log_ret[log_ret.abs() < 1.5*sigma].std()
            sigma_s = log_ret[log_ret.abs() >= 1.5*sigma].std()
            
            floor_params = dict(floor_params_tuple)
            floor_prices = np.array(list(floor_params.values()))
            n_levels = len(floor_prices)
            
            CHUNK = min(n_sims, 2000)
            total_reached = np.zeros(n_levels, dtype=int)
            total_reached_bs = np.zeros(n_levels, dtype=int)
            all_final = []
            
            remaining = n_sims
            while remaining > 0:
                cs = min(CHUNK, remaining)
                remaining -= cs
                
                regime_draws = rng.random((cs, horizon_days))
                vols = np.where(regime_draws < 0.15, sigma_s, sigma_n)
                shocks = rng.normal(mu, 1.0, (cs, horizon_days)) * vols
                
                has_bs = rng.random(cs) < bs_prob
                bs_days = np.where(has_bs, rng.randint(1, max(horizon_days // 2, 2), size=cs), -1)
                bs_shocks_arr = bs_impact + rng.uniform(-0.10, 0.05, cs)
                
                prices = np.full(cs, current_price)
                min_prices = np.full(cs, current_price)
                bs_occurred = np.zeros(cs, dtype=bool)
                min_in_bs = np.zeros(cs, dtype=bool)
                
                for t in range(horizon_days):
                    d = t + 1
                    bs_today = has_bs & (bs_days == d) & (~bs_occurred)
                    if bs_today.any():
                        prices[bs_today] *= (1 + bs_shocks_arr[bs_today])
                        bs_occurred[bs_today] = True
                    normal = ~bs_today
                    prices[normal] *= np.exp(shocks[normal, t])
                    
                    new_min = prices < min_prices
                    in_bs_win = bs_occurred & (d <= bs_days + 60)
                    min_in_bs[new_min & in_bs_win] = True
                    min_in_bs[new_min & ~in_bs_win] = False
                    min_prices = np.minimum(min_prices, prices)
                
                all_final.append(prices.copy())
                for j in range(n_levels):
                    rm = min_prices <= floor_prices[j]
                    total_reached[j] += rm.sum()
                    total_reached_bs[j] += (rm & min_in_bs).sum()
            
            avg_final = float(np.mean(np.concatenate(all_final)))
            results = []
            for j, (level, price) in enumerate(floor_params.items()):
                results.append({
                    'level': level, 'price': price,
                    'probability': total_reached[j] / n_sims,
                    'prob_via_bs': total_reached_bs[j] / n_sims,
                    'avg_final_price': avg_final
                })
            return pd.DataFrame(results)
        
        with st.spinner("Calcolando probabilità..."):
            prob_df = calc_entry_probs_v2(
                df['Close'].values, CURRENT_PRICE, tuple(entry_levels.items()),
                n_sims=10000, horizon_days=int(horizon_months * 30.44), seed=mc_seed,
                bs_prob=es_bs_prob, bs_impact=es_bs_impact
            )
        
        # Build strategy table with numeric values (sortable)
        strategy_rows = []
        for _, row in prob_df.iterrows():
            level = row['level']
            entry_price = row['price']
            prob_reach = row['probability']
            prob_bs = row['prob_via_bs']
            
            if level == 'Prezzo Attuale':
                btc_bought = budget_invest / entry_price
                ev = btc_bought * target_price_2y + budget_reserve  # reserve stays cash
                expected_return = (ev / budget - 1) * 100
            else:
                if prob_reach > 0:
                    btc_invest = budget_invest / entry_price
                    # Reserve deployed during BS if floor reached during BS
                    if prob_bs > 0 and budget_reserve > 0:
                        btc_reserve = budget_reserve / entry_price
                        ev_invest = prob_reach * (btc_invest * target_price_2y)
                        ev_reserve = prob_bs * (btc_reserve * target_price_2y)
                        ev_cash = (1 - prob_reach) * budget_invest + (1 - prob_bs) * budget_reserve
                        ev = ev_invest + ev_reserve + ev_cash
                    else:
                        ev = prob_reach * (btc_invest * target_price_2y) + (1 - prob_reach) * budget_invest + budget_reserve
                    expected_return = (ev / budget - 1) * 100
                    btc_bought = budget_invest / entry_price
                else:
                    ev = budget
                    expected_return = 0.0
                    btc_bought = 0
            
            strategy_rows.append({
                'Livello': level,
                'Prezzo Entry ($)': round(entry_price, 0),
                'Sconto (%)': round((entry_price / CURRENT_PRICE - 1) * 100, 1),
                'Prob. (%)': round(prob_reach * 100, 1),
                'via BS (%)': round(prob_bs * 100, 1),
                'BTC se eseguito': round(btc_bought, 6),
                'Valore Atteso (€)': round(ev, 0),
                'Rend. Atteso (%)': round(expected_return, 1),
            })
        
        col_cfg = {
            'Prezzo Entry ($)': st.column_config.NumberColumn(format="$,.0f"),
            'Sconto (%)': st.column_config.NumberColumn(format="+.1f"),
            'Prob. (%)': st.column_config.NumberColumn(format=".1f"),
            'via BS (%)': st.column_config.NumberColumn(format=".1f"),
            'BTC se eseguito': st.column_config.NumberColumn(format=".6f"),
            'Valore Atteso (€)': st.column_config.NumberColumn(format="$,.0f"),
            'Rend. Atteso (%)': st.column_config.NumberColumn(format="+.1f"),
        }
        st.dataframe(pd.DataFrame(strategy_rows), width="stretch", hide_index=True, column_config=col_cfg)
        
        # --- Split Strategies with Reserve ---
        st.markdown("---")
        st.markdown("### 🔀 Strategie Split (con Riserva)")
        
        split_strategies = [
            {'name': '100% Ora', 'allocations': [('Prezzo Attuale', 1.0)]},
            {'name': '50% Ora + 50% Q10', 'allocations': [('Prezzo Attuale', 0.5), ('Q10 Floor', 0.5)]},
            {'name': '50% Ora + 50% Q05', 'allocations': [('Prezzo Attuale', 0.5), ('Q05 Floor', 0.5)]},
            {'name': '33/33/34 (Ora/Q10/Q05)', 'allocations': [('Prezzo Attuale', 0.33), ('Q10 Floor', 0.33), ('Q05 Floor', 0.34)]},
            {'name': 'Ladder 25% (Ora/Q20/Q10/Q05)', 'allocations': [('Prezzo Attuale', 0.25), ('Q20 Floor', 0.25), ('Q10 Floor', 0.25), ('Q05 Floor', 0.25)]},
        ]
        
        split_rows = []
        for strat in split_strategies:
            total_btc = 0
            total_ev = 0
            executed_pct = 0
            
            for level, alloc in strat['allocations']:
                level_price = entry_levels[level]
                prob_row = prob_df[prob_df['level'] == level].iloc[0]
                prob = prob_row['probability'] if level != 'Prezzo Attuale' else 1.0
                prob_bs = prob_row['prob_via_bs'] if level != 'Prezzo Attuale' else 0.0
                
                alloc_invest = budget_invest * alloc
                btc_if_exec = alloc_invest / level_price
                total_btc += prob * btc_if_exec
                
                ev_part = prob * (btc_if_exec * target_price_2y) + (1 - prob) * alloc_invest
                total_ev += ev_part
                executed_pct += prob * alloc
            
            # Reserve: deployed during BS at best available floor
            if budget_reserve > 0:
                # Use Q10 as the BS entry price for reserve
                q10_price = entry_levels['Q10 Floor']
                prob_bs_q10 = prob_df[prob_df['level'] == 'Q10 Floor']['prob_via_bs'].values[0]
                btc_reserve = budget_reserve / q10_price
                total_ev += prob_bs_q10 * (btc_reserve * target_price_2y) + (1 - prob_bs_q10) * budget_reserve
                total_btc += prob_bs_q10 * btc_reserve
            
            expected_return = (total_ev / budget - 1) * 100
            
            split_rows.append({
                'Strategia': strat['name'],
                'Invest (%)': round((1 - reserve_pct) * 100, 0),
                'Riserva (%)': round(reserve_pct * 100, 0),
                'Prob. Esec. (%)': round(executed_pct * 100, 1),
                'BTC Attesi': round(total_btc, 6),
                'Valore Atteso (€)': round(total_ev, 0),
                'Rend. Atteso (%)': round(expected_return, 1),
            })
        
        split_cfg = {
            'Invest (%)': st.column_config.NumberColumn(format=".0f"),
            'Riserva (%)': st.column_config.NumberColumn(format=".0f"),
            'Prob. Esec. (%)': st.column_config.NumberColumn(format=".1f"),
            'BTC Attesi': st.column_config.NumberColumn(format=".6f"),
            'Valore Atteso (€)': st.column_config.NumberColumn(format="$,.0f"),
            'Rend. Atteso (%)': st.column_config.NumberColumn(format="+.1f"),
        }
        st.dataframe(pd.DataFrame(split_rows), width="stretch", hide_index=True, column_config=split_cfg)
        
        if reserve_pct > 0:
            st.caption(f"💡 La riserva (€{budget_reserve:,.0f}) viene deployata a Q10 durante eventi Black Swan. Se il BS non avviene, resta in cash.")
    
    with strat_tab2:
        st.subheader("📈 Backtest Storico")
        
        st.markdown("""
        **Come avrebbero performato queste strategie nel passato?**
        Per ogni giorno storico simuliamo l'esecuzione della strategia.
        """)
        
        bt_col1, bt_col2 = st.columns(2)
        with bt_col1:
            bt_horizon = st.selectbox("Orizzonte Backtest", [6, 12, 18, 24], index=1, format_func=lambda x: f"{x} mesi")
        with bt_col2:
            bt_start_year = st.selectbox("Inizio Backtest", [2015, 2016, 2017, 2018, 2019, 2020], index=0)
        
        @st.cache_data(ttl=3600)
        def run_backtest(_df_data, qr_010_a, qr_010_b, qr_005_a, qr_005_b, horizon_days, start_year):
            """Backtest entry strategies on historical data"""
            df_bt = _df_data[_df_data['Date'].dt.year >= start_year].copy().reset_index(drop=True)
            
            results = {
                'Lump Sum (Compra Subito)': [],
                'Wait Q10': [],
                'Wait Q05': [],
                'DCA Mensile': [],
                '50% Ora + 50% Q10': [],
            }
            
            # Track execution rates
            exec_count = {'Wait Q10': 0, 'Wait Q05': 0}
            
            dates = []
            
            for i in range(len(df_bt) - horizon_days):
                start_price = df_bt['Close'].iloc[i]
                start_days = df_bt['Days'].iloc[i]
                
                window = df_bt.iloc[i:i+horizon_days+1]
                end_price = window['Close'].iloc[-1]
                min_price = window['Low'].min()
                
                q10_floor = qr_010_a * (start_days ** qr_010_b)
                q05_floor = qr_005_a * (start_days ** qr_005_b)
                
                # Lump Sum
                lump_return = (end_price / start_price - 1) * 100
                results['Lump Sum (Compra Subito)'].append(lump_return)
                
                # Wait Q10: if floor not reached, buy at end (opportunity cost = missed appreciation)
                if min_price <= q10_floor:
                    wait_return = (end_price / q10_floor - 1) * 100
                    exec_count['Wait Q10'] += 1
                else:
                    # Capital stayed in cash — 0% return (cash)
                    wait_return = 0.0
                results['Wait Q10'].append(wait_return)
                
                # Wait Q05
                if min_price <= q05_floor:
                    wait_return = (end_price / q05_floor - 1) * 100
                    exec_count['Wait Q05'] += 1
                else:
                    wait_return = 0.0
                results['Wait Q05'].append(wait_return)
                
                # DCA
                monthly_prices = window['Close'].iloc[::30]
                if len(monthly_prices) > 0:
                    avg_entry = monthly_prices.mean()
                    dca_return = (end_price / avg_entry - 1) * 100
                else:
                    dca_return = lump_return
                results['DCA Mensile'].append(dca_return)
                
                # 50/50 Split
                btc_now = 0.5 / start_price
                if min_price <= q10_floor:
                    btc_q10 = 0.5 / q10_floor
                    total_btc = btc_now + btc_q10
                    final_value = total_btc * end_price
                else:
                    final_value = btc_now * end_price + 0.5
                split_return = (final_value / 1.0 - 1) * 100
                results['50% Ora + 50% Q10'].append(split_return)
                
                dates.append(df_bt['Date'].iloc[i])
            
            return pd.DataFrame(results, index=dates), exec_count, len(dates)
        
        with st.spinner(f"Eseguendo backtest dal {bt_start_year}..."):
            bt_results, bt_exec_count, bt_total = run_backtest(
                df[['Date', 'Close', 'Low', 'Days']],
                qr_models[0.10]['a'], qr_models[0.10]['b'],
                qr_models[0.05]['a'], qr_models[0.05]['b'],
                bt_horizon * 30, bt_start_year
            )
        
        # Show execution rates for Wait strategies
        exec_col1, exec_col2 = st.columns(2)
        with exec_col1:
            q10_exec_pct = bt_exec_count['Wait Q10'] / bt_total * 100 if bt_total > 0 else 0
            st.metric("Wait Q10 - Tasso Esecuzione", f"{q10_exec_pct:.1f}%", 
                      help="% di volte che il floor Q10 è stato raggiunto nell'orizzonte")
        with exec_col2:
            q05_exec_pct = bt_exec_count['Wait Q05'] / bt_total * 100 if bt_total > 0 else 0
            st.metric("Wait Q05 - Tasso Esecuzione", f"{q05_exec_pct:.1f}%",
                      help="% di volte che il floor Q05 è stato raggiunto nell'orizzonte")
        
        st.caption("⚠️ Le strategie Wait mostrano 0% quando il floor non viene raggiunto (capitale resta in cash). Considera il costo opportunità vs Lump Sum.")
        
        st.markdown("#### 📊 Statistiche Backtest")
        
        stats_data = []
        # Approximate annualized risk-free rate (use ~4% for current environment)
        annual_rf = 0.04
        period_rf = annual_rf * (bt_horizon / 12)  # risk-free return over backtest horizon
        
        for strategy in bt_results.columns:
            returns = bt_results[strategy]
            excess = returns - period_rf * 100  # convert to percentage
            sharpe_val = round(excess.mean() / returns.std(), 2) if returns.std() > 0 else None
            stats_data.append({
                'Strategia': strategy,
                'Rend. Medio (%)': round(returns.mean(), 1),
                'Mediana (%)': round(returns.median(), 1),
                'Migliore (%)': round(returns.max(), 1),
                'Peggiore (%)': round(returns.min(), 1),
                'Std Dev (%)': round(returns.std(), 1),
                'Sharpe*': sharpe_val,
                '% Positivi': round((returns > 0).mean() * 100, 1)
            })
        
        bt_col_cfg = {
            'Rend. Medio (%)': st.column_config.NumberColumn(format="+.1f"),
            'Mediana (%)': st.column_config.NumberColumn(format="+.1f"),
            'Migliore (%)': st.column_config.NumberColumn(format="+.1f"),
            'Peggiore (%)': st.column_config.NumberColumn(format="+.1f"),
            'Std Dev (%)': st.column_config.NumberColumn(format=".1f"),
            'Sharpe*': st.column_config.NumberColumn(format=".2f"),
            '% Positivi': st.column_config.NumberColumn(format=".1f"),
        }
        st.dataframe(pd.DataFrame(stats_data), width="stretch", hide_index=True, column_config=bt_col_cfg)
        
        st.markdown("#### 📈 Distribuzione Rendimenti")
        
        fig_bt = go.Figure()
        colors = ['#F7931A', '#dc3545', '#8B0000', '#28a745', '#17a2b8']
        
        for i, strategy in enumerate(bt_results.columns):
            fig_bt.add_trace(go.Box(
                y=bt_results[strategy],
                name=strategy.replace(' ', '\n'),
                marker_color=colors[i % len(colors)],
                boxmean=True
            ))
        
        fig_bt.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        fig_bt.update_layout(
            title=f'Distribuzione Rendimenti ({bt_horizon} mesi, dal {bt_start_year})',
            yaxis_title='Rendimento (%)',
            template='plotly_dark',
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig_bt, width="stretch")
        
        st.markdown("#### 📈 Rendimento Medio nel Tempo")
        
        fig_equity = go.Figure()
        
        for i, strategy in enumerate(bt_results.columns):
            rolling_mean = bt_results[strategy].rolling(60, min_periods=30).mean()
            fig_equity.add_trace(go.Scatter(
                x=bt_results.index,
                y=rolling_mean,
                name=strategy,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig_equity.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        fig_equity.update_layout(
            title='Rendimento Medio Mobile (60 giorni)',
            yaxis_title='Rendimento (%)',
            template='plotly_dark',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_equity, width="stretch")
    
    with strat_tab3:
        st.subheader("🏆 Raccomandazione Personalizzata")
        
        st.markdown(f"""
        ### Il tuo scenario:
        - 💰 **Budget**: €{budget:,}
        - 📅 **Orizzonte**: {horizon_months} mesi
        - 🎯 **Target Price**: ${target_price_2y:,}
        - 📊 **Prezzo Attuale**: ${CURRENT_PRICE:,.0f}
        """)
        
        st.markdown("---")
        
        prob_q10 = prob_df[prob_df['level'] == 'Q10 Floor']['probability'].values[0]
        prob_q05 = prob_df[prob_df['level'] == 'Q05 Floor']['probability'].values[0]
        
        dist_to_q10 = (CURRENT_PRICE / entry_levels['Q10 Floor'] - 1) * 100
        dist_to_q05 = (CURRENT_PRICE / entry_levels['Q05 Floor'] - 1) * 100
        
        # Black Swan risk adjustment
        # Use BS parameters from the sidebar (default 5% prob, -50% impact)
        bs_annual_prob = bs_probability  # already set in sidebar
        horizon_years_strat = horizon_months / 12
        # Probability of at least one BS event in the horizon
        prob_no_bs = (1 - bs_annual_prob) ** horizon_years_strat
        prob_bs_horizon = 1 - prob_no_bs
        
        # Tail-risk adjusted expected value multiplier
        # If BS happens, expected drawdown from entry price
        bs_expected_loss = bs_impact / 100  # e.g. -0.50
        # Adjust target: with prob_bs chance, you lose bs_impact; otherwise target stands
        risk_adj_factor = (1 - prob_bs_horizon) + prob_bs_horizon * (1 + bs_expected_loss * 0.5)  # partial recovery assumed
        
        st.markdown(f"""
        #### 🦢 Fattore di Rischio Black Swan
        - **Prob. BS nell'orizzonte ({horizon_months}m)**: {prob_bs_horizon*100:.1f}%
        - **Impatto BS configurato**: {bs_impact}%
        - **Fattore aggiustamento rischio**: {risk_adj_factor:.2f}x
        """)
        
        st.markdown("---")
        
        # Decision logic (enhanced with BS awareness)
        if dist_to_q10 <= 5:
            rec_emoji = "🟢"
            rec_text = "COMPRA ORA (Lump Sum)"
            rec_reason = f"Il prezzo è già vicino al floor Q10 ({dist_to_q10:+.1f}%). Ottimo punto di ingresso!"
            recommendation = "LUMP_SUM"
        elif dist_to_q10 <= 15 and prob_q10 > 0.4:
            rec_emoji = "🟡"
            rec_text = "50% ORA + 50% A Q10"
            rec_reason = f"Prezzo vicino a Q10 ({dist_to_q10:+.1f}%) con {prob_q10*100:.0f}% prob. di toccarlo."
            recommendation = "SPLIT_Q10"
        elif prob_q10 > 0.5:
            rec_emoji = "🟡"
            rec_text = "50% ORA + 50% A Q10"
            rec_reason = f"Buona probabilità ({prob_q10*100:.0f}%) di raggiungere Q10."
            recommendation = "SPLIT_Q10"
        elif prob_q05 > 0.3:
            rec_emoji = "🟠"
            rec_text = "LADDER (25% per livello)"
            rec_reason = f"Probabilità discreta di dip (Q10: {prob_q10*100:.0f}%, Q05: {prob_q05*100:.0f}%)."
            recommendation = "LADDER"
        else:
            rec_emoji = "🟢"
            rec_text = "COMPRA ORA (Lump Sum)"
            rec_reason = f"Bassa probabilità di dip significativi (Q10: {prob_q10*100:.0f}%)."
            recommendation = "LUMP_SUM"
        
        # If Black Swan risk is high (>15%), suggest more cautious approach
        if prob_bs_horizon > 0.15 and recommendation == "LUMP_SUM":
            rec_emoji = "🟡"
            rec_text = "50% ORA + 50% A Q10 (risk-adjusted)"
            rec_reason += f" ⚠️ Ma il rischio BS è elevato ({prob_bs_horizon*100:.0f}% nell'orizzonte): consigliato split."
            recommendation = "SPLIT_Q10"
        
        st.success(f"## {rec_emoji} {rec_text}")
        st.info(f"**Motivazione**: {rec_reason}")
        
        st.markdown("---")
        st.markdown("### 📋 Piano di Esecuzione")
        
        if recommendation == "LUMP_SUM":
            btc_amount = budget / CURRENT_PRICE
            target_value = btc_amount * target_price_2y * risk_adj_factor
            st.markdown(f"""
            | Azione | Valore | Risk-Adjusted |
            |--------|--------|---------------|
            | 💶 Investi | €{budget:,} | |
            | 💰 Prezzo | ${CURRENT_PRICE:,.0f} | |
            | ₿ BTC | {btc_amount:.6f} | |
            | 🎯 Valore a target | €{btc_amount * target_price_2y:,.0f} | €{target_value:,.0f} |
            | 📈 Rendimento | {(target_price_2y/CURRENT_PRICE - 1)*100:+.1f}% | {(target_value/budget - 1)*100:+.1f}% |
            """)
            
        elif recommendation == "SPLIT_Q10":
            btc_now = (budget * 0.5) / CURRENT_PRICE
            btc_q10 = (budget * 0.5) / entry_levels['Q10 Floor']
            
            st.markdown(f"""
            **FASE 1 - ORA**:
            | | |
            |---|---|
            | 💶 Investi | €{budget*0.5:,.0f} (50%) |
            | 💰 Prezzo | ${CURRENT_PRICE:,.0f} |
            | ₿ BTC | {btc_now:.6f} |
            
            **FASE 2 - SE TOCCA Q10** (${entry_levels['Q10 Floor']:,.0f}):
            | | |
            |---|---|
            | 💶 Investi | €{budget*0.5:,.0f} (50%) |
            | ₿ BTC | {btc_q10:.6f} |
            | 📊 Probabilità | {prob_q10*100:.0f}% |
            
            **Risultato se Q10 raggiunto**: {btc_now + btc_q10:.6f} BTC → €{(btc_now + btc_q10) * target_price_2y:,.0f} (€{(btc_now + btc_q10) * target_price_2y * risk_adj_factor:,.0f} risk-adj.)
            """)
            
        elif recommendation == "LADDER":
            st.markdown("**Piano Ladder (25% per livello)**:")
            ladder_data = []
            for name, level_key in [('Ora', 'Prezzo Attuale'), ('Q20', 'Q20 Floor'), ('Q10', 'Q10 Floor'), ('Q05', 'Q05 Floor')]:
                price = entry_levels[level_key]
                btc = (budget * 0.25) / price
                ladder_data.append({
                    'Livello': name,
                    'Prezzo': f"${price:,.0f}",
                    'Importo': f"€{budget*0.25:,.0f}",
                    'BTC': f"{btc:.6f}"
                })
            st.dataframe(pd.DataFrame(ladder_data), width="stretch", hide_index=True)
        
        # Sensitivity analysis: how recommendation changes with different seeds
        st.markdown("---")
        st.markdown("### 🔀 Sensitivity Analysis (Robustezza)")
        st.caption("Come cambiano le probabilità MC con seed diversi? Se i risultati sono stabili, il modello è robusto.")
        
        sensitivity_seeds = [mc_seed, mc_seed + 1, mc_seed + 7, mc_seed + 42, mc_seed + 100]
        sens_results = []
        
        # Pre-compute return stats once
        log_ret_sens = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        mu_s, sigma_s_all = log_ret_sens.mean(), log_ret_sens.std()
        sigma_n_s = log_ret_sens[log_ret_sens.abs() < 1.5*sigma_s_all].std()
        sigma_s_s = log_ret_sens[log_ret_sens.abs() >= 1.5*sigma_s_all].std()
        n_quick = 2000
        h_days = int(horizon_months * 30.44)
        
        for s_seed in sensitivity_seeds:
            rng_sens = np.random.RandomState(s_seed)
            
            # Chunked quick MC
            CHUNK_S = min(n_quick, 1000)
            all_min_s = []
            rem_s = n_quick
            while rem_s > 0:
                cs = min(CHUNK_S, rem_s)
                rem_s -= cs
                regime_d = rng_sens.random((cs, h_days))
                v = np.where(regime_d < 0.15, sigma_s_s, sigma_n_s)
                sh = rng_sens.normal(mu_s, 1.0, (cs, h_days)) * v
                p = np.full(cs, CURRENT_PRICE)
                min_p = np.full(cs, CURRENT_PRICE)
                for t in range(h_days):
                    p = p * np.exp(sh[:, t])
                    min_p = np.minimum(min_p, p)
                all_min_s.append(min_p)
            
            all_min_concat = np.concatenate(all_min_s)
            reached_q10 = np.sum(all_min_concat <= entry_levels['Q10 Floor'])
            reached_q05 = np.sum(all_min_concat <= entry_levels['Q05 Floor'])
            
            sens_results.append({
                'Seed': s_seed,
                'P(Q10)': f"{reached_q10/n_quick*100:.1f}%",
                'P(Q05)': f"{reached_q05/n_quick*100:.1f}%"
            })
        
        st.dataframe(pd.DataFrame(sens_results), width="stretch", hide_index=True)
        
        st.markdown("---")
        st.error("""
        ⚠️ **ATTENZIONE**: Questa è un'analisi quantitativa basata su modelli statistici.
        - I modelli possono essere sbagliati
        - Il passato non garantisce il futuro
        - Il fattore Black Swan è un'approssimazione, non una previsione
        - **Investi solo ciò che puoi permetterti di perdere**
        """)
    
    st.markdown("---")
    
    # ============================================================================
    # EXIT STRATEGY
    # ============================================================================
    st.header("🚪 Exit Strategy — Power Law Ceiling Models")
    st.markdown("""
    I **quantili superiori** sui prezzi HIGH definiscono livelli di resistenza probabilistici — 
    "soffitti" sopra cui il prezzo si spinge raramente.
    Utili per pianificare **prese di profitto graduali** entro una finestra temporale.
    """)
    
    # Exit parameters
    exit_col1, exit_col2, exit_col3 = st.columns(3)
    with exit_col1:
        exit_horizon_years = st.selectbox("📅 Finestra Exit (anni)", [1, 2, 3, 5, 7, 10], index=2, key="exit_horizon")
    with exit_col2:
        btc_holdings = st.number_input("₿ BTC in portafoglio", min_value=0.0001, max_value=1000.0, value=0.1, step=0.01, format="%.4f")
    with exit_col3:
        exit_strategy_type = st.selectbox("📋 Strategia di uscita", 
                                          ["Graduale (Ladder)", "Target singolo", "Take-Profit progressivo"],
                                          key="exit_strat_type")
    
    # Compute ceiling levels for today and projection
    exit_tab1, exit_tab2, exit_tab3 = st.tabs(["📊 Livelli Ceiling", "🔮 Proiezione Exit", "📋 Piano di Uscita"])
    
    with exit_tab1:
        st.subheader("📊 Livelli Ceiling Attuali")
        st.markdown("Quantile Regression sui **prezzi HIGH** — definisce i soffitti di resistenza probabilistici.")
        
        ceiling_data = []
        for q in upper_quantiles:
            ceil_price = predict_pl(TODAY_DAYS, qr_upper[q]['a'], qr_upper[q]['b'])
            dist_pct = (ceil_price / CURRENT_PRICE - 1) * 100
            gain_pct = (ceil_price / CURRENT_PRICE - 1) * 100
            value_at_ceil = btc_holdings * ceil_price
            ceiling_data.append({
                'Ceiling': f"Q{int(q*100)}",
                'Prezzo ($)': round(ceil_price, 0),
                'vs Oggi (%)': round(dist_pct, 1),
                'Valore Portfolio ($)': round(value_at_ceil, 0),
                'Interpretazione': f"Il prezzo supera questo livello solo il {(1-q)*100:.0f}% del tempo"
            })
        
        ceil_col_cfg = {
            'Prezzo ($)': st.column_config.NumberColumn(format="$,.0f"),
            'vs Oggi (%)': st.column_config.NumberColumn(format="+.1f"),
            'Valore Portfolio ($)': st.column_config.NumberColumn(format="$,.0f"),
        }
        st.dataframe(pd.DataFrame(ceiling_data), width="stretch", hide_index=True, column_config=ceil_col_cfg)
        
        # NFA (Never Fallen After) — inverse of NLB for tops
        st.markdown("---")
        st.markdown("### 🔝 ATH Drawdown Analysis — Ceiling di riferimento")
        st.markdown("Analisi dei massimi storici: quanto è durato ogni ciclo sopra certi multipli del fair value?")
        
        # Calculate how often price exceeded PL fair value by different multiples
        df['PL_Fair'] = predict_pl(df['Days'], pl_standard['a'], pl_standard['b'])
        df['Price_to_Fair'] = df['High'] / df['PL_Fair']
        
        overshoot_data = []
        for mult in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            days_above = (df['Price_to_Fair'] >= mult).sum()
            pct_above = days_above / len(df) * 100
            if days_above > 0:
                last_above = df[df['Price_to_Fair'] >= mult]['Date'].iloc[-1].strftime('%Y-%m-%d')
            else:
                last_above = "Mai"
            overshoot_data.append({
                'Multiplo Fair Value': f"{mult}x",
                'Giorni Sopra': days_above,
                '% Tempo': round(pct_above, 1),
                'Ultimo': last_above
            })
        st.dataframe(pd.DataFrame(overshoot_data), width="stretch", hide_index=True)
        st.caption("Più basso il % tempo, più raro e insostenibile quel livello — ottimo punto di vendita.")
    
    with exit_tab2:
        st.subheader(f"🔮 Proiezione Ceiling ({exit_horizon_years}Y)")
        
        exit_days = exit_horizon_years * 365
        exit_periods = {}
        for y in range(1, exit_horizon_years + 1):
            exit_periods[f'{y}Y'] = y * 365
        
        exit_proj_data = {'Ceiling': []}
        for p in exit_periods.keys():
            exit_proj_data[p] = []
        
        for q in upper_quantiles:
            exit_proj_data['Ceiling'].append(f"Q{int(q*100)}")
            for pname, days in exit_periods.items():
                val = predict_pl(TODAY_DAYS + days, qr_upper[q]['a'], qr_upper[q]['b'])
                exit_proj_data[pname].append(round(val, 0))
        
        # Add PL Fair Value row
        exit_proj_data['Ceiling'].append('PL Fair')
        for pname, days in exit_periods.items():
            val = predict_pl(TODAY_DAYS + days, pl_standard['a'], pl_standard['b'])
            exit_proj_data[pname].append(round(val, 0))
        
        # Format columns
        exit_proj_cfg = {}
        for p in exit_periods.keys():
            exit_proj_cfg[p] = st.column_config.NumberColumn(format="$,.0f")
        
        st.dataframe(pd.DataFrame(exit_proj_data), width="stretch", hide_index=True, column_config=exit_proj_cfg)
        
        # Chart: ceiling projection
        fig_exit = go.Figure()
        
        proj_exit = pd.DataFrame({
            'Date': pd.date_range(TODAY, periods=exit_days, freq='D'),
            'Days': np.arange(TODAY_DAYS, TODAY_DAYS + exit_days)
        })
        
        colors_exit = ['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
        for i, q in enumerate(upper_quantiles):
            proj_exit[f'Q{int(q*100)}'] = predict_pl(proj_exit['Days'], qr_upper[q]['a'], qr_upper[q]['b'])
            fig_exit.add_trace(go.Scatter(
                x=proj_exit['Date'], y=proj_exit[f'Q{int(q*100)}'],
                name=f'Q{int(q*100)} Ceiling',
                line=dict(color=colors_exit[i], width=2, dash='dot' if q >= 0.95 else 'solid')
            ))
        
        # PL Fair
        proj_exit['PL_Fair'] = predict_pl(proj_exit['Days'], pl_standard['a'], pl_standard['b'])
        fig_exit.add_trace(go.Scatter(
            x=proj_exit['Date'], y=proj_exit['PL_Fair'],
            name='PL Fair Value', line=dict(color='white', width=2, dash='dash')
        ))
        
        # Current price line
        fig_exit.add_hline(y=CURRENT_PRICE, line_dash="dot", line_color="gray", 
                          annotation_text=f"Oggi: ${CURRENT_PRICE:,.0f}")
        
        fig_exit.update_layout(
            title=f'Ceiling Projection ({exit_horizon_years}Y)',
            yaxis_title='Prezzo ($)', yaxis_type='log',
            template='plotly_dark', height=500, hovermode='x unified'
        )
        st.plotly_chart(fig_exit, width="stretch")
    
    with exit_tab3:
        st.subheader("📋 Piano di Uscita Personalizzato")
        
        if exit_strategy_type == "Graduale (Ladder)":
            st.markdown("""
            **Ladder Exit**: Vendi una quota fissa a ogni livello di ceiling raggiunto.
            Riduce il rischio di vendere troppo presto o troppo tardi.
            """)
            
            # Use Q80, Q85, Q90, Q95 as exit levels
            exit_levels = [0.80, 0.85, 0.90, 0.95]
            alloc_per_level = 1.0 / len(exit_levels)
            
            ladder_exit_data = []
            total_expected_value = 0
            
            for q in exit_levels:
                # Price at midpoint of exit horizon
                mid_days = TODAY_DAYS + exit_horizon_years * 365 // 2
                ceil_price = predict_pl(mid_days, qr_upper[q]['a'], qr_upper[q]['b'])
                btc_sell = btc_holdings * alloc_per_level
                value = btc_sell * ceil_price
                
                # Prob of reaching ceiling (rough: based on quantile meaning)
                prob_reach = 1 - q  # Q90 means price is above only 10% of time → ~10% prob
                
                total_expected_value += prob_reach * value + (1 - prob_reach) * (btc_sell * CURRENT_PRICE)
                
                ladder_exit_data.append({
                    'Livello': f'Q{int(q*100)}',
                    'Prezzo Target ($)': round(ceil_price, 0),
                    'BTC da Vendere': round(btc_sell, 6),
                    'Valore se Raggiunto ($)': round(value, 0),
                    'Prob. Appross. (%)': round(prob_reach * 100, 0),
                })
            
            exit_ladder_cfg = {
                'Prezzo Target ($)': st.column_config.NumberColumn(format="$,.0f"),
                'BTC da Vendere': st.column_config.NumberColumn(format=".6f"),
                'Valore se Raggiunto ($)': st.column_config.NumberColumn(format="$,.0f"),
                'Prob. Appross. (%)': st.column_config.NumberColumn(format=".0f"),
            }
            st.dataframe(pd.DataFrame(ladder_exit_data), width="stretch", hide_index=True, column_config=exit_ladder_cfg)
            
            st.info(f"**Valore atteso totale** (ponderato per probabilità): ${total_expected_value:,.0f}")
            
        elif exit_strategy_type == "Target singolo":
            q_target = st.select_slider("Seleziona Ceiling Target", 
                                        options=[0.80, 0.85, 0.90, 0.95, 0.99],
                                        value=0.90,
                                        format_func=lambda x: f"Q{int(x*100)}")
            
            target_prices = {}
            for y in range(1, exit_horizon_years + 1):
                target_prices[f'{y}Y'] = predict_pl(TODAY_DAYS + y * 365, qr_upper[q_target]['a'], qr_upper[q_target]['b'])
            
            st.markdown(f"**Target: Q{int(q_target*100)}** — Il prezzo supera questo livello solo il {(1-q_target)*100:.0f}% del tempo.")
            
            target_data = []
            for period, price in target_prices.items():
                value = btc_holdings * price
                gain = (price / CURRENT_PRICE - 1) * 100
                target_data.append({
                    'Periodo': period,
                    'Target ($)': round(price, 0),
                    'Gain (%)': round(gain, 1),
                    'Valore Portfolio ($)': round(value, 0),
                })
            
            target_cfg = {
                'Target ($)': st.column_config.NumberColumn(format="$,.0f"),
                'Gain (%)': st.column_config.NumberColumn(format="+.1f"),
                'Valore Portfolio ($)': st.column_config.NumberColumn(format="$,.0f"),
            }
            st.dataframe(pd.DataFrame(target_data), width="stretch", hide_index=True, column_config=target_cfg)
            
        elif exit_strategy_type == "Take-Profit progressivo":
            st.markdown("""
            **Take-Profit Progressivo**: Vendi quote crescenti man mano che il prezzo sale.
            Conservativo all'inizio, aggressivo verso i top.
            """)
            
            tp_levels = [
                (0.80, 0.10, "Primo take-profit — 10%"),
                (0.85, 0.15, "Momentum — 15%"),
                (0.90, 0.25, "Euforia — 25%"),
                (0.95, 0.30, "Top potenziale — 30%"),
                (0.99, 0.20, "Blow-off top — 20% residuo"),
            ]
            
            tp_data = []
            remaining_btc = btc_holdings
            mid_days = TODAY_DAYS + exit_horizon_years * 365 // 2
            
            for q, pct, note in tp_levels:
                ceil_price = predict_pl(mid_days, qr_upper[q]['a'], qr_upper[q]['b'])
                btc_sell = btc_holdings * pct
                value = btc_sell * ceil_price
                remaining_btc -= btc_sell
                
                tp_data.append({
                    'Ceiling': f'Q{int(q*100)}',
                    'Vendita (%)': round(pct * 100, 0),
                    'BTC Venduti': round(btc_sell, 6),
                    'Prezzo ($)': round(ceil_price, 0),
                    'Incasso ($)': round(value, 0),
                    'BTC Residui': round(max(remaining_btc, 0), 6),
                    'Note': note,
                })
            
            tp_cfg = {
                'Vendita (%)': st.column_config.NumberColumn(format=".0f"),
                'BTC Venduti': st.column_config.NumberColumn(format=".6f"),
                'Prezzo ($)': st.column_config.NumberColumn(format="$,.0f"),
                'Incasso ($)': st.column_config.NumberColumn(format="$,.0f"),
                'BTC Residui': st.column_config.NumberColumn(format=".6f"),
            }
            st.dataframe(pd.DataFrame(tp_data), width="stretch", hide_index=True, column_config=tp_cfg)
            
            total_income = sum(r['Incasso ($)'] for r in tp_data)
            st.info(f"**Incasso totale stimato**: ${total_income:,.0f} | **BTC residui** (HODL forever): {max(remaining_btc, 0):.6f}")
        
        st.markdown("---")
        st.warning("""
        ⚠️ **I ceiling sono stime statistiche, non garanzie.** 
        BTC potrebbe superare Q99 in un blow-off top, o non raggiungere mai Q80 nel periodo indicato.
        Usali come guida per un piano di exit disciplinato, non come previsioni puntuali.
        """)
    
    st.markdown("---")
    
    # Model Parameters (expandable)
    with st.expander("🔧 Model Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quantile Regression (on LOW prices)**")
            params_data = {
                'Model': ['Q01', 'Q02', 'Q05', 'Q10', 'Q15', 'Q20'],
                'a': [f"{qr_models[q]['a']:.4e}" for q in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]],
                'b (slope)': [f"{qr_models[q]['b']:.4f}" for q in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]],
                'SE(b)': [f"{qr_models[q]['se_slope']:.4f}" for q in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]],
                'p-value': [f"{qr_models[q]['pval_slope']:.2e}" for q in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]]
            }
            st.dataframe(pd.DataFrame(params_data), width="stretch", hide_index=True)
            st.caption("SE = Standard Error dello slope. p-value < 0.05 indica significatività statistica.")
        
        with col2:
            st.markdown("**Power Law (on CLOSE prices)**")
            st.markdown(f"""
            - **a** = {pl_standard['a']:.4e}
            - **b** = {pl_standard['b']:.4f}
            - **R²** = {pl_standard['r2']:.4f}
            - **Residual σ** = {pl_standard['residual_std']:.4f}
            
            Formula: `P = a × Days^b`
            """)
            
            st.markdown("---")
            st.markdown("**Upper Quantile Regression (on HIGH prices) — Ceiling Models**")
            upper_params = {
                'Model': [f'Q{int(q*100)}' for q in upper_quantiles],
                'a': [f"{qr_upper[q]['a']:.4e}" for q in upper_quantiles],
                'b (slope)': [f"{qr_upper[q]['b']:.4f}" for q in upper_quantiles],
                'SE(b)': [f"{qr_upper[q]['se_slope']:.4f}" for q in upper_quantiles],
                'p-value': [f"{qr_upper[q]['pval_slope']:.2e}" for q in upper_quantiles]
            }
            st.dataframe(pd.DataFrame(upper_params), width="stretch", hide_index=True)
    
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
        
        ### Sharpe* Ratio (Backtest)
        
        Lo Sharpe* nel backtest è calcolato come (rendimento medio - risk-free) / deviazione standard,
        dove il risk-free è approssimato al 4% annuo. Non è annualizzato — si riferisce all'orizzonte del backtest.
        
        ### EVT (Extreme Value Theory)
        
        Usa la Generalized Pareto Distribution (GPD) sulle perdite estreme (sopra il 90° percentile).
        Il floor EVT è scalato su un orizzonte di 30 giorni con approx. √t.
        ⚠️ Lo scaling √t è un'approssimazione — BTC ha autocorrelazione e code pesanti.
        
        ### NLB (Never Look Back)
        
        Il NLB degli ultimi ~6 mesi è provvisorio: non è ancora confermato che quei prezzi 
        non verranno rivisitati in futuro.
        
        ### Exit Strategy — Ceiling Models
        
        I modelli di ceiling usano Quantile Regression sui prezzi **HIGH** (massimi giornalieri)
        ai quantili superiori (80°, 85°, 90°, 95°, 99°).
        
        - **Q80**: Prezzo superato il 20% del tempo → punto di vendita conservativo
        - **Q90**: Superato solo il 10% → zona di take-profit aggressivo
        - **Q95-Q99**: Territorio blow-off top → vendita delle ultime posizioni
        
        L'analisi del multiplo rispetto al Fair Value (Price/PL_Fair) fornisce un ulteriore 
        indicatore: storicamente, BTC raramente rimane sopra 3x il fair value per più di pochi giorni.
        """)
    
    # ============================================================================
    # PDF EXPORT FOR AI ANALYSIS
    # ============================================================================
    st.markdown("---")
    st.header("📄 Esporta Report per Analisi AI")
    st.markdown("Genera un PDF strutturato contenente tutti i dati e un prompt professionale da dare in pasto a un'AI per ottenere un'analisi completa.")
    
    def generate_ai_report_pdf():
        """Generate a comprehensive PDF report structured as an AI analysis prompt"""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.units import mm
        
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=15*mm, leftMargin=15*mm, rightMargin=15*mm)
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle('SectionHead', parent=styles['Heading2'], textColor=colors.HexColor('#F7931A'), spaceAfter=6))
        styles.add(ParagraphStyle('DataBlock', parent=styles['Code'], fontSize=8, leading=10, spaceAfter=4))
        styles.add(ParagraphStyle('Prompt', parent=styles['Normal'], fontSize=9, leading=12, backColor=colors.HexColor('#f0f0f0'), borderPadding=8, spaceAfter=8))
        styles.add(ParagraphStyle('SmallNote', parent=styles['Normal'], fontSize=7, textColor=colors.HexColor('#666666')))
        
        story = []
        
        # Title
        story.append(Paragraph("BITCOIN FLOOR ANALYSIS - AI ANALYSIS REPORT", styles['Title']))
        story.append(Paragraph(f"Generated: {TODAY.strftime('%Y-%m-%d %H:%M UTC')} | BTC Price: ${CURRENT_PRICE:,.0f}", styles['Normal']))
        story.append(Spacer(1, 8))
        
        # AI PROMPT SECTION
        story.append(Paragraph("PROMPT PER ANALISI AI", styles['SectionHead']))
        
        prompt_text = (
            "You are a quantitative finance analyst specializing in cryptocurrency markets. "
            "Below is a comprehensive dataset from a Bitcoin Floor Analysis tool that uses Power Law regression, "
            "Quantile Regression on historical LOW and HIGH prices, Monte Carlo simulations with regime-switching "
            "volatility, and Extreme Value Theory (EVT). "
            "<br/><br/>"
            "Please analyze this data and provide:<br/>"
            "1. <b>Current Market Position</b>: Where is BTC relative to its floor/ceiling models? Is it cheap, fair, or expensive?<br/>"
            "2. <b>Entry Strategy Assessment</b>: Given the Monte Carlo probabilities, which entry strategy offers the best risk-adjusted expected value? Consider the reserve fund allocation for Black Swan events.<br/>"
            "3. <b>Exit Strategy Assessment</b>: Based on the ceiling models, what is the optimal exit ladder for the given time horizon?<br/>"
            "4. <b>Risk Analysis</b>: What are the key risks the models may be underestimating? How does the halving cycle phase affect the outlook?<br/>"
            "5. <b>Model Limitations</b>: Identify potential weaknesses in the methodology.<br/>"
            "6. <b>Actionable Recommendations</b>: Provide specific, time-bound recommendations with confidence levels.<br/>"
            "<br/>Base your analysis ONLY on the data provided below. Do not hallucinate or assume data not present."
        )
        story.append(Paragraph(prompt_text, styles['Prompt']))
        story.append(Spacer(1, 6))
        
        # MARKET DATA
        story.append(Paragraph("1. MARKET SNAPSHOT", styles['SectionHead']))
        market_data = [
            ["Metric", "Value"],
            ["Current Price (USD)", f"${CURRENT_PRICE:,.0f}"],
            ["Date", TODAY.strftime('%Y-%m-%d')],
            ["Days Since Genesis (2009-01-03)", f"{TODAY_DAYS:,}"],
            ["NLB Floor (Never Look Back)", f"${current_nlb:,.0f}"],
            ["PL Fair Value", f"${pl_fair:,.0f}"],
            ["PL Lower Band (-1 sigma)", f"${pl_lower:,.0f}"],
            ["PL Upper Band (+1 sigma)", f"${pl_upper:,.0f}"],
            ["Price / PL Fair Value", f"{CURRENT_PRICE/pl_fair:.2f}x"],
        ]
        t = Table(market_data, colWidths=[180, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F7931A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        
        # HALVING CYCLE
        story.append(Paragraph("2. HALVING CYCLE", styles['SectionHead']))
        halving_data = [
            ["Metric", "Value"],
            ["Last Halving", last_halving.strftime('%Y-%m-%d')],
            ["Days Since Halving", f"{days_since_halving}"],
            ["Next Halving (est.)", NEXT_HALVING_EST.strftime('%Y-%m')],
            ["Cycle Progress", f"{cycle_progress:.1f}%"],
        ]
        if cycle_progress < 25:
            halving_data.append(["Phase", "Accumulation Post-Halving (historically bullish)"])
        elif cycle_progress < 50:
            halving_data.append(["Phase", "Bull Run (historically rapid growth)"])
        elif cycle_progress < 75:
            halving_data.append(["Phase", "Cycle Maturity (possible tops, increasing caution)"])
        else:
            halving_data.append(["Phase", "Late Cycle / Pre-Halving (historically consolidation)"])
        t2 = Table(halving_data, colWidths=[180, 200])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F7931A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ]))
        story.append(t2)
        story.append(Spacer(1, 8))
        
        # FLOOR MODELS
        story.append(Paragraph("3. FLOOR MODELS (Quantile Regression on LOG(LOW) prices)", styles['SectionHead']))
        floor_table = [["Quantile", "Floor Price (USD)", "Distance from Current", "Slope (b)", "SE(b)", "p-value"]]
        for q in quantiles:
            qm = qr_models[q]
            floor_p = predict_pl(TODAY_DAYS, qm['a'], qm['b'])
            dist = (CURRENT_PRICE / floor_p - 1) * 100
            floor_table.append([
                f"Q{int(q*100):02d}",
                f"${floor_p:,.0f}",
                f"+{dist:.1f}% above",
                f"{qm['b']:.4f}",
                f"{qm['se_slope']:.4f}",
                f"{qm['pval_slope']:.2e}"
            ])
        t3 = Table(floor_table, colWidths=[55, 85, 85, 60, 60, 65])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ]))
        story.append(t3)
        story.append(Spacer(1, 8))
        
        # CEILING MODELS
        story.append(Paragraph("4. CEILING MODELS (Quantile Regression on LOG(HIGH) prices)", styles['SectionHead']))
        ceil_table = [["Quantile", "Ceiling Price (USD)", "Distance from Current", "Slope (b)", "Interpretation"]]
        for q in upper_quantiles:
            qm = qr_upper[q]
            ceil_p = predict_pl(TODAY_DAYS, qm['a'], qm['b'])
            dist = (ceil_p / CURRENT_PRICE - 1) * 100
            ceil_table.append([
                f"Q{int(q*100)}",
                f"${ceil_p:,.0f}",
                f"{dist:+.1f}%",
                f"{qm['b']:.4f}",
                f"Price exceeds this only {(1-q)*100:.0f}% of time"
            ])
        t4 = Table(ceil_table, colWidths=[55, 95, 80, 60, 150])
        t4.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (3, -1), 'RIGHT'),
        ]))
        story.append(t4)
        story.append(Spacer(1, 8))
        
        # POWER LAW MODEL
        story.append(Paragraph("5. POWER LAW OLS MODEL (on CLOSE prices)", styles['SectionHead']))
        pl_text = (
            f"Formula: Price = a * Days^b<br/>"
            f"a = {pl_standard['a']:.6e}, b = {pl_standard['b']:.4f}, "
            f"R-squared = {pl_standard['r2']:.4f}, Residual sigma = {pl_standard['residual_std']:.4f}<br/>"
            f"Fair Value today = ${pl_fair:,.0f}, "
            f"Lower band (-1 sigma) = ${pl_lower:,.0f}, "
            f"Upper band (+1 sigma) = ${pl_upper:,.0f}"
        )
        story.append(Paragraph(pl_text, styles['DataBlock']))
        story.append(Spacer(1, 6))
        
        # CEILING PROJECTIONS
        story.append(Paragraph("6. CEILING PROJECTIONS (1-5 Year)", styles['SectionHead']))
        proj_table = [["Ceiling"] + [f"+{y}Y" for y in range(1, 6)]]
        for q in upper_quantiles:
            row = [f"Q{int(q*100)}"]
            for y in range(1, 6):
                val = predict_pl(TODAY_DAYS + y * 365, qr_upper[q]['a'], qr_upper[q]['b'])
                row.append(f"${val:,.0f}")
            proj_table.append(row)
        # Add PL Fair
        row_fair = ["PL Fair"]
        for y in range(1, 6):
            val = predict_pl(TODAY_DAYS + y * 365, pl_standard['a'], pl_standard['b'])
            row_fair.append(f"${val:,.0f}")
        proj_table.append(row_fair)
        
        t5 = Table(proj_table, colWidths=[55] + [75]*5)
        t5.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17a2b8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ]))
        story.append(t5)
        story.append(Spacer(1, 8))
        
        story.append(PageBreak())
        
        # FLOOR PROJECTIONS
        story.append(Paragraph("7. FLOOR PROJECTIONS (1-5 Year)", styles['SectionHead']))
        floor_proj_table = [["Floor"] + [f"+{y}Y" for y in range(1, 6)]]
        for q in quantiles:
            row = [f"Q{int(q*100):02d}"]
            for y in range(1, 6):
                val = predict_pl(TODAY_DAYS + y * 365, qr_models[q]['a'], qr_models[q]['b'])
                row.append(f"${val:,.0f}")
            floor_proj_table.append(row)
        
        t6 = Table(floor_proj_table, colWidths=[55] + [75]*5)
        t6.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ]))
        story.append(t6)
        story.append(Spacer(1, 8))
        
        # ENTRY STRATEGY DATA
        story.append(Paragraph(f"8. ENTRY STRATEGY ANALYSIS (Budget: EUR {budget:,.0f}, Horizon: {horizon_months}m, Target: ${target_price_2y:,.0f})", styles['SectionHead']))
        story.append(Paragraph(f"Reserve: {reserve_pct*100:.0f}% (EUR {budget_reserve:,.0f}) | BS Prob: {es_bs_prob*100:.0f}% | BS Impact: {es_bs_impact*100:.0f}%", styles['DataBlock']))
        
        entry_table = [["Level", "Entry $", "Discount", "Prob.", "via BS", "BTC", "EV (EUR)", "Exp. Ret."]]
        for row in strategy_rows:
            entry_table.append([
                row['Livello'],
                f"${row['Prezzo Entry ($)']:,.0f}",
                f"{row['Sconto (%)']:+.1f}%",
                f"{row['Prob. (%)']:.1f}%",
                f"{row.get('via BS (%)', 0):.1f}%",
                f"{row['BTC se eseguito']:.6f}",
                f"EUR {row['Valore Atteso (€)']:,.0f}",
                f"{row['Rend. Atteso (%)']:+.1f}%",
            ])
        t7 = Table(entry_table, colWidths=[65, 60, 50, 40, 40, 55, 65, 50])
        t7.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6f42c1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ]))
        story.append(t7)
        story.append(Spacer(1, 8))
        
        # METHODOLOGY NOTES
        story.append(Paragraph("9. METHODOLOGY NOTES FOR AI", styles['SectionHead']))
        method_text = (
            "<b>Quantile Regression</b>: Fits specific percentiles of log(price) vs log(days) using Huber sandwich SE. "
            "Lower quantiles (Q01-Q20) fitted on daily LOW prices define floors. "
            "Upper quantiles (Q80-Q99) fitted on daily HIGH prices define ceilings.<br/><br/>"
            "<b>Monte Carlo</b>: Regime-switching model with normal (85%) and stress (15%) volatility regimes. "
            "Soft floor bounce at 85% of power law floor (70% bounce probability, 30% pass-through). "
            "Black Swan events: single shock within first half of horizon. Chunked vectorization for memory efficiency.<br/><br/>"
            "<b>EVT</b>: Generalized Pareto Distribution on losses exceeding 90th percentile. "
            "30-day VaR scaled via sqrt(t) approximation (known limitation for autocorrelated assets).<br/><br/>"
            "<b>NLB (Never Look Back)</b>: Highest price that was never revisited. Last ~6 months are provisional.<br/><br/>"
            "<b>Known Limitations</b>: MC assumes i.i.d. returns (no momentum/mean-reversion). "
            "Regime threshold (1.5 sigma) is fixed, not GARCH-adaptive. "
            "Power Law may break down if Bitcoin adoption dynamics fundamentally change. "
            "Floor models are backward-looking; structural breaks could invalidate them."
        )
        story.append(Paragraph(method_text, styles['SmallNote']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(
            "This report was auto-generated by BTC Floor Analysis tool. All models are statistical estimates, not predictions. "
            "Past performance does not guarantee future results. This is NOT financial advice.",
            styles['SmallNote']
        ))
        
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    
    if st.button("📄 Genera Report PDF per AI", type="primary", use_container_width=True):
        with st.spinner("Generando PDF..."):
            try:
                pdf_bytes = generate_ai_report_pdf()
                st.download_button(
                    label="⬇️ Scarica Report PDF",
                    data=pdf_bytes,
                    file_name=f"BTC_Floor_Analysis_{TODAY.strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ Report generato! Clicca il bottone sopra per scaricarlo.")
            except Exception as e:
                st.error(f"Errore nella generazione del PDF: {e}")
                st.info("Assicurati che `reportlab` sia nel requirements.txt")
    
    st.caption("Il PDF contiene un prompt strutturato e tutti i dati necessari. Può essere dato in pasto a Claude, GPT-4, o qualsiasi LLM per ottenere un'analisi finanziaria approfondita.")
    
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
