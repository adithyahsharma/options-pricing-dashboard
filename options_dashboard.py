import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.interpolate import CubicSpline
from datetime import datetime


@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry):
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)
    return chain.calls, chain.puts


@st.cache_data(ttl=300)
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.options, stock.info['regularMarketPrice']


def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    call = S * nd1 - K * np.exp(-r * T) * nd2
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put


def delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    nd1 = norm.cdf(d1)
    delta_call = nd1
    delta_put = nd1 - 1
    return delta_call, delta_put


def gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    nd1 = norm.pdf(d1)
    gamma_result = nd1 / (S * sigma * np.sqrt(T))
    return gamma_result


def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    nd1 = norm.pdf(d1)
    vega_result = (S * nd1 * np.sqrt(T)) / 100
    return vega_result


def theta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    nd2 = norm.cdf(d2)
    theta_call = ((-S * nd1 * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * nd2) / 365
    theta_put = ((-S * nd1 * sigma / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return theta_call, theta_put


def rho(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd2 = norm.cdf(d2)
    rho_call = (K * T * np.exp(-r * T) * nd2) / 100
    rho_put = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
    return rho_call, rho_put


st.set_page_config(layout="wide")
st.title("Options Pricing Dashboard")

S = st.slider("Stock Price (S)", min_value=10.0, max_value=1000.0, value=200.0, step=1.0)
K = st.slider("Strike Price (K)", min_value=10.0, max_value=1000.0, value=200.0, step=1.0)
T = st.slider("Time to Expiry (T)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
r = st.slider("Risk-Free Rate (r)", min_value=0.01, max_value=0.15, value=0.05, step=0.01)
sigma = st.slider("Volatility (sigma)", min_value=0.05, max_value=1.0, value=0.30, step=0.01)


call, put = black_scholes(S, K, T, r, sigma)
delta_call, delta_put = delta(S, K, T, r, sigma)
gamma_value = gamma(S, K, T, r, sigma)
vega_value = vega(S, K, T, r, sigma)
theta_call, theta_put = theta(S, K, T, r, sigma)
rho_call, rho_put = rho(S, K, T, r, sigma)


st.subheader("Option Prices")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Call Price", f"${call:.2f}")
with col2:
    st.metric("Put Price", f"${put:.2f}")

st.write("---")

st.subheader("Greeks")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Delta Call", f"{delta_call:.4f}")
    st.metric("Delta Put", f"{delta_put:.4f}")
with col2:
    st.metric("Gamma", f"{gamma_value:.4f}")
    st.metric("Vega", f"{vega_value:.4f}")
with col3:
    st.metric("Theta Call", f"{theta_call:.4f}")
    st.metric("Theta Put", f"{theta_put:.4f}")
with col4:
    st.metric("Rho Call", f"{rho_call:.4f}")
    st.metric("Rho Put", f"{rho_put:.4f}")


st.write("---")


st.subheader("Signal Analysis")

if S < K:
    call_option = "OTM"
    put_option = "ITM"
    moneyness = "Call is OTM, Put is ITM"
elif abs(S - K) / K < 0.01:
    call_option = "ATM"
    put_option = "ATM"
    moneyness = "Both ATM"
elif S > K:
    call_option = "ITM"
    put_option = "OTM"
    moneyness = "Call is ITM, Put is OTM"

CallIntrinsicValue = max(S - K, 0)
PutIntrinsicValue = max(K - S, 0)
CallTimeValue = call - CallIntrinsicValue
PutTimeValue = put - PutIntrinsicValue


st.write(f"**Call Intrinsic Value:** \${CallIntrinsicValue:.2f} | **Call Time Value:** \${CallTimeValue:.2f}")
st.write(f"**Put Intrinsic Value:** \${PutIntrinsicValue:.2f} | **Put Time Value:** \${PutTimeValue:.2f}")


st.write("")
st.write("")


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"**Delta (Call):** A \$1.00 move in the stock moves the call price by \${delta_call:.2f}")
    st.markdown(f"**Delta (Put):** A \$1.00 move in the stock moves the put price by -\${abs(delta_put):.2f}")
with col2:
    st.markdown(f"**Gamma:** A \$1.00 move in the stock changes Delta by {gamma_value:.4f}")
    st.markdown(f"**Vega:** A 1% move in volatility changes the option price by \${vega_value:.4f}")
with col3:
    st.markdown(f"**Theta (Call):** Each day that passes, the call loses \${abs(theta_call):.4f} in value")
    st.markdown(f"**Theta (Put):** Each day that passes, the put loses \${abs(theta_put):.4f} in value")
with col4:
    st.markdown(f"**Rho (Call):** A 1% change in rates moves the call price by \${rho_call:.2f}")
    st.markdown(f"**Rho (Put):** A 1% change in rates moves the put price by -\${abs(rho_put):.2f}")

st.write("")
st.write("")

if call_option == "ITM":
    st.markdown(f"This call option is **ITM**, with \${CallIntrinsicValue:.2f} of intrinsic value. If exercised right now, the holder would profit \${CallIntrinsicValue:.2f} minus the premium paid.")
elif call_option == "OTM":
    st.markdown(f"This call option is **{call_option}**, with no intrinsic value. Its entire premium is time value.")
else:
    st.markdown(f"This option is **{call_option}**, with no intrinsic value. Its entire premium is time value.")


if put_option == "ITM":
    st.markdown(f"This put option is **ITM**, with \${PutIntrinsicValue:.2f} of intrinsic value. If exercised right now, the holder would profit \${PutIntrinsicValue:.2f} minus the premium paid.")
elif put_option == "OTM":
    st.markdown(f"This put option is **{put_option}**, with no intrinsic value. Its entire premium is time value.")

st.write("")
st.write("")

if delta_call > 0.7:
    direction_text = f"strong directional exposure (Delta of {delta_call:.2f}), behaving similarly to owning the underlying stock"
elif delta_call > 0.3:
    direction_text = f"moderate directional exposure (Delta of {delta_call:.2f})"
else:
    direction_text = f"limited directional exposure (Delta of {delta_call:.2f}), requiring a significant move to become profitable"

theta_text = f"This position loses \${abs(theta_call):.4f} per day to time decay, or approximately \${abs(theta_call)*7:.2f} over a week in eroded value if all else stays constant"

vega_pct = (vega_value / call) * 100

if vega_pct > 5:
    vega_text = f"If implied volatility shifts by 1 percentage point, this option's price would change by \${vega_value:.4f}, about {vega_pct:.1f}% of its current value. This position is highly sensitive to volatility changes"
elif vega_pct > 2:
    vega_text = f"If implied volatility shifts by 1 percentage point, this option's price would change by \${vega_value:.4f}, about {vega_pct:.1f}% of its current value. This position has moderate sensitivity to volatility changes"
else:
    vega_text = f"If implied volatility shifts by 1 percentage point, this option's price would change by \${vega_value:.4f}, about {vega_pct:.1f}% of its current value. This position has relatively low sensitivity to volatility changes"

if T > 1:
    rho_text = f"With {T:.1f} years to expiry, interest rate sensitivity (Rho of {rho_call:.2f}) becomes more meaningful to the position's value"
else:
    rho_text = f"With only {T:.1f} years to expiry, interest rate sensitivity (Rho of {rho_call:.2f}) is relatively minor compared to the other Greeks"

st.markdown(f"Overall, this option has {direction_text}. {theta_text}. {vega_text}. {rho_text}.")


st.write("---")


st.subheader("Greeks Visualization")

S_range = np.linspace(S * 0.5, S * 1.5, 200)

delta_calls = [delta(s, K, T, r, sigma)[0] for s in S_range]
delta_puts = [delta(s, K, T, r, sigma)[1] for s in S_range]
gamma_values = [gamma(s, K, T, r, sigma) for s in S_range]
vega_values = [vega(s, K, T, r, sigma) for s in S_range]
theta_calls = [theta(s, K, T, r, sigma)[0] for s in S_range]
theta_puts = [theta(s, K, T, r, sigma)[1] for s in S_range]
rho_calls = [rho(s, K, T, r, sigma)[0] for s in S_range]
rho_puts = [rho(s, K, T, r, sigma)[1] for s in S_range]


col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(S_range, delta_calls, label="Delta Call")
    ax.plot(S_range, delta_puts, label="Delta Put")
    ax.axvline(x=S, color="green", linestyle="--", label="Current Stock Price")
    ax.axvline(x=K, color="gray", linestyle="--", label="Strike Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Delta")
    ax.set_title("Delta vs Stock Price")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)
with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(S_range, gamma_values, label="Gamma")
    ax.axvline(x=S, color="green", linestyle="--", label="Current Stock Price")
    ax.axvline(x=K, color="gray", linestyle="--", label="Strike Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Gamma")
    ax.set_title("Gamma vs Stock Price")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)
with col3:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(S_range, vega_values, label="Vega")
    ax.axvline(x=S, color="green", linestyle="--", label="Current Stock Price")
    ax.axvline(x=K, color="gray", linestyle="--", label="Strike Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Vega")
    ax.set_title("Vega vs Stock Price")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(S_range, theta_calls, label="Theta Call")
    ax.plot(S_range, theta_puts, label="Theta Put")
    ax.axvline(x=S, color="green", linestyle="--", label="Current Stock Price")
    ax.axvline(x=K, color="gray", linestyle="--", label="Strike Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Theta")
    ax.set_title("Theta vs Stock Price")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)
with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(S_range, rho_calls, label="Rho Call")
    ax.plot(S_range, rho_puts, label="Rho Put")
    ax.axvline(x=S, color="green", linestyle="--", label="Current Stock Price")
    ax.axvline(x=K, color="gray", linestyle="--", label="Strike Price")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Rho")
    ax.set_title("Rho vs Stock Price")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)


st.write("---")


st.subheader("Live Options Data")
st.caption("Note: built on free yfinance data, which has quality limitations. For the cleanest skew, select a liquid expiry 1-3 months out during market hours.")
ticker_input = st.text_input("Enter Ticker", value="SPY")

try:
    expiry_dates, current_price = get_stock_info(ticker_input)
except Exception as e:
    st.error("Unable to fetch live data — Yahoo Finance rate limit reached. Please try again in a few minutes, or run locally for full functionality.")
    st.stop()

expiry_input = st.selectbox("Select Expiry", expiry_dates)
calls, puts = get_options_chain(ticker_input, expiry_input)

calls_clean = calls[(calls['impliedVolatility'] > 0.01) & (calls['impliedVolatility'] < 2.0) & (calls['volume'] > 5) & (calls['strike'] >= current_price)]
puts_clean = puts[(puts['impliedVolatility'] > 0.01) & (puts['impliedVolatility'] < 2.0) & (puts['volume'] > 5) & (puts['strike'] < current_price)]

expiry_date = datetime.strptime(expiry_input, "%Y-%m-%d")
today = datetime.today()
T_expiry = max((expiry_date - today).days / 365, 1/365)

combined = pd.concat([puts_clean, calls_clean]).sort_values('strike')
combined = combined[combined['volume'] > 20]
combined = combined.copy()
combined['iv_median'] = combined['impliedVolatility'].rolling(window=5, center=True, min_periods=1).median()
combined = combined[abs(combined['impliedVolatility'] - combined['iv_median']) < 0.1]
combined = combined.drop(columns=['iv_median'])

cs = CubicSpline(combined['strike'], combined['impliedVolatility'])

strike_smooth = np.linspace(combined['strike'].min(), combined['strike'].max(), 300)
iv_smooth = cs(strike_smooth)
iv_smooth = np.clip(iv_smooth, 0, None)

atm_iv = float(cs(current_price))
low_strike = combined['strike'].quantile(0.1)
high_strike = combined['strike'].quantile(0.9)
low_strike_iv = float(cs(low_strike))
high_strike_iv = float(cs(high_strike))

left_wing_diff = ((low_strike_iv - atm_iv) / atm_iv) * 100
right_wing_diff = ((high_strike_iv - atm_iv) / atm_iv) * 100

if right_wing_diff <= 0:
    shape_text = "The right wing shows no elevation relative to ATM. The skew is concentrated entirely on the downside (put) wing, a strong equity-style skew."
elif left_wing_diff <= 0:
    shape_text = "The left wing shows no elevation relative to ATM. This suggests limited downside hedging demand at this expiry."
else:
    wing_ratio = left_wing_diff / right_wing_diff
    if wing_ratio > 1.5:
        shape_text = f"The left wing (downside) is elevated about {wing_ratio:.1f}x more than the right wing. This skew is generally driven by downside hedging demand."
    elif wing_ratio < 0.67:
        shape_text = f"The right wing (upside) is elevated about {1/wing_ratio:.1f}x more than the left wing. This is typically unusual, and could reflect expectations of a sharp upside move."
    else:
        shape_text = f"Both wings are elevated by roughly similar amounts (ratio of {wing_ratio:.1f}). This results in a smile shape, which can reflect an upcoming event with a huge swing either way."

st.markdown("Black-Scholes assumes a single constant volatility across all strikes. In reality, implied volatility varies by strike, reflecting market demand for protection against large moves. A downward-sloping skew (elevated put-side vol) generally reflects demand for downside protection. A smile (both wings elevated) is more common where large moves in either direction are expected.")

st.markdown(f"**For {ticker_input} at the {expiry_input} expiry:** ATM implied vol is {atm_iv*100:.1f}%. The downside wing is {left_wing_diff:.0f}% higher than ATM, and the upside wing is {right_wing_diff:.0f}% higher than ATM.")

st.markdown(f"**Volatility Shape:** {shape_text}")

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(strike_smooth, iv_smooth, label="Implied Volatility (fitted)", linewidth=1.5)
ax.scatter(combined['strike'], combined['impliedVolatility'], s=5, alpha=0.3, label="Raw IV")
ax.axvline(x=current_price, color="green", linestyle="--", label="Current Stock Price")
ax.set_xlabel("Strike Price")
ax.set_ylabel("Implied Volatility")
ax.set_title(f"Volatility Skew - {ticker_input} {expiry_input}")
ax.legend(fontsize=6)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig, use_container_width=True)


st.write("---")
st.subheader("Market-Implied Greeks")
st.caption("Greeks computed using fitted implied volatility at each strike, rather than Black-Scholes constant vol assumption.")

strike_range = np.linspace(combined['strike'].min(), combined['strike'].max(), 200)
fitted_vols = np.clip(cs(strike_range), 0.01, None)

market_delta_calls = [delta(current_price, k, T_expiry, 0.05, v)[0] for k, v in zip(strike_range, fitted_vols)]
market_delta_puts = [delta(current_price, k, T_expiry, 0.05, v)[1] for k, v in zip(strike_range, fitted_vols)]
market_gamma = [gamma(current_price, k, T_expiry, 0.05, v) for k, v in zip(strike_range, fitted_vols)]
market_vega = [vega(current_price, k, T_expiry, 0.05, v) for k, v in zip(strike_range, fitted_vols)]

col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(strike_range, market_delta_calls, label="Delta Call (market vol)")
    ax.plot(strike_range, market_delta_puts, label="Delta Put (market vol)")
    ax.axvline(x=current_price, color="green", linestyle="--", label="Current Stock Price")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Delta")
    ax.set_title("Market-Implied Delta vs Strike")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)
with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(strike_range, market_gamma, label="Gamma (market vol)")
    ax.axvline(x=current_price, color="green", linestyle="--", label="Current Stock Price")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Gamma")
    ax.set_title("Market-Implied Gamma vs Strike")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)
with col3:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(strike_range, market_vega, label="Vega (market vol)")
    ax.axvline(x=current_price, color="green", linestyle="--", label="Current Stock Price")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Vega")
    ax.set_title("Market-Implied Vega vs Strike")
    ax.legend(fontsize=6)
    st.pyplot(fig, use_container_width=True)