"""
================================================================================
  FASHION RETAIL PRICE OPTIMISER — STREAMLIT DASHBOARD
  FYP Project · Sri Lanka Apparel Retail · Jan 2023 – Dec 2024
--------------------------------------------------------------------------------
  USAGE:
      pip install streamlit pandas numpy plotly
      streamlit run dashboard.py

  Methodology (mirrors FYP-CODES.ipynb):
     Demand Model
       Final_Demand = XGBoost_Base_Demand × (New_Price / Current_Price)^Elasticity

     • XGBoost predicts base demand at the CURRENT price using the product's
       full, consistent feature vector (lags, rolling stats, margins, encoded
       attributes, calendar features). Prices are NOT swapped inside the feature
       vector — that would create out-of-distribution rows.
     • Elasticity (midpoint method, clipped to [-8, 2] in the optimisation loop)
       scales the XGBoost base demand whenever a different candidate price is
       evaluated.
     • Candidate prices follow psychological "x90" anchors (90, 190, …, 990)
       within the business constraints defined in the notebook (Phase 4).

  Required data files (place in the same directory as this script):
     result_df.csv           — per-product optimisation output (Phase 4)
     data_engineered.csv     — monthly engineered features per PATTERN
     model_metrics.json      — (optional) XGBoost R² / MAPE / RMSE / MAE

  Column aliases handled (legacy → current notebook names):
     REVENUE_UPLIFT_%       → REVENUE_IMPROVEMENT_%
     REVENUE_UPLIFT_$       → REVENUE_IMPROVEMENT_$
     PROFIT_IMPROVEMENT_PCT → PROFIT_IMPROVEMENT_%
     PREDICTED_QTY          → PREDICTED_QTY_HYBRID
================================================================================
"""

# ── Standard-library imports ──────────────────────────────────────────────────
import os
import json

# ── Third-party imports ───────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ================================================================
#  BUSINESS-CONSTRAINT CONSTANTS  (mirror notebook Phase 4)
# ================================================================

PRICE_FLOOR_MULT   = 0.55   # Candidate price ≥ current_price × 0.55
PRICE_CEIL_MULT    = 1.35   # Candidate price ≤ current_price × 1.35
COST_MARGIN_MIN    = 1.06   # Candidate price ≥ cost_price  × 1.06  (Cell 95)
COST_MARGIN_MAX    = 2.00   # Candidate price ≤ cost_price  × 2.00
MIN_MARGIN_PCT     = 5.0    # Reject candidates with gross margin < 5 %
MIN_QUANTITY       = 1      # Reject candidates with predicted demand < 1 unit
QTY_DROP_GUARD_PCT = -40.0  # Reject candidates that collapse demand by > 40 %

# Psychological price anchors used in Phase 4 of the notebook
PSYCH_ANCHORS = [90, 190, 290, 390, 490, 590, 690, 790, 890, 990]

# Elasticity clipping bounds used inside the optimisation loop (Cell 98)
ELASTICITY_CLIP = (-8.0, 2.0)


# ================================================================
#  PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Fashion Retail Price Optimiser — FYP",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
#  CUSTOM CSS  (dark / gold luxury aesthetic)
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Base typography & background ── */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #0f0f0f; color: #f0ece4; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #161616;
        border-right: 1px solid #2a2a2a;
    }
    section[data-testid="stSidebar"] * { color: #f0ece4 !important; }

    /* ── Page title / subtitle ── */
    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem; color: #f0ece4;
        letter-spacing: -0.5px; margin-bottom: 0; line-height: 1.1;
    }
    .main-subtitle {
        font-size: 0.95rem; color: #888; margin-top: 4px;
        font-weight: 300; letter-spacing: 0.3px;
    }

    /* ── Generic metric card ── */
    .metric-card {
        background: #1a1a1a; border: 1px solid #2a2a2a;
        border-radius: 12px; padding: 20px 24px; text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #c8a96e; }
    .metric-label {
        font-size: 0.75rem; color: #888; text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 500; margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem; color: #f0ece4; line-height: 1;
    }
    .metric-value.gold  { color: #c8a96e; }
    .metric-value.green { color: #5cb85c; }
    .metric-value.red   { color: #e05c5c; }

    /* ── Optimal-price hero badge ── */
    .optimal-badge {
        background: linear-gradient(135deg, #c8a96e 0%, #e8c98e 100%);
        color: #0f0f0f; font-family: 'DM Serif Display', serif;
        font-size: 3rem; font-weight: 700;
        border-radius: 16px; padding: 28px 36px; text-align: center;
        letter-spacing: -1px; box-shadow: 0 8px 32px rgba(200,169,110,0.25);
    }
    .optimal-label {
        font-family: 'DM Sans', sans-serif; font-size: 0.8rem;
        font-weight: 600; letter-spacing: 2px; text-transform: uppercase;
        margin-bottom: 6px; opacity: 0.7;
    }

    /* ── Section headings ── */
    .section-header {
        font-family: 'DM Serif Display', serif; font-size: 1.3rem;
        color: #f0ece4; border-bottom: 1px solid #2a2a2a;
        padding-bottom: 10px; margin-bottom: 20px;
    }

    /* ── Subtle horizontal rule ── */
    .custom-divider {
        height: 1px; margin: 24px 0;
        background: linear-gradient(90deg, transparent, #2a2a2a, transparent);
    }

    /* ── Native st.metric overrides ── */
    div[data-testid="stMetric"] {
        background: #1a1a1a; border: 1px solid #2a2a2a;
        border-radius: 12px; padding: 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #c8a96e !important;
        font-family: 'DM Serif Display', serif !important;
    }
    div[data-testid="stMetricLabel"] { color: #888 !important; }
    div[data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }

    /* ── Sidebar widget labels ── */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #aaa !important; font-size: 0.8rem !important;
        text-transform: uppercase !important; letter-spacing: 1px !important;
    }

    /* ── Primary (gold) button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #c8a96e, #e8c98e) !important;
        color: #0f0f0f !important; border: none !important;
        border-radius: 8px !important; font-weight: 600 !important;
        font-size: 0.9rem !important; letter-spacing: 0.5px !important;
        padding: 10px 24px !important; width: 100% !important;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.85 !important; }

    /* ── Secondary (ghost) button ── */
    .stButton > button[kind="secondary"] {
        background: transparent !important; color: #888 !important;
        border: 1px solid #333 !important; border-radius: 8px !important;
        font-weight: 400 !important; font-size: 0.8rem !important;
        padding: 6px 16px !important; width: 100% !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #c8a96e !important; color: #c8a96e !important;
    }

    /* ── Active-filter chip row ── */
    .tag-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    .tag {
        background: #222; border: 1px solid #333; color: #c8a96e;
        font-size: 0.78rem; padding: 4px 10px; border-radius: 20px;
        font-weight: 500;
    }

    /* ── Informational callout box ── */
    .info-box {
        background: #1a1a1a; border-left: 3px solid #c8a96e;
        border-radius: 0 8px 8px 0; padding: 14px 18px;
        font-size: 0.88rem; color: #aaa; margin: 12px 0;
    }

    /* ── Methodology / formula panel ── */
    .method-box {
        background: #1a1a1a; border: 1px solid #2a2a2a;
        border-radius: 12px; padding: 18px 22px; margin: 0 0 20px 0;
    }
    .method-box .title {
        font-family: 'DM Serif Display', serif;
        color: #c8a96e; font-size: 1rem; margin-bottom: 8px;
        letter-spacing: 0.3px;
    }
    .method-box .formula {
        background: #0f0f0f; border: 1px dashed #333;
        color: #e8c98e; font-family: 'Courier New', monospace;
        padding: 8px 12px; border-radius: 6px;
        font-size: 0.85rem; margin: 8px 0;
    }
    .method-box .caption { color: #888; font-size: 0.82rem; line-height: 1.5; }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161616; border-radius: 10px;
        padding: 4px; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important; color: #888 !important;
        border-radius: 8px !important; font-size: 0.88rem !important;
        font-weight: 500 !important; padding: 8px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a2a2a !important; color: #c8a96e !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ================================================================
#  DATA LOADING  (cached so reloads only happen on file changes)
# ================================================================

@st.cache_data
def load_results():
    """
    Load result_df.csv (or the legacy price_optimisation_results.csv).

    This file is produced by Phase 4 of FYP-CODES.ipynb and contains one row
    per product with the model-chosen optimal price, predicted demand, revenue /
    profit at the optimal, elasticity, and recommendation.

    Legacy column renames are applied so the dashboard works with both old and
    new notebook outputs without any manual pre-processing.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ["result_df.csv", "price_optimisation_results.csv"]:
        path = os.path.join(script_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["PATTERN"] = df["PATTERN"].astype(str)

            # Rename legacy columns so the rest of the dashboard can use a
            # single consistent set of column names (notebook naming convention)
            legacy_renames = {
                "REVENUE_UPLIFT_%":       "REVENUE_IMPROVEMENT_%",
                "REVENUE_UPLIFT_$":       "REVENUE_IMPROVEMENT_$",
                "PROFIT_IMPROVEMENT_PCT": "PROFIT_IMPROVEMENT_%",
                "PREDICTED_QTY":          "PREDICTED_QTY_HYBRID",
            }
            df.rename(
                columns={k: v for k, v in legacy_renames.items() if k in df.columns},
                inplace=True,
            )
            return df
    return None


@st.cache_data
def load_engineered():
    """
    Load data_engineered.csv — monthly feature-engineered history per PATTERN.

    Used in the Product Explorer tab for sales-velocity charts and the
    similar-products comparison. YEAR_MONTH is parsed to datetime so that
    Plotly can render a proper time axis.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "data_engineered.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["PATTERN"]    = df["PATTERN"].astype(str)
    df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])
    return df


@st.cache_data
def load_model_metrics():
    """
    Load optional model_metrics.json saved by the notebook after XGBoost training.

    Expected keys: R2, MAPE_% (or MAPE), RMSE, MAE.
    Returns None silently when the file is absent — the dashboard degrades
    gracefully without the model-performance strip.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "model_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ================================================================
#  NOTEBOOK-ALIGNED HELPERS
# ================================================================

def generate_candidate_prices(current_price, cost_price):
    """
    Mirror of generate_candidate_prices() in Phase 4 of the notebook.

    Builds the set of psychological-anchor prices ("x90" endings) that fall
    within the feasible band:
        [max(current × FLOOR_MULT, cost × MARGIN_MIN),
         min(current × CEIL_MULT,  cost × MARGIN_MAX)]

    If no anchors land in the band, the floor, current, and ceiling prices
    are returned as rounded integers to guarantee at least one candidate.

    Parameters
    ----------
    current_price : float  — active selling price
    cost_price    : float  — unit cost / buy price

    Returns
    -------
    list of int — sorted candidate prices
    """
    if current_price <= 0 or cost_price <= 0 or cost_price >= current_price:
        return []

    price_floor = max(current_price * PRICE_FLOOR_MULT,
                      cost_price    * COST_MARGIN_MIN)
    price_ceil  = min(current_price * PRICE_CEIL_MULT,
                      cost_price    * COST_MARGIN_MAX)

    if price_ceil <= price_floor:
        return []

    # Enumerate every x90 anchor across all relevant 1 000-unit blocks
    candidates = set()
    for block in range(int(price_floor // 1000), int(price_ceil // 1000) + 2):
        for anchor in PSYCH_ANCHORS:
            p = block * 1000 + anchor
            if price_floor <= p <= price_ceil:
                candidates.add(p)

    # Fallback: guarantee at least three candidates when no anchors fit
    if not candidates:
        candidates = {round(price_floor), round(current_price), round(price_ceil)}

    return sorted(candidates)


def predict_demand(candidate_price, current_price, xgb_base_demand, elasticity):
    """
    Mirror of predict_demand() in the notebook.

    Applies point-elasticity scaling to the XGBoost base demand:
        Final_Demand = XGB_base × (candidate_price / current_price) ^ elasticity

    A floor of 1.0 unit is imposed so downstream revenue / profit calculations
    never go negative from rounding.

    Parameters
    ----------
    candidate_price  : float — price to evaluate
    current_price    : float — reference price at which XGBoost was evaluated
    xgb_base_demand  : float — XGBoost demand prediction at current_price
    elasticity       : float — price elasticity (typically negative)

    Returns
    -------
    float — predicted demand at candidate_price
    """
    if candidate_price <= 0 or current_price <= 0 or xgb_base_demand <= 0:
        return xgb_base_demand if xgb_base_demand > 0 else 1.0
    ratio  = candidate_price / current_price
    demand = xgb_base_demand * (ratio ** elasticity)
    return max(1.0, demand)


def check_constraints(candidate_price, cost_price, predicted_qty, base_qty=None):
    """
    Mirror of check_constraints() in the notebook (Phase 4).

    A candidate price is feasible only if ALL of the following hold:
      1. Price > cost  (strictly positive gross margin)
      2. Gross margin % ≥ MIN_MARGIN_PCT (5 %)
      3. Predicted demand ≥ MIN_QUANTITY (1 unit)
      4. Demand drop vs XGBoost base ≤ QTY_DROP_GUARD_PCT (40 %) — optional

    Parameters
    ----------
    candidate_price : float
    cost_price      : float
    predicted_qty   : float — predicted demand at candidate_price
    base_qty        : float | None — XGBoost base demand (enables drop guard)

    Returns
    -------
    bool — True if all constraints pass
    """
    if candidate_price <= cost_price:
        return False
    margin_pct = (candidate_price - cost_price) / candidate_price * 100
    if margin_pct < MIN_MARGIN_PCT:
        return False
    if predicted_qty < MIN_QUANTITY:
        return False
    if base_qty is not None and base_qty > 0:
        drop_pct = (predicted_qty - base_qty) / base_qty * 100
        if drop_pct < QTY_DROP_GUARD_PCT:
            return False
    return True


def compute_candidate_curve(current_price, cost_price, xgb_base_demand, elasticity):
    """
    Build a full candidate-price DataFrame using the notebook's demand formula.

    For every psychological-anchor price in the feasible band, computes:
      • Predicted demand  (predict_demand)
      • Revenue           = price × demand
      • Profit            = (price − cost) × demand
      • Gross margin %
      • Feasibility flag  (check_constraints)

    The profit-maximising feasible row is returned as best_row.

    Parameters
    ----------
    current_price    : float
    cost_price       : float
    xgb_base_demand  : float — XGBoost base demand at current_price
    elasticity       : float — clipped elasticity for this product

    Returns
    -------
    (pd.DataFrame, dict | None) — (candidate table, best feasible row or None)
    """
    candidates = generate_candidate_prices(current_price, cost_price)
    rows = []
    best_profit = -np.inf
    best_row    = None

    for price in candidates:
        pred_qty   = predict_demand(price, current_price, xgb_base_demand, elasticity)
        revenue    = price * pred_qty
        profit     = (price - cost_price) * pred_qty
        margin_pct = (price - cost_price) / price * 100
        feasible   = check_constraints(
            price, cost_price, pred_qty, base_qty=xgb_base_demand
        )

        row = {
            "Candidate Price": price,
            "Predicted QTY":   round(pred_qty,   1),
            "Revenue":         round(revenue,    2),
            "Profit":          round(profit,     2),
            "Margin %":        round(margin_pct, 1),
            "Feasible":        feasible,
        }
        rows.append(row)

        if feasible and profit > best_profit:
            best_profit = profit
            best_row    = row

    return pd.DataFrame(rows), best_row


def simulate_single_price(sim_price, current_price, cost_price,
                          xgb_base_demand, elasticity):
    """
    What-if calculation for any manually chosen price using the demand formula.

    Called by the Live Price Simulator slider in Tab 1. Returns the four KPIs
    shown in the simulator metric row.

    Parameters
    ----------
    sim_price       : int   — slider-chosen price
    current_price   : float — reference price
    cost_price      : float — unit cost
    xgb_base_demand : float — XGBoost base demand at current_price
    elasticity      : float

    Returns
    -------
    (pred_qty, revenue, profit, margin_pct) — all floats
    """
    pred_qty   = predict_demand(sim_price, current_price, xgb_base_demand, elasticity)
    revenue    = sim_price * pred_qty
    profit     = (sim_price - cost_price) * pred_qty
    margin_pct = (sim_price - cost_price) / sim_price * 100 if sim_price > 0 else 0.0
    return pred_qty, revenue, profit, margin_pct


def get_product_history(pattern, eng_df):
    """
    Return chronologically sorted monthly history rows for a single PATTERN.

    Parameters
    ----------
    pattern : str | int — product pattern code
    eng_df  : pd.DataFrame | None — data_engineered.csv

    Returns
    -------
    pd.DataFrame — empty if pattern not found or eng_df is None
    """
    if eng_df is None:
        return pd.DataFrame()
    sub = eng_df[eng_df["PATTERN"] == str(pattern)].copy()
    return sub.sort_values("YEAR_MONTH").reset_index(drop=True)


def get_sale_velocity(history):
    """
    Compute sale-velocity KPIs from a product's monthly history.

    Compares the average of the last 3 months vs the prior 3 months to
    classify trend direction as Accelerating / Stable / Decelerating.

    Parameters
    ----------
    history : pd.DataFrame — output of get_product_history()

    Returns
    -------
    dict with keys:
        current_velocity, trend_direction, trend_pct,
        recent_avg (last 3M mean), older_avg (prior 3M mean)
    """
    if history.empty:
        return {
            "current_velocity": 0,
            "trend_direction": "No data",
            "trend_pct": 0,
            "recent_avg": 0,
            "older_avg": 0,
        }

    qty = history["QTY"].dropna()
    n   = len(qty)
    if n == 0:
        return {
            "current_velocity": 0,
            "trend_direction": "No data",
            "trend_pct": 0,
            "recent_avg": 0,
            "older_avg": 0,
        }

    current_velocity = float(qty.iloc[-1])
    recent = qty.iloc[-3:].mean()   if n >= 3 else qty.mean()
    older  = qty.iloc[-6:-3].mean() if n >= 6 else qty.iloc[: max(1, n // 2)].mean()
    trend_pct = (recent - older) / older * 100 if older > 0 else 0.0

    if   trend_pct >  5: direction = "Accelerating"
    elif trend_pct < -5: direction = "Decelerating"
    else:                direction = "Stable"

    return {
        "current_velocity": current_velocity,
        "trend_direction":  direction,
        "trend_pct":        trend_pct,
        "recent_avg":       recent,
        "older_avg":        older,
    }


def get_similar_patterns(pattern, results_df, n=8):
    """
    Find products similar to the selected pattern.

    Match priority (mirrors the notebook's segmentation logic):
      1. Same FIT + MATERIAL  (returns if ≥ 3 matches)
      2. Same FIT + COLOUR    (returns if ≥ 2 matches)
      3. Same FIT only        (fallback)
      4. Any n products        (last-resort fallback)

    Parameters
    ----------
    pattern    : str — selected pattern code
    results_df : pd.DataFrame — full result_df
    n          : int — maximum number of similar products to return

    Returns
    -------
    pd.DataFrame — up to n rows (excluding the selected pattern itself)
    """
    row = results_df[results_df["PATTERN"] == str(pattern)]
    if row.empty:
        return pd.DataFrame()

    row      = row.iloc[0]
    fit      = row.get("FIT")
    material = row.get("MATERIAL")
    colour   = row.get("COLOUR")

    others = results_df[results_df["PATTERN"] != str(pattern)].copy()

    # Priority 1: same FIT + MATERIAL
    if fit and material:
        mask    = (others.get("FIT", pd.Series()) == fit) & \
                  (others.get("MATERIAL", pd.Series()) == material)
        matched = others[mask]
        if len(matched) >= 3:
            return matched.head(n)

    # Priority 2: same FIT + COLOUR
    if fit and colour:
        mask    = (others.get("FIT", pd.Series()) == fit) & \
                  (others.get("COLOUR", pd.Series()) == colour)
        matched = others[mask]
        if len(matched) >= 2:
            return matched.head(n)

    # Priority 3: same FIT only
    if fit:
        mask = others.get("FIT", pd.Series()) == fit
        return others[mask].head(n)

    # Last resort: return any n products
    return others.head(n)


def dark_chart_layout(title="", height=340):
    """
    Return a Plotly layout dict styled for the dark / gold dashboard theme.

    Parameters
    ----------
    title  : str — chart title (rendered in DM Serif Display)
    height : int — chart height in pixels

    Returns
    -------
    dict — passed directly to fig.update_layout(**dark_chart_layout(...))
    """
    return dict(
        title=dict(
            text=title,
            font=dict(family="DM Serif Display", size=14, color="#f0ece4"),
        ),
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="#aaa", family="DM Sans", size=10),
        xaxis=dict(gridcolor="#2a2a2a", color="#888", tickformat=","),
        yaxis=dict(gridcolor="#2a2a2a", color="#888", tickformat=","),
        legend=dict(bgcolor="#222", bordercolor="#333", font=dict(color="#aaa")),
        hovermode="x unified",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
    )


# ================================================================
#  SESSION STATE  (initialise keys that survive reruns)
# ================================================================
for key, default in [("show_results", False), ("sim_price", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ================================================================
#  LOAD DATA
# ================================================================
results_df    = load_results()
engineered_df = load_engineered()
model_metrics = load_model_metrics()


# ================================================================
#  SIDEBAR
# ================================================================
with st.sidebar:
    # ── Branding header ──────────────────────────────────────────
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-family: DM Serif Display, serif; font-size:1.4rem; color:#f0ece4;'>
            🏷️ Pricing Optimiser
        </div>
        <div style='font-size:0.75rem; color:#666; margin-top:4px;'>
            Sri Lanka Apparel Retail · FYP Project
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Cost configuration ────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#888;text-transform:uppercase;"
        "letter-spacing:1.5px;margin-bottom:8px;'>💰 Cost Configuration</div>",
        unsafe_allow_html=True,
    )
    cost_price = st.number_input(
        "Reference Cost Price (Rs.)",
        min_value=100, max_value=50000, value=800, step=50,
        help=(
            "Used as the fallback cost when browsing the Aggregate view. "
            "Individual product costs are read directly from result_df.csv."
        ),
    )
    cost_range = st.slider(
        "Cost Price Range (Rs.)",
        min_value=100, max_value=50000,
        value=(int(cost_price * 0.8), int(cost_price * 1.2)),
        step=50,
        help="Filter to products whose COST_PRICE falls within this range.",
    )
    st.markdown("---")

    # ── Product attribute filters (cascading) ─────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#888;text-transform:uppercase;"
        "letter-spacing:1.5px;margin-bottom:8px;'>👔 Product Attributes</div>",
        unsafe_allow_html=True,
    )

    selected_filters = {}

    if results_df is not None:
        # Apply cost-range filter first so attribute drop-downs only show
        # values that exist within the selected cost band (cascading UX)
        cost_filtered_df = results_df.copy()
        if "COST_PRICE" in results_df.columns:
            cost_filtered_df = results_df[
                (results_df["COST_PRICE"] >= cost_range[0])
                & (results_df["COST_PRICE"] <= cost_range[1])
            ]

        st.markdown(
            f"<div style='font-size:0.75rem;color:#c8a96e;margin-bottom:10px;'>"
            f"📦 {len(cost_filtered_df):,} products in cost range</div>",
            unsafe_allow_html=True,
        )

        if cost_filtered_df.empty:
            st.markdown(
                "<div style='font-size:0.8rem;color:#888;padding:8px 0;'>"
                "⚠️ No products found in this cost range.</div>",
                unsafe_allow_html=True,
            )
        else:
            feature_icons = {
                "COLLAR":   "👔  Collar",
                "FIT":      "📐  Fit",
                "MATERIAL": "🧵  Material",
                "SLEEVE":   "👕  Sleeve",
                "TEXTURE":  "🔲  Texture",
                "COLOUR":   "🎨  Colour",
            }

            # One-click reset clears all per-attribute session-state keys
            if st.button("✕  Clear All Filters", use_container_width=True):
                for col in feature_icons:
                    st.session_state.pop(f"filter_{col}", None)
                st.rerun()

            # Each selectbox further narrows the dataset for the next widget
            cascading_df = cost_filtered_df.copy()
            for col, label in feature_icons.items():
                if col not in cascading_df.columns:
                    continue
                available = sorted(cascading_df[col].dropna().unique().tolist())
                val = st.selectbox(label, ["All"] + available, key=f"filter_{col}")
                if val != "All":
                    selected_filters[col] = val
                    cascading_df = cascading_df[cascading_df[col] == val]
    else:
        st.warning("No `result_df.csv` found.\nRun Phase 4 of the notebook first.")

    st.markdown("---")

    # ── Demand model settings ─────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#888;text-transform:uppercase;"
        "letter-spacing:1.5px;margin-bottom:8px;'>📈 Demand Model Settings</div>",
        unsafe_allow_html=True,
    )
    use_product_elasticity = st.checkbox(
        "Use per-product elasticity",
        value=True,
        help=(
            "Reads the ELASTICITY column from result_df.csv for the selected "
            "product and clips it to the notebook's optimisation bounds "
            f"[{ELASTICITY_CLIP[0]}, {ELASTICITY_CLIP[1]}]. "
            "Uncheck to override with the manual slider below."
        ),
    )
    if not use_product_elasticity:
        manual_elasticity = st.slider(
            "Manual Elasticity",
            min_value=float(ELASTICITY_CLIP[0]),
            max_value=float(ELASTICITY_CLIP[1]),
            value=-1.5,
            step=0.1,
            help=(
                f"More negative ⟹ demand more sensitive to price changes. "
                f"Notebook clips to [{ELASTICITY_CLIP[0]}, {ELASTICITY_CLIP[1]}]."
            ),
        )
    else:
        manual_elasticity = -1.5  # default; overridden by product value when checkbox is on

    st.markdown("---")

    # Trigger the main content area by setting session state
    if st.button("🔍  Load Model Results", type="primary"):
        st.session_state.show_results = True
        st.session_state.sim_price    = None


# ================================================================
#  MAIN HEADER
# ================================================================
st.markdown("""
<div style='margin-bottom: 20px;'>
    <div class='main-title'>Fashion Retail Price Optimiser</div>
    <div class='main-subtitle'>
        XGBoost + Elasticity · Sri Lanka Apparel Retail (Jan 2023 – Dec 2024)
    </div>
</div>
""", unsafe_allow_html=True)

# ── Methodology panel — explains the demand model to the examiner ─────────────
st.markdown("""
<div class='method-box'>
    <div class='title'>⚙ Methodology — Demand Model</div>
    <div class='formula'>Final_Demand = XGBoost_Base × (New_Price / Current_Price)<sup>ε</sup></div>
    <div class='caption'>
        XGBoost predicts base demand at the <em>current</em> price using each
        product's real feature vector (lags, rolling stats, margins, encoded
        attributes, calendar features). The elasticity <em>ε</em> (midpoint method,
        per-product median, clipped to [−8, 2]) then scales that base whenever
        a different candidate price is evaluated. The optimiser selects the
        psychological-anchor price that maximises profit, subject to a minimum 5 %
        gross margin, a minimum 1-unit demand, and a 40 % demand-drop guardrail.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Optional XGBoost performance strip (only shown when model_metrics.json exists) ──
if model_metrics:
    mp_cols = st.columns(4)
    r2   = model_metrics.get("R2")
    mape = model_metrics.get("MAPE_%") or model_metrics.get("MAPE")
    rmse = model_metrics.get("RMSE")
    mae  = model_metrics.get("MAE")
    if r2   is not None: mp_cols[0].metric("XGBoost R²", f"{r2:.4f}")
    if mape is not None: mp_cols[1].metric("MAPE %",     f"{mape:.2f}%")
    if rmse is not None: mp_cols[2].metric("RMSE",       f"{rmse:.2f}")
    if mae  is not None: mp_cols[3].metric("MAE",        f"{mae:.2f}")

# ── Guard: stop rendering if no results file was found ───────────────────────
if results_df is None:
    st.markdown("""
    <div class='info-box'>
        <strong style='color:#c8a96e;'>No results data found.</strong><br>
        Place <code>result_df.csv</code> in the same directory as this script
        and restart the Streamlit server.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ================================================================
#  APPLY SIDEBAR FILTERS TO MAIN DATAFRAME
# ================================================================
filtered_df = results_df.copy()

# Step 1: apply cost-range filter
if "COST_PRICE" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["COST_PRICE"] >= cost_range[0])
        & (filtered_df["COST_PRICE"] <= cost_range[1])
    ]

# Step 2: apply each selected attribute filter
for col, val in selected_filters.items():
    if col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col] == val]


# ── Active-filter chip display ────────────────────────────────────────────────
col_left, col_right = st.columns([3, 1])
with col_left:
    if selected_filters:
        chips = "<div class='tag-row'>"
        for col, val in selected_filters.items():
            chips += f"<span class='tag'>{col}: {val}</span>"
        chips += (
            f"<span class='tag' style='border-color:#444;color:#888;'>"
            f"{len(filtered_df):,} matched</span></div>"
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='color:#666;font-size:0.85rem;padding:8px 0;'>"
            "No attribute filters — showing all products in cost range</div>",
            unsafe_allow_html=True,
        )
with col_right:
    st.markdown(
        f"<div style='text-align:right;color:#888;font-size:0.85rem;padding:8px 0;'>"
        f"{len(filtered_df):,} / {len(results_df):,} products</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)


# ================================================================
#  TABS
# ================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Price Calculator",
    "📈  Portfolio Analytics",
    "🔎  Product Explorer",
    "📋  Segment Analysis",
])


# ================================================================
#  TAB 1 — PRICE CALCULATOR
#  Shows the model-optimised price for a single product together
#  with a revenue / profit curve and a live what-if simulator.
# ================================================================
with tab1:

    if not st.session_state.show_results or filtered_df.empty:
        # Prompt the user to configure filters and click the sidebar button
        st.markdown("""
        <div class='info-box' style='text-align:center; padding:40px;'>
            <div style='font-size:1.2rem; color:#c8a96e; margin-bottom:8px;'>
                Set your filters and click <strong>Load Model Results</strong>
            </div>
            <div style='color:#666; font-size:0.9rem;'>
                Use the sidebar to configure cost price, product attributes,
                and elasticity — then click the gold button.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Product selector (shown when more than one product matches) ──────
        n_matched = len(filtered_df)
        if n_matched > 1:
            st.markdown(
                f"<div style='background:#1a1a1a;border:1px solid #2a2a2a;"
                f"border-radius:10px;padding:14px 18px;margin-bottom:16px;'>"
                f"<span style='color:#c8a96e;font-size:0.8rem;"
                f"text-transform:uppercase;letter-spacing:1.2px;font-weight:600;'>"
                f"📋 {n_matched:,} products matched — select one to inspect "
                f"its model-optimised price</span></div>",
                unsafe_allow_html=True,
            )
            _product_options = ["— Aggregate (medians) —"] + \
                               filtered_df["PATTERN"].astype(str).tolist()
            _selected_product = st.selectbox(
                "Select Product",
                _product_options,
                key="tab1_product_select",
                label_visibility="collapsed",
            )
            _use_single = _selected_product != "— Aggregate (medians) —"
        else:
            # Only one product matches — skip the selector
            _use_single       = True
            _selected_product = (
                filtered_df["PATTERN"].astype(str).iloc[0] if n_matched == 1 else None
            )

        # ── Resolve the selected product row ────────────────────────────────
        if _use_single and _selected_product:
            _product_row = filtered_df[
                filtered_df["PATTERN"].astype(str) == _selected_product
            ].iloc[0]
        else:
            _product_row = None

        # ── Derive inputs depending on single-product vs aggregate mode ─────
        if _product_row is not None:
            # Single-product path — all values come directly from result_df.csv
            current_price  = float(_product_row.get("CURRENT_PRICE", cost_price * 2.0))
            effective_cost = float(_product_row.get("COST_PRICE", cost_price))

            # Prefer XGBoost base demand; fall back to historical avg monthly qty
            xgb_base = _product_row.get("XGB_BASE_DEMAND", None)
            if xgb_base is None or pd.isna(xgb_base) or float(xgb_base) <= 0:
                xgb_base = float(_product_row.get("CURRENT_QTY", 10.0))
            xgb_base = float(xgb_base)

            # Use the notebook's stored elasticity (clipped) unless overridden
            raw_elas = _product_row.get("ELASTICITY", None)
            if (
                use_product_elasticity
                and raw_elas is not None
                and not pd.isna(raw_elas)
            ):
                elas = float(np.clip(raw_elas, *ELASTICITY_CLIP))
            else:
                elas = manual_elasticity

            # Authoritative optimal values stored by the notebook (Phase 4)
            model_optimal_price = _product_row.get("OPTIMAL_PRICE", None)
            model_optimal_qty   = _product_row.get("PREDICTED_QTY_HYBRID", None)
            model_optimal_rev   = _product_row.get("OPTIMAL_REVENUE", None)
            model_optimal_prof  = _product_row.get("PROFIT_AT_OPTIMAL", None)
            price_change_pct    = _product_row.get("PRICE_CHANGE_%", None)
            recommendation      = _product_row.get("RECOMMENDATION", "")
            elasticity_type     = _product_row.get("ELASTICITY_TYPE", "")

        else:
            # Aggregate path — representative scenario using filtered-set medians
            current_price  = (
                filtered_df["CURRENT_PRICE"].median()
                if "CURRENT_PRICE" in filtered_df.columns
                else cost_price * 2.0
            )
            effective_cost = (
                filtered_df["COST_PRICE"].median()
                if "COST_PRICE" in filtered_df.columns
                else cost_price
            )
            xgb_base_raw = (
                filtered_df["XGB_BASE_DEMAND"].median()
                if "XGB_BASE_DEMAND" in filtered_df.columns
                else filtered_df.get("CURRENT_QTY", pd.Series([10.0])).median()
            )
            xgb_base = float(xgb_base_raw) if not pd.isna(xgb_base_raw) else 10.0

            if use_product_elasticity and "ELASTICITY" in filtered_df.columns:
                elas_median = filtered_df["ELASTICITY"].dropna().median()
                elas = float(np.clip(elas_median, *ELASTICITY_CLIP))
                if pd.isna(elas):
                    elas = manual_elasticity
            else:
                elas = manual_elasticity

            # Re-run the optimiser on the median scenario
            _, best_row = compute_candidate_curve(
                current_price, effective_cost, xgb_base, elas
            )
            if best_row is not None:
                model_optimal_price = best_row["Candidate Price"]
                model_optimal_qty   = best_row["Predicted QTY"]
                model_optimal_rev   = best_row["Revenue"]
                model_optimal_prof  = best_row["Profit"]
            else:
                # Fallback: keep current price if no feasible candidate found
                model_optimal_price = current_price
                model_optimal_qty   = xgb_base
                model_optimal_rev   = current_price * xgb_base
                model_optimal_prof  = (current_price - effective_cost) * xgb_base

            price_change_pct = (
                (model_optimal_price - current_price) / current_price * 100
                if current_price > 0
                else 0
            )
            recommendation = (
                "INCREASE PRICE" if model_optimal_price > current_price
                else "DECREASE PRICE" if model_optimal_price < current_price
                else "MAINTAIN PRICE"
            )
            elasticity_type = "AGGREGATE"

        # ── Guard against invalid / missing upstream values ──────────────────
        if pd.isna(current_price) or current_price <= 0:
            current_price = effective_cost * 2.0
        if pd.isna(xgb_base) or xgb_base <= 0:
            xgb_base = 10.0

        # Optimal gross margin % (used in badge + simulator delta)
        optimal_margin = (
            (model_optimal_price - effective_cost) / model_optimal_price * 100
            if model_optimal_price and model_optimal_price > 0
            else 0
        )

        # ── KPI summary row ─────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Cost Price", f"Rs. {effective_cost:,.0f}")
        with m2:
            ref_label = (
                "Current Price" if _product_row is not None else "Median Current Price"
            )
            st.metric(ref_label, f"Rs. {current_price:,.0f}")
        with m3:
            st.metric(
                "XGBoost Base Demand",
                f"{xgb_base:.0f} units",
                help="XGBoost's predicted monthly demand at the current price.",
            )
        with m4:
            e_help = (
                f"Product's own elasticity (clipped to [{ELASTICITY_CLIP[0]}, "
                f"{ELASTICITY_CLIP[1]}])"
                if use_product_elasticity
                else "Manual override (per-product checkbox is off)."
            )
            st.metric("Elasticity (ε)", f"{elas:.2f}", help=e_help)
        with m5:
            count_label = (
                "1 Product"
                if (_product_row is not None and n_matched > 1)
                else f"{n_matched:,} Products"
            )
            st.metric("Viewing", count_label)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Optimal-price badge (left) + revenue/profit curve (right) ────────
        opt_col, chart_col = st.columns([1, 2])

        # Build the full candidate curve (drives the chart AND the simulator)
        cand_df, _best_row = compute_candidate_curve(
            current_price, effective_cost, xgb_base, elas
        )

        with opt_col:
            rec_color = (
                "#5cb85c" if recommendation == "INCREASE PRICE"
                else "#e05c5c" if recommendation == "DECREASE PRICE"
                else "#c8a96e"
            )
            st.markdown(
                f"""
                <div class='optimal-badge'>
                    <div class='optimal-label'>Model-Optimal Price</div>
                    Rs. {int(model_optimal_price):,}
                </div>
                <div style='margin-top:12px; text-align:center;
                            color:{rec_color}; font-size:0.85rem;
                            letter-spacing:1.5px; font-weight:600;'>
                    {recommendation}
                </div>
                <div style='margin-top:16px; display:flex; gap:10px;'>
                    <div class='metric-card' style='flex:1;'>
                        <div class='metric-label'>Predicted QTY</div>
                        <div class='metric-value'>{float(model_optimal_qty):.0f}</div>
                    </div>
                    <div class='metric-card' style='flex:1;'>
                        <div class='metric-label'>Margin</div>
                        <div class='metric-value green'>{optimal_margin:.1f}%</div>
                    </div>
                </div>
                <div style='margin-top:10px;'>
                    <div class='metric-card'>
                        <div class='metric-label'>Optimal Revenue</div>
                        <div class='metric-value gold'>Rs. {float(model_optimal_rev):,.0f}</div>
                    </div>
                </div>
                <div style='margin-top:10px;'>
                    <div class='metric-card'>
                        <div class='metric-label'>Optimal Profit</div>
                        <div class='metric-value green'>Rs. {float(model_optimal_prof):,.0f}</div>
                    </div>
                </div>
                <div style='margin-top:10px;'>
                    <div class='metric-card'>
                        <div class='metric-label'>Price Change</div>
                        <div class='metric-value {"green" if (price_change_pct or 0) >= 0 else "red"}'>
                            {float(price_change_pct or 0):+.1f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show elasticity class tag when available (not shown in aggregate mode)
            if elasticity_type and elasticity_type != "AGGREGATE":
                st.markdown(
                    f"<div style='margin-top:12px; text-align:center; "
                    f"color:#888; font-size:0.78rem; letter-spacing:1px;'>"
                    f"Elasticity class: <strong style='color:#c8a96e;'>"
                    f"{elasticity_type}</strong></div>",
                    unsafe_allow_html=True,
                )

        with chart_col:
            fig = go.Figure()

            # Split into feasible (plotted fully) and infeasible (ghosted markers)
            feasible_df   = cand_df[cand_df["Feasible"]]
            infeasible_df = cand_df[~cand_df["Feasible"]]

            # Revenue curve — feasible candidates only
            fig.add_trace(go.Scatter(
                x=feasible_df["Candidate Price"],
                y=feasible_df["Revenue"],
                mode="lines+markers",
                name="Revenue",
                line=dict(color="#c8a96e", width=2.5),
                marker=dict(size=5, color="#c8a96e"),
                hovertemplate="Price: Rs. %{x:,}<br>Revenue: Rs. %{y:,.0f}<extra></extra>",
            ))

            # Profit curve — feasible candidates only
            fig.add_trace(go.Scatter(
                x=feasible_df["Candidate Price"],
                y=feasible_df["Profit"],
                mode="lines+markers",
                name="Profit",
                line=dict(color="#5cb85c", width=2.5, dash="dot"),
                marker=dict(size=5, color="#5cb85c"),
                hovertemplate="Price: Rs. %{x:,}<br>Profit: Rs. %{y:,.0f}<extra></extra>",
            ))

            # Infeasible candidates — shown as grey crosses so the user can see
            # why certain prices were excluded by the constraint checks
            if not infeasible_df.empty:
                fig.add_trace(go.Scatter(
                    x=infeasible_df["Candidate Price"],
                    y=infeasible_df["Profit"],
                    mode="markers",
                    name="Infeasible",
                    marker=dict(size=5, color="#555", symbol="x"),
                    hovertemplate=(
                        "Price: Rs. %{x:,}<br>Profit: Rs. %{y:,.0f}<br>"
                        "<i>Fails constraints</i><extra></extra>"
                    ),
                    opacity=0.5,
                ))

            # Star markers pinpointing the optimal price on both curves
            for col_name in ["Revenue", "Profit"]:
                opt_val = cand_df.loc[
                    cand_df["Candidate Price"] == model_optimal_price, col_name
                ]
                if not opt_val.empty:
                    fig.add_trace(go.Scatter(
                        x=[model_optimal_price],
                        y=[opt_val.values[0]],
                        mode="markers",
                        name=f"Optimal ({col_name})",
                        marker=dict(
                            size=13, color="#ff6b6b", symbol="star",
                            line=dict(color="white", width=1.5),
                        ),
                        showlegend=(col_name == "Revenue"),  # avoid duplicate legend entries
                        hovertemplate=(
                            f"<b>OPTIMAL</b><br>Price: Rs. %{{x:,}}<br>"
                            f"{col_name}: Rs. %{{y:,.0f}}<extra></extra>"
                        ),
                    ))

            # Vertical dashed line marking the current (baseline) price
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="#555",
                annotation_text=f"Current: Rs. {int(current_price):,}",
                annotation_font_color="#888",
                annotation_font_size=10,
            )

            layout = dark_chart_layout("Revenue & Profit Curves", 380)
            layout["yaxis"]["title"] = "Rs."
            layout["xaxis"]["title"] = "Candidate Price (Rs.)"
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # ── Live Price Simulator ─────────────────────────────────────────────
        st.markdown(
            "<div class='section-header'>🎛️  Live Price Simulator</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#666;font-size:0.85rem;margin-bottom:16px;'>"
            "Drag the slider to see what any price produces under the "
            "<code>XGB_base × (new/current)^ε</code> model — and how "
            "it compares to the model's profit-maximising choice.</div>",
            unsafe_allow_html=True,
        )

        # Simulator price range = candidate price range (so curves stay comparable)
        if not cand_df.empty:
            sim_min = int(cand_df["Candidate Price"].min())
            sim_max = int(cand_df["Candidate Price"].max())
        else:
            sim_min = int(current_price * 0.7)
            sim_max = int(current_price * 1.3)

        # Initialise slider at the model-optimal price on first load
        sim_default = (
            model_optimal_price
            if st.session_state.sim_price is None
            else st.session_state.sim_price
        )
        sim_default = int(max(sim_min, min(sim_max, sim_default)))

        sim_price = st.slider(
            "Simulated Price (Rs.)",
            min_value=sim_min,
            max_value=sim_max,
            value=sim_default,
            step=100,
            key="sim_slider",
        )
        st.session_state.sim_price = sim_price  # persist across reruns

        # Compute demand-model outputs at the simulated price
        sim_qty, sim_rev, sim_profit, sim_margin = simulate_single_price(
            sim_price, current_price, effective_cost, xgb_base, elas
        )

        # Deltas vs the model-optimal (shown in metric delta labels)
        delta_rev    = sim_rev    - float(model_optimal_rev  or 0)
        delta_profit = sim_profit - float(model_optimal_prof or 0)
        delta_margin = sim_margin - optimal_margin
        delta_qty    = sim_qty    - float(model_optimal_qty  or 0)

        # Check whether the simulated price passes all business constraints
        feasible = check_constraints(sim_price, effective_cost, sim_qty,
                                     base_qty=xgb_base)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Predicted QTY",     f"{sim_qty:.0f}",
                      delta=f"{delta_qty:+.0f} vs optimal")
        with sc2:
            st.metric("Simulated Revenue", f"Rs. {sim_rev:,.0f}",
                      delta=f"Rs. {delta_rev:+,.0f}")
        with sc3:
            st.metric("Simulated Profit",  f"Rs. {sim_profit:,.0f}",
                      delta=f"Rs. {delta_profit:+,.0f}")
        with sc4:
            st.metric("Margin %",          f"{sim_margin:.1f}%",
                      delta=f"{delta_margin:+.1f}%")

        # Warn the user when the slider price violates one or more constraints
        if not feasible:
            st.markdown(
                "<div class='info-box' style='border-left-color:#e05c5c;'>"
                "⚠️ This simulated price fails one or more business constraints "
                f"(min margin {MIN_MARGIN_PCT:.0f} %, or demand drop > "
                f"{abs(QTY_DROP_GUARD_PCT):.0f} % from XGBoost base).</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # ── Full candidate-price table ───────────────────────────────────────
        st.markdown(
            "<div class='section-header'>All Candidate Prices</div>",
            unsafe_allow_html=True,
        )

        # Build a display copy — add a label column and format all numbers
        display_cands = cand_df.copy()
        display_cands.insert(
            0,
            "",
            display_cands["Candidate Price"].apply(
                lambda x: "⭐ OPTIMAL" if x == model_optimal_price
                else ("🎛️ SIM" if x == sim_price else "")
            ),
        )
        display_cands["Feasible"]        = display_cands["Feasible"].map({True: "✓", False: "✕"})
        display_cands["Candidate Price"] = display_cands["Candidate Price"].apply(lambda x: f"Rs. {int(x):,}")
        display_cands["Predicted QTY"]   = display_cands["Predicted QTY"].apply(lambda x: f"{x:.0f}")
        display_cands["Revenue"]         = display_cands["Revenue"].apply(lambda x: f"Rs. {x:,.0f}")
        display_cands["Profit"]          = display_cands["Profit"].apply(lambda x: f"Rs. {x:,.0f}")
        display_cands["Margin %"]        = display_cands["Margin %"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(
            display_cands,
            use_container_width=True,
            hide_index=True,
            height=min(400, len(cand_df) * 38 + 40),
        )

        # Explanation of the selection logic for the examiner
        st.markdown(
            f"<div class='info-box'>"
            f"<strong style='color:#c8a96e;'>How the optimal price is chosen</strong><br>"
            f"For every psychological-anchor price in "
            f"<code>[{PRICE_FLOOR_MULT:.2f}×current, {PRICE_CEIL_MULT:.2f}×current] "
            f"∩ [{COST_MARGIN_MIN:.2f}×cost, {COST_MARGIN_MAX:.2f}×cost]</code>, "
            f"demand is predicted via "
            f"<code>XGB_base × (new/current)^{elas:.2f}</code>. Candidates "
            f"failing min margin {MIN_MARGIN_PCT:.0f} % or causing a "
            f"demand drop > {abs(QTY_DROP_GUARD_PCT):.0f} % vs XGBoost base "
            f"are excluded. The survivor with highest <strong>profit</strong> "
            f"wins (revenue breaks ties)."
            f"</div>",
            unsafe_allow_html=True,
        )


# ================================================================
#  TAB 2 — PORTFOLIO ANALYTICS
#  Aggregate KPIs, price-change distribution, recommendation split,
#  elasticity distribution, profit-improvement scatter, and a
#  parity plot comparing XGBoost base demand vs actual monthly qty.
# ================================================================
with tab2:

    if filtered_df.empty:
        st.markdown(
            "<div class='info-box'>No products match filters. "
            "Broaden your selection.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='section-header'>Portfolio Overview</div>",
            unsafe_allow_html=True,
        )

        # ── Top-level portfolio KPIs ─────────────────────────────────────────
        p1, p2, p3, p4, p5 = st.columns(5)

        avg_current  = filtered_df["CURRENT_PRICE"].mean()  if "CURRENT_PRICE"  in filtered_df.columns else 0
        avg_optimal  = filtered_df["OPTIMAL_PRICE"].mean()  if "OPTIMAL_PRICE"  in filtered_df.columns else 0
        avg_rev_imp  = filtered_df["REVENUE_IMPROVEMENT_%"].mean() if "REVENUE_IMPROVEMENT_%" in filtered_df.columns else 0
        avg_prof_imp = filtered_df["PROFIT_IMPROVEMENT_%"].mean()  if "PROFIT_IMPROVEMENT_%"  in filtered_df.columns else 0
        pct_success  = (
            (filtered_df["OPTIMIZATION_STATUS"] == "SUCCESS").mean() * 100
            if "OPTIMIZATION_STATUS" in filtered_df.columns
            else 0
        )

        with p1: st.metric("Products",           f"{len(filtered_df):,}")
        with p2: st.metric("Avg Current Price",  f"Rs. {avg_current:,.0f}")
        with p3: st.metric("Avg Optimal Price",  f"Rs. {avg_optimal:,.0f}",
                           delta=f"Rs. {avg_optimal - avg_current:+,.0f}")
        with p4: st.metric("Avg Revenue Uplift", f"{avg_rev_imp:+.2f}%")
        with p5: st.metric("Repriced",           f"{pct_success:.0f}%",
                           help="Percentage of products where a better price was found.")

        # ── Portfolio total financial impact strip ────────────────────────────
        if all(c in filtered_df.columns for c in [
            "CURRENT_REVENUE", "OPTIMAL_REVENUE",
            "CURRENT_PROFIT",  "PROFIT_AT_OPTIMAL",
        ]):
            total_cur_rev  = filtered_df["CURRENT_REVENUE"].sum()
            total_opt_rev  = filtered_df["OPTIMAL_REVENUE"].sum()
            total_cur_prof = filtered_df["CURRENT_PROFIT"].sum()
            total_opt_prof = filtered_df["PROFIT_AT_OPTIMAL"].sum()
            rev_uplift     = total_opt_rev  - total_cur_rev
            prof_uplift    = total_opt_prof - total_cur_prof
            rev_pct  = rev_uplift  / total_cur_rev  * 100 if total_cur_rev  > 0 else 0
            prof_pct = prof_uplift / total_cur_prof * 100 if total_cur_prof > 0 else 0

            st.markdown("<br>", unsafe_allow_html=True)
            f1, f2, f3, f4 = st.columns(4)
            with f1: st.metric("Total Current Revenue", f"Rs. {total_cur_rev:,.0f}")
            with f2: st.metric("Total Optimal Revenue", f"Rs. {total_opt_rev:,.0f}",
                               delta=f"Rs. {rev_uplift:+,.0f} ({rev_pct:+.2f}%)")
            with f3: st.metric("Total Current Profit",  f"Rs. {total_cur_prof:,.0f}")
            with f4: st.metric("Total Optimal Profit",  f"Rs. {total_opt_prof:,.0f}",
                               delta=f"Rs. {prof_uplift:+,.0f} ({prof_pct:+.2f}%)")

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # ── Row 1: price-change histogram (left) + recommendation pie (right) ─
        row1_l, row1_r = st.columns(2)

        with row1_l:
            if "PRICE_CHANGE_%" in filtered_df.columns:
                pct_vals = filtered_df["PRICE_CHANGE_%"].dropna()
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=pct_vals, nbinsx=30,
                    marker=dict(color="#c8a96e", line=dict(color="#0f0f0f", width=0.5)),
                    opacity=0.85,
                    hovertemplate="Range: %{x:.1f}%<br>Count: %{y}<extra></extra>",
                ))
                fig_hist.add_vline(x=0, line_dash="dash", line_color="#888")
                fig_hist.add_vline(
                    x=pct_vals.mean(), line_dash="solid", line_color="#ff6b6b",
                    annotation_text=f"Mean {pct_vals.mean():.1f}%",
                    annotation_font_color="#ff6b6b", annotation_font_size=10,
                )
                layout_h = dark_chart_layout("Recommended Price Change Distribution", 320)
                layout_h["xaxis"]["title"] = "Price Change %"
                layout_h["yaxis"]["title"] = "Products"
                fig_hist.update_layout(**layout_h)
                st.plotly_chart(fig_hist, use_container_width=True)

        with row1_r:
            if "RECOMMENDATION" in filtered_df.columns:
                rec_counts = filtered_df["RECOMMENDATION"].value_counts()
                color_map  = {
                    "INCREASE PRICE": "#5cb85c",
                    "MAINTAIN PRICE": "#c8a96e",
                    "DECREASE PRICE": "#e05c5c",
                }
                colors_pie = [color_map.get(lbl, "#888") for lbl in rec_counts.index]

                fig_pie = go.Figure(go.Pie(
                    labels=rec_counts.index,
                    values=rec_counts.values,
                    marker=dict(colors=colors_pie, line=dict(color="#0f0f0f", width=2)),
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>%{value} products (%{percent})<extra></extra>",
                    hole=0.4,
                ))
                layout_p = dark_chart_layout("Recommendation Breakdown", 320)
                layout_p["showlegend"] = False
                fig_pie.update_layout(**layout_p)
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # ── Row 2: elasticity histogram (left) + profit-improvement scatter (right) ─
        row2_l, row2_r = st.columns(2)

        with row2_l:
            if "ELASTICITY" in filtered_df.columns:
                elas_vals = filtered_df["ELASTICITY"].dropna()
                fig_elas  = go.Figure()
                fig_elas.add_trace(go.Histogram(
                    x=elas_vals, nbinsx=30,
                    marker=dict(color="#7b68ee", line=dict(color="#0f0f0f", width=0.5)),
                    opacity=0.85,
                    hovertemplate="Elasticity: %{x:.2f}<br>Count: %{y}<extra></extra>",
                ))
                # Mark the elastic / inelastic boundary at ε = -1
                fig_elas.add_vline(
                    x=-1, line_dash="dash", line_color="#e05c5c",
                    annotation_text="Elastic boundary (−1)",
                    annotation_font_color="#e05c5c", annotation_font_size=10,
                )
                fig_elas.add_vline(
                    x=elas_vals.mean(), line_dash="solid", line_color="#c8a96e",
                    annotation_text=f"Mean {elas_vals.mean():.2f}",
                    annotation_font_color="#c8a96e", annotation_font_size=10,
                )
                layout_e = dark_chart_layout("Elasticity Distribution", 320)
                layout_e["xaxis"]["title"] = "Elasticity"
                layout_e["yaxis"]["title"] = "Products"
                fig_elas.update_layout(**layout_e)
                st.plotly_chart(fig_elas, use_container_width=True)

        with row2_r:
            if all(c in filtered_df.columns
                   for c in ["CURRENT_PRICE", "PROFIT_IMPROVEMENT_$", "ELASTICITY"]):
                fig_scatter = go.Figure(go.Scatter(
                    x=filtered_df["CURRENT_PRICE"],
                    y=filtered_df["PROFIT_IMPROVEMENT_$"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=filtered_df["ELASTICITY"],
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="Elasticity", thickness=12, len=0.8,
                                      tickfont=dict(color="#888")),
                        opacity=0.75,
                        line=dict(color="#222", width=0.5),
                    ),
                    hovertemplate=(
                        "Pattern: %{text}<br>"
                        "Price: Rs. %{x:,}<br>"
                        "Profit Uplift: Rs. %{y:,.0f}<extra></extra>"
                    ),
                    text=filtered_df.get("PATTERN", pd.Series([""] * len(filtered_df))),
                ))
                fig_scatter.add_hline(y=0, line_dash="dash", line_color="#555")
                layout_s = dark_chart_layout("Profit Improvement vs Current Price", 320)
                layout_s["xaxis"]["title"] = "Current Price (Rs.)"
                layout_s["yaxis"]["title"] = "Profit Improvement (Rs.)"
                fig_scatter.update_layout(**layout_s)
                st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Row 3: XGBoost base demand parity plot ───────────────────────────
        # Shows how well XGBoost's base demand compares to actual avg monthly qty.
        # Points on the y=x parity line indicate a perfect match; points above /
        # below indicate over- / under-prediction respectively.
        if all(c in filtered_df.columns for c in ["XGB_BASE_DEMAND", "CURRENT_QTY"]):
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            max_val = max(
                filtered_df["XGB_BASE_DEMAND"].max(),
                filtered_df["CURRENT_QTY"].max(),
            )
            fig_parity = go.Figure()
            fig_parity.add_trace(go.Scatter(
                x=filtered_df["CURRENT_QTY"],
                y=filtered_df["XGB_BASE_DEMAND"],
                mode="markers",
                marker=dict(size=7, color="#c8a96e", opacity=0.65,
                            line=dict(color="#222", width=0.5)),
                hovertemplate=(
                    "Pattern: %{text}<br>"
                    "Actual avg qty: %{x:.1f}<br>"
                    "XGBoost base: %{y:.1f}<extra></extra>"
                ),
                text=filtered_df.get("PATTERN", pd.Series([""] * len(filtered_df))),
                name="Patterns",
            ))
            # y = x parity line
            fig_parity.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                line=dict(color="#555", dash="dash"),
                name="Parity (y=x)",
                hoverinfo="skip",
            ))
            layout_pr = dark_chart_layout(
                "XGBoost Base Demand vs Actual Avg Monthly Qty", 360
            )
            layout_pr["xaxis"]["title"] = "Actual Avg Monthly Qty"
            layout_pr["yaxis"]["title"] = "XGBoost Base Demand"
            fig_parity.update_layout(**layout_pr)
            st.plotly_chart(fig_parity, use_container_width=True)


# ================================================================
#  TAB 3 — PRODUCT EXPLORER
#  Searchable, sortable table of all filtered products with a deep-
#  dive section for a single pattern: attributes, pricing KPIs,
#  monthly sales history with velocity trend, and a comparison
#  against similar patterns.
# ================================================================
with tab3:

    st.markdown(
        "<div class='section-header'>Product Explorer</div>",
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.markdown(
            "<div class='info-box'>No products match filters.</div>",
            unsafe_allow_html=True,
        )
    else:
        # ── Search + sort controls ────────────────────────────────────────────
        search_col, sort_col = st.columns([2, 1])
        with search_col:
            search_term = st.text_input(
                "🔍  Search by Pattern ID",
                placeholder="e.g. 35223",
                label_visibility="collapsed",
            )
        with sort_col:
            sort_options = [c for c in [
                "PROFIT_IMPROVEMENT_$",
                "REVENUE_IMPROVEMENT_%",
                "CURRENT_PRICE",
                "OPTIMAL_PRICE",
                "PRICE_CHANGE_%",
            ] if c in filtered_df.columns]
            sort_by = st.selectbox(
                "Sort by", sort_options, label_visibility="collapsed"
            )

        explorer_df = filtered_df.copy()
        if search_term.strip():
            explorer_df = explorer_df[
                explorer_df["PATTERN"].str.contains(
                    search_term.strip(), case=False, na=False
                )
            ]
        if sort_by in explorer_df.columns:
            explorer_df = explorer_df.sort_values(sort_by, ascending=False)

        # Show only columns that exist in the DataFrame (graceful degradation)
        display_cols = [c for c in [
            "PATTERN", "COLOUR", "FIT", "MATERIAL", "SLEEVE", "COLLAR",
            "CURRENT_PRICE", "OPTIMAL_PRICE", "PRICE_CHANGE_%",
            "XGB_BASE_DEMAND", "PREDICTED_QTY_HYBRID",
            "REVENUE_IMPROVEMENT_%", "PROFIT_IMPROVEMENT_$",
            "ELASTICITY", "ELASTICITY_TYPE",
            "RECOMMENDATION", "OPTIMIZATION_STATUS",
        ] if c in explorer_df.columns]

        st.markdown(
            f"<div style='color:#666;font-size:0.82rem;margin-bottom:8px;'>"
            f"Showing {len(explorer_df):,} products</div>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            explorer_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=min(550, len(explorer_df) * 38 + 40),
            hide_index=True,
        )

        # Allow the user to export the filtered table
        csv_data = explorer_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Filtered Results (CSV)",
            data=csv_data,
            file_name="filtered_price_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

        # ── Single-product deep dive ──────────────────────────────────────────
        selected_pattern = None
        if "PATTERN" in explorer_df.columns and len(explorer_df) > 0:
            st.markdown(
                "<div class='section-header'>Single Product Deep Dive</div>",
                unsafe_allow_html=True,
            )

            pattern_options  = explorer_df["PATTERN"].tolist()
            selected_pattern = st.selectbox(
                "Select a Pattern ID to inspect",
                pattern_options,
                label_visibility="collapsed",
            )

            product = explorer_df[
                explorer_df["PATTERN"] == selected_pattern
            ].iloc[0]

            dd1, dd2, dd3 = st.columns(3)

            # Column 1: product attributes
            with dd1:
                attrs_html = "<div class='metric-card'>"
                for attr in ["COLOUR", "FIT", "MATERIAL", "SLEEVE", "COLLAR", "TEXTURE"]:
                    if attr in product.index and pd.notna(product[attr]):
                        attrs_html += (
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:4px 0;border-bottom:1px solid #2a2a2a;'>"
                            f"<span style='color:#666;font-size:0.8rem;'>{attr}</span>"
                            f"<span style='color:#c8a96e;font-size:0.85rem;"
                            f"font-weight:500;'>{product[attr]}</span></div>"
                        )
                attrs_html += "</div>"
                st.markdown(attrs_html, unsafe_allow_html=True)

            # Column 2: pricing and elasticity metrics
            with dd2:
                for label, col, fmt in [
                    ("Current Price",    "CURRENT_PRICE",   "Rs. {:,.0f}"),
                    ("Optimal Price",    "OPTIMAL_PRICE",   "Rs. {:,.0f}"),
                    ("Cost Price",       "COST_PRICE",      "Rs. {:,.0f}"),
                    ("Elasticity",       "ELASTICITY",      "{:.3f}"),
                    ("Elasticity Type",  "ELASTICITY_TYPE", "{}"),
                    ("Price Change",     "PRICE_CHANGE_%",  "{:+.2f}%"),
                ]:
                    if col in product.index and pd.notna(product[col]):
                        try:
                            val_str = fmt.format(float(product[col]))
                        except (ValueError, TypeError):
                            val_str = str(product[col])
                        st.metric(label, val_str)

            # Column 3: demand and financial outcome metrics
            with dd3:
                for label, col, fmt in [
                    ("XGBoost Base Demand",  "XGB_BASE_DEMAND",      "{:.1f} units"),
                    ("Predicted Qty", "PREDICTED_QTY_HYBRID", "{:.1f} units"),
                    ("Revenue Uplift",       "REVENUE_IMPROVEMENT_%","{ :+.2f}%"),
                    ("Profit Uplift",        "PROFIT_IMPROVEMENT_$", "Rs. {:+,.0f}"),
                    ("Current Profit",       "CURRENT_PROFIT",       "Rs. {:,.0f}"),
                    ("Optimal Profit",       "PROFIT_AT_OPTIMAL",    "Rs. {:,.0f}"),
                    ("Recommendation",       "RECOMMENDATION",       "{}"),
                ]:
                    if col in product.index and pd.notna(product[col]):
                        val = product[col]
                        try:
                            val_str = f"{float(val):,.2f}"
                        except (ValueError, TypeError):
                            val_str = str(val)
                        st.metric(label, val_str)

        # ── Past sales history and velocity charts ───────────────────────────
        if engineered_df is not None and selected_pattern:
            history = get_product_history(selected_pattern, engineered_df)

            if not history.empty:
                st.markdown(
                    "<div class='custom-divider'></div>", unsafe_allow_html=True
                )
                st.markdown(
                    "<div class='section-header'>"
                    "📦  Past Sold Quantity & Sale Velocity</div>",
                    unsafe_allow_html=True,
                )

                velocity = get_sale_velocity(history)

                vk1, vk2, vk3, vk4, vk5 = st.columns(5)
                with vk1:
                    st.metric("Total Months", f"{len(history)}")
                with vk2:
                    st.metric("Total Qty Sold", f"{int(history['QTY'].sum()):,}")
                with vk3:
                    st.metric("Avg Monthly Qty", f"{history['QTY'].mean():.1f}")
                with vk4:
                    direction = velocity["trend_direction"]
                    icon      = (
                        "↑" if direction == "Accelerating"
                        else "↓" if direction == "Decelerating"
                        else "→"
                    )
                    dir_color = (
                        "#5cb85c" if direction == "Accelerating"
                        else "#e05c5c" if direction == "Decelerating"
                        else "#c8a96e"
                    )
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-label'>Sale Velocity Trend</div>"
                        f"<div class='metric-value' style='color:{dir_color};"
                        f"font-size:1.5rem;'>{icon} {direction}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with vk5:
                    st.metric(
                        "Velocity Change",
                        f"{velocity['trend_pct']:+.1f}%",
                        delta=(
                            f"Recent 3M: {velocity['recent_avg']:.1f} "
                            f"vs Prior 3M: {velocity['older_avg']:.1f}"
                        ),
                        help="Compares avg qty of last 3 months vs the 3 before.",
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Monthly qty bar chart with colour coding above/below the mean
                # and optional rolling-mean overlays from data_engineered.csv
                fig_hist = go.Figure()
                overall_mean = history["QTY"].mean()
                bar_colors   = [
                    "#5cb85c" if q >= overall_mean else "#e05c5c"
                    for q in history["QTY"]
                ]

                fig_hist.add_trace(go.Bar(
                    x=history["YEAR_MONTH"].dt.strftime("%Y-%m"),
                    y=history["QTY"],
                    name="Monthly Qty",
                    marker=dict(color=bar_colors, line=dict(color="#0f0f0f", width=0.3)),
                    hovertemplate="Month: %{x}<br>Qty: %{y:,}<extra></extra>",
                ))

                # 3-month rolling mean (pre-computed in data_engineered.csv)
                if "QTY_ROLL_MEAN_3M" in history.columns:
                    fig_hist.add_trace(go.Scatter(
                        x=history["YEAR_MONTH"].dt.strftime("%Y-%m"),
                        y=history["QTY_ROLL_MEAN_3M"],
                        mode="lines", name="3M Rolling Avg",
                        line=dict(color="#c8a96e", width=2.5),
                        hovertemplate="Month: %{x}<br>3M Avg: %{y:.1f}<extra></extra>",
                    ))

                # 6-month rolling mean (pre-computed in data_engineered.csv)
                if "QTY_ROLL_MEAN_6M" in history.columns:
                    fig_hist.add_trace(go.Scatter(
                        x=history["YEAR_MONTH"].dt.strftime("%Y-%m"),
                        y=history["QTY_ROLL_MEAN_6M"],
                        mode="lines", name="6M Rolling Avg",
                        line=dict(color="#7b68ee", width=2, dash="dot"),
                        hovertemplate="Month: %{x}<br>6M Avg: %{y:.1f}<extra></extra>",
                    ))

                fig_hist.add_hline(
                    y=overall_mean, line_dash="dash", line_color="#555",
                    annotation_text=f"Overall Avg: {overall_mean:.1f}",
                    annotation_font_color="#888", annotation_font_size=10,
                )
                layout_h = dark_chart_layout(
                    f"Monthly Sales History — Pattern {selected_pattern}", 380
                )
                layout_h["xaxis"]["title"]    = "Month"
                layout_h["yaxis"]["title"]    = "Units Sold"
                layout_h["xaxis"]["tickangle"] = -45
                layout_h["barmode"]           = "overlay"
                fig_hist.update_layout(**layout_h)
                st.plotly_chart(fig_hist, use_container_width=True)

                # Velocity trend — linear regression over all months
                if len(history) >= 3:
                    history_v          = history.copy()
                    history_v["MONTH_NUM"] = range(1, len(history_v) + 1)
                    z          = np.polyfit(history_v["MONTH_NUM"], history_v["QTY"], 1)
                    trend_line = np.poly1d(z)(history_v["MONTH_NUM"])

                    fig_vel = go.Figure()
                    fig_vel.add_trace(go.Scatter(
                        x=history_v["YEAR_MONTH"].dt.strftime("%Y-%m"),
                        y=history_v["QTY"],
                        mode="lines+markers", name="Monthly Qty",
                        line=dict(color="#c8a96e", width=2),
                        marker=dict(size=5, color="#c8a96e"),
                        hovertemplate="Month: %{x}<br>Qty: %{y:,}<extra></extra>",
                    ))
                    fig_vel.add_trace(go.Scatter(
                        x=history_v["YEAR_MONTH"].dt.strftime("%Y-%m"),
                        y=trend_line,
                        mode="lines", name="Sales Trend",
                        line=dict(
                            color="#ff6b6b" if z[0] < 0 else "#5cb85c",
                            width=2.5, dash="dot",
                        ),
                        hovertemplate="Month: %{x}<br>Trend: %{y:.1f}<extra></extra>",
                    ))

                    slope_dir = "growing" if z[0] > 0 else "declining"
                    layout_v  = dark_chart_layout(
                        f"Sale Velocity Trend — {slope_dir.title()} "
                        f"({z[0]:+.1f} units/month)",
                        280,
                    )
                    layout_v["xaxis"]["title"]    = "Month"
                    layout_v["yaxis"]["title"]    = "Units Sold"
                    layout_v["xaxis"]["tickangle"] = -45
                    fig_vel.update_layout(**layout_v)
                    st.plotly_chart(fig_vel, use_container_width=True)

            # ── Similar-patterns comparison ───────────────────────────────────
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-header'>"
                "👥  Similar Pattern Products — Qty & Velocity Comparison</div>",
                unsafe_allow_html=True,
            )

            similar_df = get_similar_patterns(selected_pattern, results_df, n=10)

            if similar_df.empty:
                st.markdown(
                    "<div class='info-box'>No similar products found "
                    "with the same FIT + MATERIAL attributes.</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Show the attribute basis used for the similarity match
                ref_row   = results_df[results_df["PATTERN"] == str(selected_pattern)]
                sim_basis = ""
                if not ref_row.empty:
                    r = ref_row.iloc[0]
                    sim_basis = (
                        f"FIT: <b>{r.get('FIT','?')}</b> &nbsp;·&nbsp; "
                        f"MATERIAL: <b>{r.get('MATERIAL','?')}</b>"
                    )
                st.markdown(
                    f"<div style='color:#888;font-size:0.83rem;margin-bottom:16px;'>"
                    f"Matched by {sim_basis}</div>",
                    unsafe_allow_html=True,
                )

                # Combine selected product + similar products into one frame
                current_row = results_df[results_df["PATTERN"] == str(selected_pattern)]
                compare_df  = (
                    pd.concat([current_row, similar_df], ignore_index=True)
                    .drop_duplicates(subset="PATTERN")
                )

                sc1, sc2 = st.columns(2)

                # Left: horizontal bar chart of avg monthly qty per pattern
                with sc1:
                    if "CURRENT_QTY" in compare_df.columns:
                        comp_sorted  = compare_df.sort_values("CURRENT_QTY", ascending=True)
                        bar_colors_s = [
                            "#c8a96e" if str(p) == str(selected_pattern) else "#4a4a6a"
                            for p in comp_sorted["PATTERN"]
                        ]
                        fig_cmp = go.Figure(go.Bar(
                            x=comp_sorted["CURRENT_QTY"],
                            y=comp_sorted["PATTERN"].astype(str),
                            orientation="h",
                            marker=dict(color=bar_colors_s,
                                        line=dict(color="#0f0f0f", width=0.3)),
                            hovertemplate=(
                                "Pattern: %{y}<br>"
                                "Avg Monthly Qty: %{x:,}<extra></extra>"
                            ),
                            text=comp_sorted["CURRENT_QTY"].apply(lambda x: f"{int(x):,}"),
                            textposition="outside",
                        ))
                        layout_cmp = dark_chart_layout(
                            "Avg Monthly Qty — Selected vs Similar", 380
                        )
                        layout_cmp["xaxis"]["title"] = "Avg Monthly Units"
                        layout_cmp["yaxis"]["title"] = "Pattern"
                        layout_cmp["showlegend"]     = False
                        fig_cmp.update_layout(**layout_cmp)
                        st.plotly_chart(fig_cmp, use_container_width=True)

                # Right: overlapping last-12-month sales lines
                with sc2:
                    fig_multi   = go.Figure()
                    all_patterns = compare_df["PATTERN"].astype(str).tolist()
                    for pat in all_patterns:
                        pat_hist    = get_product_history(pat, engineered_df)
                        if pat_hist.empty:
                            continue
                        pat_hist    = pat_hist.tail(12)
                        is_selected = pat == str(selected_pattern)
                        fig_multi.add_trace(go.Scatter(
                            x=pat_hist["YEAR_MONTH"].dt.strftime("%Y-%m"),
                            y=pat_hist["QTY"],
                            mode="lines",
                            name=f"★ {pat}" if is_selected else pat,
                            line=dict(
                                color="#c8a96e" if is_selected else "#3a3a5a",
                                width=3 if is_selected else 1.2,
                                dash="solid" if is_selected else "dot",
                            ),
                            opacity=1.0 if is_selected else 0.6,
                            hovertemplate=(
                                f"Pattern {pat}<br>"
                                "Month: %{x}<br>"
                                "Qty: %{y:,}<extra></extra>"
                            ),
                        ))
                    layout_ml = dark_chart_layout(
                        "Last 12 Months — Sales History Comparison", 380
                    )
                    layout_ml["xaxis"]["title"]    = "Month"
                    layout_ml["yaxis"]["title"]    = "Units Sold"
                    layout_ml["xaxis"]["tickangle"] = -45
                    layout_ml["legend"]["font"]["size"] = 9
                    fig_multi.update_layout(**layout_ml)
                    st.plotly_chart(fig_multi, use_container_width=True)

                # Velocity summary table for all compared patterns
                st.markdown(
                    "<div style='color:#888;font-size:0.85rem;"
                    "margin:16px 0 8px 0;'>Sale Velocity Summary</div>",
                    unsafe_allow_html=True,
                )

                vel_rows = []
                for pat in compare_df["PATTERN"].astype(str):
                    pat_hist = get_product_history(pat, engineered_df)
                    v        = get_sale_velocity(pat_hist)
                    pat_info = compare_df[compare_df["PATTERN"] == pat]
                    vel_rows.append({
                        "Pattern": "★ " + pat if pat == str(selected_pattern) else pat,
                        "Avg Monthly Qty": (
                            round(pat_hist["QTY"].mean(), 1) if not pat_hist.empty else "-"
                        ),
                        "Last Month Qty": (
                            int(pat_hist["QTY"].iloc[-1]) if not pat_hist.empty else "-"
                        ),
                        "Trend":           v["trend_direction"],
                        "Velocity Change": f"{v['trend_pct']:+.1f}%",
                        "Months of Data":  len(pat_hist),
                        "Optimal Price": (
                            f"Rs. {int(pat_info['OPTIMAL_PRICE'].iloc[0]):,}"
                            if "OPTIMAL_PRICE" in pat_info.columns and not pat_info.empty
                            else "-"
                        ),
                    })

                vel_df = pd.DataFrame(vel_rows)
                st.dataframe(
                    vel_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(420, len(vel_df) * 38 + 40),
                )


# ================================================================
#  TAB 4 — SEGMENT ANALYSIS
#  Reproduces the feature-wise summary tables from the notebook
#  (Price Simulation Loop, Cell 102). Groups result_df.csv by any
#  product attribute (COLOUR, FIT, MATERIAL, etc.) and aggregates
#  pricing KPIs, financial impact, and recommendation counts.
# ================================================================
with tab4:
    st.markdown(
        "<div class='section-header'>Feature-Wise Segment Analysis</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#666;font-size:0.85rem;margin-bottom:16px;'>"
        "Aggregates optimisation results by product attribute — "
        "reproduces the feature-wise summary tables from the notebook "
        "(Price Simulation Loop, Cell 102).</div>",
        unsafe_allow_html=True,
    )

    SEGMENT_FEATURES = ["COLOUR", "FIT", "MATERIAL", "SLEEVE", "TEXTURE", "COLLAR"]

    if filtered_df.empty:
        st.markdown(
            "<div class='info-box'>No products match current filters.</div>",
            unsafe_allow_html=True,
        )
    else:
        available_seg = [
            c for c in SEGMENT_FEATURES
            if c in filtered_df.columns and filtered_df[c].nunique(dropna=True) > 0
        ]

        if not available_seg:
            st.markdown(
                "<div class='info-box'>No attribute columns found in the data.</div>",
                unsafe_allow_html=True,
            )
        else:
            seg_choice = st.selectbox(
                "Analyse by attribute", available_seg, key="seg_feature_select"
            )

            # Columns required to build the full segment summary
            req_cols = [
                "PATTERN", "CURRENT_PRICE", "OPTIMAL_PRICE", "PRICE_CHANGE_%",
                "REVENUE_IMPROVEMENT_%", "PROFIT_IMPROVEMENT_%",
                "CURRENT_REVENUE", "OPTIMAL_REVENUE",
                "CURRENT_PROFIT", "PROFIT_AT_OPTIMAL",
                "RECOMMENDATION",
            ]
            has_all = all(c in filtered_df.columns for c in req_cols)

            if not has_all:
                st.warning("Some required columns are missing from result_df.csv.")
            else:
                # Build segment-level aggregation (mirrors notebook Cell 102)
                seg_df = (
                    filtered_df.groupby(seg_choice)
                    .agg(
                        Products             =("PATTERN",               "count"),
                        Avg_Current_Price    =("CURRENT_PRICE",         "mean"),
                        Avg_Optimal_Price    =("OPTIMAL_PRICE",         "mean"),
                        Avg_Price_Change_Pct =("PRICE_CHANGE_%",        "mean"),
                        Avg_Revenue_Imp_Pct  =("REVENUE_IMPROVEMENT_%", "mean"),
                        Avg_Profit_Imp_Pct   =("PROFIT_IMPROVEMENT_%",  "mean"),
                        Total_Current_Rev    =("CURRENT_REVENUE",       "sum"),
                        Total_Optimal_Rev    =("OPTIMAL_REVENUE",       "sum"),
                        Total_Current_Profit =("CURRENT_PROFIT",        "sum"),
                        Total_Optimal_Profit =("PROFIT_AT_OPTIMAL",     "sum"),
                        # Recommendation counts — using UPPERCASE labels from the notebook
                        Price_Up   =("RECOMMENDATION", lambda x: (x == "INCREASE PRICE").sum()),
                        Price_Down =("RECOMMENDATION", lambda x: (x == "DECREASE PRICE").sum()),
                        Keep       =("RECOMMENDATION", lambda x: (x == "MAINTAIN PRICE").sum()),
                    )
                    .round(2)
                    .reset_index()
                )

                # Derived uplift percentages at segment level
                seg_df["Revenue_Uplift_%"] = (
                    (seg_df["Total_Optimal_Rev"] - seg_df["Total_Current_Rev"])
                    / seg_df["Total_Current_Rev"].replace(0, np.nan) * 100
                ).round(2)

                seg_df["Profit_Uplift_%"] = (
                    (seg_df["Total_Optimal_Profit"] - seg_df["Total_Current_Profit"])
                    / seg_df["Total_Current_Profit"].replace(0, np.nan) * 100
                ).round(2)

                # Sort by profit uplift descending so the best segment leads
                seg_df = seg_df.sort_values("Profit_Uplift_%", ascending=False)

                # ── Segment KPI chips ────────────────────────────────────────
                top_seg    = seg_df.iloc[0][seg_choice]   if len(seg_df) > 0 else "—"
                top_uplift = seg_df.iloc[0]["Profit_Uplift_%"] if len(seg_df) > 0 else 0

                sk1, sk2, sk3 = st.columns(3)
                with sk1: st.metric(f"Segments ({seg_choice})", f"{len(seg_df)}")
                with sk2: st.metric("Top Segment by Profit Uplift", str(top_seg))
                with sk3: st.metric("Top Segment Profit Uplift",   f"{top_uplift:+.2f}%")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Profit-uplift horizontal bar chart ───────────────────────
                seg_sorted_bar = seg_df.sort_values("Profit_Uplift_%", ascending=True)
                bar_cols = [
                    "#5cb85c" if v >= 0 else "#e05c5c"
                    for v in seg_sorted_bar["Profit_Uplift_%"]
                ]

                fig_seg = go.Figure(go.Bar(
                    x=seg_sorted_bar["Profit_Uplift_%"],
                    y=seg_sorted_bar[seg_choice].astype(str),
                    orientation="h",
                    marker=dict(color=bar_cols, line=dict(color="#0f0f0f", width=0.3)),
                    text=seg_sorted_bar["Profit_Uplift_%"].apply(lambda x: f"{x:+.1f}%"),
                    textposition="outside",
                    hovertemplate=(
                        f"{seg_choice}: %{{y}}<br>"
                        "Profit Uplift: %{x:+.2f}%<extra></extra>"
                    ),
                ))
                fig_seg.add_vline(x=0, line_dash="dash", line_color="#555")
                layout_seg = dark_chart_layout(
                    f"Profit Uplift % by {seg_choice}",
                    max(320, len(seg_df) * 36 + 80),
                )
                layout_seg["xaxis"]["title"] = "Profit Uplift %"
                layout_seg["yaxis"]["title"] = seg_choice
                layout_seg["showlegend"]     = False
                fig_seg.update_layout(**layout_seg)
                st.plotly_chart(fig_seg, use_container_width=True)

                # ── Recommendation-split stacked bar ─────────────────────────
                rec_sorted = seg_df.sort_values("Products", ascending=True)
                fig_rec    = go.Figure()
                fig_rec.add_trace(go.Bar(
                    name="Increase",
                    x=rec_sorted["Price_Up"],
                    y=rec_sorted[seg_choice].astype(str),
                    orientation="h",
                    marker_color="#5cb85c",
                    hovertemplate=f"{seg_choice}: %{{y}}<br>Increase: %{{x}}<extra></extra>",
                ))
                fig_rec.add_trace(go.Bar(
                    name="Maintain",
                    x=rec_sorted["Keep"],
                    y=rec_sorted[seg_choice].astype(str),
                    orientation="h",
                    marker_color="#c8a96e",
                    hovertemplate=f"{seg_choice}: %{{y}}<br>Maintain: %{{x}}<extra></extra>",
                ))
                fig_rec.add_trace(go.Bar(
                    name="Decrease",
                    x=rec_sorted["Price_Down"],
                    y=rec_sorted[seg_choice].astype(str),
                    orientation="h",
                    marker_color="#e05c5c",
                    hovertemplate=f"{seg_choice}: %{{y}}<br>Decrease: %{{x}}<extra></extra>",
                ))
                layout_rec = dark_chart_layout(
                    f"Recommendation Split by {seg_choice}",
                    max(320, len(seg_df) * 36 + 80),
                )
                layout_rec["barmode"]        = "stack"
                layout_rec["xaxis"]["title"] = "Products"
                layout_rec["yaxis"]["title"] = seg_choice
                fig_rec.update_layout(**layout_rec)
                st.plotly_chart(fig_rec, use_container_width=True)

                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

                # ── Full segment summary table ────────────────────────────────
                st.markdown(
                    "<div class='section-header'>Full Segment Summary Table</div>",
                    unsafe_allow_html=True,
                )

                # Format columns for display (price → Rs., pct → +x.xx%)
                display_seg = seg_df.copy()
                for col in ["Avg_Current_Price", "Avg_Optimal_Price"]:
                    display_seg[col] = display_seg[col].apply(lambda x: f"Rs. {x:,.0f}")
                for col in [
                    "Avg_Price_Change_Pct", "Avg_Revenue_Imp_Pct",
                    "Avg_Profit_Imp_Pct",   "Revenue_Uplift_%",
                    "Profit_Uplift_%",
                ]:
                    display_seg[col] = display_seg[col].apply(
                        lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
                    )
                for col in [
                    "Total_Current_Rev", "Total_Optimal_Rev",
                    "Total_Current_Profit", "Total_Optimal_Profit",
                ]:
                    display_seg[col] = display_seg[col].apply(lambda x: f"Rs. {x:,.0f}")

                st.dataframe(
                    display_seg.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    height=min(550, len(display_seg) * 38 + 40),
                )

                csv_seg = seg_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️  Download {seg_choice} Segment Summary (CSV)",
                    data=csv_seg,
                    file_name=f"segment_summary_{seg_choice.lower()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ================================================================
#  FOOTER
# ================================================================
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#444; font-size:0.75rem; padding:8px 0 20px 0;'>
    Apparel Retail Predictive Pricing Model &nbsp;·&nbsp; FYP Project
    &nbsp;·&nbsp; XGBoost + Midpoint Elasticity
</div>
""", unsafe_allow_html=True)
