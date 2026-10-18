import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             classification_report, confusion_matrix, accuracy_score)
from sklearn.model_selection import train_test_split
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA C-MAPSS Engine Health",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background-color: #0a0e1a; color: #e2e8f0; }
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%); }
    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
        border: 1px solid #2d3a5a; border-radius: 12px;
        padding: 20px; text-align: center; margin: 8px 0;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #60a5fa; font-family: 'JetBrains Mono', monospace; }
    .metric-card .label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #1e3a5f 0%, transparent 100%);
        border-left: 4px solid #3b82f6; padding: 12px 20px;
        border-radius: 0 8px 8px 0; margin: 20px 0 12px 0;
        font-size: 1.1rem; font-weight: 600; color: #93c5fd; letter-spacing: 0.5px;
    }
    .info-box {
        background: #162032; border: 1px solid #1e3a5f; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0; font-size: 0.9rem; color: #94a3b8;
    }
    .health-badge-healthy { background: #064e3b; color: #34d399; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .health-badge-warning { background: #451a03; color: #fb923c; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .health-badge-critical { background: #450a0a; color: #f87171; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
    .stSelectbox label, .stSlider label, .stRadio label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    div[data-testid="metric-container"] { background: #1a2035; border: 1px solid #2d3a5a; border-radius: 10px; padding: 16px; }
    .stTabs [data-baseweb="tab-list"] { background: #0f1629; border-bottom: 1px solid #2d3a5a; }
    .stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; border-bottom: 2px solid #3b82f6 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        color: white; border: none; border-radius: 8px; font-weight: 600;
        font-family: 'Space Grotesk', sans-serif; padding: 10px 24px; transition: all 0.2s;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); transform: translateY(-1px); }
    h1, h2, h3 { color: #e2e8f0 !important; }
    p { color: #94a3b8; }
    .sensor-pill {
        display: inline-block; background: #1e3a5f; color: #93c5fd;
        padding: 2px 10px; border-radius: 12px; font-size: 0.78rem;
        margin: 2px; font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


COLS = ['unit', 'cycles', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

SENSOR_NAMES = {
    's2' : 'Fan Inlet Temp (T2)',
    's3' : 'LPC Outlet Temp (T24)',
    's4' : 'HPC Outlet Temp (T30)',
    's7' : 'HPC Outlet Pressure (P30)',
    's8' : 'Fan Speed (Nf)',
    's9' : 'Core Speed (Nc)',
    's11': 'HPC Static Pressure (Ps30)',
    's12': 'Fuel Flow Ratio (phi)',
    's13': 'Corrected Fan Speed (NRf)',
    's14': 'Corrected Core Speed (NRc)',
    's15': 'Bypass Ratio (BPR)',
    's17': 'Bleed Enthalpy (htBleed)',
    's20': 'HP Turbine Coolant Bleed (W31)',
    's21': 'LP Turbine Coolant Bleed (W32)',
}

USEFUL_SENSORS = list(SENSOR_NAMES.keys())
SENSOR_LABELS  = list(SENSOR_NAMES.values())
LABEL_TO_COL   = {v: k for k, v in SENSOR_NAMES.items()}

DATASET_INFO = {
    'FD001': {'conditions': 1, 'fault_modes': 1, 'desc': 'Sea Level · HPC Degradation'},
    'FD002': {'conditions': 6, 'fault_modes': 1, 'desc': '6 Conditions · HPC Degradation'},
    'FD003': {'conditions': 1, 'fault_modes': 2, 'desc': 'Sea Level · HPC + Fan Degradation'},
    'FD004': {'conditions': 6, 'fault_modes': 2, 'desc': '6 Conditions · HPC + Fan Degradation'},
}


@st.cache_data
def load_data(train_file, test_file, rul_file):
    train = pd.read_csv(train_file, sep=r'\s+', header=None, names=COLS)
    test  = pd.read_csv(test_file,  sep=r'\s+', header=None, names=COLS)
    rul   = pd.read_csv(rul_file,   sep=r'\s+', header=None, names=['RUL'])

    max_cyc = train.groupby('unit')['cycles'].max().rename('max_cycle')
    train   = train.join(max_cyc, on='unit')
    train['RUL'] = train['max_cycle'] - train['cycles']
    train.drop(columns='max_cycle', inplace=True)

    train['health'] = pd.cut(train['RUL'],
                             bins=[-1, 50, 125, 9999],
                             labels=[2, 1, 0]).astype(int)

    test_last = test.groupby('unit').last().reset_index()
    test_last['RUL'] = rul['RUL'].values
    test_last['health'] = pd.cut(test_last['RUL'],
                                  bins=[-1, 50, 125, 9999],
                                  labels=[2, 1, 0]).astype(int)
    return train, test_last


def dark_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0f1629')
    ax.set_facecolor('#141d30')
    for spine in ax.spines.values():
        spine.set_color('#2d3a5a')
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    return fig, ax

def dark_figs(nrows, ncols, figsize=(12, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#0f1629')
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    axes = np.array(axes).flatten()
    for ax in axes:
        ax.set_facecolor('#141d30')
        for spine in ax.spines.values():
            spine.set_color('#2d3a5a')
        ax.tick_params(colors='#64748b', labelsize=9)
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("###   NASA C-MAPSS")
    st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-top:-8px;'>Turbofan Engine Health Monitor</p>", unsafe_allow_html=True)
    st.markdown("---")

    dataset = st.selectbox("Dataset", ['FD001', 'FD002', 'FD003', 'FD004'],
                           help="FD001 is simplest; FD004 is hardest")
    st.markdown(f"<div class='info-box'> <b>{dataset}</b><br>{DATASET_INFO[dataset]['desc']}</div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#475569;'>Upload your dataset files</p>", unsafe_allow_html=True)

    train_file = st.file_uploader(f"train_{dataset}.txt", type='txt', key='train')
    test_file  = st.file_uploader(f"test_{dataset}.txt',  type='txt", key='test')
    rul_file   = st.file_uploader(f"RUL_{dataset}.txt",   type='txt', key='rul')

    st.markdown("---")
    st.markdown("<p style='font-size:0.72rem;color:#334155;text-align:center;'>Data: NASA Ames · Saxena et al. 2008</p>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#   NASA C-MAPSS Engine Health Dashboard")
st.markdown("<p style='color:#64748b;margin-top:-12px;margin-bottom:24px;'>Predictive Maintenance · Remaining Useful Life · Data Inference</p>",
            unsafe_allow_html=True)

if not (train_file and test_file and rul_file):
    st.markdown("""
    <div class='info-box' style='border-color:#1d4ed8;background:#0f1e3d;font-size:0.95rem;padding:24px;'>
    <b style='color:#93c5fd;font-size:1rem;'>👈  Upload your files in the sidebar to get started</b><br><br>
    You need three files for each dataset:<br>
    <span class='sensor-pill'>train_FD001.txt</span>
    <span class='sensor-pill'>test_FD001.txt</span>
    <span class='sensor-pill'>RUL_FD001.txt</span><br><br>
    Start with <b style='color:#60a5fa;'>FD001</b> — simplest (1 operating condition, 1 fault mode).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'> Dataset Overview</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    info = [
        ("FD001", "100 engines", "1 condition",  "HPC only"),
        ("FD002", "260 engines", "6 conditions", "HPC only"),
        ("FD003", "100 engines", "1 condition",  "HPC + Fan"),
        ("FD004", "248 engines", "6 conditions", "HPC + Fan"),
    ]
    for col, (name, eng, cond, fault) in zip(cols, info):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value' style='font-size:1.3rem;'>{name}</div>
                <div class='label'>{eng}</div>
                <div style='color:#60a5fa;font-size:0.8rem;margin-top:6px;'>{cond}</div>
                <div style='color:#94a3b8;font-size:0.75rem;'>{fault}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'> Sensor Reference (14 useful sensors)</div>",
                unsafe_allow_html=True)
    sensor_ref = pd.DataFrame([
        {'Code': k, 'Physical Measurement': v} for k, v in SENSOR_NAMES.items()
    ])
    st.dataframe(sensor_ref, use_container_width=True, hide_index=True)
    st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────
train, test = load_data(train_file, test_file, rul_file)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Total Engines",   train['unit'].nunique())
with k2: st.metric("Training Rows",   f"{len(train):,}")
with k3: st.metric("Avg Engine Life", f"{train.groupby('unit')['cycles'].max().mean():.0f} cycles")
with k4: st.metric("Sensors Used",    len(USEFUL_SENSORS))
with k5:
    pct_crit = (train['health'] == 2).mean() * 100
    st.metric("Critical Readings", f"{pct_crit:.1f}%")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Explore Data",
    " Linear Regression",
    " Classification",
    " ANOVA",
    " Predict Engine",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPLORE DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Engine Degradation Trajectories</div>",
                unsafe_allow_html=True)

    # ── Sensor selector + view toggle ─────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 1, 1])

    with col_a:
        sensor_label = st.selectbox("Select sensor to visualise", SENSOR_LABELS, index=5)
        sensor_col   = LABEL_TO_COL[sensor_label]

    with col_b:
        view_mode = st.radio("View mode", ["Single Engine", "Multiple Engines"], index=0)

    with col_c:
        all_units = sorted(train['unit'].unique())
        if view_mode == "Single Engine":
            selected_unit = st.selectbox("Select engine ID", all_units, index=0)
        else:
            n_engines = st.slider("Number of engines to show", 3, 15, 8)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = dark_fig((11, 4))

    if view_mode == "Single Engine":
        eng_data = train[train['unit'] == selected_unit]

        ax.plot(
            eng_data['cycles'],
            eng_data[sensor_col],
            color='#60a5fa',
            lw=1.8,
            alpha=0.9,
            label=f'Engine {selected_unit}'
        )

        # 🔥 Health zone shading
        for _, row in eng_data.iterrows():
            color = '#34d399' if row['health'] == 0 else (
                    '#fb923c' if row['health'] == 1 else '#f87171')
            ax.axvspan(row['cycles'] - 0.5, row['cycles'] + 0.5,
                       alpha=0.04, color=color, linewidth=0)

        # 🔥 Threshold lines
        warn_cross = eng_data[eng_data['RUL'] <= 125]['cycles'].min()
        crit_cross = eng_data[eng_data['RUL'] <= 50]['cycles'].min()

        if pd.notna(warn_cross):
            ax.axvline(warn_cross, color='#fb923c', linestyle='--',
                       label=f'Warning starts ({int(warn_cross)})')

        if pd.notna(crit_cross):
            ax.axvline(crit_cross, color='#f87171', linestyle='--',
                       label=f'Critical starts ({int(crit_cross)})')

        max_cycle = eng_data['cycles'].max()
        ax.set_title(f'{sensor_label} — Engine {selected_unit} (Life: {max_cycle} cycles)')

    else:
        sample_units = train['unit'].unique()[:n_engines]
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, n_engines))

        for i, uid in enumerate(sample_units):
            eng = train[train['unit'] == uid]
            ax.plot(
                eng['cycles'],
                eng[sensor_col],
                alpha=0.7,
                lw=1.2,
                color=palette[i],
                label=f'Engine {uid}'
            )

        ax.set_title(f'{sensor_label} — degradation ({n_engines} engines)')

    ax.set_xlabel('Cycle')
    ax.set_ylabel(sensor_label)

    ax.legend(fontsize=7, ncol=3, framealpha=0.2,
              facecolor='#1a2035',
              edgecolor='#2d3a5a',
              labelcolor='#94a3b8')

    st.pyplot(fig)
    plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # RUL Distribution & Health Classes (UNCHANGED)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>RUL Distribution & Health Classes</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = dark_fig((6, 3.5))
        ax.hist(train['RUL'], bins=50, color='#3b82f6', alpha=0.8, edgecolor='#1d4ed8')
        ax.axvline(50,  color='#f87171', lw=1.5, linestyle='--', label='Critical boundary (50)')
        ax.axvline(125, color='#fb923c', lw=1.5, linestyle='--', label='Warning boundary (125)')
        ax.set_xlabel('RUL (cycles)')
        ax.set_ylabel('Count')
        ax.set_title('RUL Distribution — Training Set')
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

    with c2:
        counts = train['health'].value_counts().sort_index()
        bar_labels = ['Healthy\n(RUL>125)', 'Warning\n(50–125)', 'Critical\n(RUL<50)']
        colors = ['#34d399', '#fb923c', '#f87171']

        fig, ax = dark_fig((6, 3.5))
        bars = ax.bar(bar_labels, counts.values, color=colors)

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height(),
                    f'{val:,}',
                    ha='center',
                    va='bottom',
                    color='#94a3b8')

        ax.set_title('Health Class Distribution')
        ax.set_ylabel('Count')

        st.pyplot(fig)
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Correlation Plot (UNCHANGED)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Sensor Correlation with RUL</div>",
                unsafe_allow_html=True)

    corrs = train[USEFUL_SENSORS + ['RUL']].corr()['RUL'].drop('RUL').sort_values()
    corrs.index = [SENSOR_NAMES[k] for k in corrs.index]

    fig, ax = dark_fig((11, 4.5))
    colors_bar = ['#f87171' if v < 0 else '#34d399' for v in corrs.values]

    ax.barh(corrs.index, corrs.values, color=colors_bar, alpha=0.8)
    ax.axvline(0, color='#475569', lw=0.8)

    ax.set_xlabel('Correlation with RUL')
    ax.set_title('Which sensors correlate most with Remaining Useful Life?')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LINEAR REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Linear Regression — Predict RUL</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='info-box'> <b>Goal:</b> Use sensor readings to predict how many cycles remain before engine failure. This is a <b>regression</b> problem — the output is a continuous number.</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        default_reg_labels = [SENSOR_NAMES[s] for s in
                              ['s9','s11','s12','s14','s15','s17','s20','s21']]
        sel_labels = st.multiselect("Feature sensors", SENSOR_LABELS,
                                    default=default_reg_labels)
        sel_cols   = [LABEL_TO_COL[l] for l in sel_labels]

        clip_rul = st.slider("Clip max RUL (piecewise approach)", 50, 300, 130,
                             help="Cap RUL — engines are only predictable near end of life")
        run_reg  = st.button("▶  Run Regression", key='run_reg')

    with col2:
        st.markdown("""
        <div class='info-box'>
        <b style='color:#93c5fd;'>How RUL is computed:</b><br>
        <code style='color:#60a5fa;font-size:0.85rem;'>RUL = max_cycle_for_engine − current_cycle</code><br><br>
        A <b>piecewise RUL</b> clips the maximum — it assumes the engine is "nominally healthy"
        for most of its life and only becomes predictable near failure.
        </div>""", unsafe_allow_html=True)

    if run_reg and sel_cols:
        df_train             = train.copy()
        df_train['RUL_clip'] = df_train['RUL'].clip(upper=clip_rul)
        X_tr = df_train[sel_cols]; y_tr = df_train['RUL_clip']
        X_te = test[sel_cols];     y_te = test['RUL'].clip(upper=clip_rul)

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        model       = LinearRegression()
        model.fit(X_tr_s, y_tr)
        y_pred      = model.predict(X_te_s)
        y_pred_clip = np.clip(y_pred, 0, None)

        rmse = np.sqrt(mean_squared_error(y_te, y_pred_clip))
        mae  = mean_absolute_error(y_te, y_pred_clip)
        r2   = r2_score(y_te, y_pred_clip)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("RMSE", f"{rmse:.2f} cycles")
        with m2: st.metric("MAE",  f"{mae:.2f} cycles")
        with m3: st.metric("R²",   f"{r2:.4f}")

        fig, axes = dark_figs(1, 2, figsize=(12, 4))

        axes[0].scatter(y_te, y_pred_clip, alpha=0.5, s=20,
                        color='#60a5fa', edgecolors='none')
        mn, mx = min(y_te.min(), y_pred_clip.min()), max(y_te.max(), y_pred_clip.max())
        axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
        axes[0].set_xlabel('Actual RUL'); axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title('Actual vs Predicted RUL')
        axes[0].legend(fontsize=8, framealpha=0.2,
                       facecolor='#1a2035', edgecolor='#2d3a5a', labelcolor='#94a3b8')

        residuals = y_te - y_pred_clip
        axes[1].hist(residuals, bins=30, color='#a78bfa', alpha=0.8, edgecolor='#0f1629')
        axes[1].axvline(0, color='#f87171', lw=1.5, linestyle='--')
        axes[1].set_xlabel('Residual (Actual − Predicted)'); axes[1].set_ylabel('Count')
        axes[1].set_title(f'Residuals  (mean={residuals.mean():.1f})')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("<div class='section-header'>Feature Coefficients</div>",
                    unsafe_allow_html=True)
        coef_df = pd.DataFrame({'Sensor': sel_labels, 'Coefficient': model.coef_})\
                    .sort_values('Coefficient')
        fig, ax = dark_fig((10, 3.5))
        colors_c = ['#f87171' if v < 0 else '#34d399' for v in coef_df['Coefficient']]
        ax.barh(coef_df['Sensor'], coef_df['Coefficient'],
                color=colors_c, alpha=0.85, edgecolor='#0f1629')
        ax.axvline(0, color='#475569', lw=0.8)
        ax.set_title('Linear Regression Coefficients (standardised features)')
        ax.set_xlabel('Coefficient value')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── VIF Table ──────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>Variance Inflation Factor (VIF)</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        <b style='color:#93c5fd;'>What is VIF?</b><br>
        VIF measures <b>multicollinearity</b> — how much one sensor can be predicted
        by the others. High VIF means redundant sensors that inflate coefficient uncertainty.<br><br>
        <b>Rule of thumb:</b>
        &nbsp;<span style='color:#34d399;font-weight:700;'>■</span> VIF &lt; 5 → Low (OK)
        &nbsp;<span style='color:#fb923c;font-weight:700;'>■</span> VIF 5–10 → Moderate
        &nbsp;<span style='color:#f87171;font-weight:700;'>■</span> VIF &gt; 10 → High multicollinearity
        </div>""", unsafe_allow_html=True)

        vif_vals = [variance_inflation_factor(X_tr_s, i) for i in range(X_tr_s.shape[1])]
        vif_data = pd.DataFrame({'Sensor': sel_labels, 'VIF': vif_vals})
        vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

        def vif_color(v):
            return '#34d399' if v < 5 else ('#fb923c' if v < 10 else '#f87171')

        def vif_label(v):
            return '✅ Low' if v < 5 else ('⚠️ Moderate' if v < 10 else '🔴 High')

        vif_display = vif_data.copy()
        vif_display['VIF']      = vif_data['VIF'].round(2)
        vif_display['Severity'] = vif_data['VIF'].apply(vif_label)
        st.dataframe(vif_display, use_container_width=True, hide_index=True)

        fig, ax = dark_fig((10, 3.5))
        ax.barh(vif_data['Sensor'], vif_data['VIF'],
                color=[vif_color(v) for v in vif_data['VIF']],
                alpha=0.85, edgecolor='#0f1629')
        ax.axvline(5,  color='#fb923c', lw=1.2, linestyle='--', label='Moderate threshold (5)')
        ax.axvline(10, color='#f87171', lw=1.2, linestyle='--', label='High threshold (10)')
        ax.set_xlabel('VIF Value')
        ax.set_title('Variance Inflation Factor by Sensor')
        ax.legend(fontsize=8, framealpha=0.2,
                  facecolor='#1a2035', edgecolor='#2d3a5a', labelcolor='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        high_vif = vif_data[vif_data['VIF'] > 10]['Sensor'].tolist()
        mod_vif  = vif_data[(vif_data['VIF'] >= 5) & (vif_data['VIF'] <= 10)]['Sensor'].tolist()
        if high_vif:
            vif_note = f"Sensors with high multicollinearity (VIF &gt; 10): <b style='color:#f87171;'>{', '.join(high_vif)}</b> — consider removing these to improve coefficient reliability."
        elif mod_vif:
            vif_note = f"Moderate multicollinearity (VIF 5–10) in: <b style='color:#fb923c;'>{', '.join(mod_vif)}</b> — worth monitoring but acceptable."
        else:
            vif_note = "All selected sensors have low VIF (&lt; 5) — no significant multicollinearity detected. ✅"

        st.markdown(f"""
        <div class='info-box'>
        <b style='color:#93c5fd;'>Interpretation for your report:</b><br>
        The model achieved R² = <b style='color:#60a5fa;'>{r2:.4f}</b>, meaning it explains
        {r2*100:.1f}% of the variance in RUL. RMSE of {rmse:.1f} cycles means predictions
        are off by ~{rmse:.0f} cycles on average. Positive coefficients indicate sensors that
        rise as the engine ages; negative coefficients drop with age.<br><br>
        <b>VIF Analysis:</b> {vif_note}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION (Logistic Regression only)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Classification — Engine Health State</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='info-box'> <b>Goal:</b> Classify each engine reading into <b style='color:#34d399;'>Healthy</b>, <b style='color:#fb923c;'>Warning</b>, or <b style='color:#f87171;'>Critical</b> based on sensor data.</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        clf_labels = st.multiselect("Feature sensors", SENSOR_LABELS,
                                    default=SENSOR_LABELS, key='clf_sens')
        clf_cols   = [LABEL_TO_COL[l] for l in clf_labels]
        run_clf    = st.button("▶  Run Classification", key='run_clf')

    with col2:
        st.markdown("""
        <div class='info-box'>
        <b style='color:#93c5fd;'>Class definitions:</b><br>
        <span class='health-badge-healthy'>🟢 Healthy</span> RUL &gt; 125 cycles — engine fine, no action needed<br><br>
        <span class='health-badge-warning'>🟡 Warning</span> RUL 50–125 — schedule maintenance soon<br><br>
        <span class='health-badge-critical'>🔴 Critical</span> RUL &lt; 50 — immediate intervention required
        </div>""", unsafe_allow_html=True)

    if run_clf and clf_cols:
        X_tr = train[clf_cols]; y_tr = train['health']
        X_te = test[clf_cols];  y_te = test['health']

        scaler2 = StandardScaler()
        X_tr_s  = scaler2.fit_transform(X_tr)
        X_te_s  = scaler2.transform(X_te)

        clf    = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        acc    = accuracy_score(y_te, y_pred)

        st.metric("Test Accuracy", f"{acc*100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(y_te, y_pred)
            fig, ax = dark_fig((5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Healthy', 'Warning', 'Critical'],
                        yticklabels=['Healthy', 'Warning', 'Critical'],
                        ax=ax, linewidths=0.5,
                        annot_kws={'color': 'white', 'size': 12})
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            ax.tick_params(colors='#94a3b8')
            st.pyplot(fig); plt.close()

        with c2:
            report = classification_report(y_te, y_pred,
                                           target_names=['Healthy', 'Warning', 'Critical'],
                                           output_dict=True)
            rdf = pd.DataFrame(report).T.iloc[:3][['precision', 'recall', 'f1-score']]
            fig, ax = dark_fig((5, 4))
            x = np.arange(3); w = 0.25
            ax.bar(x - w, rdf['precision'], w, color='#3b82f6', alpha=0.85, label='Precision')
            ax.bar(x,     rdf['recall'],    w, color='#34d399', alpha=0.85, label='Recall')
            ax.bar(x + w, rdf['f1-score'], w, color='#a78bfa', alpha=0.85, label='F1-Score')
            ax.set_xticks(x); ax.set_xticklabels(['Healthy', 'Warning', 'Critical'])
            ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
            ax.set_title('Precision / Recall / F1 by Class')
            ax.legend(fontsize=8, framealpha=0.2,
                      facecolor='#1a2035', edgecolor='#2d3a5a', labelcolor='#94a3b8')
            st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class='info-box'>
        <b style='color:#93c5fd;'>Interpretation for your report:</b><br>
        Logistic Regression achieved <b style='color:#60a5fa;'>{acc*100:.1f}%</b> accuracy classifying engine health states.
        For safety-critical applications, <b>recall on the Critical class</b> is the most important metric —
        missing a critical engine is far worse than a false alarm.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOVA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>One-Way ANOVA — Sensor Values by Health Class</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
     <b>ANOVA Question:</b> Do sensor readings differ significantly across
    Healthy / Warning / Critical groups?
    H₀: group means are equal  |  Reject if p &lt; 0.05
    </div>""", unsafe_allow_html=True)

    anova_label = st.selectbox("Select sensor for ANOVA", SENSOR_LABELS, index=3)
    anova_col   = LABEL_TO_COL[anova_label]

    grp0 = train[train['health'] == 0][anova_col].dropna()
    grp1 = train[train['health'] == 1][anova_col].dropna()
    grp2 = train[train['health'] == 2][anova_col].dropna()

    f_stat, p_val = stats.f_oneway(grp0, grp1, grp2)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("F-statistic", f"{f_stat:.2f}")
    with col2: st.metric("p-value",     f"{p_val:.2e}")
    with col3: st.metric("Significant?", " Yes" if p_val < 0.05 else "❌ No")

    fig, axes = dark_figs(1, 2, figsize=(12, 4))
    data_bp = [grp0.sample(min(len(grp0), 2000)), grp1, grp2]

    bp = axes[0].boxplot(data_bp, patch_artist=True,
                         medianprops=dict(color='white', lw=2),
                         whiskerprops=dict(color='#475569'),
                         capprops=dict(color='#475569'),
                         flierprops=dict(marker='o', markersize=2, alpha=0.3, color='#64748b'))
    for patch, c in zip(bp['boxes'], ['#34d399', '#fb923c', '#f87171']):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_xticklabels(['Healthy', 'Warning', 'Critical'])
    axes[0].set_ylabel(anova_label)
    axes[0].set_title(f'{anova_label}\nby Health Class (Boxplot)')

    parts = axes[1].violinplot(data_bp, positions=[1, 2, 3], showmedians=True)
    for pc, c in zip(parts['bodies'], ['#34d399', '#fb923c', '#f87171']):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    parts['cmedians'].set_color('white')
    parts['cbars'].set_color('#475569')
    parts['cmaxes'].set_color('#475569')
    parts['cmins'].set_color('#475569')
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(['Healthy', 'Warning', 'Critical'])
    axes[1].set_ylabel(anova_label)
    axes[1].set_title(f'{anova_label} — Violin Plot')

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("<div class='section-header'>Post-hoc Pairwise t-tests</div>",
                unsafe_allow_html=True)
    pairs = [('Healthy vs Warning',  grp0, grp1),
             ('Healthy vs Critical', grp0, grp2),
             ('Warning vs Critical', grp1, grp2)]
    tukey_rows = []
    for name, g1, g2 in pairs:
        t, p = stats.ttest_ind(g1.sample(min(len(g1), 1000), random_state=42),
                               g2.sample(min(len(g2), 1000), random_state=42))
        tukey_rows.append({'Comparison': name, 't-statistic': f'{t:.3f}',
                           'p-value': f'{p:.2e}',
                           'Significant': ' Yes' if p < 0.05 else '❌ No'})
    st.dataframe(pd.DataFrame(tukey_rows), use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class='info-box'>
    <b style='color:#93c5fd;'>Interpretation for your report:</b><br>
    One-way ANOVA for <b>{anova_label}</b>: F = {f_stat:.2f}, p = {p_val:.2e}.
    {'Since p &lt; 0.05, we <b>reject H₀</b> — this sensor differs significantly across health classes.' if p_val < 0.05
     else 'Since p ≥ 0.05, we <b>fail to reject H₀</b> — no significant difference across classes.'}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'> Real-time Engine Health Predictor</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Enter sensor readings and get an instant RUL prediction and health classification.</div>",
                unsafe_allow_html=True)

    scaler_p = StandardScaler()
    X_all    = scaler_p.fit_transform(train[USEFUL_SENSORS])
    reg_p    = LinearRegression().fit(X_all, train['RUL'].clip(upper=130))
    clf_p    = LogisticRegression(max_iter=1000).fit(X_all, train['health'])

    st.markdown("**Adjust sensor readings:**")
    sensor_means = train[USEFUL_SENSORS].mean()

    user_vals = {}
    sensor_rows = [USEFUL_SENSORS[i:i+3] for i in range(0, len(USEFUL_SENSORS), 3)]
    for row in sensor_rows:
        cols_r = st.columns(3)
        for col, s in zip(cols_r, row):
            with col:
                user_vals[s] = st.number_input(
                    SENSOR_NAMES[s],
                    value=float(round(sensor_means[s], 3)),
                    format="%.3f",
                    key=f'pred_{s}'
                )

    if st.button(" Predict Health", key='predict_btn'):
        input_df     = pd.DataFrame([user_vals])[USEFUL_SENSORS]
        input_scaled = scaler_p.transform(input_df)

        rul_pred  = max(0, reg_p.predict(input_scaled)[0])
        cls_pred  = clf_p.predict(input_scaled)[0]
        cls_prob  = clf_p.predict_proba(input_scaled)[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>{rul_pred:.0f}</div>
                <div class='label'>Predicted RUL (cycles)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            label = ['🟢 Healthy', '🟡 Warning', '🔴 Critical'][cls_pred]
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value' style='font-size:1.4rem;'>{label}</div>
                <div class='label'>Health Classification</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value' style='font-size:1.1rem;'>{max(cls_prob)*100:.1f}%</div>
                <div class='label'>Model Confidence</div>
            </div>""", unsafe_allow_html=True)

        fig, ax = dark_fig((7, 2.5))
        bars_p = ax.barh(['Healthy', 'Warning', 'Critical'], cls_prob,
                         color=['#34d399', '#fb923c', '#f87171'],
                         alpha=0.85, edgecolor='#0f1629')
        for bar, prob in zip(bars_p, cls_prob):
            ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', color='#94a3b8', fontsize=10)
        ax.set_xlim(0, 1.15); ax.set_xlabel('Probability')
        ax.set_title('Classification Probabilities')
        st.pyplot(fig); plt.close()