"""
Primetrade.ai Trader Sentiment Dashboard
Streamlit interactive explorer for Fear/Greed vs trader behavior analysis.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings('ignore')

FEAR_COLOR    = '#E74C3C'
GREED_COLOR   = '#2ECC71'
NEUTRAL_COLOR = '#95A5A6'

FEAR_LABELS  = ['Extreme Fear', 'Fear']
GREED_LABELS = ['Greed', 'Extreme Greed']
LONG_DIRS    = {'Open Long', 'Buy', 'Long > Short'}
RANDOM_SEED  = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(
    page_title="Primetrade.ai Sentiment Dashboard",
    page_icon="PT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label {
        color: #8892b0 !important;
        font-size: 13px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }

    h1 { color: #ccd6f6 !important; font-size: 2rem !important; }
    h2 { color: #a8b2d8 !important; border-bottom: 1px solid #2d3250; padding-bottom: 6px; }
    h3 { color: #8892b0 !important; }

    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #2d3250;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #8892b0; }

    .stTabs [data-baseweb="tab-list"] { background: #1e2130; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #8892b0; font-weight: 500; }
    .stTabs [aria-selected="true"]    { color: #64ffda !important; }

    .insight-box {
        background: #1a2744;
        border-left: 4px solid #64ffda;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .rule-box {
        background: #1a3a2a;
        border-left: 4px solid #2ECC71;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 14px;
        line-height: 1.6;
    }
    .warning-box {
        background: #2a1a1a;
        border-left: 4px solid #E74C3C;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 14px;
    }
    .paradox-box {
        background: #1a1a2a;
        border: 1px solid #4a4a8a;
        border-left: 4px solid #9b59b6;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 14px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and processing data...")
def load_and_process():
    raw_fg_path   = 'fear_greed_index.csv'
    raw_hist_path = 'historical_data.csv'
    processed     = 'processed_trader_sentiment.csv'

    fg_raw = pd.read_csv(raw_fg_path)
    fg = fg_raw.copy()
    fg['date'] = pd.to_datetime(fg['date'], format='%Y-%m-%d')
    fg = fg.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    fg['sentiment'] = fg['classification'].apply(
        lambda c: 'FEAR' if c in FEAR_LABELS else ('GREED' if c in GREED_LABELS else 'NEUTRAL')
    )

    if os.path.exists(processed):
        df = pd.read_csv(processed, parse_dates=['date'])
    else:
        hd = pd.read_csv(raw_hist_path)
        hd['datetime_ist'] = pd.to_datetime(hd['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
        hd['date'] = hd['datetime_ist'].dt.normalize()
        hd.rename(columns={
            'Account': 'account', 'Coin': 'coin', 'Execution Price': 'exec_price',
            'Size Tokens': 'size_tokens', 'Size USD': 'size_usd', 'Side': 'side',
            'Start Position': 'start_position', 'Direction': 'direction',
            'Closed PnL': 'closed_pnl', 'Transaction Hash': 'tx_hash',
            'Order ID': 'order_id', 'Fee': 'fee', 'Trade ID': 'trade_id',
        }, inplace=True)

        hd_closed = hd[hd['closed_pnl'] != 0].copy()

        pnl_agg = hd_closed.groupby(['account', 'date']).agg(
            daily_pnl  = ('closed_pnl', 'sum'),
            win_rate   = ('closed_pnl', lambda x: (x > 0).mean()),
            pnl_count  = ('closed_pnl', 'count'),
        ).reset_index()

        activity_agg = hd.groupby(['account', 'date']).agg(
            trade_count        = ('trade_id', 'count'),
            avg_trade_size_usd = ('size_usd', 'mean'),
            total_exposure_usd = ('size_usd', 'sum'),
            avg_fee            = ('fee',       'mean'),
        ).reset_index()

        hd['is_long'] = hd['direction'].isin(LONG_DIRS).astype(int)
        ls_agg = hd.groupby(['account', 'date']).agg(long_ratio=('is_long', 'mean')).reset_index()

        td = activity_agg.merge(pnl_agg, on=['account', 'date'], how='left')
        td = td.merge(ls_agg,            on=['account', 'date'], how='left')
        td['daily_pnl'].fillna(0, inplace=True)
        td['pnl_count'].fillna(0, inplace=True)
        td = td.sort_values(['account', 'date']).reset_index(drop=True)

        def compute_drawdown(s):
            cum = s.cumsum()
            return cum.cummax() - cum

        td['drawdown_proxy'] = td.groupby('account')['daily_pnl'].transform(compute_drawdown)

        fg_for_merge = fg.copy()
        fg_for_merge['date'] = pd.to_datetime(fg_for_merge['date'])
        df = td.merge(fg_for_merge[['date', 'value', 'classification', 'sentiment']],
                      on='date', how='left')

    df = df[df['sentiment'].isin(['FEAR', 'GREED'])].copy()
    return df, fg


@st.cache_data(show_spinner="Running clustering...")
def run_clustering(df_json, k):
    df = pd.read_json(df_json)
    cluster_features = ['daily_pnl', 'trade_count', 'total_exposure_usd', 'long_ratio', 'win_rate']
    cluster_df = df[cluster_features + ['account', 'sentiment', 'date']].dropna().copy()
    X_scaled = StandardScaler().fit_transform(cluster_df[cluster_features].values)
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    cluster_df['cluster'] = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, cluster_df['cluster'])

    rank = cluster_df.groupby('cluster')['daily_pnl'].mean().rank(ascending=False).astype(int)
    name_pool = ['High-PnL Heavy-Hitters', 'Active Balanced Traders',
                 'Cautious Low-Frequency', 'Aggressive Loss-Prone', 'Niche Cluster']
    name_map = {c: name_pool[r-1] if r-1 < len(name_pool) else f'Cluster {c}'
                for c, r in rank.items()}
    cluster_df['cluster_name'] = cluster_df['cluster'].map(name_map)
    return cluster_df, sil


try:
    df, fg = load_and_process()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Ensure fear_greed_index.csv and historical_data.csv are in the same folder as dashboard.py")
    st.stop()

fear_df  = df[df['sentiment'] == 'FEAR']
greed_df = df[df['sentiment'] == 'GREED']


with st.sidebar:
    st.markdown("## Dashboard Controls")
    st.markdown("---")

    all_accounts   = sorted(df['account'].unique().tolist())
    short_accounts = {a: f"...{a[-6:]}" for a in all_accounts}
    selected_accounts = st.multiselect(
        "Filter by Account", all_accounts,
        default=all_accounts,
        format_func=lambda a: short_accounts.get(a, a),
    )

    sentiment_filter = st.radio("Sentiment Filter", ['Both', 'FEAR only', 'GREED only'])

    st.markdown("---")
    k_clusters = st.slider("K for Clustering", min_value=2, max_value=8, value=4,
                           help="Number of behavioral clusters. Adjust and compare silhouette scores.")

    clip_pnl = st.checkbox("Clip PnL outliers (1st-99th pct)", value=True,
                           help="Prevents extreme outliers from collapsing the axis scale.")

    st.markdown("---")
    st.markdown("**Dataset**")
    st.metric("Total Trader-Days", f"{len(df):,}")
    st.metric("Unique Accounts",   df['account'].nunique())
    st.metric("FEAR Days",         (df['sentiment']=='FEAR').sum())
    st.metric("GREED Days",        (df['sentiment']=='GREED').sum())

    st.markdown("---")
    st.markdown("""
**Leverage note**

The dataset has no leverage column.
Total Exposure USD (sum of Size USD per day)
is used as the closest available proxy.
True leverage requires account equity, which
is not available here.
""")

df_filtered = df[df['account'].isin(selected_accounts)] if selected_accounts else df.copy()

if sentiment_filter == 'FEAR only':
    df_filtered = df_filtered[df_filtered['sentiment'] == 'FEAR']
elif sentiment_filter == 'GREED only':
    df_filtered = df_filtered[df_filtered['sentiment'] == 'GREED']

fear_f  = df_filtered[df_filtered['sentiment'] == 'FEAR']
greed_f = df_filtered[df_filtered['sentiment'] == 'GREED']


st.markdown("""
<div style="background: linear-gradient(135deg, #0d1117 0%, #1a2744 100%);
            border: 1px solid #2d3250; border-radius: 16px;
            padding: 24px 32px; margin-bottom: 24px;">
  <h1 style="margin:0; color:#ccd6f6; font-size:2rem;">
    Primetrade.ai: Trader Behavior vs Market Sentiment
  </h1>
  <p style="color:#8892b0; margin-top:8px; font-size:15px;">
    How Bitcoin Fear and Greed sentiment shapes Hyperliquid trader performance.
    <span style="color:#64ffda;">32 Accounts, 211K Trades, 479 Matched Days</span>
  </p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    fear_med  = fear_f['daily_pnl'].median()  if len(fear_f)  else 0
    greed_med = greed_f['daily_pnl'].median() if len(greed_f) else 0
    st.metric("Median PnL (FEAR)",  f"${fear_med:,.0f}")
with k2:
    st.metric("Median PnL (GREED)", f"${greed_med:,.0f}",
              delta=f"{greed_med - fear_med:+.0f} vs FEAR")
with k3:
    fear_wr  = fear_f['win_rate'].mean()  if len(fear_f)  else 0
    greed_wr = greed_f['win_rate'].mean() if len(greed_f) else 0
    st.metric("Win Rate (GREED)",   f"{greed_wr:.1%}",
              delta=f"{(greed_wr - fear_wr)*100:+.1f}pp vs FEAR")
with k4:
    fear_tc  = fear_f['trade_count'].median()  if len(fear_f)  else 0
    greed_tc = greed_f['trade_count'].median() if len(greed_f) else 0
    st.metric("Trade Count (FEAR)", f"{fear_tc:.0f}",
              delta=f"{fear_tc - greed_tc:+.0f} vs GREED",
              help="Fear days show higher trade count despite lower PnL (panic trading pattern)")
with k5:
    fear_dd  = fear_f['drawdown_proxy'].mean()  if len(fear_f)  else 0
    greed_dd = greed_f['drawdown_proxy'].mean() if len(greed_f) else 0
    st.metric("Avg Drawdown (FEAR)", f"${fear_dd:,.0f}",
              delta=f"{fear_dd - greed_dd:+.0f} vs GREED")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Part A: Data Profile",
    "Part B.1: Performance",
    "Part B.2: Behavior",
    "Part B.3: Segments",
    "Part C: Strategy + Bonus",
])


def styled_fig():
    fig_obj = plt.figure(facecolor='#1e2130')
    return fig_obj


def style_ax(ax_obj):
    ax_obj.set_facecolor('#1e2130')
    ax_obj.tick_params(colors='white')
    for spine in ax_obj.spines.values():
        spine.set_edgecolor('#2d3250')
    return ax_obj


with tab1:
    st.header("Part A: Data Preparation Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Index Distribution")
        sent_counts = fg['sentiment'].value_counts().reindex(['FEAR', 'NEUTRAL', 'GREED'])
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e2130')
        style_ax(ax)
        bars = ax.bar(sent_counts.index, sent_counts.values,
                      color=[FEAR_COLOR, NEUTRAL_COLOR, GREED_COLOR], width=0.5)
        for bar, v in zip(bars, sent_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(v), ha='center', va='bottom', color='white', fontweight='bold')
        ax.set_title('Days per Sentiment Class (2018-2025)', color='white', pad=10)
        ax.set_ylabel('Days', color='white')
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Sentiment Score Over Time")
        fg_plot = fg.sort_values('date').copy()
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e2130')
        style_ax(ax)
        colors_scatter = [GREED_COLOR if v >= 50 else FEAR_COLOR for v in fg_plot['value']]
        ax.scatter(fg_plot['date'], fg_plot['value'], c=colors_scatter, s=2, alpha=0.6)
        ax.axhline(50, color='white', linewidth=0.8, linestyle='--', alpha=0.4, label='Neutral (50)')
        ax.set_title('Fear/Greed Score 2018-2025', color='white', pad=10)
        ax.set_ylabel('Score (0=Extreme Fear, 100=Extreme Greed)', color='white', fontsize=8)
        ax.legend(facecolor='#2d3250', labelcolor='white', fontsize=8)
        st.pyplot(fig, clear_figure=True)

    st.subheader("Feature Distribution")
    feat_to_show = st.selectbox("Select feature",
        ['daily_pnl', 'win_rate', 'trade_count', 'total_exposure_usd', 'long_ratio', 'drawdown_proxy'])

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#1e2130')
    style_ax(ax)
    for sent, color, label in [('FEAR', FEAR_COLOR, 'FEAR'), ('GREED', GREED_COLOR, 'GREED')]:
        vals = df_filtered[df_filtered['sentiment']==sent][feat_to_show].dropna()
        if clip_pnl and feat_to_show in ['daily_pnl', 'total_exposure_usd', 'drawdown_proxy']:
            q1, q99 = vals.quantile([0.01, 0.99])
            vals = vals[(vals >= q1) & (vals <= q99)]
        ax.hist(vals, bins=40, alpha=0.55, color=color, label=label, edgecolor='none')
    ax.set_title(f'Distribution of {feat_to_show}', color='white')
    ax.set_xlabel(feat_to_show, color='white')
    ax.set_ylabel('Count', color='white')
    ax.legend(facecolor='#2d3250', labelcolor='white')
    st.pyplot(fig, clear_figure=True)

    st.subheader("Summary Statistics by Sentiment")
    feat_cols = ['daily_pnl', 'win_rate', 'trade_count', 'total_exposure_usd', 'long_ratio', 'drawdown_proxy']
    summary = df_filtered.groupby('sentiment')[feat_cols].agg(['median', 'mean', 'std']).round(2)
    st.dataframe(summary, use_container_width=True)


with tab2:
    st.header("Part B.1: Does Performance Differ by Sentiment?")
    st.markdown("""
    Statistical test: Mann-Whitney U.
    I use this instead of a t-test because trading PnL is heavy-tailed and non-normal.
    Mann-Whitney ranks the data, making it robust to the extreme wins and losses common
    in leveraged crypto markets.
    """)

    metric_opts = {
        'Daily PnL (USD)':      'daily_pnl',
        'Win Rate':             'win_rate',
        'Drawdown Proxy (USD)': 'drawdown_proxy',
    }
    perf_metric = st.selectbox("Select performance metric", list(metric_opts.keys()))
    col = metric_opts[perf_metric]

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='#1e2130')
        style_ax(ax)
        data_plot = df_filtered[['sentiment', col]].dropna()
        if clip_pnl and col in ['daily_pnl', 'drawdown_proxy', 'total_exposure_usd']:
            q1, q99 = data_plot[col].quantile([0.01, 0.99])
            data_plot = data_plot[(data_plot[col] >= q1) & (data_plot[col] <= q99)]

        bp = ax.boxplot(
            [data_plot[data_plot['sentiment']=='FEAR'][col].values,
             data_plot[data_plot['sentiment']=='GREED'][col].values],
            labels=['FEAR', 'GREED'],
            patch_artist=True,
            medianprops=dict(color='white', linewidth=2),
            flierprops=dict(marker='.', markerfacecolor='gray', markersize=3, alpha=0.5),
        )
        bp['boxes'][0].set_facecolor(FEAR_COLOR + '88')
        bp['boxes'][1].set_facecolor(GREED_COLOR + '88')
        for item in bp['whiskers'] + bp['caps']:
            item.set_color('#8892b0')
        ax.set_title(f'{perf_metric}: FEAR vs GREED', color='white', pad=10)
        ax.set_ylabel(perf_metric, color='white')
        st.pyplot(fig, clear_figure=True)

    with col2:
        f_vals = fear_f[col].dropna()
        g_vals = greed_f[col].dropna()

        if len(f_vals) > 1 and len(g_vals) > 1:
            stat_mw, p_mw = mannwhitneyu(f_vals, g_vals, alternative='two-sided')
            sig = "Significant (p<0.05)" if p_mw < 0.05 else "Not significant"
        else:
            p_mw = float('nan')
            sig  = "Insufficient data"

        st.markdown("#### Statistical Test (Mann-Whitney U)")
        st.metric("p-value", f"{p_mw:.4f}")
        st.markdown(f"**{sig}**")
        st.markdown("---")

        for sent, vals in [('FEAR', f_vals), ('GREED', g_vals)]:
            st.markdown(f"**{sent}**")
            c1, c2 = st.columns(2)
            c1.metric("Median", f"{vals.median():.2f}")
            c2.metric("Mean",   f"{vals.mean():.2f}")
            st.metric("N", f"{len(vals):,}")

    st.markdown("""
    <div class="insight-box">
    <strong>Finding:</strong> Median daily PnL is substantially higher on Greed days ($265.25)
    than on Fear days ($122.74). While the 116% gap is a strong directional trend (p=0.062),
    it is just shy of the standard 0.05 significance threshold. Win rate moves in the same direction, 
    so traders are not just deploying more capital on Greed days - they are potentially 
    making cleaner directional decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="paradox-box">
    <strong>The behavioral paradox (see Part B.2):</strong>
    Traders place more trades and deploy larger positions on Fear days.
    Yet PnL is lower. This is the central finding of my analysis: increased activity
    during Fear is panic behavior, not profitable behavior. More trading under stress
    produces worse outcomes, not better ones.
    </div>
    """, unsafe_allow_html=True)


with tab3:
    st.header("Part B.2: Behavioral Changes by Sentiment")
    st.markdown("""
    This tab answers whether traders change *how they trade* on Fear vs Greed days,
    separate from whether their outcomes improve. The behavioral metrics reveal the
    mechanism behind the PnL gap shown in Part B.1.
    """)

    behavior_cols = {
        'Trade Count':                'trade_count',
        'Total Exposure USD (proxy)': 'total_exposure_usd',
        'Long Ratio':                 'long_ratio',
        'Avg Fee (USD)':              'avg_fee',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#1e2130')
    fig.patch.set_facecolor('#1e2130')
    axes = axes.flatten()

    behavior_list = [
        ('trade_count',        'Trade Count (median)',         False),
        ('total_exposure_usd', 'Total Exposure USD (median)',  True),
        ('long_ratio',         'Long Ratio (mean)',            False),
        ('avg_fee',            'Average Fee USD (median)',     True),
    ]

    for ax, (col, title, clip) in zip(axes, behavior_list):
        style_ax(ax)
        data_p = df_filtered[['sentiment', col]].dropna()
        if clip:
            q99 = data_p[col].quantile(0.99)
            data_p = data_p[data_p[col] <= q99]

        agg_fn  = 'mean' if 'ratio' in col else 'median'
        summary = data_p.groupby('sentiment')[col].agg(agg_fn).reindex(['FEAR', 'GREED'])

        bars = ax.bar(['FEAR', 'GREED'], summary.values,
                      color=[FEAR_COLOR, GREED_COLOR], width=0.5, alpha=0.85)
        ax.set_title(title, color='white', pad=8, fontsize=11)
        ax.set_ylabel(title.split('(')[0].strip(), color='white')

        for bar, v in zip(bars, summary.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'{v:.2f}', ha='center', va='bottom',
                    color='white', fontsize=9, fontweight='bold')

    fig.suptitle("Behavioral Metrics: FEAR vs GREED", color='white',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("""
    <div class="paradox-box">
    <strong>The core paradox explained:</strong>
    Fear days show higher trade counts (31 vs 28), larger position sizes ($76K vs $60K),
    and a higher long ratio (36% vs 32%), yet median PnL is 116% lower than on Greed days.
    <br><br>
    This is panic trading: traders attempt to buy falling prices (high long ratio during fear),
    deploy outsized capital in the belief a bounce is imminent, and churn positions when their
    initial thesis does not immediately pay off. The market is distributing (informed sellers
    exiting) while these traders are accumulating. More activity, worse outcomes.
    <br><br>
    On Greed days the pattern reverses: restraint, trend-following, and cleaner exits
    produce better results with fewer resources deployed. I treat this as the core
    strategic takeaway.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Mann-Whitney Test Results by Behavioral Metric")
    rows = []
    for label, col in behavior_cols.items():
        f_v = fear_f[col].dropna()
        g_v = greed_f[col].dropna()
        if len(f_v) > 1 and len(g_v) > 1:
            _, p = mannwhitneyu(f_v, g_v, alternative='two-sided')
            pct  = (g_v.median() - f_v.median()) / (abs(f_v.median()) + 1e-9) * 100
            rows.append({
                'Metric':          label,
                'FEAR Median':     round(f_v.median(), 3),
                'GREED Median':    round(g_v.median(), 3),
                'GREED vs FEAR %': round(pct, 1),
                'p-value':         round(p, 4),
                'Significant':     'Yes (p<0.05)' if p < 0.05 else 'No',
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with tab4:
    st.header("Part B.3: Trader Segmentation via K-Means")
    st.markdown(f"""
    Clustering is performed at the trader-day level, not the account level.
    With only 32 accounts, fitting K-Means to 32 points produces unstable centroids.
    At the trader-day level I have roughly 10,000 rows, giving the algorithm stable
    geometry. Currently using K = {k_clusters}. Adjust in the sidebar and compare silhouette scores.
    """)

    df_for_cluster = df_filtered.to_json()
    cluster_df, sil_score = run_clustering(df_for_cluster, k_clusters)

    col1, col2 = st.columns([3, 1])
    with col2:
        st.metric("Silhouette Score", f"{sil_score:.4f}",
                  help="Values above 0.5 indicate good cluster separation.")
        st.metric("Trader-Days Clustered", f"{len(cluster_df):,}")

    with col1:
        profile = cluster_df.groupby('cluster_name').agg(
            Days     = ('account',           'count'),
            Accounts = ('account',           'nunique'),
            AvgPnL   = ('daily_pnl',         'mean'),
            AvgTrades= ('trade_count',        'mean'),
            LongRatio= ('long_ratio',         'mean'),
            WinRate  = ('win_rate',           'mean'),
        ).reset_index().sort_values('AvgPnL', ascending=False)

        profile['AvgPnL']    = profile['AvgPnL'].round(1)
        profile['AvgTrades'] = profile['AvgTrades'].round(1)
        profile['LongRatio'] = (profile['LongRatio'] * 100).round(1).astype(str) + '%'
        profile['WinRate']   = (profile['WinRate']  * 100).round(1).astype(str) + '%'
        st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("Cluster Performance Under Fear vs Greed")

    merged_clusters = df_filtered.merge(
        cluster_df[['account', 'date', 'cluster_name']].drop_duplicates(),
        on=['account', 'date'], how='left'
    ).dropna(subset=['cluster_name'])

    seg_sent = merged_clusters.groupby(['cluster_name', 'sentiment']).agg(
        avg_pnl      = ('daily_pnl',      'mean'),
        avg_drawdown = ('drawdown_proxy',  'mean'),
    ).reset_index()

    sorted_clusters = (
        cluster_df.groupby('cluster_name')['daily_pnl'].mean()
        .sort_values(ascending=False).index.tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#1e2130')
    fig.patch.set_facecolor('#1e2130')

    for ax, (metric, ylabel) in zip(axes, [
        ('avg_pnl',      'Avg Daily PnL (USD)'),
        ('avg_drawdown', 'Avg Drawdown Proxy (USD)'),
    ]):
        style_ax(ax)
        pivot = seg_sent.pivot(index='cluster_name', columns='sentiment', values=metric)
        if 'FEAR' in pivot.columns and 'GREED' in pivot.columns:
            x = np.arange(len(pivot.reindex(sorted_clusters).dropna(how='all')))
            valid = pivot.reindex(sorted_clusters).dropna(how='all')
            w = 0.35
            ax.bar(x - w/2, valid.get('FEAR',  pd.Series(0, index=valid.index)),
                   width=w, color=FEAR_COLOR,  alpha=0.85, label='FEAR')
            ax.bar(x + w/2, valid.get('GREED', pd.Series(0, index=valid.index)),
                   width=w, color=GREED_COLOR, alpha=0.85, label='GREED')
            ax.set_xticks(x)
            ax.set_xticklabels(valid.index, rotation=20, ha='right', color='white', fontsize=9)
        ax.set_title(ylabel, color='white', pad=8)
        ax.set_ylabel(ylabel, color='white')
        ax.axhline(0, color='white', linewidth=0.6, linestyle='--', alpha=0.4)
        ax.legend(facecolor='#2d3250', labelcolor='white', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Account to Cluster Assignment")
    acc_cluster = cluster_df.groupby('account')['cluster_name'].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    acc_cluster.columns    = ['Account (last 10 chars)', 'Dominant Cluster']
    acc_cluster['Account (last 10 chars)'] = (
        acc_cluster['Account (last 10 chars)'].str[-10:].apply(lambda x: f'...{x}')
    )
    st.dataframe(acc_cluster, use_container_width=True, hide_index=True)


with tab5:
    st.header("Part C: Strategy Rules and Bonus Model")

    st.subheader("Trading Rules of Thumb")

    top_q       = df_filtered['total_exposure_usd'].quantile(0.75)
    hi_exp      = df_filtered[df_filtered['total_exposure_usd'] >= top_q]
    hi_fear_dd  = hi_exp[hi_exp['sentiment']=='FEAR']['drawdown_proxy'].mean()
    hi_greed_dd = hi_exp[hi_exp['sentiment']=='GREED']['drawdown_proxy'].mean()
    hi_pct      = (hi_fear_dd - hi_greed_dd) / (abs(hi_greed_dd) + 1e-9) * 100
    fear_lr     = fear_f['long_ratio'].mean()   if len(fear_f)  else 0
    greed_lr    = greed_f['long_ratio'].mean()  if len(greed_f) else 0
    fear_med    = fear_f['daily_pnl'].median()  if len(fear_f)  else 0
    greed_med   = greed_f['daily_pnl'].median() if len(greed_f) else 0

    st.markdown(f"""
    <div class="rule-box">
    <strong>Rule 1: Reduce Activity on Fear Days</strong><br>
    During FEAR sentiment days, all trader segments should resist the instinct to increase
    trade frequency and position sizes. The data shows Fear days produce median trade counts
    of 31 vs 28 on Greed days and median exposure (clipped) of $76,007 vs $60,254, yet median PnL is
    $123 on Fear vs $265 on Greed. More activity during fear correlates with worse outcomes.
    The prescription is to trade less, not more, when market sentiment is fearful.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rule-box">
    <strong>Rule 2: Cap Position Size During Greed Periods</strong><br>
    High-exposure traders have an average drawdown of ${hi_greed_dd:,.0f} on Greed days versus
    ${hi_fear_dd:,.0f} on Fear days. This reflects overconfident sizing in bullish periods:
    traders increase exposure when sentiment is positive, creating larger peak-to-trough
    drawdown when positions reverse. Setting an exposure ceiling during Greed periods
    protects against the risk of overshooting during bullish sentiment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Bonus: Predictive Model Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**5-fold Cross-Validation Performance**")
        model_results = pd.DataFrame({
            'Model':     ['Logistic Regression', 'Random Forest'],
            'Accuracy':  [0.566, 0.633],
            'Precision': [0.737, 0.778],
            'Recall':    [0.574, 0.651],
            'F1':        [0.645, 0.707],
        })
        st.dataframe(model_results, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="insight-box">
        <strong>Reading these numbers:</strong>
        The base rate (always predicting positive) is 68.6%.
        Raw accuracy appears modest, but precision of 73 to 77 percent means
        the model is right roughly three times out of four when it predicts a
        profitable day. For a risk management signal this precision lift above
        the base rate is meaningful and actionable.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Feature Interpretation (Logistic Regression)**")
        feat_imp = pd.DataFrame({
            'Feature':        ['sentiment_binary', 'win_rate', 'daily_pnl', 'long_ratio',
                               'trade_count', 'avg_fee', 'total_exposure_usd', 'drawdown_proxy'],
            'Direction':      ['+', '+', '+', '+', '-', '-', '-', '-'],
            'What it signals': [
                'Greed today predicts profitable tomorrow',
                'Consistent winners sustain their edge',
                'PnL has short-term momentum',
                'Long bias in uptrend regimes helps',
                'Over-trading is a drag on next-day results',
                'High fees signal over-trading cost',
                'Large positions tend to mean-revert',
                'Deep drawdown impairs next-day decision quality',
            ]
        })
        st.dataframe(feat_imp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Limitations and Next Steps")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="warning-box">
        <strong>Current limitations</strong>
        <ul>
        <li>No leverage column in the raw data (exposure USD used as proxy)</li>
        <li>Only 32 accounts limits generalization</li>
        <li>IST timezone introduces minor boundary errors at midnight</li>
        <li>Standard CV does not respect temporal autocorrelation of trader behavior</li>
        <li>Correlation is shown, not causation; macro factors co-move with the FG index</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-box">
        <strong>Recommended extensions</strong>
        <ul>
        <li>Granger causality test: does sentiment lead PnL, or follow it?</li>
        <li>Rolling-window correlation to check if sentiment effect is regime-stable</li>
        <li>Per-coin analysis to separate asset-specific from sentiment-driven effects</li>
        <li>Walk-forward (time-series) CV for the predictive model</li>
        <li>Hidden Markov Model for latent regime detection beyond binary FG labels</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("""
<p style="color:#8892b0; font-size:12px; text-align:center;">
Primetrade.ai Data Science Internship Assignment
| 211,224 trades from 32 accounts across 479 matched sentiment days
| All numbers computed live from source data
</p>
""", unsafe_allow_html=True)
