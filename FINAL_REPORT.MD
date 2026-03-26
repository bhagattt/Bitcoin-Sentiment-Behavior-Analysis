# Analysis Report: Bitcoin Sentiment & Trader Behavior (Hyperliquid)

**Author:** Aryan Bhagat 
**Date:** March 26, 2026

---

##  My Approach & Methodology

When I first opened the dataset, my primary goal was to see if the "vibe" of the Bitcoin market (captured by the Fear/Greed Index) actually translates into measurable changes in how professional traders on Hyperliquid behave. I started with 211,224 raw trade-level rows from 32 distinct accounts, which I engineered into a unified **trader-day** dataset.

For every trader on each day, I calculated eight core engagement metrics: total PnL, win rate, trade frequency, exposure (as a leverage proxy), fees paid, and long-bias ratio. I then joined this to the daily sentiment index across 479 unique dates, resulting in 1,964 analysis-ready records (focusing exclusively on Fear and Greed regimes).

To ensure my findings were statistically sound, I chose non-parametric **Mann-Whitney U** tests rather than standard T-tests, as trading data is notoriously heavy-tailed and non-normal. I also applied **K-Means Clustering** to separate "The Heavy-Hitters" from "Active Balanced" traders to see if different archetypes react to fear differently.

---

##  Key Insights & Findings

### 1. The Behavioral Paradox: Panic Trading vs. Performance
The most fascinating discovery for me was what I’ve termed the "Behavioral Paradox." My data shows that traders are **significantly more active** when the market is fearful:
- **Trade Count:** Median of **31 trades** on Fear days vs **28 trades** on Greed (p=0.039).
- **Position Sizing:** Median daily exposure of **$76,007** on Fear days vs **$60,254** on Greed (p=0.007).

Despite this frantic activity and larger capital deployment, the median daily PnL was **116% higher during Greed periods ($265 vs $123)**. This reveals a clear "panic trading" signature where traders buy the dip (Fear long ratio: 35.8% mean) into falling markets, paying more in fees and earning less in return.

### 2. High-Exposure Overconfidence & Drawdown Risk
I noticed a counter-intuitive trend among my top-quartile traders: their average drawdown is actually **higher during Greed periods ($13,980) than during Fear ($12,243)**. This suggests that overconfident positioning during bullish regimes builds up peak equity that creates far more painful peak-to-trough losses when the trend eventually reverses.

### 3. Advanced Predictive Modeling (Walk-Forward Validation)
By moving beyond basic snapshots, I built a predictive model that captures trader "momentum." I added **3-day rolling averages** for PnL and trade activity, allowing the model to see if a trader is on a winning or losing streak. 

Most importantly, I implemented a **Walk-Forward (TimeSeriesSplit) validation strategy**. In quantitative finance, standard random splits are often invalid because they inadvertently "shout" from the future into the past. By testing only on the future, I achieved a **~72% precision** using a modern **Gradient Boosting** aggregator. When the model predicts a profitable next day, it is right nearly 3 out of 4 times—a reliable risk filter that could allow a platform to dynamically adjust position limits or issue proactive warnings.

---

##  My Strategy Recommendations

Based on these numbers, I’ve formulated two practical risk-management principles for the Primetrade.ai platform:

1.  **Impose a "Fear-Day" Activity Ceiling**: Since more activity during fear correlates with 116% lower PnL, the platform should alert traders when they exceed their baseline trade frequency during Fear regimes.
2.  **Cap Exposure during Greed Overshoots**: To prevent the $13,980 average drawdown risk I observed in heavy hitters, accounts should be advised to cap their position sizing relative to their trailing 30-day equity peak during Extreme Greed periods.

---

##  Unexpected Findings & Anomalies

During my analysis, I stumbled upon a few strange and interesting points that weren't immediately obvious:

- **The Greed Drawdown Paradox**: You would expect the biggest market drawdowns to happen during Fear regimes. However, for the high-exposure group (the top 25%), I found the average drawdown was actually **higher during Greed ($13,980) than during Fear ($12,243)**. This "weird" result happened because traders became overconfident during bullish sentiment, creating massive peak equity that subsequently collapsed when the trend reversed.
- **The Zero-PnL Trap**: Approximately 50% of the raw trade data rows have a `Closed PnL` of exactly zero. These aren't just "breakeven" trades; they are likely position entries or partial fills. If I hadn't filtered these out of the win-rate calculation, I would have reported a "breakeven" rate that was artificially high and completely misleading.
- **Sentiment Feedback Loop**: I tested for "causality" and found something interesting: market sentiment predicts trading activity with a **1-day lag (p=0.023)**, but trading volume predicts sentiment with a **3-day lag (p=0.048)**. This suggests a feedback loop where sentiment drives immediate retail action, which then takes a few days to bubble back up into the broader market mood.
- **PnL Concentration**: The distribution of performance is incredibly heavy-tailed. My clustering identified that **just 11 accounts** (Cluster 0) are responsible for almost all the high-performing trader-days, while the other 97% of the data averages a tiny fraction of that PnL.

## Final Thoughts
Through these findings, I’ve confirmed that sentiment isn't just a noisy reaction to price — it is an active driver of retail behavior that requires sophisticated, sentiment-aware risk management.
