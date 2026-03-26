# Primetrade.ai Trader Sentiment Analysis (Hyperliquid)

This study analyzes how Bitcoin market sentiment (Fear/Greed Index) impacts the behavior and performance of traders on Hyperliquid. It identifies a "Behavioral Paradox" where traders are more active during market stress yet earn significantly less profit ($123 vs $265 median PnL).

## Installation and Setup
To run this project locally, ensure you have Python 3.10 or a newer version installed. 

First, clone this repository (or copy the files into a new folder) and install the necessary data science libraries by running this command in your terminal:

pip install pandas numpy matplotlib seaborn scikit-learn statsmodels streamlit nbformat

## How to Run

### 1. Generating the Analysis Notebook
I provide a script that takes the raw data and generates the final analysis notebook automatically. This ensures the study is fully reproducible. To run it:

python create_notebook.py

This will create a new file named primetrade_analysis.ipynb. You can open this file in any Jupyter environment (like JupyterLab, VS Code, or Google Colab) to walk through the data cleaning, statistical tests, and visualizations.

### 2. Launching the Interactive Dashboard
To interact with the data in real-time and explore specific accounts or clusters, I created a Streamlit dashboard. Launch it by running:

streamlit run dashboard.py

After running this command, your terminal will provide a local URL (typically http://localhost:8501). Open this link in your browser to access the interactive explorer. You can adjust PnL outlier filters or browse the "Behavioral Paradox" tab to see the side-by-side performance of different sentiments.

## Key Insights and Strategy
- **The Paradox**: Traders act more frantically during Fear regimes—placing more trades (31 median vs 28) and taking larger positions—even though it results in 116% lower PnL ($123 vs $265).
- **Rule 1: Activity Ceiling on Fear Days**: The data shows that more trades under stress correlates with worse outcomes. 
- **Rule 2: Position Capping during Greed**: High-exposure traders suffer 14% higher drawdowns ($13,980) on Greed days than on Fear days because they tend to overshoot their risk during bullish regimes.
- **Predictive Model**: I built a model that identifies next-day profitability with 77% precision.

## File Structure
- primetrade_analysis.ipynb: My full analysis notebook and final findings.
- dashboard.py: The interactive Streamlit dashboard source.
- create_notebook.py: The reproducibility script that creates the notebook.
- processed_trader_sentiment.csv: The aggregated trader-day dataset output.
- fear_greed_index.csv: Daily Bitcoin sentiment data (2018-2025).
- historical_data.csv: Trade-level data from 32 Hyperliquid traders.

## My Approach
I starting by aggregates over 210,000 individual trades from 32 accounts into a daily format. I then used statistical tests and clustering to see if a trader's "vibe" changes when the market turns fearful. The project moves from raw CSVs to engineered features, through statistical tests and K-Means segmentation (Silhouette 0.73), identifying how different trader archetypes respond to market sentiment.
