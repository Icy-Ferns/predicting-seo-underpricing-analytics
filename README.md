# Predicting SEO Underpricing for Trading Strategy Optimization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost-orange.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance-brightgreen.svg)]()

> **A data-driven approach to capitalize on Seasoned Equity Offering (SEO) underpricing inefficiencies using machine learning**

---

## Executive Summary

This project explores **SEO underpricing**—a recurring market inefficiency where new equity shares are priced below their immediate trading value. Using **11,756 historical SEO observations (2000-2023)** and comprehensive macroeconomic/microeconomic data, We developed a machine learning framework to predict underpricing and simulate a trading strategy.

### Key Results
| Metric | Value | Significance |
|--------|-------|-------------|
| **Total Return** | **22.6%** | Cumulative gain from strategy execution |
| **Sharpe Ratio** | **1.15** | Strong risk-adjusted returns (>1 = efficient) |
| **CAGR** | **5.2%** | Steady annualized growth |
| **Max Drawdown** | **-3.9%** | Exceptional capital preservation |
| **Daily Alpha** | **0.03%** | Consistent outperformance vs S&P 500 |
| **Model R²** | **0.0053** | Low explanatory power (see Limitations) |

### Strategic Value
For quantitative trading firms, this framework demonstrates:
- **Exploiting market inefficiencies**: Captured alpha from systematic underpricing patterns
- **Risk management**: Low drawdown (-3.9%) ensures capital preservation
- **Scalable infrastructure**: Production-ready ML pipeline with real-time scoring capabilities

---

## Critical Learning: The R² Paradox

### The Challenge
My best model (LightGBM) achieved an **R² of only 0.0053**, meaning it explains less than 1% of variance in underpricing—well below the 10-15% threshold for practical utility in academic finance.

### The Disconnect
Despite low statistical explanatory power, the trading simulation showed strong returns (22.6% total, Sharpe 1.15). This **mismatch suggests potential data leakage or overfitting** in the backtesting framework.

### Key Learnings
1. **Statistical performance ≠ Trading performance**: A model can have low R² but still capture directional signals
2. **Data leakage risks**: Without rigorous temporal validation, backtests can be misleadingly optimistic
3. **Feature engineering gaps**: The selected features may not capture all drivers of underpricing (e.g., investor sentiment, prospectus language)
4. **Market complexity**: Non-linear relationships in finance are notoriously difficult to model

### What I Would Do Differently
- **Add statistical significance testing**: Confidence intervals for performance metrics
- **Implement walk-forward validation**: Stricter temporal splits to prevent look-ahead bias
- **Incorporate NLP features**: Text analysis of SEC filings (8-K, 424B) for sentiment signals
- **More conservative backtesting**: Include slippage, bid-ask spreads, and liquidity constraints
- **Monitor model decay**: Real-time drift detection to identify when patterns change

---

## Project Overview

### What is SEO Underpricing?
When a company that is already publicly traded sells new shares, it often sells them at a discount compared to the current stock price so investors are willing to buy them. That discount is called **Seasoned equity offering**'(SEO)' underpricing.

\[ \text{Underpricing \%} = \frac{\text{Price at Close (Issue Date)} - \text{Offer Price (USD)}}{\text{Offer Price (USD)}} \times 100 \]

**Example**: If a bank underprices an SEO by 4%, and we issue $300 million worth of shares, that’s $12 million 
left on the table. If our model helps reduce that under-pricing error by just 1%, that’s $3 million in 
savings. That’s real money.

### Why This Matters
- **For issuers**: "Money left on the table"—underpricing costs firms billions annually
- **For investors**: Predictable short-term gains if patterns can be identified
- **For quant funds**: Alpha generation in a semi-efficient market

---

## Methodology

### Data Sources
| Category | Variables | Source |
|----------|-----------|--------|
| **SEO Data** | 14,308 observations, 43 variables (Offer Price, Trading Volume, Issue Type) | Proprietary client data |
| **Macroeconomic** | 25 indicators (GDP growth, CPI, Treasury yields, VIX) | FRED, BEA |
| **Industry Returns** | Fama-French 5-factor + Momentum, 17 industry portfolios | Kenneth French Data Library |
| **Firm Financials** | 11 metrics (P/E ratio, ROA, ROE, EPS) | Compustat (WRDS) |
| **Market Data** | Daily prices, dividends, volume | WRDS Security Daily |

### Data Preprocessing
- **Date alignment**: Ensured all features are aligned to SEO issue date (no look-ahead bias)
- **Missing data**: Imputed using median for continuous, mode for categorical
- **Outlier treatment**: Winsorized top/bottom 1% of continuous variables

### Feature Engineering
- **Log transformations**: `log(trading_volume)` to handle skewness
- **Temporal features**: Year, month, quarter extracted from issue dates
- **Lagged variables**: Macroeconomic indicators aligned to pre-offer dates (avoiding look-ahead bias)
- **One-hot encoding**: Industry categories (17 groups), exchange types
- **Lasso feature selection**: Reduced to 33 features with non-zero coefficients

### Models Evaluated
| Model | RMSE | R² | Notes |
|-------|------|-----|-------|
| **LightGBM** ⭐ | **0.05277** | **0.0053** | Best overall; fast training |
| XGBoost | 0.05272 | 0.0073 | Close second |
| Random Forest | 0.05278 | 0.0103 | Highest R² but slower |
| Gradient Boosting | 0.05300 | -0.0031 | Negative R² (worse than baseline) |
| Ridge | 0.05320 | -0.0122 | Linear models struggled |
| Lasso | 0.05310 | -0.0068 | Regularization didn't help |

**Validation**: TimeSeriesSplit (respects temporal order) + GridSearchCV for hyperparameter tuning

---

## Trading Strategy Simulation

### Strategy Logic
1. **Signal Generation**: Predict underpricing for each SEO using LightGBM
2. **Entry Criteria**: 
   - Expected daily return > **4.5%** (based on median underpricing from EDA)
   - Investment capped at **10%** of SEO market value OR **5%** of portfolio (whichever is lower)
3. **Risk Management**:
   - Diversification: Max 5% allocation per SEO
   - Sequential deployment: Capital locked until exit (no reuse)
4. **Exit**: Close position 1 day after issue date
5. **Costs**: 5% transaction costs (brokerage, slippage)

### Backtest Results (May 2021 - Early 2024)
- **Initial Capital**: $1,000,000
- **Final Value**: $1,226,000
- **Total Return**: 22.6%
- **Sharpe Ratio**: 1.15 (strong risk-adjusted performance)
- **Max Drawdown**: -3.9% (excellent capital preservation)

**Interpretation**: The strategy identified 20-30 profitable SEO opportunities annually, compounding gains over time.

---

## Key Predictors of Underpricing

Top 10 features by LightGBM importance:

| Feature | Business Interpretation |
|---------|------------------------|
| **Offer Price (USD)** | Higher prices → More dilution concerns → Greater underpricing |
| **Trading Volume** | Low volume = Low institutional interest → Negative signal |
| **Industry Monthly Return** | Strong sector performance attracts capital → Less underpricing |
| **Projected Growth Rate** | High growth offsets dilution fears → Positive sentiment |
| **VIX (Volatility)** | Market uncertainty increases risk premiums → More underpricing |
| **Return on Assets (ROA)** | Operational efficiency signals stability → Better pricing |
| **P/E Ratio** | Valuation metric for fair pricing assessment |
| **Ownership Type** | Institutional vs retail mix affects demand |
| **Share Count** | Larger offerings → More dilution → Negative impact |
| **EPS** | Weak earnings → Lower investor confidence |

---

## Repository Structure

```
├── Z_1_Data_Cleaning.ipynb          # Raw data formatting & validation
├── Z_2_x_Enrichment_*.ipynb         # Feature engineering (macro, micro, industry data)
├── Z_3_Data_Merging.ipynb           # Consolidation of all data sources
├── Z_4_EDA.ipynb                    # Exploratory data analysis & visualization
├── Z_5_ML_Dataset_Prep.ipynb        # Feature selection, scaling, train-test split
├── Z_6_Modelling_Evaluation.ipynb   # Model training, validation, trading simulation
├── Report-Predicting-SEO-Underpricing.pdf  # Comprehensive business report (34 pages)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Excludes proprietary data files
```

**Note**: Raw data files (`.parquet`, `.csv`) are **not included** due to confidentiality agreements.

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab


### Running the Notebooks
**Data Dependency**: Notebooks reference proprietary `.parquet` files not included in this repo. To replicate:
1. Replace file paths in `Z_1_Data_Cleaning.ipynb` with your own SEO dataset
2. Ensure data schema matches expected columns (see report Appendix A)

### Key Notebooks
- **Start here**: `Z_4_EDA.ipynb` for visualizations and pattern exploration
- **Modeling**: `Z_6_Modelling_Evaluation.ipynb` for LightGBM training and backtesting
- **Full pipeline**: Run notebooks sequentially (Z_1 → Z_6)

---

## Business Applications (Hypothetical Framework for Citadel)

### 1. Real-Time Scoring Infrastructure
- **Data Pipeline**: Stream SEO filings via SEC EDGAR API → Apache Kafka → Feature Store (Tecton/Feast)
- **Model Deployment**: FastAPI microservice (Docker) scoring new SEOs every 15 minutes
- **Output**: Ranked list of high-probability underpricing opportunities

### 2. Portfolio Integration
- **Proprietary Trading**: Tactical intraday positions on flagged SEOs (5-10 trades/month)
- **Asset Management**: Incorporate scores into multi-factor risk models for quarterly rebalancing
- **Risk Controls**: Monte Carlo stress testing (VIX spikes, sector shocks)

### 3. Advanced Features (Future Scope)
- **NLP Pipeline**: BERT-based sentiment analysis of prospectus filings (8-K, 424B)
- **Drift Detection**: Retrain every 3 months or when prediction variance > threshold
- **Alternative Data**: Social media sentiment (Twitter/StockTwits), Google Trends

---

## Performance Visualizations

### Portfolio Growth Over Time
The chart in the report (Figure 5.2) shows steady capital appreciation from $1M → $1.23M over 2.5 years, with minimal volatility.

### Underpricing Distribution by Industry
Manufacturing and Commercial Banking sectors exhibited higher underpricing variance (Figure 3.3), suggesting sector-specific strategies.

### Geographic Patterns
SEOs from Minnesota, Utah, and Nevada showed 2-3x higher underpricing than Illinois/Virginia (Figure 3.4), indicating regional market dynamics.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Languages** | Python 3.8+ |
| **ML Frameworks** | LightGBM, XGBoost, scikit-learn |
| **Data Processing** | pandas, numpy, pyarrow |
| **Visualization** | matplotlib, seaborn |
| **Data Sources** | WRDS API, SEC EDGAR, pandas_datareader |
| **Financial Libraries** | Alpha Vantage, numpy_financial |
| **Validation** | TimeSeriesSplit, GridSearchCV |

---

## Key Learnings & Reflections

### What Worked Well
 - **Comprehensive data integration**: 25+ macro variables + 11 firm metrics + industry returns  
 - **Rigorous preprocessing**: Outlier treatment, missing data imputation, feature engineering  
 - **Business-first approach**: Linked model outputs to actionable trading strategy  
 - **Production thinking**: Designed scalable deployment architecture (Kafka, microservices)

### Areas for Improvement
 - **Model explainability**: R² = 0.0053 suggests missing critical features (e.g., investor sentiment, order book data)  
 - **Backtesting rigor**: Potential look-ahead bias in strategy simulation  
 - **Statistical testing**: Lack of confidence intervals for return metrics  
 - **Real-world constraints**: Simplified assumptions on liquidity, transaction costs

### Professional Growth
This project taught me to:
1. **Balance statistical rigor with business value**: Even weak models can inform strategy if used carefully
2. **Question strong results**: The R²/returns disconnect forced me to examine data leakage risks
3. **Think like a PM**: Designed end-to-end deployment plan (data → model → monitoring)
4. **Communicate limitations**: Honest reflection builds credibility in technical roles

---

## Full Report

For detailed methodology, EDA visualizations, and strategic recommendations, see:  
📎 **[Report-Predicting-SEO-Underpricing.pdf](Report-Predicting-SEO-Underpricing.pdf)** (34 pages)

**Highlights**:
- Executive summary for non-technical stakeholders
- Comprehensive literature review (Kim & Shin 2001, Fama-French factors)
- Detailed backtesting assumptions and Monte Carlo simulations
- Strategic implementation roadmap for hedge funds

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Data Usage**: SEO data is proprietary and not included. Public data sources (FRED, Fama-French) are cited per their terms.

---

## Acknowledgments

- **WRDS (Wharton Research Data Services)**: Security daily data, Compustat financials
- **Kenneth French Data Library**: Fama-French factors & industry returns
- **Federal Reserve Economic Data (FRED)**: Macroeconomic indicators
- **SEC EDGAR**: Company filings and CIK/CUSIP mapping
- **scikit-learn & LightGBM communities**: Open-source ML tools

---

** Thank you for your time, and suggestions are welcome!**
