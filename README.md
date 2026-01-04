# Casino Analytics Proof of Concept

**Work in Progress**

This repository contains synthetic data generation tools and exploratory analysis for an online casino predictive analytics proof of concept.

## Project Structure
- `data/` - Casino game datasets
- `knime/` - KNIME workflows for ETL
- `notebooks/` - Exploratory data analysis
- `src/` - Python prediction models
- `sql/` - Data transformation queries

## Tech Highlights
- **Data Processing:** KNIME, SQL, Python/Pandas
- **Analysis:** Jupyter notebooks
- **Prediction:** Time series models
- **Stack:** Multi-tool approach (KNIME + Python + SQL)

## Why KNIME?

Used for:
- **Rapid prototyping** - Test data pipeline logic in hours, not days
- **Stakeholder demos** - Visual workflows non-technical teams can understand
- **Validation** - Prove concepts before investing in production code

Production pipelines built in Databricks after validation.
  
## Contents

- **`generator.py`** - Python script for generating realistic synthetic casino transaction data
  - Simulates 10,000 players across 4 games over 12 months
  - Implements player segmentation (Whale, VIP, High, Regular, Casual)
  - Models behavioral patterns including loss chasing, hot hand effect, and session dynamics
  - Uses statistical distributions (log-normal, Pareto, gamma) based on gambling research

- **Databricks Notebooks** - Data profiling and exploratory analysis
  - Player behavior patterns
  - Game performance metrics
  - Temporal trends and seasonality

- **Plotly Visualizations** - Interactive descriptive analytics
  - Player segment distributions
  - Revenue breakdowns
  - Behavioral metrics

## Current Status

✅ Synthetic data generation  
✅ Descriptive analytics  
🚧 Predictive models (coming soon)

## Future Work

- Time series forecasting (ARIMA, Prophet, LSTM)
- Player churn prediction
- GGR projections
- Responsible gambling risk scoring

## References

Mathematical modeling based on:
- Deng et al. (2021) - Pareto distributions in online casino gambling
- Academic research on gambling behavior and RTP mechanics


![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PySpark](https://img.shields.io/badge/pyspark-3.5-orange.svg)
![Databricks](https://img.shields.io/badge/databricks-MLFlow-red.svg)
![Knime](https://img.shields.io/badge/knime-local-yellow.svg)

---

*This is a demonstration project for analytics capabilities. All data is synthetically generated.*
