# Predicting Power Plant Energy Output

A machine learning model that predicts the net hourly electrical energy output of a Combined Cycle Power Plant from four environmental sensor readings, achieving an average error of just **0.5%** on unseen data.

## The problem

Combined Cycle Power Plants generate electricity using gas turbines, steam turbines and heat recovery steam generators. Their output shifts with environmental conditions, so accurately forecasting it helps operators plan generation and supply with confidence.

This project uses **9,568 hourly readings** to predict energy output (PE, 420 to 496 MW) from:

| Feature | Description | Correlation with output |
|---|---|---|
| AT | Ambient Temperature (°C) | 0.95 negative |
| V | Exhaust Vacuum (cm Hg) | 0.87 negative |
| AP | Ambient Pressure (millibar) | 0.52 positive |
| RH | Relative Humidity (%) | 0.39 positive |

## Approach

1. **Task type:** supervised regression, since the target is a continuous value with historical answers available.
2. **Data split:** 80% training (7,654 rows) and 20% held back as an untouched test set (1,914 rows).
3. **Validation:** five fold cross validation on the training set to compare models fairly.
4. **Metric:** Mean Absolute Error (MAE) in MW, supported by RMSE, MAPE and R².

## Model comparison (cross validation MAE)

| Model | MAE (MW) |
|---|---|
| Linear Regression, AT only | 4.29 |
| Linear Regression, all 4 features | 3.63 |
| **Random Forest, all 4 features** | **2.47** |

The Random Forest outperformed both linear models by capturing nonlinear relationships between the features and output.

## Final results on the test set

| Metric | Result |
|---|---|
| Mean Absolute Error | 2.3 MW |
| Root Mean Squared Error | 3.23 MW |
| Mean Absolute Percentage Error | 0.51% |
| R² | 0.964 |

The model predicts hourly output to within about 2.3 MW on average and explains over 96% of the variation in energy output. Ambient temperature drives around 90% of its decisions, matching the underlying physics: hotter air is less dense, so turbines produce less power.

## Run it yourself

Open `CCPP_Energy_Prediction.ipynb` in [Google Colab](https://colab.research.google.com) or Jupyter, upload `CCPP_data.csv` and run all cells.

**Tools:** Python, pandas, scikit learn, matplotlib

## Data source

Pınar Tüfekci, *Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods*, International Journal of Electrical Power & Energy Systems, Volume 60, September 2014, Pages 126 to 140.
