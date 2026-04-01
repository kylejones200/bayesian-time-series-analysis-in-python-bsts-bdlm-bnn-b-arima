# Description: Short example for Bayesian Time Series Analysis in Python BSTS BDLM BNN B Arima.



from data_io import read_csv
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import Adam
from sklearn.metrics import mean_squared_error
from tqdm import trange
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pmd
import pybsts
import pymc as pm
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# === Data Loading ===
def load_data(file_name="ercot_load_data.csv"):
    df = read_csv(file_name, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = df.resample('h').mean().dropna()
    return df

# === Modular Visualization Function ===
def plot_forecast(df, forecast_index, forecast_mean, forecast_lower, forecast_upper, model_name):
    """
    Plots the historical data, forecast, and confidence intervals for the last 25 points.

    Parameters:
    - df: DataFrame containing the historical data
    - forecast_index: Index for the forecasted values
    - forecast_mean: Forecasted mean values
    - forecast_lower: Lower bound of the confidence interval
    - forecast_upper: Upper bound of the confidence interval
    - model_name: Name of the model for the title and filename
    """
    # Plot all historical data
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['values'], label='Actual', color='blue')

    # Plot the forecast for the last 25 points
    plt.plot(forecast_index, forecast_mean, label='Forecast', color='red', linestyle='--')
    plt.fill_between(
        forecast_index, 
        forecast_lower, 
        forecast_upper, 
        color='red', alpha=0.2, label='95% Confidence Interval'
    )

    # Add dashed vertical line where holdout set begins
    holdout_start = df.index[-len(forecast_index)]
    plt.axvline(x=holdout_start, color='black', linestyle='--', label='Holdout Start')

    # Customizations
    plt.title(f'{model_name} Forecast')
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_forecast.png')
    plt.show()

# === Simplified Bayesian Structural Time Series (BSTS) ===
def bayesian_sts(df, forecast_horizon=25):
    train = df['values'].iloc[:-forecast_horizon].values

    # Define and fit the BSTS model
    specification = {
        "ar_order": 1, 
        "local_trend": {"local_level": True},
        "sigma_prior": np.std(train, ddof=1), 
        "initial_value": train[0]
    }

    model = pybsts.PyBsts(
        "gaussian", 
        specification, 
        {
            "ping": 10, 
            "niter": 100, 
            "burn": 10, 
            "forecast_horizon": forecast_horizon, 
            "seed": 42
        }
    )

    model.fit(train, seed=42)
    forecast = model.predict(seed=42)
    forecast_mean = np.mean(forecast, axis=0)
    forecast_std = np.std(forecast, axis=0)

    # Forecast index for the last 25 points
    forecast_index = df.index[-forecast_horizon:]

    # Use modular visualization
    plot_forecast(
        df, 
        forecast_index, 
        forecast_mean, 
        forecast_mean - 1.96 * forecast_std, 
        forecast_mean + 1.96 * forecast_std, 
        model_name="BSTS"
    )

    return forecast_mean

# === Simplified BDLM with Confidence Interval ===
def bayesian_bdlm(df, forecast_horizon=25):
    train = df.iloc[:-forecast_horizon]

    # Fit BDLM model
    with pm.Model() as model:
        sigma = pm.HalfNormal('sigma', sigma=1)
        trend_sigma = pm.HalfNormal('trend_sigma', sigma=0.1)
        seasonal_sigma = pm.HalfNormal('seasonal_sigma', sigma=0.1)
        
        trend = pm.GaussianRandomWalk('trend', sigma=trend_sigma, shape=len(train))
        
        period = 24
        seasonal = pm.Normal('seasonal', mu=0, sigma=seasonal_sigma, shape=period)
        
        idx = np.arange(len(train)) % period
        mu = trend + seasonal[idx]
        
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=train['values'])
        
        trace = pm.sample(2000, tune=1000, return_inferencedata=False)
    
    # Forecasting
    trend_pred = np.mean(trace['trend'], axis=0)
    seasonal_pred = np.mean(trace['seasonal'], axis=0)
    predictions = trend_pred + seasonal_pred[idx]

    # Calculate 95% credible intervals
    lower_bound = np.percentile(trace['trend'], 2.5, axis=0) + seasonal_pred[idx]
    upper_bound = np.percentile(trace['trend'], 97.5, axis=0) + seasonal_pred[idx]

    # Forecast index for the last 25 points
    forecast_index = df.index[-forecast_horizon:]

    # Use modular visualization
    plot_forecast(
        df, 
        forecast_index, 
        predictions[-forecast_horizon:], 
        lower_bound[-forecast_horizon:], 
        upper_bound[-forecast_horizon:], 
        model_name="BDLM"
    )

    return predictions[-forecast_horizon:]

# === Simplified Bayesian Neural Network (BNN) ===
def bayesian_nn(df, forecast_horizon=25):
    # Prepare data for BNN
    def prepare_data(data, lookback=7):
        X, y = [], []
        values = data['values'].values
        for i in range(len(values) - lookback):
            X.append(values[i:i+lookback])
            y.append(values[i+lookback])
        return torch.FloatTensor(X), torch.FloatTensor(y)

    X_train, y_train = prepare_data(df.iloc[:-forecast_horizon])
    X_test, y_test = prepare_data(df.iloc[-(forecast_horizon+7):])

    # Define BNN model
    class TimeSeriesBNN(PyroModule):
        def __init__(self, input_dim=7, hidden_dim=32, output_dim=1):
            super().__init__()
            self.hidden1 = PyroModule[nn.Linear](input_dim, hidden_dim)
            self.hidden2 = PyroModule[nn.Linear](hidden_dim, hidden_dim)
            self.output = PyroModule[nn.Linear](hidden_dim, output_dim)
            self.activation = nn.ReLU()

            self.hidden1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
            self.hidden1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
            self.hidden2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, hidden_dim]).to_event(2))
            self.hidden2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
            self.output.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
            self.output.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

        def forward(self, x, y=None):
            x = self.activation(self.hidden1(x))
            x = self.activation(self.hidden2(x))
            mu = self.output(x).squeeze(-1)
            
            sigma = pyro.sample("sigma", dist.Gamma(1.0, 1.0))
            with pyro.plate("data", x.shape[0]):
                obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
            return mu

    # Train the BNN model
    model = TimeSeriesBNN()
    guide = AutoDiagonalNormal(model)
    adam = Adam({"lr": 0.01})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())
    
    num_epochs = 1000
    for epoch in trange(num_epochs):
        svi.step(X_train, y_train)

    # Predict using the BNN model
    def predict_bnn(model, guide, X_input, n_samples=100):
        predictive = pyro.infer.Predictive(model, guide=guide, num_samples=n_samples)
        samples = predictive(X_input)
        preds = samples['obs'].detach().numpy()
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        return mean_pred, std_pred

    mean_pred, std_pred = predict_bnn(model, guide, X_test)

    # Forecast index for the last 25 points
    forecast_index = df.index[-forecast_horizon:]

    # Use modular visualization
    plot_forecast(
        df, 
        forecast_index, 
        mean_pred, 
        mean_pred - 1.96 * std_pred, 
        mean_pred + 1.96 * std_pred, 
        model_name="BNN"
    )

    return mean_pred

# ===Bayesian ARIMA ===
def bayesian_arima(df, forecast_horizon=25):
    train = df['values'].iloc[:-forecast_horizon].values

    # Train ARIMA on the training set
    arima = pmd.auto_arima(
        train,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        m=24,
        seasonal=True,
        d=None,
        D=1,
        test='adf',
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Forecast the last 25 points
    barima_forecast, conf_int = arima.predict(n_periods=forecast_horizon, return_conf_int=True)
    forecast_index = df.index[-forecast_horizon:]

    # Use modular visualization
    plot_forecast(
        df, 
        forecast_index, 
        barima_forecast, 
        conf_int[:, 0], 
        conf_int[:, 1], 
        model_name="Bayesian ARIMA"
    )

    return barima_forecast

Bayesian ARIMA   90.518746   9.514134  2.311101  2.343940
BNN with Pyro   748.787444  27.363981  7.818061  7.414179
BDLM            326.776625  18.076964  4.722612  4.531827
BSTS            383.895118  19.593242  5.781497  5.676614

# === Main Function ===
def main():
    df = load_data()

    barima_forecast = bayesian_arima(df)
    bnn_forecast = bayesian_nn(df)
    bdlm_forecast = bayesian_bdlm(df)
    bsts_forecast = bayesian_sts(df)

    y_true = df['values'].values[-25:]
    
    metrics = {
        "Bayesian ARIMA": calculate_metrics(y_true, barima_forecast),
        "BNN with Pyro": calculate_metrics(y_true, bnn_forecast),
        "BDLM": calculate_metrics(y_true, bdlm_forecast),
        "BSTS": calculate_metrics(y_true, bsts_forecast)
    }

    # Convert to DataFrame for better readability
    df_metrics = pd.DataFrame(metrics, index=["MSE", "RMSE", "MAPE", "sMAPE"]).T
    logger.info(df_metrics)

    # Plot comparison
    df_metrics.plot(kind='bar', figsize=(15, 8))
    plt.title('Model Comparison')
    plt.ylabel('Error')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
