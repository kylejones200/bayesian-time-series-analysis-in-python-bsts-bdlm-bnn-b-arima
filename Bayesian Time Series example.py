"""Generated from Jupyter notebook: Bayesian Time Series example

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pmd
import pymc as pm
from pmdarima import model_selection


def simulated_time_series_data() -> None:
    np.random.seed(42)

    n = 200

    x = np.arange(n)

    y = 5 + 0.5 * x + np.random.normal(0, 2, n)

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label="Observed Data")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.tight_layout()

    plt.savefig("observed_data.png")

    plt.show()

    with pm.Model() as model:
        phi = pm.Normal("phi", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        init_dist = pm.Normal.dist(0, 10)
        y_obs = pm.GaussianRandomWalk(
            "y_obs", sigma=sigma, init_dist=init_dist, shape=n
        )
        y_like = pm.Normal("y_like", mu=y_obs, sigma=sigma, observed=y)
        trace = pm.sample(1000, tune=1000, return_inferencedata=True, cores=1)

    az.plot_trace(trace)

    plt.tight_layout()

    plt.savefig("bayesian.png")

    plt.show()

    with model:
        posterior_predictive = pm.sample_posterior_predictive(trace)

    print(f"Original shape of posterior_samples: {posterior_samples.shape}")

    posterior_samples = posterior_samples.squeeze()

    print(f"Shape after squeeze: {posterior_samples.shape}")

    if posterior_samples.ndim > 2:
        posterior_samples = posterior_samples[:, 0, :]
        print(f"Shape after indexing: {posterior_samples.shape}")

    posterior_mean = posterior_samples.mean(axis=0)

    lower_bound = np.percentile(posterior_samples, 2.5, axis=0)

    upper_bound = np.percentile(posterior_samples, 97.5, axis=0)

    print(f"x shape: {x.shape}, posterior_mean shape: {posterior_mean.shape}")

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label="Observed Data")

    plt.plot(x, posterior_mean, label="Posterior Mean", color="r")

    plt.fill_between(
        x, lower_bound, upper_bound, color="r", alpha=0.3, label="95% Credible Interval"
    )

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.tight_layout()

    plt.savefig("observed_predicted.png")

    plt.show()


def inspect_the_shape_of_posterior_samples() -> None:
    print(f"Original shape of posterior_samples: {posterior_samples.shape}")

    posterior_samples = posterior_samples.squeeze()

    print(f"Shape after squeeze: {posterior_samples.shape}")

    if posterior_samples.ndim > 2:
        posterior_samples = posterior_samples[:, 0, :]
        print(f"Shape after indexing: {posterior_samples.shape}")

    posterior_mean = posterior_samples.mean(axis=0)

    lower_bound = np.percentile(posterior_samples, 2.5, axis=0)

    upper_bound = np.percentile(posterior_samples, 97.5, axis=0)

    print(f"x shape: {x.shape}, posterior_mean shape: {posterior_mean.shape}")

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, label="Observed Data")

    plt.plot(x, posterior_mean, label="Posterior Mean", color="r")

    plt.fill_between(
        x, lower_bound, upper_bound, color="r", alpha=0.3, label="95% Credible Interval"
    )

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.tight_layout()

    plt.show()


def compute_mape() -> None:
    mape = np.mean(np.abs((y - posterior_mean) / y)) * 100

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


def train_test_split() -> None:
    train, test = model_selection.train_test_split(y, train_size=150)

    arima = pmd.auto_arima(
        train,
        error_action="ignore",
        trace=True,
        suppress_warnings=True,
        seasonal=False,
        maxiter=5,
    )

    forecasts = arima.predict(n_periods=test.shape[0])

    mape = np.mean(np.abs((test - forecasts) / test)) * 100

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    x_test = np.arange(test.shape[0])

    plt.figure(figsize=(10, 6))

    plt.plot(x_test, test, color="red", label="Actual")

    plt.plot(x_test, forecasts, color="blue", label="Forecast")

    plt.title("Actual Test Samples vs. Forecasts")

    plt.xlabel("Time Index")

    plt.ylabel("Value")

    plt.legend()

    plt.tight_layout()

    plt.savefig("pdmarima.png")

    plt.show()


def main() -> None:
    simulated_time_series_data()
    inspect_the_shape_of_posterior_samples()
    compute_mape()
    train_test_split()


if __name__ == "__main__":
    main()
