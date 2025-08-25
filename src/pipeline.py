"""Simple pipeline entrypoints for data -> model -> strategy"""

from src.data import load_data
from src.models import Model
from src.trading import Strategy


def run_pipeline(config):
    data = load_data(config["data"])  # placeholder
    model = Model(config["model"])   # placeholder
    strat = Strategy(config["strategy"])  # placeholder
    predictions = model.predict(data)
    trades = strat.generate_trades(predictions)
    return trades


if __name__ == "__main__":
    print("Run pipeline with a config")
