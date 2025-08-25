#!/usr/bin/env python3
"""Run a backtest using the pipeline and backtest engine"""
from src.pipeline import run_pipeline


def main():
    config = {"data":{"source":"data/raw"}, "model":{}, "strategy":{}}
    trades = run_pipeline(config)
    print("Generated trades:", trades)


if __name__ == '__main__':
    main()
