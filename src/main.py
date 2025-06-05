from load_data import load_data
from wrangle import wrangle
import pandas as pd
from train import train_model


if __name__ == "__main__":
    holidays, oil, stores, transactions, train = load_data()
    df = wrangle(holiday=holidays, oil=oil, stores=stores, transactions=transactions, train=train)
    # print(df.head())
    train_model(df)
