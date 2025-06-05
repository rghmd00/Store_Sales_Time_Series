import pandas as pd


def load_data():
    holidays = pd.read_csv("./data/raw/holidays_events.csv")
    oil = pd.read_csv("./data/raw/oil.csv")
    stores = pd.read_csv("./data/raw/stores.csv")
    transactions = pd.read_csv("./data/raw/transactions.csv")
    train = pd.read_csv("./data/raw/train.csv")

    return holidays, oil, stores, transactions, train