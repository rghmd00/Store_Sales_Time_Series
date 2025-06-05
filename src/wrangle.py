from sklearn.preprocessing import LabelEncoder
import pandas as pd 

def wrangle(holiday:None, oil=None, stores=None, transactions=None, train=None,test=None):
    

    if train is not None:
        df = train.copy()
    else:
        df = test.copy()


    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(holiday, on='date', how='left')
    df = df.merge(oil, on='date', how='left')
    df = df.merge(transactions, on=['store_nbr', 'date'], how='left')


    # df['transactions'] = df['transactions'].fillna(0)
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    df.rename(columns={'type_x': 'type_store', 'type_y': 'type_holiday'}, inplace=True)
    df.drop(columns=['id','type_holiday','locale','locale_name','description','transferred'], inplace=True)
    
    df.sort_values(["store_nbr", "date"], inplace=True)
    df["transactions"] = df.groupby("store_nbr")["transactions"].ffill().bfill()



    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["day"] = pd.to_datetime(df["date"]).dt.day
    df["dayofweek"] = pd.to_datetime(df["date"]).dt.dayofweek
    df = df.drop(columns=["date"])





    for col in ["family", "city", "state", "type_store"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])



    # Save processed data to a CSV file
    df.to_csv("./data/processed/processed_data.csv", index=False)

    return df 