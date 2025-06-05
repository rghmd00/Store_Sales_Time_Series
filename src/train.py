import xgboost as xgb
import numpy as np
import pickle

def train_model(df):
    
    y_train = df["sales"]
    X_train = df.drop(columns=["sales"])


    df["log_sales"] = np.log1p(df["sales"])  # log(1 + sales)
    y_train_logged = df["log_sales"]



    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist" 
    )

    model.fit(X_train, y_train_logged)
    with open("./models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

