import pickle
import xgboost as xgb


# #### Load the model

def predict(data):
    with open('bank-model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)


    sample = dv.transform([data])
    feature_names = dv.get_feature_names_out().tolist()

    X =xgb.DMatrix(sample, feature_names=feature_names)


    y_pred = model.predict(X)

    result = (y_pred >= 0.5).astype(int)
    return int(result)

