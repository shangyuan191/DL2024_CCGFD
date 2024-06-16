import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.ocsvm import OCSVM
from pyod.models.xgbod import XGBOD
from pyod.models.lunar import LUNAR
from pyod.utils.data import evaluate_print
import pickle
# 讀csv，回傳data frame
def DataLoader(file_name):
    df=pd.read_csv(file_name)
    # print(df.shape)
    # print(df.columns)
    return df

df=DataLoader('creditcard.csv')
df_2023=DataLoader('creditcard_2023.csv')
df_benign=df[df['Class']==0]
df_fraud=df[df['Class']==1]
# print(df_benign)
# print(df_fraud)
X=df_benign.drop(columns=['Class'])
y=df_benign['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test=pd.concat([X_test,y_test],axis=1)
test=pd.concat([test,df_fraud],axis=0,ignore_index=True)
X_test=test.drop(columns=['Class'])
y_test=test['Class']
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
models = {
    # 'KNN': KNN(),
    # 'LOF': LOF(),
    # 'IForest': IForest(),
    # 'ECOD': ECOD(),
    # 'FeatureBagging': FeatureBagging(),
    'XGBOD': XGBOD()
    # 'LUNAR': LUNAR()
}
with open('PyOD_result.txt','w') as f:
    for model_name, model in tqdm(models.items(),desc="Train OD model..."):
        f.write(f"Model = {model_name}\n")
        # if model_name=="XGBOD":
        #     model.fit(X_train,y_train)
        # else:
        model.fit(X_train,y_train)
        y_train_pred=model.labels_
        y_test_pred = model.predict(X_test)
        y_test_scores = model.decision_function(X_test)  
        f.write(f"\nOn Test Data with {model_name}:\n")
        evaluate_print(model_name, y_test, y_test_scores)
        f.write(f"save model: {model_name}\n")
        filename = f'./PyOD_model/{model_name}_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)