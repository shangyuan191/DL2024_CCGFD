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
from sklearn.preprocessing import StandardScaler
# 讀csv，回傳data frame
def DataLoader(file_name):
    df=pd.read_csv(file_name)
    # print(df.shape)
    # print(df.columns)
    return df

def data_sampling(df):
    fraud_data=df[df['Class']==1]
    normal_data=df[df['Class']==0].sample(n=len(fraud_data)*10,random_state=42)
    
    return normal_data,fraud_data

def data_standardization(df):
    # 假設 df 是你的 DataFrame，columns 是要標準化的欄位名稱列表
    columns_to_normalize = ['Time', 'Amount']  # 填入要標準化的欄位名稱
    # 初始化 StandardScaler 物件
    scaler = StandardScaler()
    # 對指定列進行標準化
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

df=DataLoader('creditcard.csv')
print(df['Class'][0])
print(type(df['Class'][0]))
df_2023=DataLoader('creditcard_2023.csv')
df_benign,df_fraud=data_sampling(df)
df_benign=data_standardization(df_benign)
df_fraud=data_standardization(df_fraud)


# 合并两个数据集
combined_data = pd.concat([df_fraud, df_benign])

# 打乱数据
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 切分数据为训练集和测试集，按 80:20 比例
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

# 提取特征和标签
X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']


# X=df_benign.drop(columns=['Class'])
# y=df_benign['Class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test=pd.concat([X_test,y_test],axis=1)
# test=pd.concat([test,df_fraud],axis=0,ignore_index=True)
# X_test=test.drop(columns=['Class'])
# y_test=test['Class']



# df_train=df_train.drop('Time', axis=1)
# df_eval=df_eval.drop('Time', axis=1)

# X_train=pd.DataFrame(df_train.drop(columns=['Class']))
# X_test=pd.DataFrame(df_train['Class'])
# y_train=pd.DataFrame(df_eval.drop(columns=['Class']))
# y_test=pd.DataFrame(df_eval['Class'])
# X=df_benign.drop(columns=['Class'])
# y=df_benign['Class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# test=pd.concat([X_test,y_test],axis=1)
# test=pd.concat([test,df_fraud],axis=0,ignore_index=True)
# X_test=test.drop(columns=['Class'])
# y_test=test['Class']
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())
models = {
    'KNN': KNN(),
    'LOF': LOF(),
    'IForest': IForest(),
    'ECOD': ECOD(),
    'FeatureBagging': FeatureBagging(),
    'XGBOD': XGBOD(),
    'LUNAR': LUNAR()
}
with open('PyOD_result.txt','w') as f:
    for model_name, model in tqdm(models.items(),desc="Train OD model..."):
        f.write(f"Model = {model_name}\n")
        if model_name=="XGBOD":
            model.fit(X_train,y_train)
        else:
            model.fit(X_train)
        y_train_pred=model.labels_
        y_test_pred = model.predict(X_test)
        y_test_scores = model.decision_function(X_test)  
        f.write(f"\nOn Test Data with {model_name}:\n")
        evaluate_print(model_name, y_test, y_test_scores)
        f.write(f"save model: {model_name}\n")
        filename = f'./PyOD_model/{model_name}_model.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)