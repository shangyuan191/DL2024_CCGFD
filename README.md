# DL2024_CCGFD
## 資料集
- 下載信用卡詐騙資料集到此repository底下，並命名為"creditcard.csv"。
- 資料集來源：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
## 兩種程式執行方式：
1. 使用Jupyer Notebook執行
- 訓練資料中混入異常資料的實驗結果：main.ipynb
- 訓練資料中r僅有正常資料的實驗結果：main_clear.ipynb
2. 執行python檔
- 創建存放圖資料的資料夾
    ```
    !mkdir graph
    !mkdir graph/test
    !mkdir graph/train
    !mkdir graph_clear
    !mkdir graph_clear/test
    !mkdir graph_clear/train
    ```
- 訓練資料中混入異常資料的實驗結果：
```python main.py```
- 訓練資料中r僅有正常資料的實驗結果：
```python main_clear.py```

