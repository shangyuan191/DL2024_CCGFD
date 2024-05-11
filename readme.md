# CreditCard

### 先執行preprocessing.ipynb

creditcard.csv經過preprocessing後存成creditcard.npz，裡面有features, label, cosine_sim三個numpy array

在preprocessing.ipynb中，做了3件事

1. 刪除Time and Amount兩個特徵
2. 對正常樣本做採樣，數量為異常樣本的10倍
3. 算cosine_sim

---

### 讀取成torch_geometric.data的格式

先建立./dataset資料夾，裡面要有./dataset/raw資料夾放credircard.npz。

建好資料夾以及確定creditcard.npz有在./dataset/raw裡面，就可以用以下讀資料

```python
data = CreditCard(root='./dataset', cos=0.5)
```

./dataset ./dataset/raw我傳給你們時應該都有先建好，只需要幫我把creditcard.npz拉到./dataset/raw裡面就好

第一次呼叫會自動建立graph dataset，裡面有x, y, egee_index三個屬性(方法)(我不確定用這的名詞對不對>_<)

root可以改成你想要的名稱，但之前建好的資料夾也要改名

cos值可以自己自訂，訂成不同值會新增不同graph dataset
