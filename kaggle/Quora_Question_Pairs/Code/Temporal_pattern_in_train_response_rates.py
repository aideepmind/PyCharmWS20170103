import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv( "kaggle/Quora_Question_Pairs/Data/train.csv")

df["qmax"]      = df.apply(lambda row: max(row["qid1"], row["qid2"]), axis=1)
df              = df.sort_values(by=["qmax"], ascending=True)
df["dupe_rate"] = df.is_duplicate.rolling(window=500, min_periods=500).mean()
df["timeline"]  = np.arange(df.shape[0]) / float(df.shape[0])

df.plot(x="timeline", y="dupe_rate", kind="line")
plt.show()

lr = LinearRegression()
lr.fit(df[["timeline"]], df["dupe_rate"].fillna(0))
test_data = np.array([x/1000 for x in range(0, 1000)]).reshape(1000, 1)
test_preds = lr.predict(test_data)

plt.scatter(test_data.reshape(1000), test_preds)