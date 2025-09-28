import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
#loading the csv
df = pd.read_csv("House.csv")
print(df.head())
x = df.drop(["SalePrice"], axis=1)
y = df['SalePrice']
#plotting sample
sns.lmplot(x="LotArea",y="SalePrice",data=df,ci=None)
plt.show()
sns.lmplot(x="OverallQual",y="SalePrice",data=df,ci=None)
plt.show()
sns.lmplot(x="OverallCond",y="YearBuilt",data=df,ci=None)
plt.show()
#model fitting
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state = 42)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_train,y_train))

model.score(x_train,y_train)
y_predict = model.predict(x_test)

#score
print(f"R2 {round(r2_score(y_test,y_predict),2)}")