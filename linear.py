from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import seaborn as sns
#data labelling
df = pd.read_csv("Salary_Data.csv")
print(df.head())
sns.lmplot(x="YearsExperience",y="Salary",data=df,ci=None)
plt.show()
x = df[["YearsExperience"]]
y = df["Salary"]
#training the model
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=43)

#fit the model
model = LinearRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

#visualise
plt.scatter(x,y, color="blue")
plt.plot(x_test,y_predict, color = "red")
plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.show()

