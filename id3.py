import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
df = pd.read_csv("diabetes.csv")
df.head()
X=df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y=df['Outcome']

x_train,x_test,y_train,y_test =  train_test_split(X , Y, test_size =0.3,random_state = 1)
model = DecisionTreeClassifier(max_depth =3) 
model.fit(x_train,y_train)

import matplotlib.pyplot as plt
y_pred = model.predict(x_test)
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"], filled=True)
plt.show()
accuracy =  accuracy_score(y_test,y_pred)
print(f"Accuracy :{accuracy:.2f}")

