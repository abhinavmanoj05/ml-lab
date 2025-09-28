from sklearn.naive_bayes import  GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##1- classify labels (data preprocessing)
df = pd.read_csv("iris.csv")
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

##2- data allocation
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

## model creation
model  = GaussianNB()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

##Printing results
accuracy_score = accuracy_score(y_test,y_predict)
print(f"accuracy score:{accuracy_score:.4f}")
recall_score = recall_score(y_test,y_predict, average="macro")
precision_score = precision_score(y_test,y_predict, average="macro")
print(f"Precision Score: {precision_score:.4f}")
print(f"Recall Score: {recall_score:.4f}")
#confusion matrix



cm = confusion_matrix(y_test, y_predict, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#bar graph
categories = ['Accuracy','Precision','Recall']
values = [accuracy_score,recall_score,precision_score]
plt.bar(categories,values)
plt.xlabel('Categoies')
plt.ylabel('Values')
plt.title("Bar graph")
plt.show()
