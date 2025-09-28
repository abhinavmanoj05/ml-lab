import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np

Year = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]  
Marks = [99,86,87,88,91,86,93,87,94,78]

mymodel = np.poly1d(np.polyfit(Year, Marks, 3))
myline = np.linspace(2011,2020,100)

percentage = mymodel(2017)
print(percentage)
print(r2_score(Marks,mymodel(Year)))

plt.scatter(Year,Marks)
plt.plot(myline,mymodel(myline))
plt.show()