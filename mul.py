#product of 2 arrays
import numpy as np
arr1 = np.array([[1,2],[2,3]])
arr2= np.array([[5,6],[6,2]])
print("First matrix:")
print(arr1)
print("Second matrix")
print(arr2)
print("Product matrix")
prod=np.matmul(arr1,arr2)
print(prod)
#square root a no
import math
n=int(input("Enter an number"))
print("Square root of",n,"=",math.sqrt(n))
#days since birth
import datetime as dt
dob = dt.datetime(2005,10,5)
print(dt.datetime.today()-dob)