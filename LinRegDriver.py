import numpy as np
from LinearRegression import LinearReg

linearReg=LinearReg(100000,0.00001,.01,0,1)

data=np.genfromtxt('Data/housing.txt')

X=data[:,0:13]
y=data[:,13:]



linearReg.curve(X,y)

data_predict=np.genfromtxt('Data/housing_predict.txt')
x=data_predict[:,0:13]
Y=data_predict[:,13:]
y_hat=linearReg.predict(x)
cross_error=linearReg.cross_validate(x,Y)

print("predictions",y_hat)
print("cross validation error",cross_error)