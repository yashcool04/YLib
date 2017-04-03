import numpy as np
from NeuralNetwork import NN

network=NN(3,3,3,100000,.00001,.0001,0,0)


X=np.matrix('1,2,3;4,3,6;1,2,5;0,1,2;1,4,5;1,0,1')
y=np.matrix('0;2;1;1;1;2')
X_cv=np.matrix('1,0,0;1,2,4;10,7,4')
y_cv=np.matrix('0;1;1')


print(network.fit(X,y))
print(network.predict(X_cv))
network.cross_validate(X_cv,y_cv)

