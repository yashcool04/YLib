# Jlib
JLib is a basic classifier and regression library created to test implementations of basic learning algorithms.The classification is carried out by 3-layered neural network with user defined number of nodes in all three layers(Though neural network can also be used for regression).The regression part is carried by linear regression with squared-error cost function.Cost function v iteration can also be plotted.

This implemenation contains 4 files along with data files.The files `LinearRegression.py` and `NeuralNetwork.py` consists the core regression and classification logics respectively while the files `LinRegDriver.py` and `NNdriver.py` deals with testing these implementations.

### Using this implementation

#### Regression
To use the regression implementation first import the LinearReg class and instantiate it:
        
        from LinearRegression import LinearReg
        linearReg=LinearReg(iterations,learning_rate,l2_reg,max_error,plot_cost)
        
        iterations:number of iterations for which gradient descent would run.
        learning_rate: learning rate of gradient descent.
        l2_reg: regularization constant
        max_error:error at which learning stops
        plot_cost:weather to plot cost function versus iteration
  
  Following methods are available to implement regression:
  
        curve(self,X,y):
        argument X(training set in form of matrix arranged such that each training example must constitute one row of matrix)
        argument y(labels such that each label forms a row of matrix y)
        This method trains the model and returns modified values of weights or theta
        
        
        predict(self,X)
        argument X(The set of new inputs in form of matrix similar to that of training set)
        This method returns prediction in form of matrix similar to labels
        
        
        cross_validate(self,X,y)
        argument X(Cross validation set in form of matrix similar to input matrix)
        argument y(labels corrosponding to cross validation set in form of matrix such that each label forms a row)
 
 
#### Neural network
 
To use the neural network implementation,first import the NN class and then instantiate it:
          
    from NeuralNetwork import NN
    network=NN(input,hidden,output,iterations,l_rate,l2_reg,decay,momentum,plot_cost)
    
    input: number of nodes in input layer
    hidden: number of nodes in hidden layer
    output: number of nodes in output layer
    iterations: number of iterations
    l_rate: initial learning rate
    l2_reg: regularization term
    decay: decay in learning rate
    momentum: momentum
    plot_cost:weather to plot cost v iteration
    
Following methods are available to implement this network:
    
    forwardProp(self,X)
    argument X:training set in form of matrix arranged such that each training example must constitute one row of matrix
    This method forward propogates the network and returns the modified output activations
    
    backProp(self,y)
    argument y:labels such that each label forms a row of matrix y
    This is backpropagation regime through the network
    
    loss(self,y)
    argument y:labels such that each label forms a row of matrix y
    This method computes the loss of network
    
    fit(self,X,y):
    argument X:training set in form of matrix arranged such that each training example must constitute one row of matrix
    argument y:labels such that each label forms a row of matrix y
    This method trains the network and prints loss in each iteration
    
    predict(self,X)
    argument X:The set of new inputs on which predictions have to be made in form of matrix similar to that of training set
    This method returns the prediction in form of matrix similar to label matrix
    
    cross_validate(self,X,y)
    argument X:The cross validation set similar to input set in shape
    argument y:labels corrosponding to cross validation set
    This method returns accuracy of prediction.

The files `LinRegDriver.py` and `NNdriver.py` consists of sample implementations of these implementations!
    


 
       
        
        
