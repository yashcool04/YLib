import numpy as np
import matplotlib.pyplot as plt


class LinearReg:

    def __init__(self,iterations=100,learning_rate=0.1,l2_reg=0,max_error=0,plot_cost=1):

        """

        :param iterations:number of iterations for which gradient descent would run.
        :param learning_rate: learning rate of gradient descent.
        :param l2_reg: regularization constant
        :param max_error:error at which learning stops
        :param plot_cost:weather to plot cost function versus iteration
        """

        self.iterations=iterations
        self.learning_rate=learning_rate
        self.l2_reg=l2_reg
        self.max_error=max_error
        self.plot_cost=plot_cost
        self.theta=None


    def curve(self,X,y):

        #bias of ones
        bias = np.ones((np.shape(X)[0], 1))
        #add a bias column of ones to input X
        X=np.concatenate((bias,X),1)

        #extract number of examples and number of features from dimensions of X
        examples,features=np.shape(X)

        #initialize theta
        self.theta=np.ones((features,1))

        for i in range(self.iterations):

            #prediction by dot product of input X and transpose of row vector theta
            prediction=np.dot(X,self.theta)

            #the error in prediction is predicted value subtracted by observed value
            error=prediction-y



            #squared error cost function
            cost=np.sum(error**2)/(2*examples) + self.l2_reg*np.sum((self.theta)**2)

            #skipping initial iterations so that graph scales nicely
            if self.plot_cost and i>20:
                plt.scatter(i,cost)

            #updating value of theta
            self.theta=self.theta-(self.learning_rate/examples)*(np.dot(np.transpose(X),error))-(self.l2_reg/examples)*self.theta

            if i%10==0:
                print("iteration",i)
                print("cost",cost)
                #print("theta",self.theta)

            if cost<self.max_error:
                plt.show()
                return self.theta
        plt.show()

        return self.theta




    def predict(self,X):

        # bias of ones
        bias = np.ones((np.shape(X)[0],1))
        # add a bias column of ones to input X
        X = np.concatenate((bias, X), 1)

        #prediction
        y_hat=np.dot(X,self.theta)



        return y_hat


    def cross_validate(self,X,y):

        # bias of ones
        bias = np.ones((np.shape(X)[0], 1))
        # add a bias column of ones to input X
        X = np.concatenate((bias, X), 1)

        #prediction
        y_hat = np.dot(X, self.theta)

        #cross validation error
        cross_error = np.sum((y_hat - y) ** 2) / 2 * (np.shape(X)[0])

        return cross_error


















