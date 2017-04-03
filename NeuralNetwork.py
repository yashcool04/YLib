import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def diff_sigmoid(z):
    return z*(1-z)





class NN:

    def __init__(self,input,hidden,output,iterations=1000,l_rate=.01,l2_reg=0,decay=0,momentum=0,plot_cost=1):
        """

        :param input: number of nodes in input layer
        :param hidden: number of nodes in hidden layer
        :param output: number of nodes in output layer
        :param iterations: number of iterations
        :param l_rate: initial learning rate
        :param l2_reg: regularization term
        :param decay: decay in learning rate
        :param momentum: momentum
        :param plot_cost:weather to plot cost v iteration
        """
        self.input=input
        self.hidden=hidden
        self.output=output
        self.iterations=iterations
        self.l_rate=l_rate
        self.l2_reg=l2_reg
        self.decay=decay
        self.momentum=momentum
        self.plot_cost=1

        #initialize weights normally as referred in cs231n

        self.w_in=np.random.normal(size=(self.input,hidden))/(np.sqrt(self.input))
        self.w_out=np.random.normal(size=(self.hidden,self.output))/(np.sqrt(self.hidden))

        #initialize bias
        self.b_in=np.zeros((1,hidden))
        self.b_hid=np.zeros((1,output))

        #initialize activations
        self.act_in=np.ones(input)
        self.act_hid=np.ones(hidden)
        self.act_out=np.ones(output)

    def forwardProp(self,X):

        #X=M*Input (M->number of examples)
        self.act_in=np.copy(X)
        hidden_layer=np.dot(X,self.w_in)+self.b_in  #M*hidden
        self.act_hid=np.maximum(0,hidden_layer)    #M*hidden ReLU activations

        output_layer=np.dot(self.act_hid,self.w_out)+self.b_in #M*output
        self.act_out=np.exp(output_layer)/np.sum(np.exp(output_layer),axis=1,keepdims=True) #M*output,for row axis==1


        return self.act_out

    def backProp(self,y):

        #y=M*1
        num_examples=np.shape(y)[0]

        dout=np.copy(self.act_out)



        dout[range(num_examples),y]-=1 #M*output

        dout/=num_examples

        #backprop into w_out and b_out
        dw_out=np.dot(self.act_hid.T,dout)
        db_hid=np.sum(dout,axis=0)

        #backprop through hidden layer
        dhidden=np.dot(dout,self.w_out.T)

        #backprop the relu
        dhidden[self.act_hid<=0]=0

        #backprop into w_in and b_in
        dw_in=np.dot(self.act_in.T,dhidden)
        db_in=np.sum(dhidden,axis=0)

        #regulariaztion gradient
        dw_out+=self.l2_reg*self.w_out
        dw_in+=self.l2_reg*self.w_in

        #parameter update
        self.w_in-=self.l_rate*dw_in
        self.b_in-=self.l_rate*db_in

        self.w_out-=self.l_rate*dw_out
        self.b_hid-=self.l_rate*db_hid


    def loss(self,y):
        num_examples = np.shape(y)[0]
        #print("loss print",self.act_out[range(num_examples), y])
        correct_probs = -np.log(self.act_out[range(num_examples), y])
        data_loss = np.sum(correct_probs) / num_examples
        reg_loss = 0.5 * self.l2_reg * np.sum(self.w_in * self.w_in) + 0.5 * self.l2_reg * np.sum(
            self.w_out * self.w_out)
        total_loss = data_loss + reg_loss
        return total_loss

    def fit(self,X,y):

        for i in range(self.iterations):

            self.forwardProp(X)
            if self.plot_cost and i>20:
                plt.scatter(i,self.loss(y))
            self.backProp(y)
            #print("before loss",self.act_out)

            print(self.loss(y))
        plt.show()
    
    
    def predict(self,X):
        final_layer=self.forwardProp(X)     #M*output
        max_index=np.argmax(final_layer,axis=1)     #M*1

        return max_index

    def cross_validate(self,X,y):

        max_index=self.predict(X)
        correct_predictions_boolean=y==max_index
        correct_predictions=np.sum(correct_predictions_boolean)
        total_predictions=np.shape(y)[0]
        accuracy=(correct_predictions/total_predictions)*100        #defined as correct predictions out of total predictions
        return accuracy




































