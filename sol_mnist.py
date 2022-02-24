import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators as data
import pickle

class Network(layers.BaseNetwork):
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer, hidden_units, hidden_layers):
        # you should always call __init__ first 
        super().__init__()
        #TODO: define your network architecture here
        # self.linear_hidden_1 = layers.Linear(data_layer,hidden_units) # hidden layer
        # self.bias_hidden_1 = layers.Bias(self.linear_hidden_1)
        # self.relu = layers.Relu(self.bias_hidden_1)

        self.modules = layers.ModuleList()
        self.modules.append(data_layer)
        for i in range(hidden_layers):
            self.modules.append(layers.Linear(self.modules[-1],hidden_units))
            self.modules.append(layers.Bias(self.modules[-1]))
            self.modules.append(layers.Relu(self.modules[-1]))

        self.modules.append(layers.Linear(self.modules[-1],10)) 
        self.modules.append(layers.Bias(self.modules[-1]))
        # self.modules.append(layers.Sigmoid(self.modules[-1]))
        # For prob 3 and 4:
        # layers.ModuleList can be used to add arbitrary number of layers to the network
        # e.g.:
        # self.modules = layers.ModuleList()
        # self.modules.append(self.linear_hidden_1)
        # self.modules.append(self.bias_hidden_1)
        # self.modules.append(self.linear)
        # self.modules.append(self.bias)

                
        #TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.modules[-1])
class Trainer:
    def __init__(self):
        pass
    
    def define_network(self, data_layer, parameters=None):
        '''
        For prob 2, 3, 4 and mnist:
        parameters is a dict that might contain keys: "hidden_units" and "hidden_layers".
        "hidden_units" specify the number of hidden units for each layer. "hidden_layers" specify number of hidden layers. 
        Note: we might be testing if your network code is generic enough through define_network. Your network code can be even more general, but this is the bare minimum you need to support.
        Note: You are not required to use define_network in setup function below, although you are welcome to.
        '''
        hidden_units = parameters["hidden_units"] #needed for prob 2, 3, 4, mnist
        hidden_layers = parameters["hidden_layers"] #needed for prob 3, 4, mnist
        #TODO: construct your network here
        network = Network(data_layer, hidden_units, hidden_layers)
        return network
    
    def setup(self, training_data):
        x, y = training_data
        #TODO: define input data layer
        self.data_layer = layers.Data(x)
        #TODO: construct the network. you don't have to use define_network.
        self.network = self.define_network(self.data_layer,{"hidden_units": 10, "hidden_layers":3})
        #TODO: use the appropriate loss function here
        self.loss_layer = layers.CrossEntropySoftMax(self.network.output_layer, y)
        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optim = layers.SGDSolver(0.07, self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optim
    
    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function 

        #forward pass - from input layer - to loss layer
        loss = self.loss_layer.forward()
        self.loss_layer.backward()
        self.optim.step()
        # print(loss)
        return loss
    def get_num_iters_on_public_test(self):
        #TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 30000
    
    def train(self, num_iter):
        train_losses = []
        #TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.
        for i in range (num_iter):
            train_losses.append(self.train_step())
        # you have to return train_losses for the function
        print(train_losses[-1])
        return train_losses

    def test(self, test_data):
        x, y = test_data
        self.data_layer = layers.Data(x)

        predict = self.network.forward()
        correct = 0
        for i in range(len(predict)):
            print("predict: ", np.argmax(predict[i]), " label: ", y[i])
            print(predict[i])
            if np.argmax(predict[i]) == y[i]:
                correct+=1
        return correct/len(predict)
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        with open('mnist.pkl', 'rb') as f:
            mnist_data = pickle.load(f)

        train_x = mnist_data['training_images']
        test_x = mnist_data['test_images']
        train_y = mnist_data['training_labels']
        test_y = mnist_data['test_labels']
        # print(train_x.shape) (60000, 784) #(60000,)
        # train_x = train_x[:1000]
        # train_y = train_y[:1000]
        batch_size = 256
        num_round = int(np.ceil(train_x.shape[0]/batch_size))
        ind = 0
        for j in range (num_round):
            last_ind = min(ind+batch_size, train_x.shape[0])
            train_data = (train_x[ind:last_ind], train_y[ind:last_ind])
            if ind==0:
                trainer.setup(train_data)
            else:
                x,y = train_data
                trainer.network.modules[1].in_layer = layers.Data(x)
                # print(trainer.network.modules[1].in_layer)
                trainer.loss_layer.set_data(y)
            trainer.train(100)
            ind+=batch_size
        
        # print(train_data.shape)
        train_data_test = (train_x[-1000:],train_y[-1000:])
        test_data = (test_x,test_y)
        print(trainer.test(train_data_test))
        print(trainer.test(test_data))

    else:
        #DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out

if __name__ == "__main__":
    main()
    pass
