import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators as data

class Network(layers.BaseNetwork):
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer):
        # you should always call __init__ first 
        super().__init__()
        #TODO: define your network architecture here
        self.linear = layers.Linear(data_layer,1)
        self.bias = layers.Bias(self.linear)
        # For prob 3 and 4:
        # layers.ModuleList can be used to add arbitrary number of layers to the network
        # e.g.:
        # self.modules = layers.ModuleList()
        # self.modules.append(self.linear)
        # self.modules.append(self.bias)

        
        #TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.bias)

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
        # hidden_units = parameters["hidden_units"] #needed for prob 2, 3, 4, mnist
        # hidden_layers = parameters["hidden_layers"] #needed for prob 3, 4, mnist
        #TODO: construct your network here
        network = Network(data_layer)
        return network
    
    def setup(self, training_data):
        x, y = training_data
        #TODO: define input data layer
        self.data_layer = layers.Data(x)
        #TODO: construct the network. you don't have to use define_network.
        self.network = self.define_network(self.data_layer)
        #TODO: use the appropriate loss function here
        self.loss_layer = layers.SquareLoss(self.network.output_layer, y)
        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optim = layers.SGDSolver(0.1, self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optim
    
    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function 

        #forward pass - from input layer - to loss layer
        # modules = self.network.get_modules_with_parameters()
        # for i in range(len(modules)):
        #     modules[i].forward()


        #get loss
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
        data_layer = layers.Data(x)
        self.network.linear.in_layer = data_layer
        self.loss_layer.labels =  y

        predict = self.network.get_output_layer().forward()
        error = 0
        for i in range(len(predict)):
            error += predict[i] - y[i]
            print("predict: ", predict[i], " label: ", y[i])
        return np.sum(error)
        
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        data_dict = data.data_1b()
        train_data = data_dict['train']
        test_data = data_dict['test']
        trainer.setup(train_data)
        trainer.train(30000)
        # trainer.test(test_data)

    else:
        #DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out

if __name__ == "__main__":
    main()
    pass
