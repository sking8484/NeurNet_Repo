from NeuralNetwork import NeuralNetwork
import numpy as np




training_data = [
    {
    'inputs':[0,0],
    'targets':[0,0,0]
    },
    {
    'inputs':[0,1],
    'targets':[1,1,1]
    },
    {
    'inputs':[1,0],
    'targets':[1,1,1]
    },
    {
    'inputs':[1,1],
    'targets':[0,0,0]
    },
]


def setup():

    nn = NeuralNetwork(2,4,8,3,.1)
    #output = nn.feed_forward(inputs)
    #output.matrix[0][0]
    x = 0
    for epoch in range(50000):

        i = np.random.randint(0,4)

        nn.train(training_data[i]['inputs'],training_data[i]['targets'])

        #print(x)
        x+= 1
    print(nn.feed_forward([0,0]))
    print(nn.feed_forward([0,1]))
    print(nn.feed_forward([1,0]))
    print(nn.feed_forward([1,1]))




setup()
