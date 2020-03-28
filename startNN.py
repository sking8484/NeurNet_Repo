from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt





doodles = ['cake','planes','trains']
data_dict = {}

for doodle in doodles:
    data_dict[doodle] = pd.read_pickle(doodle + '1000.pkl')





training_data = []
testing_data = []

x=0
for doodle in doodles:
    training_data.append([])
    testing_data.append([])
    for i in range(1000*784):
        if i < 800*784:
            training_data[x].append(data_dict[doodle][0][i])

        else:
            testing_data[x].append(data_dict[doodle][0][i])
    x+=1


cake = {}
planes = {}
trains = {}

cake['training']=  np.array(training_data[0]).reshape(800,784)
cake['training'] = np.append(cake['training'],np.ones(800).reshape(800,1),1)
cake['testing'] = np.array(testing_data[0]).reshape(200,784)
cake['testing'] = np.append(cake['testing'],np.ones(200).reshape(200,1),1)

planes['training'] = np.array(training_data[1]).reshape(800,784)
planes['training'] = np.append(planes['training'],2*np.ones(800).reshape(800,1),1)
planes['testing'] = np.array(testing_data[1]).reshape(200,784)
planes['testing'] = np.append(planes['testing'],2*np.ones(200).reshape(200,1),1)

trains['training'] = np.array(training_data[2]).reshape(800,784)
trains['training'] = np.append(trains['training'],np.zeros(800).reshape(800,1),1)
trains['testing'] = np.array(testing_data[2]).reshape(200,784)
trains['testing'] = np.append(trains['testing'],np.zeros(200).reshape(200,1),1)

#Image.fromarray(trains['testing'][0].reshape(28,28)).show()




def shuffle_data():
    data = np.append(cake['training'],planes['training'],0)
    data = np.append(data, trains['training'],0)
    np.random.shuffle(data)
    testing_data = np.append(cake['testing'],planes['testing'],0)
    testing_data = np.append(testing_data,trains['testing'],0)
    np.random.shuffle(testing_data)



    inputs = []
    normalized_data = []
    test_data = []

    testing_inputs = []
    for rows in data:
        norm_row = rows[:784]/255
        normalized_data.append(norm_row)
        input_array = [0,0,0]
        input_array[int(rows[-1])] = 1
        inputs.append(input_array)
    for rows in testing_data:
        norm_row = rows[:784]/255
        test_data.append(norm_row)
        input_array = [0,0,0]
        input_array[int(rows[-1])] = 1
        testing_inputs.append(input_array)
    return normalized_data,inputs,test_data,testing_inputs



data_dict = [{'input':[0,0],'output':[0]},
{'input':[0,1],'output':[1]},
{'input':[1,0],'output':[1]},
{'input':[1,1],'output':[0]}

]

nn = NeuralNetwork(784,35,3,3,.1)

# for epoch in range(10000):
#     for i in range(len(data_dict)):
#         nn.train(data_dict[i]['input'],data_dict[i]['output'])
#
# print(nn.feed_forward(data_dict[2]['input']))


for epoch in range(15):
    normalized_data,inputs,test_data,testing_inputs = shuffle_data()
    for i in range(len(normalized_data)):
        nn.train(list(normalized_data[i]),list(inputs[i]),epoch)

num_correct = 0


for test in range(len(test_data)):

    inputs = list(test_data[test])

    nn_output = nn.feed_forward(list(test_data[test]))

    output = np.argmax(nn_output)
    ti = testing_inputs[test]


    answer = np.argmax(ti)

    if output == answer:
        num_correct+= 1

print(num_correct/len(testing_inputs))

plt.show()
