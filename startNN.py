from NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt





# doodles = ['cake','planes','trains']
# data_dict = {}
#
# for doodle in doodles:
#     data_dict[doodle] = pd.read_pickle(doodle + '1000.pkl')
#
#
#
#
#
# training_data = []
# testing_data = []
#
# x=0
# for doodle in doodles:
#     training_data.append([])
#     testing_data.append([])
#     for i in range(1000*784):
#         if i < 800*784:
#             training_data[x].append(data_dict[doodle][0][i])
#
#         else:
#             testing_data[x].append(data_dict[doodle][0][i])
#     x+=1
#
#
# cake = {}
# planes = {}
# trains = {}
#
# cake['training']=  np.array(training_data[0]).reshape(800,784)
# cake['training'] = np.append(cake['training'],np.ones(800).reshape(800,1),1)
# cake['testing'] = np.array(testing_data[0]).reshape(200,784)
# cake['testing'] = np.append(cake['testing'],np.ones(200).reshape(200,1),1)
#
# planes['training'] = np.array(training_data[1]).reshape(800,784)
# planes['training'] = np.append(planes['training'],2*np.ones(800).reshape(800,1),1)
# planes['testing'] = np.array(testing_data[1]).reshape(200,784)
# planes['testing'] = np.append(planes['testing'],2*np.ones(200).reshape(200,1),1)
#
# trains['training'] = np.array(training_data[2]).reshape(800,784)
# trains['training'] = np.append(trains['training'],np.zeros(800).reshape(800,1),1)
# trains['testing'] = np.array(testing_data[2]).reshape(200,784)
# trains['testing'] = np.append(trains['testing'],np.zeros(200).reshape(200,1),1)
#
# #Image.fromarray(trains['testing'][0].reshape(28,28)).show()
#
#
#
#
# def shuffle_data():
#     data = np.append(cake['training'],planes['training'],0)
#     data = np.append(data, trains['training'],0)
#     np.random.shuffle(data)
#     testing_data = np.append(cake['testing'],planes['testing'],0)
#     testing_data = np.append(testing_data,trains['testing'],0)
#     np.random.shuffle(testing_data)
#
#
#
#     inputs = []
#     normalized_data = []
#     test_data = []
#
#     testing_inputs = []
#     for rows in data:
#         norm_row = rows[:784]/255
#         normalized_data.append(norm_row)
#         input_array = [0,0,0]
#         input_array[int(rows[-1])] = 1
#         inputs.append(input_array)
#     for rows in testing_data:
#         norm_row = rows[:784]/255
#         test_data.append(norm_row)
#         input_array = [0,0,0]
#         input_array[int(rows[-1])] = 1
#         testing_inputs.append(input_array)
#     return normalized_data,inputs,test_data,testing_inputs
#
#
#
# data_dict = [{'input':[0,0],'output':[0]},
# {'input':[0,1],'output':[1]},
# {'input':[1,0],'output':[1]},
# {'input':[1,1],'output':[0]}
#
# ]

spy = pd.read_csv("SPY.csv", parse_dates=True,index_col='Date')
spy = spy[['Adj Close']].pct_change()
spy['values'] = spy.shift(-1)
spy.dropna(inplace = True)
spy.columns = ['Inputs','Outputs']

def shuffle_data():
    spy_train = spy.iloc[:-200]
    spy_test = spy.iloc[-200:]
    spy_train = spy_train.values
    spy_test = spy_test.values

    np.random.shuffle(spy_train)

    spy_train = pd.DataFrame(spy_train)
    spy_test = pd.DataFrame(spy_test)
    train_inputs = spy_train[spy_train.columns[0]]
    train_answers = spy_train[spy_train.columns[1]]

    test_inputs = spy_test[spy_test.columns[0]]
    test_answers = spy_test[spy_test.columns[1]]

    return train_inputs, train_answers, test_inputs,test_answers






nn = NeuralNetwork(1,6,5,1,.1,'r')

# for epoch in range(10000):
#     for i in range(len(data_dict)):
#         nn.train(data_dict[i]['input'],data_dict[i]['output'])
#
# print(nn.feed_forward(data_dict[2]['input']))


for epoch in range(50):
    normalized_data,inputs,test_data,testing_inputs = shuffle_data()

    for i in range(int(len(normalized_data))):

        nn.train(normalized_data[i],inputs[i],epoch)

num_correct = 0
points = []
correct = []
for test in range(len(test_data)):
    inputs = test_data[test]
    nn_output = nn.feed_forward(test_data[test])*10
    points.append(nn_output[0])
    output = np.argmax(nn_output)
    ti = testing_inputs[test]

    correct.append(ti)


    answer = np.argmax(ti)

    if output == answer:
        num_correct+= 1

print(num_correct/len(testing_inputs))
plt.plot(points,label='Prediction')
plt.plot(correct,label='Actual')
plt.legend()

plt.show()
