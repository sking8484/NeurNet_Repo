import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, inputnotes,hiddennodes,layers,outputnodes,learningrate):
        self.inodes = inputnotes
        self.hnodes = hiddennodes
        self.layers = layers
        self.onodes= outputnodes
        self.lr = learningrate

        """Initialize the weights and the biases"""

        self.iw = {} #The initial Weights dictionary

        for layer in range(self.layers):
            if layer == 0:
                self.iw[layer] = (np.random.rand(self.hnodes,self.inodes)-.5)
                self.iw['bias' +str(layer)] = np.random.rand(self.hnodes)
                self.iw['bias' + str(layer)] = np.array(self.iw['bias' +str(layer)],ndmin = 2).T
            elif layer == self.layers-1:
                self.iw[layer] = (np.random.rand(self.onodes,self.hnodes)-.5)
                self.iw['bias' +str(layer)] = np.random.rand(self.onodes)
                self.iw['bias' + str(layer)] = np.array(self.iw['bias' +str(layer)],ndmin = 2).T
            else:
                self.iw[layer] = (np.random.rand(self.hnodes, self.hnodes)-.5)
                self.iw['bias'+str(layer)] = np.random.rand(self.hnodes)
                self.iw['bias' + str(layer)] = np.array(self.iw['bias'+str(layer)],ndmin = 2).T

    def sigmoid(self,x):
        return (1/(1+np.e**(-x)))
    def dsigmoid(self,x):
        return (x*(1-x))

    def feed_forward(self,input_array):
        input_array = np.array(input_array,ndmin=2).T
        self.input_array = input_array





        """Run feedword algorithm"""

        self.ff = {} #Dictionary to hold the feedforward weights

        for layer in range(self.layers):
            if layer ==0:
                self.ff[layer] = self.iw[layer]@input_array

                self.ff[layer]+= self.iw['bias'+str(layer)]
                self.ff[layer] = self.sigmoid(self.ff[layer])

            elif layer == self.layers -1:
                self.ff[layer] = self.iw[layer]@self.ff[layer-1]
                self.ff[layer]+= self.iw['bias'+str(layer)]
                self.ff[layer]=self.sigmoid(self.ff[layer])
            else:
                self.ff[layer] = self.iw[layer]@self.ff[layer-1]
                self.ff[layer]+=self.iw['bias'+str(layer)]
                self.ff[layer]= self.sigmoid(self.ff[layer])

        outputs = self.ff[self.layers-1]
        return outputs

    def train(self,inputs,targets):


        outputs = self.feed_forward(inputs)
        targets = np.array(targets,ndmin=2).T

        output_errors = targets-outputs
        for layer in range(self.layers-1,-1,-1):

            if layer == self.layers -1:

                gradient = self.dsigmoid(outputs)
                gradient = np.multiply(output_errors,gradient)
                first_errors = gradient
                gradient = self.lr*gradient

                hidden_t = self.ff[layer-1].T
                deltas = gradient@hidden_t

                self.iw[layer] += deltas
                self.iw['bias' + str(layer)]+= gradient

            elif layer ==0:
                first_errors = np.transpose(self.iw[layer+1])@first_errors
                gradient = self.dsigmoid(self.ff[layer])
                gradient = np.multiply(first_errors,gradient)
                first_errors = gradient
                gradient = self.lr*gradient

                inputs_t = self.input_array.T

                deltas = gradient@inputs_t

                self.iw[layer]+=deltas

                self.iw['bias'+str(layer)]+= gradient
            else:
                first_errors = np.transpose(self.iw[layer+1])@first_errors

                gradient = self.dsigmoid(self.ff[layer])

                gradient = np.multiply(first_errors,gradient)
                first_errors = gradient

                gradient = self.lr*gradient



                prev_hidden_t = self.ff[layer-1].T

                deltas = gradient@prev_hidden_t
                self.iw[layer]+= deltas
                self.iw['bias'+str(layer)] += gradient
