# nn1.py

# Machine Learning - Neural Network
# Hyoyup Chung

# Run in command-line:
#	python nn1.py

import numpy as np
import sys, math, statistics

# default params
dataNum = 1000
neuronNum = 5
learningrate = 0.5

def Sigmoid(x):
	return 1.0 / (1.0 + math.exp(-1.0*x))

def der_Sigmoid(x):
	sigm = Sigmoid(x)
	return sigm * (1.0 - sigm)

def ReLU(x):
	if (x<0):
		return 0.0
	return x

def der_ReLU(x):
	if (x<0):
		return 0.0
	return 1.0

def activate(x):
	return Sigmoid(x)

def der_activate(x):
	return der_Sigmoid(x)

def FeedForward(data, W_input, W, W_output, bias):
	zk = np.zeros((2,neuronNum), float)
	ok = np.zeros((2,neuronNum), float)
	o_out = 0.0

	# input layer
	for neuron in range(0,neuronNum):
		_x = data[0]*W_input[0][neuron]
		_y = data[1]*W_input[1][neuron]
		zk[0][neuron] = _x + _y
		ok[0][neuron] = activate(zk[0][neuron]+bias[0][neuron])
	# b/w hidden layer
	for neuron in range(0,neuronNum):
		summation = 0.0
		for i in range(0,neuronNum):
			summation += ok[0][i]*W[neuron][i]
		zk[1][neuron] = summation
		ok[1][neuron] = activate(zk[1][neuron]+bias[1][neuron])
	# output layer
	for i in range(0, neuronNum):
		o_out += ok[1][i]*W_output[i]
	
	return [zk, ok, o_out]

def BackPropagation(zk_ok_list, data, W_input, W, W_output, bias):
	zk = zk_ok_list[0]
	ok = zk_ok_list[1]
	output = zk_ok_list[2]

	der_E_o = np.zeros((2,neuronNum), float)
	der_E_output = output - data[2]
	# output layer
	for neuron in range(0,neuronNum):
		der_E_o[1][neuron] = der_E_output*der_activate(output)*W_output[neuron]
		grad_W = der_E_output*der_activate(output)*ok[1][neuron]
		grad_b = der_E_output*der_activate(output)
		W_output[neuron] -= learningrate*grad_W
		bias[1][neuron] -= learningrate*grad_b
	# hidden layer
	for neuron in range(0,neuronNum):
		summation = 0.0
		for i in range(0,neuronNum):
			summation += der_E_o[1][neuron]*der_activate(zk[1][neuron]+bias[1][neuron])*W[i][neuron]
		der_E_o[0][neuron] = summation
	for neuron in range(0,neuronNum):
		for i in range(0,neuronNum):
			grad_W = der_E_o[1][i]*der_activate(zk[1][i]+bias[1][i])*ok[0][neuron]
			W[neuron][i] -= learningrate*grad_W
		grad_b = der_E_o[1][neuron]*der_activate(zk[1][neuron]+bias[1][neuron])
		bias[0][neuron] -= learningrate*grad_b
	# input layer: x
	for neuron in range(0,neuronNum):
		grad_W = der_E_o[0][neuron]*der_activate(zk[0][neuron]+bias[0][neuron])*data[0]
		W_input -= learningrate*grad_W
	# input layer: y
	for neuron in range(0,neuronNum):
		grad_W = der_E_o[0][neuron]*der_activate(zk[0][neuron]+bias[0][neuron])*data[1]
		W_input -= learningrate*grad_W

# main:
if __name__ == "__main__":

	# command-line input
	inputArgSize = len(sys.argv)
	if (inputArgSize > 1):
		dataNum = int(sys.argv[1])

	dist = 9.0 / dataNum
	training_x = np.arange(1.0, 10.0, dist)
	training_y = np.arange(1.0, 10.0, dist)

	# training triplet: [x, y, x/y]
	trainingData = np.zeros((dataNum,3),float)
	for i in range(0,dataNum):
		trainingData[i] = [training_x[i], training_y[i], training_x[i]/training_y[i]]

	# initialize neural network params for 2 hidden layer
	b = np.random.rand(2, neuronNum) 		 # biases for each neuron
	W_input = np.random.rand(2, neuronNum)	 # 2 input to n neurons
	W = np.random.rand(neuronNum, neuronNum) # n neurons to n neuron
	W_output = np.random.rand(neuronNum) 	 # n neurons to 1 output
	#############################
	######## TRAINING ###########
	#############################
	for data in trainingData:
		# activation function using ReLU()
		zk_ok_list = FeedForward(data, W_input, W, W_output, b)
		BackPropagation(zk_ok_list, data, W_input, W, W_output, b)

	#############################
	######### Testing ###########
	#############################
	testDataNum = 100
	dist = 9.0 / testDataNum
	testing_x = np.arange(1.0, 10.0, dist)
	np.random.shuffle(testing_x)
	testing_y = np.arange(1.0, 10.0, dist)
	np.random.shuffle(testing_y)

	testingData = np.zeros((testDataNum,3),float)
	for i in range(0,testDataNum):
		testingData[i] = [training_x[i], training_y[i], training_x[i]/training_y[i]]

	#############################
	######## Validating #########
	#############################
	res = []
	for data in testingData:
		result = FeedForward(data, W_input, W, W_output, b)
		out = result[2]
		res.append((out - data[2])**2)
	error = np.sqrt(statistics.mean(res))
	print("RMSE for NN with 2 hidden layer each with %d neurons : %f" % (neuronNum, error))