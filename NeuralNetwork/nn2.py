# nn2.py

# Machine Learning - Neural Network (tensorflow & keras)
# Hyoyup Chung

# Run in command-line:
#	python nn2.py

import numpy as np
import matplotlib.pyplot as plt
import sys, math, statistics, struct
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

numTrainImages = 10000
numTestImages = 10000



# main:
if __name__ == "__main__":

	# train images
	train_x = []
	train_num_pixels = 0
	with open("train-images.idx3-ubyte","rb") as file:
		magicnumber = file.read(4)
		numImages = struct.unpack('>i',file.read(4))[0]
		Rows = struct.unpack('>i',file.read(4))[0]
		Cols = struct.unpack('>i',file.read(4))[0]
		train_num_pixels = Rows*Cols

		for i in range(0,numTrainImages):
			image = []
			for pix in range(0,train_num_pixels):
				byte = struct.unpack('B',file.read(1))[0]
				image.append(float(byte))
			train_x.append(image)

	# train labels
	train_y = []
	with open("train-labels.idx1-ubyte","rb") as file:
		magicnumber = file.read(4)
		numLabels = struct.unpack('>i',file.read(4))[0]
		for i in range(0,numTrainImages):
			byte = struct.unpack('B',file.read(1))[0]
			train_y.append(int(byte))

	# testing images
	test_x = []
	test_num_pixels = 0
	with open("t10k-images.idx3-ubyte","rb") as file:
		magicnumber = file.read(4)
		numImages = struct.unpack('>i',file.read(4))[0]
		Rows = struct.unpack('>i',file.read(4))[0]
		Cols = struct.unpack('>i',file.read(4))[0]
		test_num_pixels = Rows*Cols

		for i in range(0,numTestImages):
			image = []
			for pix in range(0,test_num_pixels):
				byte = struct.unpack('B',file.read(1))[0]
				image.append(float(byte))
			test_x.append(image)

	# testing images
	test_y = []
	with open("t10k-labels.idx1-ubyte","rb") as file:
		magicnumber = file.read(4)
		numLabels = struct.unpack('>i',file.read(4))[0]
		for i in range(0,numTrainImages):
			byte = struct.unpack('B',file.read(1))[0]
			test_y.append(int(byte))
		
	
	#############################
	######## TRAINING ###########
	#############################
	# print((X_test[0]))
	train_x = np.asarray(train_x)
	test_x = np.asarray(test_x)
	train_x = train_x / 255
	test_x = test_x /255

	# one hot encode outputs
	train_y = np_utils.to_categorical(train_y)
	test_y = np_utils.to_categorical(test_y)
	num_classes = test_y.shape[1]

	# create model
	model = Sequential()
	model.add(Dense(train_num_pixels, input_dim=train_num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(train_x,train_y, validation_data=(test_x, test_y))

	#############################
	######### Testing ###########
	#############################
	scores = model.evaluate(test_x, test_y, verbose=0)
	print("Error: %.2f%%" % (100-scores[1]*100))
	
	#############################
	######## Validating #########
	#############################
	images = []
	labels = []

	for i in range(0,5):
		f = plt.figure(i)
		image = test_x[i]
		image = np.reshape(image, (28, 28))
		plt.imshow(image, cmap='gray_r')
		plt.title('Digit Label: {}'.format(test_y[i]))

	plt.show()

