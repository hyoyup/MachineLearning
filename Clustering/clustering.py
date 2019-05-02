# clustering.py

# Machine Learning - Clustering
# Hyoyup Chung
# 	- Old Faithful Geyser Data
#	- Source: http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat

# Run in command-line:
#	python clustering.py
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import copy

Epsilon = 0.00001
PI = 3.1415926535897932384626433

TrainSize = 250
numClusters = 2

# returns array of shuffled [train_set, test_set] 
def ShuffleData(csvData):
	# test_size = 0.08 gives 250 train 22 test split
	train, test = train_test_split(csvData, shuffle=True, test_size = 0.08)
	return [train, test]

def NormalDistribution(xi, sig_j, mu_j, phi_j):
	xi_minus_uj = xi - mu_j 
	xi_uj_sq = np.matmul(np.transpose(xi_minus_uj), xi_minus_uj)

	res = 1.0 / (2.0*PI*sig_j) * np.exp(-1.0 * xi_uj_sq / (2.0 * sig_j * sig_j))
	# print(res)
	return res

def Bayes(i, j, normdist_table, phi_table):
	summation = 0.0
	for index in range(0,numClusters):
		summation += normdist_table[i][index]*phi_table[index]
	res = normdist_table[i][j]*phi_table[j] / summation
	# print(normdist_table[i][j])
	return res

def Update(j, W_table, train_set, mu_table):
	Wij_xi_summation = 0.0
	Wij_summation = 0.0
	W_XiMu_summation = 0.0

	for i in range(0, TrainSize):
		xi = np.zeros(2)
		xi[0] = float(train_set[i:(i+1)][1])
		xi[1] = float(train_set[i:(i+1)][2])

		# Summation: W_ij * Xi
		Wij_xi_summation += W_table[i][j]*xi

		# Summation: W_ij
		Wij_summation += W_table[i][j]

		# Summation: W_ij * (Xi - Uj)^T(Xi - Uj)
		xi_minus_uj = xi - mu_table[j] 
		xi_uj_sq = np.matmul(np.transpose(xi_minus_uj), xi_minus_uj)
		W_XiMu_summation += W_table[i][j] * xi_uj_sq

	newMu = Wij_xi_summation / Wij_summation
	newSig = np.sqrt(W_XiMu_summation / Wij_summation)
	newPhi = Wij_summation / TrainSize
	return [newMu, newSig, newPhi]

def IsTrainingComplete(mu, sigma, phi, prevMu, prevSigma, prevPhi):
	mu_diff = [0.0, 0.0]
	sigma_diff = 0.0
	phi_diff = 0.0
	for j in range(0, numClusters):
		mu_diff += np.absolute(mu[j] - prevMu[j])
		# print(mu)
		sigma_diff += np.absolute(sigma[j] - prevSigma[j])
		phi_diff += np.absolute(phi[j] - prevPhi[j])
		# print(mu_diff)
		if mu_diff[0]>Epsilon or mu_diff[1]>Epsilon or sigma_diff>Epsilon or phi_diff>Epsilon:
			return False
	return True

def NextEruption(input_erupTime, theta):
	return theta[0]+theta[1]*input_erupTime

# main:
if __name__ == "__main__":

	# Preprocess Data
	csvData = pd.read_csv("test.txt",skiprows=0, header=None, usecols=[1,2], delimiter='     ', engine='python')

	#############################
	######## TRAINING ###########
	#############################
	
	# Initialize
	mu = np.random.rand(numClusters,2) * 100.0
	sigma = np.full(numClusters,100.0)
	phi = np.full(numClusters, 1/numClusters)

	prev_mu = copy.deepcopy(mu)
	prev_sigma = copy.deepcopy(sigma)
	prev_phi = copy.deepcopy(phi)

	TrainingIncomplete = True
	iterNum = 0

	# EM Model Training
	while(TrainingIncomplete):
		iterNum+=1
		# Generate Data Set
		shuffled = ShuffleData(csvData)
		train_set = shuffled[0]
		test_set = shuffled[1]

		p_xi_zj_table = np.zeros((TrainSize,numClusters),float)
		W_ij_table = np.zeros((TrainSize,numClusters), float)

		# E-Step
		i = 0
		for row in train_set.iterrows():
			xi = [row[1][1], row[1][2]]
			for j in range(0,numClusters):
				p_xi_zj_table[i][j] = NormalDistribution(xi, prev_sigma[j], prev_mu[j], prev_phi[j])

			for j in range(0,numClusters):
				W_ij_table[i][j] = Bayes(i, j, p_xi_zj_table, prev_phi)
			i += 1

		# M-Step
		for j in range(0,numClusters):
			newParams = Update(j, W_ij_table, train_set, prev_mu)
			# update mu
			mu[j] = newParams[0]
			# update sigma
			sigma[j] = newParams[1]
			# # update phi
			phi[j] = newParams[2]

		# if training done exit training
		if (iterNum>30 or IsTrainingComplete(mu, sigma, phi, prev_mu, prev_sigma, prev_phi)):
			TrainingIncomplete = False
			continue
		else:# otherwise, copy mu, sigma, phi to prev and continue
			prev_mu = copy.deepcopy(mu)
			prev_sigma = copy.deepcopy(sigma)
			prev_phi = copy.deepcopy(phi)
		
	# Generate Clusters
	cluster1 = []
	cluster2 = []
	for row in train_set.iterrows():
		xi = [row[1][1],row[1][2]]
		cluster1_prob = NormalDistribution(xi, sigma[0], mu[0], phi[0])
		cluster2_prob = NormalDistribution(xi, sigma[1], mu[1], phi[1])
		if cluster1_prob > cluster2_prob:
			cluster1.append(xi)
		else:
			cluster2.append(xi)

	# Create Linear Model

	# Cluster1
	TrainingX_c1 = np.zeros((len(cluster1),2))
	TrainingY_c1 = np.zeros((len(cluster1),1))
	minX_c1 = float("inf")
	maxX_c1 = float("-inf")
	for i in range(0,len(cluster1)):
		if (minX_c1>cluster1[i][0]):
			minX_c1 = cluster1[i][0]
		if (maxX_c1<cluster1[i][0]):
			maxX_c1 = cluster1[i][0]
		TrainingX_c1[i] = [1, cluster1[i][0]]
		TrainingY_c1[i] = [cluster1[i][1]]

	xT = np.transpose(TrainingX_c1)
	xTx = np.matmul(xT,TrainingX_c1)
	xTx_inv = np.linalg.inv(xTx)

	xTy = np.matmul(xT,TrainingY_c1)

	theta_c1 = np.matmul(xTx_inv,xTy)

	# Cluster2
	TrainingX_c2 = np.zeros((len(cluster2),2))
	TrainingY_c2 = np.zeros((len(cluster2),1))
	minX_c2 = float("inf")
	maxX_c2 = float("-inf")
	for i in range(0,len(cluster2)):
		if (minX_c2>cluster2[i][0]):
			minX_c2 = cluster2[i][0]
		if (maxX_c2<cluster2[i][0]):
			maxX_c2 = cluster2[i][0]
		TrainingX_c2[i] = [1, cluster2[i][0]]
		TrainingY_c2[i] = [cluster2[i][1]]

	xT = np.transpose(TrainingX_c2)
	xTx = np.matmul(xT,TrainingX_c2)
	xTx_inv = np.linalg.inv(xTx)

	xTy = np.matmul(xT,TrainingY_c2)

	theta_c2 = np.matmul(xTx_inv,xTy)


	#############################
	######### Testing ###########
	#############################
	for row in test_set.iterrows():
		eruptionTime = row[1][1]
		prediction = NextEruption(eruptionTime, (theta_c1+theta_c2)/2.0)
		print("EruptionTime: %f Prediction for next eruption: %f Actual: %f" %(eruptionTime, prediction, row[1][2]))


	#############################
	####### Visualization #######
	#############################
	f1 = plt.figure(1)
	plt.title('Old Faithful Geyser Data - Trained')
	plt.xlabel('eruptions')
	plt.ylabel('waiting')

	for i in range(0,len(cluster1)):
		X = np.linspace(minX_c1,maxX_c1)
		plt.plot(cluster1[i][0],cluster1[i][1], 'r+')
		plt.plot(X, theta_c1[0]+theta_c1[1]*X, 'r-')
	for i in range(0,len(cluster2)):
		X = np.linspace(minX_c2,maxX_c2)
		plt.plot(cluster2[i][0],cluster2[i][1], 'b+')
		plt.plot(X, theta_c2[0]+theta_c2[1]*X, 'b-')
	plt.show()
