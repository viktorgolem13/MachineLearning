import numpy as np

def sim_log_data(x1, y1, n1, sd1, x2, y2, n2, sd2):
	import pandas as pd
	import numpy.random as nr
	
	wx1 = nr.normal(loc = x1, scale = sd1, size = n1)
	wy1 = nr.normal(loc = y1, scale = sd1, size = n1)
	
	z1 = [1]*n1
	
	wx2 = nr.normal(loc = x2, scale = sd2, size = n2)
	wy2 = nr.normal(loc = y2, scale = sd2, size = n2)
	z2 = [0]*n2
	
	df1 = pd.DataFrame({'x' : wx1, 'y' : wy1, 'z' : z1})
	df2 = pd.DataFrame({'x' : wx2, 'y' : wy2, 'z' : z2})
	
	return pd.concat([df1, df2], axis = 0, ignore_index = True)
	
def plot_class(df):
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize = (8, 8))
	fig.clf()
	ax = fig.gca()
	
	df[df.z == 1].plot(kind = 'scatter', x = 'x', y = 'y', ax = ax, alpha = 1.0, color = 'Red', marker = 'x', s = 40)
	
	df[df.z == 0].plot(kind = 'scatter', x = 'x', y = 'y', ax = ax, alpha = 1.0, color = 'DarkBlue', marker = 'o', s = 40)
	
	plt.show()

class MyKNearestNeighbour:

	def __init__(self, k = 5):
		self.k = k
		

	def fit(self, X, y):
		self.Xtr = X
		self.ytr = y
		return self


	def predict(self, X):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		for i in range(num_test):
			distances = np.sum((self.Xtr - X[i,:])**2, axis = 1)

			min_index = np.argmin(distances)
				
			Ypred[i] = self.ytr[min_index]

		return Ypred

	def predict2(self, X):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		for i in range(num_test):
			distances = np.sum((self.Xtr - X[i,:])**2, axis = 1)

			minKdistances = np.empty(self.k)
			minKdistances[:] = np.NAN

			minIndex = []

			l = -1
			for distance in distances:
				l = l+1
				for j in range(self.k):
					if (minKdistances[j] == np.NAN) or (distance < minKdistances[j]):
						minKdistances[j] = distance
						minIndex.append(l)
						break
						
			br = 0
			for index in minIndex:
				if self.ytr[index] == self.ytr[0]:
					br += 1

			if br > self.k//2:
				Ypred[i] = self.ytr[0]
			else:
				j = 1
				while self.ytr[j] == self.ytr[0]:
					j+=1
				Ypred[i] = self.ytr[j]

		return Ypred
	
	
def classificationAlgoritam(df, dfTest, features, numOfFeatures, targetedClass, classifier):

    nrow = df.shape[0]
    nrowTest = dfTest.shape[0]
    
    X = df[features].as_matrix().reshape(nrow, numOfFeatures)
    XTest = dfTest[features].as_matrix().reshape(nrowTest, numOfFeatures)
    
    Y = df[targetedClass].as_matrix().ravel()
    
    trainedClassifier = classifier.fit(X, Y)
    
    dfTest['predicted'] = trainedClassifier.predict(XTest)
    
    return dfTest
	
def eval_logistic(df):
	import matplotlib.pyplot as plt
	import pandas as pd
	
	truePos = df[( (df['predicted'] == 1) & (df['z'] == df['predicted']) )]
	falsePos = df[((df['predicted'] == 1) & (df['z'] != df['predicted']))]
	trueNeg = df[((df['predicted'] == 0) & (df['z'] == df['predicted']))]
	falseNeg = df[((df['predicted'] == 0) & (df['z'] != df['predicted']))]
	
	fig = plt.figure(figsize = (8, 8))
	fig.clf()
	ax = fig.gca()
	
	truePos.plot(kind = 'scatter', x = 'x', y = 'y', ax = ax, alpha = 1.0, color = 'DarkBlue', marker = 'x', s = 40)
	
	falsePos.plot(kind = 'scatter', x = 'x', y = 'y', ax = ax, alpha = 1.0, color = 'Red', marker = 'o', s = 40)
	
	trueNeg.plot(kind = 'scatter', x = 'x', y = 'y', ax = ax, alpha = 1.0, color = 'DarkBlue', marker = 'o', s = 40)
	
	falseNeg.plot(kind = 'scatter', x = 'x', y = 'y', ax = ax, alpha = 1.0, color = 'Red', marker = 'x', s = 40)
	
	TP = truePos.shape[0]
	FP = falsePos.shape[0]
	TN = trueNeg.shape[0]
	FN = falseNeg.shape[0]
	
	                                             
	confusion = pd.DataFrame({'Negative' : [TN, FN], 'Positive' : [FP, TP]}, index = ['TrueNeg', 'TruePos'])
	
	print(confusion)

	plt.show()
	
from sklearn.model_selection import train_test_split
from sklearn import neighbors

df = sim_log_data(1, 1, 100, 1, -1, -1, 100, 1)
dfTrain, dfTest = train_test_split(df, train_size = 0.25)
#classifier = neighbors.KNeighborsClassifier()
classifier = MyKNearestNeighbour()
dfTest = classificationAlgoritam(dfTrain, dfTest, ['x', 'y'], 2, 'z', classifier)
eval_logistic(dfTest)