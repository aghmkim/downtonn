import numpy as np
import scipy
from scipy import optimize
import pdb
import csv

def array_csv(filename):
		switch =1
		f = open(filename)
		try:
			reader = csv.reader(f)
			floats = []
			for i in range(1,5):
				next(reader) #skip header
			for row in reader:
				if switch>0:
					floats.append(row)
				switch = switch*-1
		finally:
			f.close()

		return np.array(floats)

def normalizeInput(array):
		tempFactor = 273.15+60
		precipFactor = 100
		normFactor = np.array([precipFactor, tempFactor, precipFactor, tempFactor, precipFactor, tempFactor])
		array_f = array.astype(np.float)
		
		array_f = array_f/normFactor
				
		return array_f

def normalizeOutput(array):
		tempFactor = 273.15+60
	
		array_f = array.astype(np.float)
		array_f = array_f/tempFactor
				
		return array_f

inputfile = 'numbs.csv'


a = array_csv(inputfile)
X = normalizeInput(a)

outputfile = 'output.csv'
b = array_csv(outputfile)
y = normalizeOutput(b)


# X = np.array(([0, 279, 0, 279.5, 0,	278.15], [0, 282.6, 0, 284.55, 0, 276.75],\
#  [0, 276.5, 0, 280.9, 0, 282.05], [0, 274.3, 0, 275.4, 0, 274.8], \
#  [56, 277.35, 58, 277.3, 33, 272.05]))

# tempFactor = 273.15+60
# precipFactor = 100
# normFactor = np.array([precipFactor, tempFactor, precipFactor, tempFactor, precipFactor, tempFactor])
# X = X/normFactor

# y = np.array([[278.15],[279.25],[283.7], [273.7], [272.05]])
# y = y/tempFactor


class Neural_Network_Weather(object):
	def __init__(self):
			self.InputLayerSize = 6
			self.HiddenLayerSize = 3
			self.OutputLayerSize = 1

			#Initialize weights for synapses
			self.Weights1 = np.random.randn(self.InputLayerSize, self.HiddenLayerSize) # n x 3
			self.Weights2 = np.random.randn(self.HiddenLayerSize, self.OutputLayerSize) # 3x2
			
			self.z2 = None
			self.a2 = None
			self.z3 = None
			self.output = None
		#Initialize Values Throughout The Network
	def forward(self, X):
			self.z2 = np.dot(X, self.Weights1) 			# 1 x n
			self.a2 = self.sigmoid(self.z2)				# 1 x n
			self.z3 = np.dot(self.a2, self.Weights2)	# 1 x 3
			self.output = self.sigmoid(self.z3)			# 1 x 3

			return self.output

	def sigmoid(self, z):
			return 1/(1+np.exp(-z))

	def sigmoid_derivative(self,z):
			return np.exp(-z)/((1+np.exp(-z))**2)

	def CostFunction(self, X, y):
			self.output = self.forward(X)
			C = 0.5*sum((y - self.output)**2)
			return C

	def CostFunctionDerivative(self, X, y):
	        #Compute derivative with respect to W and W2 for a given X and y:
	        self.output = self.forward(X)
	        
	        delta3 = np.multiply(-(y-self.output), self.sigmoid_derivative(self.z3))
	        dCostdW2 = np.dot(self.a2.T, delta3)
	        
	        delta2 = np.dot(delta3, self.Weights2.T)*self.sigmoid_derivative(self.z2)
	        dCostdW1 = np.dot(X.T, delta2)  
	        
	        return dCostdW1, dCostdW2

	def getParams(self):
		#Get W1 and W2 unrolled into vector:
		params = np.concatenate((self.Weights1.ravel(), self.Weights2.ravel()))
		return params

	def setParams(self, params):
		#Set W1 and W2 using single paramater vector.
		W1_start = 0
		W1_end = self.HiddenLayerSize*self.InputLayerSize
		self.Weights1 = np.reshape(params[W1_start:W1_end], (self.InputLayerSize , self.HiddenLayerSize))
		
		W2_end = W1_end + self.HiddenLayerSize*self.OutputLayerSize
		self.Weights2 = np.reshape(params[W1_end:W2_end], (self.HiddenLayerSize, self.OutputLayerSize))

	def  computeGradients(self,X,y):
		dCostdW1, dCostdW2 = self.CostFunctionDerivative(X, y)
		return np.concatenate((dCostdW1.ravel(), dCostdW2.ravel()))

	def ComputeNumericalGradient(N, X, y):
		paramsInitial = N.getParams()
		numgrad = np.zeros(paramsInitial.shape)
		perturb = np.zeros(paramsInitial.shape)
		e = 1e-4
		for p in range(len(paramsInitial)):
			perturb[p] = eN.setParams(paramsInitial+perturb)
			loss2 = N.CostFunction(X,y)

			N.setParams(paramsInitial-perturb)
			loss1 = N.CostFunction(X,y)

			numgrad[p] = (loss2 - loss1) / (2*e)

			perturb[p] = 0

		N.setParams(paramsInitial)
		return numgrad

#-----

class Trainer(object):
	def __init__(self, N):
		self.N = N

	def CallBack(self, params):
		self.N.setParams(params)
		self.C.append(self.N.CostFunction(self.X, self.y))

	def CostFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.CostFunction(X,y)
		gradient = self.N.computeGradients(X,y)

		return cost, gradient

	def Trainer(self, X, y):
		self.X = X
		self.y = y

		self.C=[] #list to store costs as time goes on

		params0 = self.N.getParams()

		options = {'maxiter': 400, 'disp': True}
		_res = optimize.minimize(self.CostFunctionWrapper, params0, jac = True, method = 'BFGS', args=(X,y), options=options, callback=self.CallBack)

		self.N.setParams(_res.x)
		self.optimizationResults = _res

model = Neural_Network_Weather()
model.CostFunction(X,y)

model2 = Trainer(model)
model2.Trainer(X,y)

dataout = model.forward(X)
error = y - dataout
print (error)
pdb.set_trace()

