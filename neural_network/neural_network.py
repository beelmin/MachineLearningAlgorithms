import math 
import numpy as np

class NeuralNetwork:
	def __init__(self,learning_rate,activation_function_hidden,activation_function_output):
		self.input_nodes = 3
		self.hidden_nodes = 3
		self.output_nodes = 3
		self.inputs = []
		self.hidden = []
		self.copy_of_hidden = []
		self.outputs = []
		self.copy_of_outputs = []
		self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes, self.hidden_nodes))
		self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
		self.learning_rate = learning_rate
		self.activation_function_hidden = activation_function_hidden
		self.activation_function_output = activation_function_output

		

	def sigmoid(self,x):
		return 1 / (1 + math.exp(-x))

	def tanh(self,x):
		return (math.exp(x)-math.exp(-x)) / (math.exp(x)+math.exp(-x))

	def rlu(self,x):
		return max(0,x)

	def lrlu(self,x):
		return max(0.01*x,x)

	def derivate_of_sigmoid(self,x):
		result = self.sigmoid(x) * (1 - self.sigmoid(x))
		return result 

	def derivate_of_tanh(self,x):
		result = 1 - (self.tanh(x))**2
		return result 

	def derivate_of_rlu(self,x):
		if(x < 0):
			return 0
		else:
			return 1 

	def derivate_of_lrlu(self,x):
		if(x < 0):
			return 0.01
		else:
			return 1 


	def get_activation_function(self,activation_function,x):
		if(activation_function == 1):
			return self.sigmoid(x)
		elif(activation_function == 2):
			return self.tanh(x)
		elif(activation_function == 3):
			return self.rlu(x)
		else:
			return self.lrlu(x)

	def get_derivate_of_activation_function(self,activation_function,x):
		if(activation_function == 1):
			return self.derivate_of_sigmoid(x)
		elif(activation_function == 2):
			return self.derivate_of_tanh(x)
		elif(activation_function == 3):
			return self.derivate_of_rlu(x)
		else:
			return self.derivate_of_lrlu(x)



	def forward_pass(self,features):

		self.inputs = []
		for i in range(len(features)):
			self.inputs.append(features[i])

		self.hidden = []
		self.copy_of_hidden = []
		# for every hidden nodes
		for i in range(self.hidden_nodes):
			sum = 0.0
			# for every input nodes
			for j in range(self.input_nodes):
				sum += self.inputs[j] * self.weights_input_to_hidden[j][i]

			self.hidden.append(self.get_activation_function(self.activation_function_hidden,sum))
			self.copy_of_hidden.append(sum)

		self.outputs = []
		self.copy_of_outputs = []
		#for every output nodes 
		for i in range(self.output_nodes):
			sum = 0.0
			#for every hidden nodes
			for j in range(self.hidden_nodes):
				sum += self.hidden[j] * self.weights_hidden_to_output[j][i]

			self.outputs.append(self.get_activation_function(self.activation_function_output,sum))
			self.copy_of_outputs.append(sum)


	def backpropagation(self,targets):

		delta_hidden = []
		delta_outputs = []

		#calculate error for every output nodes
		for i in range(self.output_nodes):
			delta_outputs.append(self.get_derivate_of_activation_function(self.activation_function_output,self.copy_of_outputs[i]) * 2 *(targets[i] - self.outputs[i] ))


		#calculate error for every hidden nodes 
		for i in range(self.hidden_nodes):
			error = 0.0
			#for every output nodes 
			for j in range(self.output_nodes):
				error += self.weights_hidden_to_output[i][j] * delta_outputs[j]

			delta_hidden.append(self.get_derivate_of_activation_function(self.activation_function_hidden,self.copy_of_hidden[i])*error)


		#update weights between hidden and output 
		#for every output nodes 
		for i in range(self.output_nodes):
			#for every hidden nodes 
			for j in range(self.hidden_nodes):
				self.weights_hidden_to_output[j][i] += self.learning_rate * delta_outputs[i] * self.hidden[j]


		#update weights between input and hidden
		#for every hidden nodes 
		for i in range(self.hidden_nodes):
			#for every input nodes 
			for j in range(self.input_nodes):
				self.weights_input_to_hidden[j][i] += self.learning_rate * delta_hidden[i] * self.inputs[j] 



	def train(self,features,targets,number_of_iterations):

		for i in range(number_of_iterations):
			for j in range(len(features)):
				self.forward_pass(features[j])
				self.backpropagation(targets[j])
				
			

	def test(self,features,targets):

		for i in range(len(features)):
			self.forward_pass(features[i])
			print("Ocekivano: ", targets[i])
			print("Neuralna: ", self.outputs)
			print()








			

