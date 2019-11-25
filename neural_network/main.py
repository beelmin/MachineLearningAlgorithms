import neural_network
import data





learning_rate = float(input("Unesite learning rate: "))
print("Odaberite aktivacionu f-ju za skriveni i izlazni sloj:")
print("1 - sigmoidna f-ja")
print("2 - tanges hiperbolna f-ja")
print("3 - RLU f-ja")
print("4 - LRLU f-ja")
activation_function_for_hidden = int(input("Vas izbor za skriveni sloj: "))
activation_function_for_output = int(input("Vas izbor za izlazni sloj: "))



nn = neural_network.NeuralNetwork(learning_rate,activation_function_for_hidden,activation_function_for_output)


# training neural network 
number_of_iterations = 1000
number_of_instances = 100
X,Y = data.generate_data(number_of_instances)
nn.train(X,Y,number_of_iterations)


#testing neural network
x = [[3,1,2]]
y = [[0.47, 0.66, 0.51]]
nn.test(x,y)







	
	





