import random





def generate_data(number_of_instances):
	x = []
	y = []
	for i in range(number_of_instances):
		feature = []
		feature.append(random.randint(1,3))
		feature.append(random.randint(1,3))
		feature.append(random.randint(1,3))
		x.append(feature)
		target = []
		target.append(random.uniform(0,1))
		target.append(random.uniform(0,1))
		target.append(random.uniform(0,1))
		y.append(target)

	return x,y