import operator

class KNN(object):
	def __init__(self, K=3):
		self.K = K
		self.podaci_za_treniranje = []
		self.podaci_za_testiranje = []

	def postavi_podatke_za_treniranje(self,podaci):
		self.podaci_za_treniranje = podaci 

	def postavi_podatke_za_testiranje(self,podaci):
		self.podaci_za_testiranje = podaci 

	def euklidova_distanca(self,s1,s2,dimenzija):       
		distanca = 0 
		for x in range(dimenzija):
			distanca += (s1[x] - s2[x])**2
		return distanca

		
	def klasificiraj_instancu(self,susjedi):
		klase = {} 
		for x in range(len(susjedi)):
			klasa = susjedi[x][-1]
			if klasa in klase:
				klase[klasa] += 1
			else:
				klase[klasa] = 1

		rezultat = sorted(klase,key=klase.__getitem__,reverse = True)
		return rezultat[0]

		

	def vrati_susjede(self,susjed):
		euklidske_udaljenosti = []
		d = len(self.podaci_za_treniranje[0]) - 1
		for x in range(len(self.podaci_za_treniranje)):
			euklidova_distanca = self.euklidova_distanca(susjed,self.podaci_za_treniranje[x],d)
			euklidske_udaljenosti.append((self.podaci_za_treniranje[x],euklidova_distanca))

		euklidske_udaljenosti.sort(key=operator.itemgetter(1))
		susjedi = []
		for x in range(self.K):
			susjedi.append(euklidske_udaljenosti[x][0])
		return susjedi 


	def napravi_predikciju(self):
		for x in range(len(self.podaci_za_testiranje)):
			susjedi = self.vrati_susjede(self.podaci_za_testiranje[x])
			rezultat = self.klasificiraj_instancu(susjedi)
			print('Instanca ' + str(self.podaci_za_testiranje[x]) + ' pripada klasi: ' + str(rezultat) + ' kada se uzme da je K= ' + str(self.K))
			




knn = KNN()

podaci_za_treniranje = [ [4,4,0,1], [4,3,1,1], [6,0,2,1], [5,2,2,0], [5,1,1,0], [7,2,0,0] ]
knn.postavi_podatke_za_treniranje(podaci_za_treniranje)

podaci_za_testiranje = [ [4,2,1], [0,3,3] ]
knn.postavi_podatke_za_testiranje(podaci_za_testiranje)

knn.napravi_predikciju()

