import os 
from random import randint
from PIL import Image
import sys
import shutil


# Pourcentage des données de train et test par 
pourcentage_Train = 0.7
pourcentage_Test = 0.3
pourcentage_validation = 0.3 # 30 % de train 

# Noms des classes (Violence/ Non Violence)
dir = "/Users/sandisabrina/Documents/CEMTI/flask-project/data/"
List= ['violence','non_violence']

# Création de 3 dossiers train, validation et test 
if not os.path.exists(dir+"/train"):
	os.mkdir(dir+"/train")
if not os.path.exists(dir+"/test"):
	os.mkdir(dir+"/test")
if not os.path.exists(dir+"/validation"):
	os.mkdir(dir+"/validation")
# Creation des dossier avec le nom classes dans chaqu'un  
# Suppression des dossiers test, validation et train si ils existent deja 
for i in List:
	if not os.path.exists(dir+"/test"+i):
		os.mkdir(dir+"/test/"+i)
	else:
		shutil.rmtree(dir+"/test/"+i)
		os.mkdir(dir+"/test/"+i)
	if not os.path.exists(dir+"/train"+i):
		os.mkdir(dir+"/train/"+i)
	else:
		shutil.rmtree(dir+"/train/"+ i)
		os.mkdir(dir+"/train/"+ i)
	if not os.path.exists(dir+"/validation/"+i):
		os.mkdir(dir+"/validation/"+i)
	else:
		shutil.rmtree(dir+"/validation/"+ i)
		os.mkdir(dir+"/validation/"+ i)



# si  un element existe dans le tableau 
def exist_element_tab(element, tab):
	if(len(tab)==0):
		return 0
	for i in tab:
		if(element == i):
			return 1

	return 0 


# Préparation de l'ensemble de test et de train

def Creation_train_test():
	tab_train = []
	tab_test = []
	tab_validation = []
	tab_train_without_validation = []
	# Ensemble de train 
	# Puisque nos images 
	for i in range(len(List)):
		tab_train = []
		tab_test = []
		tab_validation = []
		tab_train_without_validation = []
		taille_Data = len(os.listdir(dir+"/"+ List[i]))
		r = randint(0,taille_Data-1)
		for j in range(int(taille_Data*pourcentage_Train)):
			while (exist_element_tab(r,tab_train)==1):
   				r = randint(0,taille_Data-1)
			tab_train.append(r)
		# Ensemble de test = taille ElementDataSet - Train
		
		for j in range(taille_Data):
			bool = 0
			for k in tab_train:
				if(j==k):
					bool = 1 

			if(bool==0):
				tab_test.append(j)

		 # je prend un certain pourcentage de données a partir des données d'entrainement (train)
		for j in range (int(float(len(tab_train))*pourcentage_validation)):
			tab_validation.append(tab_train[j])

		for j in tab_train:
			bool = 0
			for k in tab_validation:
				if(j==k):
					bool = 1 

			if(bool==0):
				tab_train_without_validation.append(j)



		for j in range(len(tab_train_without_validation)):
			chemin_depart = dir + '/'+ List[i] + '/' + str(tab_train_without_validation[j]) + '.jpg'
			chemin_destination = dir + '/'+ 'train' + '/' + List[i] + '/' + str(tab_train_without_validation[j])+ '.jpg'
			filepath = shutil.copy(chemin_depart,chemin_destination)
		
		for j in range(len(tab_validation)):
			chemin_depart = dir + '/'+ List[i] + '/' + str(tab_validation[j]) + '.jpg'
			chemin_destination = dir + '/'+ 'validation' + '/' + List[i] + '/' + str(tab_validation[j])+ '.jpg'
			filepath = shutil.copy(chemin_depart,chemin_destination)

		for j in range(len(tab_test)):
			chemin_depart = dir + '/'+ List[i] + '/' + str(tab_test[j])+ '.jpg'
			chemin_destination = dir + '/'+ 'test' + '/' + List[i] + '/'+ str(tab_test[j])+ '.jpg'
			filepath = shutil.copy(chemin_depart,chemin_destination)


Creation_train_test()


