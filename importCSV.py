
import numpy as np
import csv

file = 'numbs.csv'


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

def normalize(array):
	tempFactor = 273.15+60
	precipFactor = 100
	normFactor = np.array([precipFactor, tempFactor, precipFactor, tempFactor, precipFactor, tempFactor])
	array_f = array.astype(np.float)
	
	array_f = array_f/normFactor
			
	return array_f

a = array_csv(file)
d = normalize(a)


print (d)
