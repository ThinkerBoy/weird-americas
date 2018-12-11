
import numpy as np
import math
import sys
def distance(vec1, vec2):

	sq = np.square(vec2- vec1)

	sum_up = math.sqrt(np.sum(sq))

	return sum_up 
def make_comparator(word_vec):
    def compare(x, y):
        if distance(x[1], word_vec) < distance(y[1], word_vec):
            return 1
        elif distance(x[1], word_vec) > distance(y[1], word_vec):
            return -1
        else:
            return 0
    return compare

def parse_word_embeddings(file_name):

	with open(file_name) as f:
		lines = f.readlines()
	

	words = []
	vectors = []
	for line in lines:
		terms = line.split(" ")
		words.append(terms[0])
		
		
		terms.pop(0)

		for i in range(len(terms)):
			terms[i] = float(terms[i])

		x = np.asarray(terms)

		vectors.append(x)

	return words, vectors


class NearestNeighbor(object):

	def __init__(self, file_name):

		self.words, self.vectors = parse_word_embeddings(file_name)


	def get_vector(self, word):

		index = self.words.index(word)

		return self.vectors[index] 

	def get_closest_to_a_word(self, word, num):
	
		word_vec = self.get_vector(word)

		comparator = make_comparator(word_vec)

		zipped = zip(self.words, self.vectors)

		zipped.sort(cmp= comparator, reverse= True)

		words = []
		for i in range(num):
			words.append(zipped[i][0])
		
		
		return words


if __name__ == '__main__':

	nn = NearestNeighbor(sys.argv[1])

	f = open('results.txt', 'a')
	for word in nn.words:

		nearest = nn.get_closest_to_a_word(word, 20)[1:]

		msg = "{} : {} \n \n".format(word, nearest)

		f.write(msg)
	

			
