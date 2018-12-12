import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import codecs
def get_interesting_words(word_results_file, people):
	with open(word_results_file) as f:
		res = f.readlines()

		res = [r.strip() for r in res] 
		res = [r for r in res if r not in (' \n', "")]

		all_words = set()

		all_words = all_words.union(people)
		dic = {}
		for string in res:
			
			name, words = string.split(" : ")
			words = str(words[1:-1])
			terms = words.split(", ")

			new_word = []

			for i in terms:
				new_word.append(i[1:-1])
			dic[name] = new_word 
					


		for name in people:
			words = set(dic[name])
			all_words = all_words.union(words)

	return all_words

def get_word_cloud_input(word_results_file, person):
	with open(word_results_file) as f:
		res = f.readlines()

		res = [r.strip() for r in res] 
		res = [r for r in res if r not in (' \n', "")]
		

		dic = {}
		for string in res:
			
			name, words = string.split(" : ")
			words = str(words[1:-1])
			terms = words.split(", ")

			new_word = []

			for i in terms:
				new_word.append(i[1:-1])
			dic[name] = new_word 

		dic = correct_keys(dic)					
		words = dic[person]

		weights = {}

		weight = 20 
		for word in words:
			weights[word] = weight
			weight -=1
		print(weights)
		weights = correct_keys(weights)
		s = ""
		for key in weights:
			for num in range(weights[key]):
				s += key + ", "

		for i in range(50):
			s += person + ", "

	return s 

def correct_keys(dic):

	new_dic = {}
	for i in dic:
		if i == "buenda":
			new_dic["buendia"] = dic[i]
		elif i == "rsula":
			new_dic["ursula"]= dic[i] 

		elif i == "jos":
			new_dic["jose"] = dic[i]	

		elif i == "weve":
			new_dic["we've"] = dic[i]

		elif i == "melquades":
			new_dic["melquiades"] = dic[i]
		else:
			new_dic[i] = dic[i]
	return new_dic 

def correct_labels(labels):

	for i in range(len(labels)):
		if labels[i] == "buenda":
			labels[i] = "buendia"
		if labels[i] == "rsula":
			labels[i] = "ursula"

		if labels[i] == "jos":
			labels[i] = "jose"

		if labels[i] == "weve":
			labels[i] = "we've"

		if labels[i] == "melquades":
			labels[i] = "melquiades"

	return labels
def tsne_plot_subset(word_vector_file, word_set):
	with open(word_vector_file) as f:
		vecs = f.readlines()

	tokens= []
	labels = []
	for embed in vecs:
		vals = embed.split(" ")
		if vals[0] in word_set:
			labels.append(vals[0])
			vals.pop(0)
		
			for i in range(len(vals)):
				vals[i] = float(vals[i])

			x = np.array(vals)
			tokens.append(x)



	labels = correct_labels(labels) 
	print(labels)
	y = np.array(tokens)
	tsne_model = TSNE(perplexity=60, n_components=2, init='pca', n_iter=2500, random_state=23)
	new_values = tsne_model.fit_transform(y)

	x = []
	y = []

	for value in new_values:
		x.append(value[0])
		y.append(value[1])


	plt.figure(figsize=(16,16))

	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.text(x[i] *(1 + 0.01), y[i] * (1 + 0.01), labels[i], fontsize = 10) 
		print(labels[i])
	
	plt.show()




def tsne_plot(word_vector_file):
	with open(word_vector_file) as f:
		vecs = f.readlines()

	tokens= []
	labels = []
	for embed in vecs:
		vals = embed.split(" ")
		labels.append(vals[0])
		vals.pop(0)
	
		for i in range(len(vals)):
			vals[i] = float(vals[i])

		x = np.array(vals)
		tokens.append(x)


	y = np.array(tokens)
	tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=2500, random_state=23)
	new_values = tsne_model.fit_transform(y)
	
	x = []
	y = []

	for value in new_values:
		x.append(value[0])
		y.append(value[1])


	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.annotate(labels[i], xy = (x[i], y[i]), textcoords = 'offset points', ha='right', va='bottom')
		print(labels[i])
	
	plt.axis([-200, 200, -200, 200])
	plt.show()


if __name__ == '__main__':
	people = set(["aureliano", 'rsula', 'jos', 'arcadio', 'buenda', 'remedios', 'colonel',
	'amaranta', 'macondo', 'fernanda', 'rebeca', 'melquades', 'meme', 'petra', 'crespi',
	'pilar', 'ternera', 'moscote', 'mauricio', 'gypsy'])


#	word_set = get_interesting_words("word_associations.txt",  people)
#	tsne_plot_subset('word2vec_embeddings.txt', word_set)

	print(get_word_cloud_input("word_associations.txt", 'man'))       

	
