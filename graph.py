import re
import nltk
import codecs
from operator import itemgetter
import pandas as pd
import string
def build_regexp(c):
    """
    Creates the appropriate regex for the expressions
    N(M) or N,M
    """
    if '(' in c:
        a, b = c.split(' ')
        b = b[1:-1]
        r = r"{}( {})?".format(a, b)
        return r
    if ',' in c:
        a, b = c.split(',')
        r = r"({}|{})".format(a, b)
        return r    
    return r"{}".format(c)


## save everything
with codecs.open('characters_list.txt', encoding='iso-8859-1') as x:
    characters = [l.strip().lower() for l in x]

with codecs.open('processed_book.txt', encoding='iso-8859-1') as x:
    text = [l.strip() for l in x]
raw = ' '.join(text)

## map the characters
chars = {}

for it, c in enumerate(characters):
    chars[it] = c    
    template = '{}'.format(it)
    regexp = re.compile(build_regexp(c))
    raw = re.sub(regexp, template, raw)




class Graph:

	def __init__(self):
		self.vertices = set()
		self.edges = {}

	def add_node(self, node):
		self.vertices.add(node)
		self.edges[node] = {}
	def add_edge(self, src_node, target_node, weight):
		if target_node not in self.edges[src_node]:
			self.edges[src_node][target_node] = {}
			self.edges[src_node][target_node]['weight'] = 0
			self.edges[src_node][target_node]['weight'] += weight

		else:
			self.edges[src_node][target_node]['weight'] += weight


	def nodes(self):
		return self.vertices

	def remove_node(self, node):

		self.vertices.remove(node)

		del self.edges[node]

		for n in self.vertices:
			if node in self.edges[n]:
				del self.edges[n][node]

	def rename_node(self, new_name, old_name):
		new_name =new_name
		old_name = old_name.lower()
		if old_name in self.vertices:
			self.vertices.remove(old_name)
			self.vertices.add(new_name)

		if old_name in self.edges:
			self.edges[new_name] = self.edges[old_name]
			del self.edges[old_name]
		for n in self.edges:
			if old_name in self.edges[n]:
				self.edges[n][new_name] = self.edges[n][old_name]
				del self.edges[n][old_name]


	def capitalize(self):

		vertices = set()
		for n in self.vertices:
			names = n.split(" ")
			capital = []
			for i in names:
				capital.append(i.capitalize())

			new_name = " ".join(capital)

			vertices.add(new_name)

		edges = {}

		mapping = {}
		for k in self.edges:
			names = k.split(" ")
			capital = []
			for i in names:
				capital.append(i.capitalize())

			new_name = " ".join(capital)

			edges[new_name] = {}	

			mapping[k] = new_name
		

		for k in self.edges:
			edges[mapping[k]] = []
			for n in self.edges[k]:

				names = n.split(" ")
				capital = []
				for i in names:
					capital.append(i.capitalize())

				new_name = " ".join(capital)

				edges[mapping[k]].append(new_name)	

		
		self.edges = edges
		self.vertices = vertices
# build the graph
g = Graph()
# add nodes

for c in characters:
	g.add_node(c)
	print(c)

# tokenize the text
words = [n for n in nltk.word_tokenize(raw) if n != ',' and n != '.']

print(words)
# utils list
characters_rep = [str(i) for i in range(len(characters))]

print(characters_rep)
# forward threshold
fwd_t = 30
# check for each character
for it, c in enumerate(characters):
    for i, word in enumerate(words):
        if word == str(it):
            for d in range(i, i + fwd_t + 1):
                if d < len(words):
                    if words[d] in characters_rep and words[d] != word:
                        src_node = chars[int(word)]
                        target_node = chars[int(words[d])]
                        if target_node not in g.edges[src_node]:
                            g.add_edge(src_node, target_node, weight=1)
                        else:
                            g.edges[src_node][target_node]['weight'] += 1

# remove nodes w/o edges
removed = set()
for node in g.nodes():
    if not g.edges[node]:
        print('Node w/o edges:', node)
        removed.add(node)

for node in removed:
        g.remove_node(node)
print('Total characters minus solitude nodes:', len(g.nodes()))

tups = [("Melquades", "Melquiades"), ("Jos Arcadio Buenda", 'Jose Arcadio Buendia'), ("Colonel Aureliano Buenda", 'Colonel Aureliano Buendia'), ("Visitacin", 'Visitacion'), ("Seora (Moscote)", "Senora (Moscote)"),("Santa Sofa de la Piedad", "Santa Sofia de la Piedad"), ("Magnfico (Visbal)", "Magnifico (Visbal)"), ("Gerineldo (Mrquez)", "Gerineldo (Marquez)"), ("Colonel Gerineldo Mrquez", "Colonel Gerineldo Marquez"), ("Jos Arcadio Segundo", "Jose Arcadio Segundo"), ("Jos Raquel Moncada", "Jose Raquel Moncada"), ("lvaro", "Alvaro"), ("Germn", "German"), ("Doa Fernanda del Carpio de Buenda", "Dona Fernanda del Carpio de Buendia"), ("rsula (Iguarn)", "Ursula (Iguaran)")]
for i in tups:
	old_name, new_name = i

	g.rename_node(new_name, old_name)

g.capitalize()
import networkx as nx 
import matplotlib.pyplot as plt
graph = nx.Graph()
for i in g.edges:
	for j in g.edges[i]:
		graph.add_edge(i, j)


node_and_degree = graph.degree()

colors = {}
colors['ID'] = []
colors['myvalue'] = []

for node, degree in node_and_degree:

	if degree > 30:
		colors['ID'].append(node)
		colors['myvalue'].append(4)
	elif degree > 25:
		colors['ID'].append(node)
		colors['myvalue'].append(3.8)

	elif degree > 20:
		colors['ID'].append(node)
		colors['myvalue'].append(3.6)
	elif degree > 15:
		colors['ID'].append(node)
		colors['myvalue'].append(3.4)

	elif degree > 10:
		colors['ID'].append(node)
		colors['myvalue'].append(3.2)

	elif degree > 5:
		colors['ID'].append(node)
		colors['myvalue'].append(3)

	else:
		colors['ID'].append(node)
		colors['myvalue'].append(2.8)



(largest_hub, degree) = sorted(node_and_degree, key = itemgetter(1))[-1]

hub_ego = nx.ego_graph(graph, largest_hub)

colors = pd.DataFrame(colors)
colors = colors.set_index('ID')
colors = colors.reindex(hub_ego.nodes())
pos = nx.spring_layout(hub_ego, scale=5)


nx.draw_networkx_edges(hub_ego, pos, alpha=0.2)
nx.draw_networkx_labels(hub_ego, pos)  
nx.draw_networkx_nodes(hub_ego, pos, node_color=colors['myvalue'],  cmap= plt.cm.jet, node_size=500, with_labels= False)

plt.show()
