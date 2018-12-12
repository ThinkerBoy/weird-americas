import re
import nltk


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
with open('characters_list.txt') as x:
    characters = [l.strip() for l in x]

with open('text') as x:
    text = [l.strip() for l in x]
raw = ' '.join(text)

## map the characters
chars = {}

for it, c in enumerate(characters):
    chars[it] = c    
    template = '{}'.format(it)
    regexp = re.compile(build_regexp(c))
    raw = re.sub(regexp, template, raw)


import networkx as nx

print('Total characters:', len(characters))

# build the graph
g = nx.Graph()
# add nodes
for c in characters:
    g.add_node(c)

# tokenize the text
words = [n for n in nltk.word_tokenize(raw) if n != ',' and n != '.']

# utils list
characters_rep = [str(i) for i in range(len(characters))]

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
                        if target_node not in g[src_node]:
                            g.add_edge(src_node, target_node, weight=1)
                        else:
                            g[src_node][target_node]['weight'] += 1

# remove nodes w/o edges
removed = set()
for node in g.nodes():
    if not g[node]:
        print('Node w/o edges:', node)
        g.remove_node(node)
        removed.add(node)

print('Total characters minus solitude nodes:', len(g.nodes()))

nx.write_graphml(g, 'output.graphml')

