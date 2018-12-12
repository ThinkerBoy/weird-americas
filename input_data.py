"""

This code was modified from https://github.com/Adoni/word2vec_pytorch
"""
import numpy
from collections import deque
import re
import nltk
from nltk.corpus import stopwords 
import codecs
import string
numpy.random.seed(12345)
nltk.download('punkt')
nltk.download('stopwords')
class InputData:
    """Store data for word2vec, such as word map, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count):
        self.input_file_name = file_name
        self.get_words(min_count)
	self.count = 0
        self.word_pair_catch = deque()
        self.init_sample_table()
        print('Word Count: %d' % len(self.word2id))
	print('Sentence count: %d' % self.sentence_count)

    def get_words(self, min_count):
	
	self.input_file = codecs.open(self.input_file_name, encoding='iso-8859-1')	
	f = self.input_file.readlines()
	self.sentence_count = len(f)
	raw = " ".join(f)
	tokens = [t for t in nltk.word_tokenize(raw)]

	distinct_tokens = set(tokens)

	freq = nltk.FreqDist(tokens)

	wid = 0

	self.word2id = {}
	self.id2word = {}
	self.word_frequency = {}
	self.sentence_length = 0

	
	for w, c in freq.most_common(10654):
		if c <= min_count:
			continue
		self.word2id[w] = wid
		self.id2word[wid] = w
		self.word_frequency[wid] =c
		wid +=1
		self.sentence_length += c
	self.word_count = len(self.word2id)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    # @profile
    
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = codecs.open(self.input_file_name, encoding='iso-8859-1')
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.word2id[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
	return batch_pairs

# @profile
    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size


def test():
    a = InputData('./book2.txt')


if __name__ == '__main__':
    test()
