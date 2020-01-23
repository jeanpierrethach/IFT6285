import gensim
from gensim.test.utils import datapath
from gensim import utils
import time

# Load pre-trained models
# wv = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# Train model from corpus
class Corpus(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
    	for line in open(self.filename, encoding="utf8"):
    		yield line.split()

# Train model
sentences = Corpus('data/train_posts.txt')
start = time.time()
model = gensim.models.Word2Vec(sentences, min_count=10)
end = time.time()
print(end - start)

# Save model to convert to spacy
model.wv.save_word2vec_format("models/gensim-model-min-count-10.txt")