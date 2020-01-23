from gensim import utils
from gensim.models import Word2Vec, KeyedVectors

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = 'train_posts.csv'
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


sentences = MyCorpus()
model = Word2Vec(sentences=sentences, compute_loss=True, min_count=100)
model.wv.save_word2vec_format('./bin/gensim_model-min_count100.bin', binary=True)

loss = model.get_latest_training_loss()
print(loss)