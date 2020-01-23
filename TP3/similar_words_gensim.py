from gensim.models import Word2Vec, KeyedVectors

wv_from_bin = KeyedVectors.load_word2vec_format("./bin/gensim_model-min_count100.bin", binary=True)

def get_similar_word_count(threshold=0.6):
    for i, word in enumerate(wv_from_bin.index2word):
        yield [word, len([x for x,y in wv_from_bin.similar_by_word(word, topn=wv_from_bin.vectors.shape[0]) if y > threshold])]

most_similars_precalc = {word : wv_from_bin.similar_by_word(word, topn=10) for word in wv_from_bin.index2word}
sorted_words = sorted(list(get_similar_word_count()), reverse=True, key=lambda kv: kv[1])

with open("res-gensim_model-min_count100.txt", "w") as file:
    for word, count in sorted_words:
        similar_words = ' '.join([w[0] for w in most_similars_precalc[word]])
        file.write(word + "\t\t" + str(count) + " " + similar_words + "\n")